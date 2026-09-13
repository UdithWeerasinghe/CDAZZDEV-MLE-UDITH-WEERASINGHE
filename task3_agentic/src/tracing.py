"""
Task 3C - observability.

Produces `logs/agent_trace.jsonl`: one JSON object per tool call recording the
tool name, its input arguments, its output truncated to 200 characters, and the
wall-clock duration, exactly as the brief specifies.

DESIGN NOTES
------------
* JSONL, NOT JSON. A run that crashes halfway still leaves a valid, parseable
  trace, because every line is independently complete. A single JSON array
  written at the end is lost entirely if the process dies - which is precisely
  when you most want the trace.

* FLUSHED PER RECORD. Same reasoning. The cost is negligible at agent
  timescales (tool calls take hundreds of milliseconds; an fsync takes
  microseconds).

* TRUNCATION IS RECORDED, NOT HIDDEN. When output is cut to 200 characters we
  also store `output_full_length`, so a reader can tell the difference between
  "the tool returned 200 characters" and "the tool returned 40KB and you are
  seeing the first 200".

* THE TRACER NEVER RAISES. Observability that can break the pipeline it observes
  is worse than none. Every write is wrapped; a failure logs a warning and the
  agent continues.

* CORRELATION IDS. Each run gets a `run_id` and each call a monotonic `seq`, so
  concurrent or interleaved runs can be separated after the fact and the exact
  ordering of a run is recoverable.

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Build a JSONL agent tracer
# recording tool name, inputs, truncated output and duration per call, with
# a context manager API', Date: 2026-09-10
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

OUTPUT_TRUNCATION = 200          # Specified by the brief.
ARG_TRUNCATION = 500             # Arguments are usually short; be generous.
DEFAULT_TRACE_PATH = "task3_agentic/logs/agent_trace.jsonl"


def _stringify(value: Any) -> str:
    """Render any tool payload as a string without ever raising."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return repr(value)


@dataclass
class TraceRecord:
    """One tool invocation."""

    run_id: str
    seq: int
    agent: str
    tool: str
    inputs: dict[str, Any]
    output: str
    output_full_length: int
    duration_ms: float
    ok: bool
    error: str | None
    started_at: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "seq": self.seq,
                "timestamp": self.started_at,
                "agent": self.agent,
                "tool": self.tool,
                "inputs": self.inputs,
                "output": self.output,
                "output_truncated": self.output_full_length > len(self.output),
                "output_full_length": self.output_full_length,
                "duration_ms": self.duration_ms,
                "duration_s": round(self.duration_ms / 1000.0, 4),
                "ok": self.ok,
                "error": self.error,
            },
            ensure_ascii=False,
            default=str,
        )


@dataclass
class AgentTracer:
    """Append-only tracer shared by every tool in a run."""

    path: Path = field(default_factory=lambda: Path(DEFAULT_TRACE_PATH))
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    echo: bool = True                     # Print to the notebook as calls happen.
    records: list[TraceRecord] = field(default_factory=list)
    _seq: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- recording -----------------------------------------------------------
    @contextmanager
    def record(self, tool: str, inputs: dict[str, Any], agent: str = "agent") -> Iterator[dict]:
        """Time a tool call and write the trace record on exit.

        Usage:
            with tracer.record("get_news", {"ticker": "NVDA"}) as slot:
                result = do_work()
                slot["output"] = result

        The slot dict lets the caller hand back the output without the tracer
        needing to wrap the return value, which keeps the tool signatures clean.
        """
        self._seq += 1
        seq = self._seq
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        slot: dict[str, Any] = {"output": None}
        error: str | None = None

        try:
            yield slot
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            rendered = _stringify(slot.get("output"))
            record = TraceRecord(
                run_id=self.run_id,
                seq=seq,
                agent=agent,
                tool=tool,
                inputs=self._safe_inputs(inputs),
                output=rendered[:OUTPUT_TRUNCATION],
                output_full_length=len(rendered),
                duration_ms=duration_ms,
                ok=error is None,
                error=error,
                started_at=started_at,
            )
            self._write(record)

    def _safe_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Truncate long arguments and redact anything that smells like a secret."""
        safe: dict[str, Any] = {}
        for key, value in (inputs or {}).items():
            if any(marker in key.lower() for marker in ("key", "token", "secret", "password")):
                safe[key] = "<redacted>"
                continue
            rendered = _stringify(value)
            safe[key] = rendered[:ARG_TRUNCATION] + ("…" if len(rendered) > ARG_TRUNCATION else "")
        return safe

    def _write(self, record: TraceRecord) -> None:
        self.records.append(record)
        if self.echo:
            status = "ok " if record.ok else "ERR"
            print(
                f"  [trace {record.seq:>2}] {status} {record.agent:<8} {record.tool:<20} "
                f"{record.duration_ms:>8.1f}ms  {record.output[:70]}"
            )
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json() + "\n")
                handle.flush()
        except Exception as exc:  # noqa: BLE001 - never break the pipeline
            logger.warning("Trace write failed (continuing): %s", exc)

    # -- reporting -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Aggregate view for the notebook and the README screenshot."""
        if not self.records:
            return {"run_id": self.run_id, "calls": 0}

        per_tool: dict[str, dict[str, Any]] = {}
        for record in self.records:
            entry = per_tool.setdefault(
                record.tool, {"calls": 0, "failures": 0, "total_ms": 0.0}
            )
            entry["calls"] += 1
            entry["failures"] += 0 if record.ok else 1
            entry["total_ms"] += record.duration_ms

        for entry in per_tool.values():
            entry["mean_ms"] = round(entry["total_ms"] / entry["calls"], 1)
            entry["total_ms"] = round(entry["total_ms"], 1)

        return {
            "run_id": self.run_id,
            "calls": len(self.records),
            "failures": sum(not r.ok for r in self.records),
            "total_duration_s": round(sum(r.duration_ms for r in self.records) / 1000, 2),
            "distinct_tools_used": sorted({r.tool for r in self.records}),
            "per_tool": per_tool,
            "call_order": [f"{r.agent}:{r.tool}" for r in self.records],
        }

    def print_summary(self) -> None:
        summary = self.summary()
        print(f"\nTrace summary — run {summary['run_id']}")
        print("─" * 74)
        print(f"{summary['calls']} tool calls, {summary.get('failures', 0)} failed, "
              f"{summary.get('total_duration_s', 0)}s total")
        for tool, stats in sorted(summary.get("per_tool", {}).items()):
            print(f"  {tool:<24} {stats['calls']:>2} calls  "
                  f"mean {stats['mean_ms']:>7.1f}ms  failures {stats['failures']}")
        print(f"\nCall order: {' → '.join(summary.get('call_order', []))}")
        print(f"Trace file: {self.path}")

    def load_all(self) -> list[dict[str, Any]]:
        """Read back every record in the file, including earlier runs."""
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed trace line.")
        return rows
