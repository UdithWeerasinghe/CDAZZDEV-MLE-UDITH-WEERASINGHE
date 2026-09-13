"""
Offline verification of the Task 3 components that do not need a live LLM.

Covers the handoff schema contract, the report validators, the tracer's JSONL
output format, the persistent cache lifecycle, and the tool-restriction
declarations. The graph itself needs a model with tool calling, so it is
exercised in the notebook against live inference.

Run:  python task3_agentic/tests/test_task3_offline.py

# AI-ASSISTED: Claude (claude-sonnet-5), Prompt: 'Write offline tests for the
# Task 3 handoff schema, tracer, cache and tool restriction', Date: 2026-09-10
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "task3_agentic" / "src"), str(ROOT / "task1_financial" / "src")]

from memory import ResearchCache                                    # noqa: E402
from schemas import (                                               # noqa: E402
    ClarificationRequest, PriceSnapshot, QuantBrief, ResearchReport,
    Risk, SentimentLabel, SentimentReading, TrendRegime, VolatilityReading,
)
from tracing import AgentTracer                                     # noqa: E402
from tools import AGENT_A_TOOLS, AGENT_B_TOOLS, ALL_TOOL_NAMES, fail, ok  # noqa: E402

FAILURES = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  PASS {label:<48} {detail}")
    else:
        FAILURES += 1
        print(f"  FAIL {label:<48} {detail}")


def rejects(label: str, build, fragment: str) -> None:
    try:
        build()
    except ValidationError as exc:
        message = " | ".join(e["msg"] for e in exc.errors())
        check(label, fragment.lower() in message.lower(), message[:80])
        return
    check(label, False, "accepted when it should have been rejected")


def valid_brief(**overrides) -> QuantBrief:
    payload = dict(
        ticker="NVDA", as_of="2026-09-10",
        price=PriceSnapshot(current_price=181.9, sma_50=174.2, sma_200=158.4,
                            rsi_14=61.3, macd_histogram=1.42, bollinger_pct_b=0.71,
                            pct_from_52w_high=-9.1, ytd_return_pct=34.2, sessions_analysed=502),
        volatility=VolatilityReading(window_days=30, annualised_volatility=0.412,
                                     percentile_vs_two_year=68.0),
        sentiment=SentimentReading(label=SentimentLabel.POSITIVE, score=0.34,
                                   headlines_analysed=12, positive=7, negative=2, neutral=3),
        trend_regime=TrendRegime.UPTREND,
        quantitative_findings=[
            "SMA-50 at 174.20 sits 9.9% above SMA-200 at 158.40, a sustained golden-cross regime",
            "30-day annualised volatility of 41.2% is at the 68th percentile of its 2-year range",
        ],
        data_gaps=[],
        confidence=0.74,
    )
    payload.update(overrides)
    return QuantBrief(**payload)


# ---------------------------------------------------------------------------
def test_handoff_schema() -> None:
    print("\nHandoff schema (Task 3B — 8 marks)")
    brief = valid_brief()
    check("Valid brief accepted", brief.ticker == "NVDA",
          f"confidence {brief.confidence}, {len(brief.quantitative_findings)} findings")

    rejects(
        "Finding without a figure rejected",
        lambda: valid_brief(quantitative_findings=[
            "Momentum looks strong and the trend is healthy",
            "Volatility appears somewhat elevated versus history",
        ]),
        "contains no figure",
    )
    rejects("Fewer than two findings rejected",
            lambda: valid_brief(quantitative_findings=["RSI-14 at 61.3, below the 70 threshold"]),
            "at least 2")
    rejects("Out-of-range RSI rejected",
            lambda: valid_brief(price=PriceSnapshot(rsi_14=140.0)), "less than or equal to 100")
    rejects("Unknown field rejected (extra='forbid')",
            lambda: QuantBrief(**{**valid_brief().model_dump(), "smuggled_prose": "hello"}),
            "not permitted")

    check("data_gaps present even when empty", "data_gaps" in brief.model_dump(),
          "field is mandatory, so silence is explicit")

    text = brief.as_handoff_text()
    for token in ("QUANTITATIVE BRIEF", "PRICE AND TECHNICALS", "VOLATILITY",
                  "NEWS SENTIMENT", "QUANTITATIVE FINDINGS", "DECLARED DATA GAPS"):
        check(f"Handoff renders '{token}'", token in text)

    gapped = valid_brief(data_gaps=["Could not measure news sentiment: no news tool available"],
                         confidence=0.51)
    check("Declared gaps surface in the handoff text",
          "no news tool" in gapped.as_handoff_text(), "Agent B can see what was not measured")

    check("Brief round-trips through JSON",
          QuantBrief.model_validate(json.loads(brief.model_dump_json())).ticker == "NVDA",
          "serialisable across a process boundary")


def test_clarification_contract() -> None:
    print("\nCritique loop contract (Task 3B — 8 marks)")
    request = ClarificationRequest(
        field_in_question="volatility.percentile_vs_two_year",
        question="What percentile does the current 30-day volatility sit at versus two years?",
        why_it_matters="I need it to size the protective put and cannot fetch price data myself.",
    )
    check("Specific clarification accepted", "percentile" in request.question)
    rejects("Vague clarification rejected",
            lambda: ClarificationRequest(
                field_in_question="everything",
                question="Please tell me more about the analysis you did",
                why_it_matters="It would help me write a more complete report overall",
            ),
            "specific answerable question")


def test_report_validators() -> None:
    print("\nFinal report validators (Task 3A — 10 marks)")
    risks = [
        Risk(risk="Export licensing restrictions could curtail a material revenue channel",
             supporting_evidence="Reported in 3 of 12 retrieved headlines concerning "
                                 "regulatory review of export approvals",
             severity="high"),
        Risk(risk="Valuation compression if the 200-day moving average is breached",
             supporting_evidence="Price sits 9.9% above the 200-day at 158.40; a retest "
                                 "implies roughly 13% downside",
             severity="medium"),
        Risk(risk="Elevated realised volatility raises the cost of carrying the position",
             supporting_evidence="30-day annualised volatility of 41.2% at the 68th "
                                 "percentile of its 2-year range",
             severity="medium"),
    ]
    good_hedge = (
        "Buy 3-month 10%-out-of-the-money protective puts covering roughly 50% of the "
        "position. With 30-day realised volatility at 41.2% and sitting at the 68th "
        "percentile, option premia are elevated but not extreme, so financing the puts "
        "by selling 15%-out-of-the-money calls into a collar keeps net cost near zero "
        "while capping upside above the 52-week high."
    )
    report = ResearchReport(
        ticker="NVDA",
        financial_health_summary=(
            "The company is in a sustained uptrend, with the 50-day average 9.9% above "
            "the 200-day and year-to-date return of 34.2%. Momentum confirms the trend, "
            "with the MACD histogram positive at 1.42, though RSI at 61.3 leaves moderate "
            "headroom before overbought territory. News flow is net positive across "
            "twelve headlines, at a score of +0.34."
        ),
        top_three_risks=risks,
        hedge_strategy=good_hedge,
    )
    check("Complete report accepted", len(report.top_three_risks) == 3,
          f"{len(report.render().splitlines())} rendered lines")

    rejects("Two risks rejected (exactly three required)",
            lambda: ResearchReport(ticker="NVDA",
                                   financial_health_summary=report.financial_health_summary,
                                   top_three_risks=risks[:2], hedge_strategy=good_hedge),
            "at least 3")
    rejects("Hedge with no figure rejected",
            lambda: ResearchReport(ticker="NVDA",
                                   financial_health_summary=report.financial_health_summary,
                                   top_three_risks=risks,
                                   hedge_strategy="Consider hedging your exposure using "
                                                  "protective puts if you are concerned "
                                                  "about downside risk in this position."),
            "measured quantity")
    rejects("Hedge naming no instrument rejected",
            lambda: ResearchReport(ticker="NVDA",
                                   financial_health_summary=report.financial_health_summary,
                                   top_three_risks=risks,
                                   hedge_strategy="Reduce the position by 30% and hold more "
                                                  "cash given the 41.2% volatility reading "
                                                  "observed over the last 30 days here."),
            "concrete instrument")
    rejects("Risk with hand-waving evidence rejected",
            lambda: Risk(risk="General market conditions could deteriorate meaningfully",
                         supporting_evidence="Markets can go down as well as up over time",
                         severity="low"),
            "neither a figure nor a source")

    for section in ("FINANCIAL HEALTH SUMMARY", "TOP THREE RISKS", "HEDGE STRATEGY"):
        check(f"Rendered report contains '{section}'", section in report.render())


def test_tracer() -> None:
    print("\nObservability (Task 3C — 5 marks)")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent_trace.jsonl"
        tracer = AgentTracer(path=path, echo=False)

        with tracer.record("get_price_data", {"ticker": "NVDA", "period": "2y"}) as slot:
            time.sleep(0.02)
            slot["output"] = {"current_price": 181.9, "rsi_14": 61.3}

        with tracer.record("web_search", {"query": "NVDA outlook", "api_key": "sk-secret"}) as slot:
            slot["output"] = "x" * 5000

        try:
            with tracer.record("get_news", {"ticker": "ZZZZ"}) as slot:
                raise RuntimeError("feed unavailable")
        except RuntimeError:
            pass

        rows = [json.loads(line) for line in path.read_text().strip().splitlines()]
        check("One JSONL line per call", len(rows) == 3, f"{len(rows)} lines")

        required = {"tool", "inputs", "output", "duration_ms", "run_id", "seq", "ok"}
        check("Every brief-required field present", all(required <= set(r) for r in rows),
              "tool, inputs, output, duration")
        check("Output truncated to 200 chars", len(rows[1]["output"]) == 200,
              f"full length {rows[1]['output_full_length']} recorded separately")
        check("Truncation flagged", rows[1]["output_truncated"] is True)
        check("Secrets redacted from inputs", rows[1]["inputs"]["api_key"] == "<redacted>")
        check("Duration recorded", rows[0]["duration_ms"] >= 20,
              f"{rows[0]['duration_ms']}ms for a 20ms sleep")
        check("Failure recorded, not swallowed",
              rows[2]["ok"] is False and "feed unavailable" in rows[2]["error"])
        check("Sequence is monotonic", [r["seq"] for r in rows] == [1, 2, 3])
        check("Run id is shared across calls", len({r["run_id"] for r in rows}) == 1)

        summary = tracer.summary()
        check("Summary aggregates correctly",
              summary["calls"] == 3 and summary["failures"] == 1,
              f"{summary['distinct_tools_used']}")

        # A tracer whose file cannot be written must not break the agent.
        broken = AgentTracer(path=Path(tmp) / "nested", echo=False)
        broken.path = Path("/proc/definitely/not/writable/trace.jsonl")
        try:
            with broken.record("x", {}) as slot:
                slot["output"] = "ok"
            check("Unwritable trace path does not raise", True, "observability degrades safely")
        except Exception as exc:  # noqa: BLE001
            check("Unwritable trace path does not raise", False, str(exc)[:60])


def test_cache() -> None:
    print("\nPersistent memory (Task 3C — 5 marks)")
    with tempfile.TemporaryDirectory() as tmp:
        cache = ResearchCache(tmp, max_age_hours=24)

        miss = cache.load("NVDA")
        check("Cold lookup is a miss", not miss.hit, miss.reason)

        report = {"ticker": "NVDA", "financial_health_summary": "…", "top_three_risks": []}
        path = cache.save("NVDA", report, metadata={"tool_calls": 6})
        check("Cache file keyed by ticker AND date",
              path.name.startswith("NVDA_") and len(path.stem.split("_")[1]) == 10, path.name)

        hit = cache.load("NVDA")
        check("Warm lookup is a hit", hit.hit and hit.payload["report"]["ticker"] == "NVDA", hit.reason)
        check("Metadata round-trips", hit.payload["metadata"]["tool_calls"] == 6)

        forced = cache.load("NVDA", force_refresh=True)
        check("force_refresh bypasses the cache", not forced.hit, forced.reason)

        check("Different ticker is a miss", not cache.load("AAPL").hit, "no cross-contamination")

        stale_when = datetime.now(timezone.utc) - timedelta(days=2)
        stale_path = cache.save("MSFT", report, when=stale_when)
        payload = json.loads(stale_path.read_text())
        payload["cached_at"] = (datetime.now(timezone.utc) - timedelta(hours=31)).isoformat()
        stale_path.write_text(json.dumps(payload))
        stale = cache.load("MSFT", when=stale_when)
        check("Stale entry rejected", not stale.hit, stale.reason)

        corrupt = Path(tmp) / cache.path_for("TSLA").name
        corrupt.write_text("{ this is not valid json")
        result = cache.load("TSLA")
        check("Corrupt file is a miss, not a crash", not result.hit, result.reason[:52])

        check("Entries listed", len(cache.list_entries()) >= 3,
              f"{len(cache.list_entries())} files")
        check("Clear removes files", cache.clear("NVDA") == 1)


def test_tool_restriction() -> None:
    print("\nTool restriction (Task 3B — 8 marks)")
    check("All five tools declared", len(ALL_TOOL_NAMES) == 5, str(ALL_TOOL_NAMES))
    check("Agent A has exactly the quant tools",
          set(AGENT_A_TOOLS) == {"get_price_data", "calculate_volatility", "llm_sentiment"},
          str(AGENT_A_TOOLS))
    check("Agent B has exactly the qualitative tools",
          set(AGENT_B_TOOLS) == {"web_search", "get_news"}, str(AGENT_B_TOOLS))
    check("No overlap between the two agents",
          not (set(AGENT_A_TOOLS) & set(AGENT_B_TOOLS)), "disjoint sets")
    check("Union covers all five",
          set(AGENT_A_TOOLS) | set(AGENT_B_TOOLS) == set(ALL_TOOL_NAMES), "nothing orphaned")
    check("Agent A cannot reach web_search", "web_search" not in AGENT_A_TOOLS,
          "the tool object is never passed to its node")
    check("Agent B cannot reach get_price_data", "get_price_data" not in AGENT_B_TOOLS,
          "so every number must come through the handoff")


def test_headline_sanitisation() -> None:
    """Invisible characters in scraped headlines must not reach a JSON parser.

    The tool call that crashed a live run contained a literal \\xa0 pair
    ("Shiba\\xa0un\\xa0"). Normalising at the boundary is cheaper than defending
    against it in three separate parsers.
    """
    print("\nHeadline sanitisation (regression: live tool_use_failed)")
    from tools import _clean_headline  # noqa: PLC0415

    dirty = "Shiba un Coin​ Surges﻿  —  Analysts React"
    clean = _clean_headline(dirty)
    check("Non-breaking spaces removed", " " not in clean, repr(clean[:48]))
    check("Zero-width characters removed",
          not any(c in clean for c in "​‌‍﻿"), repr(clean[:48]))
    check("Line separators removed", " " not in clean)
    check("Whitespace collapsed", "  " not in clean, repr(clean))
    check("Text preserved", clean.startswith("Shiba un Coin Surges"), repr(clean))
    check("Empty input is safe", _clean_headline("") == "" and _clean_headline(None) == "")
    check("Result is JSON-safe", json.loads(json.dumps({"h": clean}))["h"] == clean)


def test_malformed_tool_call_detection() -> None:
    """The recovery path must recognise the provider's real rejection message.

    Verbatim from a live Groq run that killed the notebook before this was
    handled. If the classifier misses it, `invoke_with_recovery` re-raises and
    we are back to a crashed run.
    """
    print("\nMalformed tool-call recovery (Task 3A — 7 marks)")
    from agents import is_tool_call_parse_error  # noqa: PLC0415

    real = (
        "Error code: 400 - {'error': {'message': 'Failed to parse tool call arguments "
        "as JSON', 'type': 'invalid_request_error', 'code': 'tool_use_failed', "
        "'failed_generation': '{\"name\": \"llm_sentiment\", \"arguments\": "
        "{\"headlines\":[\"Nscale...\",\"Shiba\\xa0un\\xa0...\"}\"}'}}"
    )
    check("Real Groq tool_use_failed detected",
          is_tool_call_parse_error(RuntimeError(real)), "verbatim from the failing run")
    check("Alternate phrasing detected",
          is_tool_call_parse_error(ValueError("invalid tool call emitted by model")))

    for benign in ("Connection reset by peer",
                   "rate limit exceeded, please retry",
                   "model not found: llama-3.1-9999"):
        check(f"Not misclassified: {benign[:28]!r}",
              not is_tool_call_parse_error(RuntimeError(benign)))


def test_tool_envelope() -> None:
    print("\nTool result envelope (Task 3A — 7 marks)")
    good = ok({"price": 181.9}, count=1)
    check("Success envelope shape", good["ok"] and good["error"] is None and "data" in good)

    bad = fail("DuckDuckGo rate-limited", "Call get_news instead and note the gap.")
    check("Failure envelope shape", not bad["ok"] and bad["data"] is None)
    check("Failure carries an actionable suggestion", "get_news" in bad["suggestion"],
          "the agent has a route to re-plan on")
    check("Envelope is JSON-serialisable", json.dumps(bad) and json.dumps(good), "safe as a ToolMessage")


if __name__ == "__main__":
    print("\nTask 3 offline verification\n" + "=" * 74)
    for fn in (test_handoff_schema, test_clarification_contract, test_report_validators,
               test_tracer, test_cache, test_tool_restriction, test_tool_envelope,
               test_headline_sanitisation, test_malformed_tool_call_detection):
        fn()
    print("\n" + "=" * 74)
    print("ALL CHECKS PASSED" if not FAILURES else f"{FAILURES} CHECK(S) FAILED")
    sys.exit(1 if FAILURES else 0)
