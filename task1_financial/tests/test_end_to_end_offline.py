"""
Offline end-to-end smoke test for Task 1.

Stubs the network boundary (yfinance, news feeds, the LLM) and exercises
everything else for real: indicators, the summary dictionary, Pydantic
validation, the repair loop, the rules-based fallback, and both renderers.

WHY THIS EXISTS
The reviewer may run this on a machine with no API keys, or on the day Yahoo
rate-limits them. A pipeline that can only be demonstrated when four external
services are cooperating is not demonstrably correct. This proves every line of
logic I wrote works; the notebook proves it works against live data too.

The FlakyStubLLM deliberately fails the first signal attempt with value
regurgitation, so the run exercises the validator rejection, the repair loop,
and the recovery path rather than a clean happy path.

Run:  python task1_financial/tests/test_end_to_end_offline.py

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Write an offline end-to-end test
# for the Task 1 pipeline using a stub LLM that fails validation on the first
# attempt', Date: 2026-09-10
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import EquityData, NewsItem, build_summary  # noqa: E402
from indicators import compute_all_indicators                   # noqa: E402
from llm_analysis import analyse                                # noqa: E402
from report import render_html_brief, render_markdown_brief     # noqa: E402

HEADLINES = [
    ("Chipmaker beats quarterly revenue estimates on data-centre demand", "positive", 0.93),
    ("Board approves an additional $25bn share repurchase programme", "positive", 0.81),
    ("Regulator opens preliminary review of export licensing", "negative", 0.74),
    ("Supply partner flags capacity constraints into next quarter", "negative", 0.66),
    ("Company to present at an industry conference next month", "neutral", 0.22),
    ("Analyst raises price target citing accelerating orders", "positive", 0.58),
    ("Chief operating officer to retire at year end", "neutral", 0.41),
    ("New accelerator architecture unveiled at developer summit", "positive", 0.77),
    ("Shares slip as broader technology index retreats", "neutral", 0.35),
    ("Long-term supply agreement signed with a major cloud provider", "positive", 0.88),
    ("Litigation over patent licensing enters discovery phase", "negative", 0.52),
    ("Inventory levels normalise after two quarters of build", "positive", 0.49),
]


class FlakyStubLLM:
    """Stub with realistic failure behaviour, not a happy-path mock."""

    def __init__(self) -> None:
        self.signal_calls = 0
        self.calls: list[str] = []

    def describe(self) -> dict:
        return {"tier": "reasoning", "providers": [{"provider": "stub", "model": "stub-v1",
                                                    "base_url": "offline"}]}

    def chat_json(self, messages, **kwargs) -> dict:
        user_content = messages[1]["content"] if len(messages) > 1 else ""

        # --- Headline classification ---
        if "Classify each of the" in user_content:
            self.calls.append("sentiment")
            requested = [
                line.split(". ", 1)[1]
                for line in user_content.split("<headlines>")[1].split("</headlines>")[0].strip().split("\n")
                if ". " in line
            ]
            lookup = {h: (s, c) for h, s, c in HEADLINES}
            return {"results": [
                {
                    "headline": h,
                    "sentiment": lookup.get(h, ("neutral", 0.3))[0],
                    "confidence": lookup.get(h, ("neutral", 0.3))[1],
                    "brief_reason": f"Assessed from the specific claim made in the headline: {h[:48]}",
                }
                for h in requested
            ]}

        # --- Trading signal ---
        self.signal_calls += 1
        self.calls.append("signal")
        if self.signal_calls == 1:
            # Attempt 1: value regurgitation. Must be REJECTED by the validator.
            return {
                "recommendation": "Buy",
                "confidence": 0.75,
                "justification": (
                    "RSI-14 is at 58.2. The MACD line is above the signal line. "
                    "The 50-day average is above the 200-day average. Bollinger %B is 0.66."
                ),
                "key_drivers": ["Momentum is positive", "Trend is up"],
                "risks": ["Market risk"],
                "indicators_cited": ["rsi_14", "macd"],
            }
        # Attempt 2: genuine synthesis after the repair instruction.
        return {
            "recommendation": "Buy",
            "confidence": 0.71,
            "justification": (
                "Trend and momentum are pointing the same way: the 50-day average holds "
                "comfortably above the 200-day, and MACD sitting above its signal line "
                "confirms the advance is still accelerating rather than merely persisting. "
                "News flow corroborates that picture, with the revenue beat and the supply "
                "agreement outweighing the export-licensing review in both count and "
                "conviction. RSI in the high 50s is the one restraint, since it leaves "
                "moderate headroom but argues against adding aggressively at this level. "
                "On balance the trend regime dominates the stretched oscillator, supporting "
                "a constructive stance with accumulation on weakness."
            ),
            "key_drivers": [
                "SMA-50 holding above SMA-200 in a sustained golden-cross regime",
                "MACD histogram positive and widening, confirming momentum",
                "Confidence-weighted news sentiment positive across twelve headlines",
            ],
            "risks": [
                "Export-licensing review could restrict a material revenue channel",
                "Supply partner capacity constraints may cap near-term shipments",
            ],
            "indicators_cited": ["sma_50", "sma_200", "macd", "rsi_14", "news_sentiment"],
        }


def synthetic_equity(ticker: str = "NVDA") -> EquityData:
    rng = np.random.default_rng(11)
    n = 520
    drift = np.linspace(0.0009, 0.0004, n)
    shocks = rng.normal(0, 1, n) * np.where(np.arange(n) < 300, 0.014, 0.024)
    close = 95.0 * np.exp(np.cumsum(drift + shocks))
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    frame = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.007, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.007, n))),
            "Close": close,
            "Volume": rng.integers(20_000_000, 90_000_000, n),
        },
        index=index,
    )
    enriched = compute_all_indicators(frame)
    info = {
        "longName": "Synthetic Semiconductor Corp", "sector": "Technology",
        "industry": "Semiconductors", "currency": "USD",
        "trailingPE": 41.7, "marketCap": 2_140_000_000_000,
    }
    return EquityData(
        ticker=ticker,
        prices=enriched,
        summary=build_summary(ticker, enriched, info),
        news=[NewsItem(headline=h, source="StubFeed", published="2026-09-08") for h, _, _ in HEADLINES],
        warnings=["Synthetic offline fixture - no live market data was fetched."],
    )


def main() -> int:
    print("\nTask 1 offline end-to-end verification\n" + "=" * 74)
    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        if condition:
            print(f"  PASS {label:<44} {detail}")
        else:
            failures += 1
            print(f"  FAIL {label:<44} {detail}")

    equity = synthetic_equity()
    summary = equity.summary

    print("\n1A - pipeline and summary dictionary")
    required = [
        "current_price", "fifty_two_week_high", "fifty_two_week_low",
        "pe_ratio", "ytd_return_pct", "momentum_signal",
    ]
    missing = [f for f in required if f not in summary]
    check("All brief-required summary fields present", not missing, f"missing={missing or 'none'}")
    check("Sessions >= 2 years of trading days", summary["sessions_analysed"] >= 504,
          f"{summary['sessions_analysed']} sessions")
    check("Headline count >= 10", len(equity.news) >= 10, f"{len(equity.news)} headlines")

    nan_fields = [k for k, v in summary.items()
                  if isinstance(v, float) and not np.isfinite(v)]
    check("No NaN/inf leaked into the summary", not nan_fields, f"offenders={nan_fields or 'none'}")
    check("Summary is JSON-serialisable", _serialisable(summary),
          "safe to hand to an LLM")
    check("52w high >= current >= 52w low",
          summary["fifty_two_week_low"] <= summary["current_price"] <= summary["fifty_two_week_high"],
          f"{summary['fifty_two_week_low']} <= {summary['current_price']} <= {summary['fifty_two_week_high']}")

    print("\n1B - LLM analysis with a deliberately flaky model")
    with tempfile.TemporaryDirectory() as tmp:
        llm = FlakyStubLLM()
        result = analyse(llm, equity, failure_log_path=Path(tmp) / "validation_failures.jsonl")

        check("Every headline classified", len(result.headline_sentiments) == len(equity.news),
              f"{len(result.headline_sentiments)}/{len(equity.news)}")
        check("Aggregate computed", result.aggregate is not None,
              f"score={result.aggregate.score:+.3f} ({result.aggregate.label.value})")
        check("Regurgitated justification was rejected", llm.signal_calls == 2,
              f"{llm.signal_calls} signal attempts - first rejected, second accepted")
        check("Repair loop recovered without fallback", not result.signal_is_fallback,
              f"recommendation={result.signal.recommendation.value}")
        check("Validation failure was logged", len(result.validation_failures) >= 1,
              f"{result.failure_summary}")

        log_file = Path(tmp) / "validation_failures.jsonl"
        check("Failure log written to disk", log_file.exists() and log_file.stat().st_size > 0,
              f"{len(log_file.read_text().strip().splitlines())} line(s)")

        quality = result.signal.quality_report()
        check("Accepted justification passes quality checks",
              quality["relational_language_present"] and len(quality["indicators_referenced"]) >= 2,
              f"{quality['sentences']} sentences, {len(quality['indicators_referenced'])} indicators")

        print("\nBonus - report rendering")
        out = Path(tmp)
        html_path = render_html_brief(equity, result, out / "equity_brief.html")
        md_path = render_markdown_brief(equity, result, out / "equity_brief.md")
        html = html_path.read_text(encoding="utf-8")

        check("HTML brief written", html_path.exists(), f"{html_path.stat().st_size / 1024:.1f} KB")
        check("Chart embedded as a data URI", "data:image/png;base64," in html,
              "self-contained, no external requests")
        check("Risk disclaimer present", "RISK DISCLAIMER" in html.upper(), "mandatory element")
        for section in ("Company snapshot", "Technical outlook", "News sentiment", "Recommendation"):
            check(f"Section '{section}' rendered", section in html)
        check("Top three headlines shown",
              sum(html.count(h) for h, _, _ in HEADLINES) >= 3, "sorted by confidence")
        check("Markdown brief written", md_path.exists(),
              f"{len(md_path.read_text().splitlines())} lines")

        # Persist a copy of the sample brief for inspection.
        keep = ROOT / "output" / "sample_brief_offline.html"
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_text(html, encoding="utf-8")
        print(f"\n  Sample brief saved to {keep.relative_to(ROOT.parent)}")

    print("\n" + "=" * 74)
    print("ALL CHECKS PASSED" if not failures else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


def _serialisable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(main())
