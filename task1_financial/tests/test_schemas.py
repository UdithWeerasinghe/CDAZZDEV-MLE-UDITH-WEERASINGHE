"""
Verification that the Task 1B validators enforce what the rubric asks for.

The interesting assertions here are the NEGATIVE ones: proving that a
plausible-looking but value-regurgitating justification is actually rejected.
A validator that accepts everything is worse than none, because it produces
false confidence.

Run:  python task1_financial/tests/test_schemas.py

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Write tests proving the
# TradingSignal validator rejects value regurgitation and accepts genuine
# synthesis', Date: 2026-09-10
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schemas import (  # noqa: E402
    AggregateSentiment,
    HeadlineSentiment,
    HeadlineSentimentBatch,
    Recommendation,
    Sentiment,
    TradingSignal,
    describe_validation_error,
)

# A justification that RESTATES values. Four sentences, mentions several
# indicators - but relates none of them. This is the exact failure the rubric
# names, and it must be rejected.
REGURGITATION = (
    "RSI-14 is currently at 62.4. The MACD line is at 1.83 and the signal line is at 1.41. "
    "The 50-day moving average is 178.20 and the 200-day moving average is 165.90. "
    "Bollinger %B is 0.72."
)

# Genuine synthesis: same indicators, but each sentence relates them.
SYNTHESIS = (
    "Trend and momentum agree here: the 50-day average sits about 7% above the 200-day, "
    "and MACD holding above its signal line confirms that the advance is still "
    "accelerating rather than merely persisting. RSI at 62 is the one point of tension, "
    "since it leaves limited headroom before the overbought zone and argues against "
    "chasing strength at this level. Bollinger %B at 0.72 says price is extended within "
    "its channel but not at the band, which is consistent with continuation rather than "
    "exhaustion. On balance the trend regime outweighs the stretched oscillator, so the "
    "constructive stance holds while accumulation is better done on weakness."
)

# Correct synthesis but only two sentences - violates the 3-5 sentence rule.
TOO_SHORT = (
    "The 50-day average sits above the 200-day while MACD confirms the momentum, which "
    "together support a constructive view. RSI at 62 tempers that slightly, however."
)

# Relational language and length, but rests on a single indicator.
SINGLE_INDICATOR = (
    "RSI at 62 suggests the advance still has room before reaching overbought territory. "
    "However, that same reading implies buyers have already done much of their work. "
    "On balance the oscillator points to continuation with diminishing upside. "
    "The reading therefore supports patience rather than urgency."
)


def valid_signal(**overrides) -> dict:
    base = dict(
        recommendation="Buy",
        confidence=0.68,
        justification=SYNTHESIS,
        key_drivers=[
            "SMA-50 sits roughly 7% above SMA-200, a durable golden-cross regime",
            "MACD histogram positive and widening, confirming momentum",
        ],
        risks=["RSI near 70 limits headroom before a mean-reversion pullback"],
        indicators_cited=["sma_50", "sma_200", "macd", "rsi_14", "bb_pct_b"],
    )
    base.update(overrides)
    return base


def expect_rejected(payload: dict, label: str, expected_fragment: str) -> None:
    try:
        TradingSignal.model_validate(payload)
    except ValidationError as exc:
        message = describe_validation_error(exc)
        assert expected_fragment in message, (
            f"{label}: rejected, but for the wrong reason.\n  got: {message}"
        )
        print(f"  PASS {label:<38} rejected: {message[:90]}")
        return
    raise AssertionError(f"{label}: should have been rejected but validated cleanly")


def test_synthesis_is_accepted() -> None:
    signal = TradingSignal.model_validate(valid_signal())
    report = signal.quality_report()
    assert report["relational_language_present"] is True
    assert len(report["indicators_referenced"]) >= 3, report
    assert 3 <= report["sentences"] <= 5, report
    print(f"  PASS {'Genuine synthesis accepted':<38} "
          f"{report['sentences']} sentences, {len(report['indicators_referenced'])} indicators")


def test_value_regurgitation_is_rejected() -> None:
    """The headline test: listing values is not reasoning."""
    expect_rejected(
        valid_signal(justification=REGURGITATION),
        "Value regurgitation",
        "without relating them",
    )


def test_too_few_sentences_rejected() -> None:
    expect_rejected(valid_signal(justification=TOO_SHORT), "Two sentences", "sentences")


def test_single_indicator_rejected() -> None:
    expect_rejected(
        valid_signal(justification=SINGLE_INDICATOR),
        "Single indicator only",
        "distinct indicator",
    )


def test_vague_drivers_rejected() -> None:
    expect_rejected(
        valid_signal(key_drivers=["Market risk", "Momentum is good and strong here"]),
        "Vague key driver",
        "too vague",
    )


def test_decimals_do_not_inflate_sentence_count() -> None:
    """'RSI is 62.4' must count as one sentence, not two."""
    with_decimals = (
        "Trend and momentum agree, with the 50-day at 178.25 above the 200-day at 165.90. "
        "MACD at 1.83 confirms this despite RSI at 62.4 approaching the overbought zone. "
        "On balance the combination supports a constructive stance."
    )
    signal = TradingSignal.model_validate(valid_signal(justification=with_decimals))
    assert signal.quality_report()["sentences"] == 3, signal.quality_report()
    print(f"  PASS {'Decimals not counted as sentences':<38} 3 sentences detected")


def test_out_of_range_values_rejected() -> None:
    for field, bad in (("confidence", 1.4), ("confidence", -0.1)):
        try:
            TradingSignal.model_validate(valid_signal(**{field: bad}))
            raise AssertionError(f"accepted {field}={bad}")
        except ValidationError:
            pass
    try:
        TradingSignal.model_validate(valid_signal(recommendation="Strong Buy"))
        raise AssertionError("accepted an out-of-enum recommendation")
    except ValidationError:
        pass
    print(f"  PASS {'Range and enum enforcement':<38} 3 cases")


def test_headline_sentiment_rejects_non_answers() -> None:
    for reason in ("positive", "good news", "bad"):
        try:
            HeadlineSentiment(
                headline="Chipmaker raises full-year guidance",
                sentiment=Sentiment.POSITIVE, confidence=0.9, brief_reason=reason,
            )
            raise AssertionError(f"accepted non-answer brief_reason={reason!r}")
        except ValidationError:
            pass

    ok = HeadlineSentiment(
        headline="Chipmaker raises full-year guidance",
        sentiment=Sentiment.POSITIVE, confidence=0.92,
        brief_reason="Raised guidance signals demand above prior expectations",
    )
    assert ok.confidence == 0.92
    print(f"  PASS {'Non-answer brief_reason rejected':<38} 3 cases, 1 valid accepted")


def test_confidence_weighted_aggregation() -> None:
    """A decisive negative must outweigh two hedged positives."""
    batch = HeadlineSentimentBatch(results=[
        HeadlineSentiment(headline="Analyst nudges target higher", sentiment=Sentiment.POSITIVE,
                          confidence=0.35, brief_reason="Minor target revision, limited impact"),
        HeadlineSentiment(headline="Index inclusion confirmed", sentiment=Sentiment.POSITIVE,
                          confidence=0.30, brief_reason="Technical flow event, modest and temporary"),
        HeadlineSentiment(headline="Regulator opens antitrust probe", sentiment=Sentiment.NEGATIVE,
                          confidence=0.95, brief_reason="Antitrust investigation threatens core revenue"),
    ])
    aggregate = AggregateSentiment.from_batch(batch)
    assert aggregate.score < 0, f"Expected negative aggregate, got {aggregate.score}"
    assert aggregate.label is Sentiment.NEGATIVE
    assert aggregate.positive_count == 2 and aggregate.negative_count == 1
    print(f"  PASS {'Confidence-weighted aggregation':<38} "
          f"score={aggregate.score:+.3f} despite 2-1 positive count")


def test_degenerate_batch_detected() -> None:
    """Identical label AND confidence across a batch means rubber-stamping."""
    rubber_stamped = HeadlineSentimentBatch(results=[
        HeadlineSentiment(headline=f"Some company headline number {i}", sentiment=Sentiment.NEUTRAL,
                          confidence=0.8, brief_reason="Routine corporate announcement with no impact")
        for i in range(5)
    ])
    assert rubber_stamped.is_degenerate is True
    assert AggregateSentiment.from_batch(rubber_stamped).degenerate_warning is True

    varied = HeadlineSentimentBatch(results=[
        HeadlineSentiment(headline="Earnings beat consensus estimates", sentiment=Sentiment.POSITIVE,
                          confidence=0.9, brief_reason="Beat on both revenue and margin lines"),
        HeadlineSentiment(headline="Chief financial officer departs", sentiment=Sentiment.NEGATIVE,
                          confidence=0.6, brief_reason="Unexpected senior departure raises governance questions"),
        HeadlineSentiment(headline="Company to present at conference", sentiment=Sentiment.NEUTRAL,
                          confidence=0.25, brief_reason="Scheduling notice carrying no new information"),
        HeadlineSentiment(headline="Buyback programme extended", sentiment=Sentiment.POSITIVE,
                          confidence=0.7, brief_reason="Extended buyback supports earnings per share"),
    ])
    assert varied.is_degenerate is False
    print(f"  PASS {'Degenerate batch detection':<38} rubber-stamp flagged, varied batch clean")


def test_fallback_signal_passes_its_own_validator() -> None:
    """The rules-based fallback must be a real substitute, not a stub."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from llm_analysis import _rules_based_signal  # noqa: PLC0415

    summary = {
        "ticker": "NVDA", "momentum_score": 0.62, "momentum_signal": "strong_bullish",
        "rsi_14": 64.2, "sma_50": 178.2, "sma_200": 165.9,
    }
    aggregate = AggregateSentiment(
        score=0.31, label=Sentiment.POSITIVE, headline_count=14,
        positive_count=8, negative_count=2, neutral_count=4, mean_confidence=0.71,
    )
    signal = _rules_based_signal(summary, aggregate)
    assert signal.recommendation is Recommendation.BUY
    assert signal.confidence <= 0.55, "A mechanical rule must not claim analyst-level confidence"
    report = signal.quality_report()
    assert report["relational_language_present"] and report["sentences"] <= 5
    print(f"  PASS {'Fallback signal self-validates':<38} "
          f"{signal.recommendation.value} @ conf {signal.confidence}, {report['sentences']} sentences")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"\nRunning {len(tests)} schema/validator tests\n" + "=" * 74)
    failures = 0
    for test in tests:
        print(f"\n{test.__name__}")
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {exc}")
    print("\n" + "=" * 74)
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
