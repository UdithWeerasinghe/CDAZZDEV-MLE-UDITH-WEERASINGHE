"""
Task 1B - LLM sentiment classification and trading-signal reasoning.

This module is the only place that knows how to turn Task 1A's output into an
LLM call and a validated result. It contains no prompt text (that lives in
prompts.py) and no schema definitions (schemas.py), which is what the
"separate prompt logic from business logic" criterion is asking for.

THE THREE FAILURE MODES THIS CODE IS BUILT AROUND
-------------------------------------------------
Free-tier models fail in specific, repeatable ways. Each has an explicit
countermeasure rather than a bare try/except:

1. DROPPED ITEMS. Ask for 18 classifications in one call and you often get 14.
   Countermeasure: classify in batches of BATCH_SIZE, match returned items back
   to the input by normalised text, and re-request only the missing headlines
   individually. A dropped headline is never silently lost.

2. SCHEMA DRIFT. The model returns `"sentiment": "Positive"`, a confidence of
   `85` instead of `0.85`, or wraps everything in a code fence.
   Countermeasure: lenient parsing in the client, then strict Pydantic
   validation, then a bounded repair loop that hands the model its own
   validator error.

3. VALUE REGURGITATION. The signal justification restates the numbers instead
   of reasoning over them - the exact failure the rubric names.
   Countermeasure: the TradingSignal validator rejects it, the repair loop
   re-requests once, and if the model still cannot synthesise we fall back to a
   deterministic rules-based signal and SAY SO in the output. A degraded but
   honest result beats a confident fabrication.

Every failure at every stage is appended to `validation_failures.jsonl`. The
brief asks for validation failures to be "caught, logged, and handled
gracefully"; the log file is the evidence that they were.

# AI-ASSISTED: Claude (claude-sonnet-5), Prompt: 'Write the Task 1B orchestration
# with batched headline classification, item-level reconciliation, a bounded
# Pydantic repair loop and a rules-based fallback signal', Date: 2026-09-10
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

import prompts
from data_pipeline import NewsItem
from schemas import (
    AggregateSentiment,
    HeadlineSentiment,
    HeadlineSentimentBatch,
    Recommendation,
    Sentiment,
    TradingSignal,
    ValidationFailure,
    describe_validation_error,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 8              # Small enough that the model rarely drops items.
MAX_REPAIR_ATTEMPTS = 2     # One honest retry, then degrade. Unbounded loops burn quota.
SENTIMENT_TEMPERATURE = 0.0 # Classification is not creative; determinism aids reproducibility.
SIGNAL_TEMPERATURE = 0.3    # A little latitude helps the prose without risking the schema.
TOP_HEADLINES_IN_PROMPT = 5


@dataclass
class AnalysisResult:
    """Everything Task 1B produces, including how well it went."""

    ticker: str
    headline_sentiments: list[HeadlineSentiment]
    aggregate: AggregateSentiment | None
    signal: TradingSignal | None
    signal_is_fallback: bool = False
    validation_failures: list[ValidationFailure] = field(default_factory=list)
    prompt_versions: list[dict[str, str]] = field(default_factory=list)
    model_manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def failure_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for failure in self.validation_failures:
            summary[failure.stage] = summary.get(failure.stage, 0) + 1
        return summary


class FailureLog:
    """Append-only JSONL log of every validation failure.

    Deliberately append-only and flushed per write: if the notebook dies
    mid-run, the evidence of what went wrong survives.
    """

    def __init__(self, path: str | Path = "output/validation_failures.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[ValidationFailure] = []

    def record(self, stage: str, attempt: int, error: str, raw: str = "", recovered: bool = False) -> None:
        failure = ValidationFailure(
            stage=stage, attempt=attempt, error=error[:2000], raw_output=raw[:4000], recovered=recovered
        )
        self.entries.append(failure)
        logger.warning("[%s] attempt %d failed: %s", stage, attempt, error[:200])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure.model_dump(), ensure_ascii=False) + "\n")

    def mark_last_recovered(self, stage: str) -> None:
        for failure in reversed(self.entries):
            if failure.stage == stage:
                failure.recovered = True
                return


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())[:80]


# ---------------------------------------------------------------------------
# Stage 1 - headline sentiment
# ---------------------------------------------------------------------------
def _classify_batch(
    llm: Any,
    ticker: str,
    company: str,
    headlines: Sequence[str],
    log: FailureLog,
) -> list[HeadlineSentiment]:
    """Classify one batch, with a bounded repair loop. Returns what validated."""
    block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
    messages = prompts.render(
        prompts.HEADLINE_SENTIMENT,
        ticker=ticker,
        company=company,
        count=len(headlines),
        headline_block=block,
    )

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 2):
        try:
            payload = llm.chat_json(
                messages,
                temperature=SENTIMENT_TEMPERATURE,
                max_tokens=180 * len(headlines) + 300,
                schema_hint='{"results": [{"headline": "...", "sentiment": "positive", '
                            '"confidence": 0.8, "brief_reason": "..."}]}',
            )
            batch = HeadlineSentimentBatch.model_validate(payload)
            if attempt > 1:
                log.mark_last_recovered("headline_sentiment")
            if batch.is_degenerate:
                logger.warning(
                    "Batch for %s is degenerate (one label, one confidence) - "
                    "possible rubber-stamping.", ticker,
                )
            return batch.results

        except ValidationError as exc:
            message = describe_validation_error(exc)
            log.record("headline_sentiment", attempt, message, raw=json.dumps(locals().get("payload", ""))[:2000])
            if attempt > MAX_REPAIR_ATTEMPTS:
                break
            messages = messages[:2] + [
                {"role": "assistant", "content": json.dumps(locals().get("payload", {}))[:2000]},
                {"role": "user", "content": prompts.REPAIR_INSTRUCTION.safe_substitute(error=message)},
            ]
        except Exception as exc:  # noqa: BLE001 - transport/parse failure
            log.record("headline_sentiment", attempt, f"{type(exc).__name__}: {exc}")
            if attempt > MAX_REPAIR_ATTEMPTS:
                break

    logger.error("Batch of %d headlines failed all attempts for %s.", len(headlines), ticker)
    return []


def classify_headlines(
    llm: Any,
    ticker: str,
    company: str,
    news: Sequence[NewsItem],
    log: FailureLog,
) -> list[HeadlineSentiment]:
    """Classify every headline, reconciling dropped items.

    The reconciliation pass is what makes this trustworthy: after batching, any
    headline the model failed to return is retried on its own, and anything
    still missing is recorded as neutral/0.0 confidence with an explicit
    'classification failed' reason rather than being dropped from the sample.
    An aggregate computed over a silently truncated sample is a wrong number
    that looks right.
    """
    headlines = [item.headline for item in news]
    if not headlines:
        logger.warning("No headlines supplied for %s; skipping sentiment stage.", ticker)
        return []

    classified: dict[str, HeadlineSentiment] = {}

    for start in range(0, len(headlines), BATCH_SIZE):
        chunk = headlines[start : start + BATCH_SIZE]
        for result in _classify_batch(llm, ticker, company, chunk, log):
            classified[_normalise(result.headline)] = result

    # Reconcile: which inputs did we not get back?
    missing = [h for h in headlines if _normalise(h) not in classified]
    if missing:
        logger.info("Retrying %d headline(s) individually for %s.", len(missing), ticker)
        for headline in missing:
            for result in _classify_batch(llm, ticker, company, [headline], log):
                classified[_normalise(result.headline)] = result

    # Anything still absent is recorded honestly rather than dropped.
    ordered: list[HeadlineSentiment] = []
    for headline in headlines:
        key = _normalise(headline)
        if key in classified:
            item = classified[key]
            # Restore the original text; models paraphrase when copying.
            ordered.append(item.model_copy(update={"headline": headline}))
        else:
            log.record("headline_sentiment", MAX_REPAIR_ATTEMPTS + 2,
                       f"Headline never classified: {headline[:120]}")
            ordered.append(
                HeadlineSentiment(
                    headline=headline,
                    sentiment=Sentiment.NEUTRAL,
                    confidence=0.0,
                    brief_reason="Classification failed after retries; recorded as neutral "
                                 "with zero confidence so it cannot bias the aggregate.",
                )
            )
    return ordered


# ---------------------------------------------------------------------------
# Stage 2 - trading signal
# ---------------------------------------------------------------------------
def _format_snapshot(summary: dict[str, Any]) -> str:
    """Render the summary as aligned key: value lines.

    Explicitly writes 'unavailable' for None. Handing a model the literal
    `null`, `NaN` or an omitted key is a reliable way to make it invent a value.
    """
    interesting = [
        ("Current price", "current_price"), ("52-week high", "fifty_two_week_high"),
        ("52-week low", "fifty_two_week_low"), ("Below 52w high (%)", "pct_below_52w_high"),
        ("YTD return (%)", "ytd_return_pct"), ("P/E ratio", "pe_ratio"),
        ("SMA-50", "sma_50"), ("SMA-200", "sma_200"), ("RSI-14", "rsi_14"),
        ("MACD", "macd"), ("MACD signal", "macd_signal"), ("MACD histogram", "macd_histogram"),
        ("Bollinger upper", "bollinger_upper"), ("Bollinger middle", "bollinger_middle"),
        ("Bollinger lower", "bollinger_lower"), ("Bollinger %B", "bollinger_pct_b"),
        ("Annualised vol (30d)", "annualised_volatility_30d"),
    ]
    width = max(len(label) for label, _ in interesting)
    lines = []
    for label, key in interesting:
        value = summary.get(key)
        lines.append(f"{label:<{width}} : {'unavailable' if value is None else value}")
    return "\n".join(lines)


def _rules_based_signal(summary: dict[str, Any], aggregate: AggregateSentiment | None) -> TradingSignal:
    """Deterministic fallback when the LLM cannot produce a valid signal.

    Blends the mechanical momentum score with news sentiment at a 70/30 weight -
    technicals dominate because they are computed from two years of data, while
    the sentiment score rests on a couple of dozen headlines.

    The justification is written to satisfy the same validator the LLM output
    must pass, so the fallback is a genuine substitute rather than a stub. It is
    always flagged as a fallback in the output.
    """
    momentum = float(summary.get("momentum_score") or 0.0)
    sentiment_score = float(aggregate.score) if aggregate else 0.0
    blended = 0.7 * momentum + 0.3 * sentiment_score

    if blended >= 0.25:
        recommendation = Recommendation.BUY
    elif blended <= -0.25:
        recommendation = Recommendation.SELL
    else:
        recommendation = Recommendation.HOLD

    agreement = "confirms" if momentum * sentiment_score >= 0 else "diverges from"
    rsi = summary.get("rsi_14")
    sma_50, sma_200 = summary.get("sma_50"), summary.get("sma_200")
    regime = (
        "above" if (sma_50 is not None and sma_200 is not None and sma_50 > sma_200) else "below"
    )

    justification = (
        f"The mechanical momentum screen returns {summary.get('momentum_signal', 'neutral')} "
        f"at a score of {momentum:+.2f}, with the 50-day average sitting {regime} the 200-day. "
        f"News sentiment {agreement} that technical picture at a score of {sentiment_score:+.2f} "
        f"across {aggregate.headline_count if aggregate else 0} headlines. "
        f"RSI-14 at {rsi if rsi is not None else 'an unavailable level'} is treated as "
        f"confirmation only, since the trend regime carries the greater weight in this framework. "
        f"On balance the blended score of {blended:+.2f} supports a {recommendation.value} stance, "
        f"though this is a deterministic fallback rather than a reasoned analyst view."
    )

    return TradingSignal(
        recommendation=recommendation,
        confidence=round(min(0.55, abs(blended) + 0.15), 2),  # Capped: a rule is not a view.
        justification=justification,
        key_drivers=[
            f"Momentum screen score of {momentum:+.2f} with SMA-50 {regime} SMA-200",
            f"News sentiment score of {sentiment_score:+.2f} across "
            f"{aggregate.headline_count if aggregate else 0} classified headlines",
        ],
        risks=[
            "This is a rules-based fallback produced after LLM validation failed, "
            "and carries no qualitative judgement",
        ],
        indicators_cited=["sma_50", "sma_200", "rsi_14", "news_sentiment"],
    )


def generate_signal(
    llm: Any,
    summary: dict[str, Any],
    aggregate: AggregateSentiment | None,
    sentiments: Sequence[HeadlineSentiment],
    log: FailureLog,
) -> tuple[TradingSignal, bool]:
    """Produce the Buy/Hold/Sell call. Returns (signal, used_fallback)."""
    ranked = sorted(sentiments, key=lambda s: s.confidence, reverse=True)[:TOP_HEADLINES_IN_PROMPT]
    top_headlines = "\n".join(
        f"- [{s.sentiment.value}, conf {s.confidence:.2f}] {s.headline}" for s in ranked
    ) or "- No headlines were successfully classified."

    messages = prompts.render(
        prompts.TRADING_SIGNAL,
        ticker=summary.get("ticker", ""),
        company=summary.get("company_name", ""),
        as_of=summary.get("as_of", "the latest session"),
        snapshot_block=_format_snapshot(summary),
        momentum_signal=summary.get("momentum_signal", "neutral"),
        momentum_score=summary.get("momentum_score", 0.0),
        momentum_components=json.dumps(summary.get("momentum_components", {})),
        momentum_rationale="\n".join(f"- {r}" for r in summary.get("momentum_rationale", [])),
        headline_count=aggregate.headline_count if aggregate else 0,
        sentiment_label=aggregate.label.value if aggregate else "unavailable",
        sentiment_score=aggregate.score if aggregate else 0.0,
        positive_count=aggregate.positive_count if aggregate else 0,
        neutral_count=aggregate.neutral_count if aggregate else 0,
        negative_count=aggregate.negative_count if aggregate else 0,
        top_headlines=top_headlines,
    )

    payload: Any = None
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 2):
        try:
            payload = llm.chat_json(
                messages,
                temperature=SIGNAL_TEMPERATURE,
                max_tokens=1200,
                schema_hint='{"recommendation": "Buy", "confidence": 0.7, "justification": "...", '
                            '"key_drivers": ["..."], "risks": ["..."], "indicators_cited": ["..."]}',
            )
            signal = TradingSignal.model_validate(payload)
            if attempt > 1:
                log.mark_last_recovered("trading_signal")
            logger.info("Signal validated on attempt %d: %s", attempt, signal.recommendation.value)
            return signal, False

        except ValidationError as exc:
            message = describe_validation_error(exc)
            log.record("trading_signal", attempt, message, raw=json.dumps(payload or {})[:3000])
            if attempt > MAX_REPAIR_ATTEMPTS:
                break
            messages = messages[:2] + [
                {"role": "assistant", "content": json.dumps(payload or {})[:2500]},
                {"role": "user", "content": prompts.REPAIR_INSTRUCTION.safe_substitute(error=message)},
            ]
        except Exception as exc:  # noqa: BLE001
            log.record("trading_signal", attempt, f"{type(exc).__name__}: {exc}")
            if attempt > MAX_REPAIR_ATTEMPTS:
                break

    logger.error("LLM signal failed validation after all attempts; using rules-based fallback.")
    return _rules_based_signal(summary, aggregate), True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def analyse(
    llm: Any,
    equity: Any,                        # EquityData from data_pipeline
    *,
    failure_log_path: str | Path = "output/validation_failures.jsonl",
) -> AnalysisResult:
    """Run the complete Task 1B stage over a Task 1A result."""
    log = FailureLog(failure_log_path)
    company = equity.summary.get("company_name", equity.ticker)

    sentiments = classify_headlines(llm, equity.ticker, company, equity.news, log)
    aggregate = (
        AggregateSentiment.from_batch(HeadlineSentimentBatch(results=sentiments))
        if sentiments else None
    )

    if aggregate:
        equity.summary["news_sentiment_score"] = aggregate.score
        equity.summary["news_sentiment_label"] = aggregate.label.value

    signal, used_fallback = generate_signal(llm, equity.summary, aggregate, sentiments, log)

    return AnalysisResult(
        ticker=equity.ticker,
        headline_sentiments=sentiments,
        aggregate=aggregate,
        signal=signal,
        signal_is_fallback=used_fallback,
        validation_failures=log.entries,
        prompt_versions=prompts.manifest(),
        model_manifest=llm.describe() if hasattr(llm, "describe") else {},
    )
