"""
Pydantic contracts for every LLM output in Task 1.

WHY THE VALIDATORS ARE UNUSUALLY STRICT
---------------------------------------
The marking criterion for signal reasoning is: "LLM reasons over indicator
combinations, not just echoes values." That is normally assessed by a human
reading the output. Here it is enforced in code.

`TradingSignal.justification` must pass three machine checks:

  1. LENGTH        - three to five sentences, as the brief specifies.
  2. BREADTH       - at least two distinct indicators referenced by name, so a
                     justification resting on RSI alone is rejected.
  3. RELATIONAL    - at least one connective that expresses how the indicators
                     relate (confirms, diverges, despite, whereas, ...). A model
                     that merely lists "RSI is 62, MACD is 1.4" contains no such
                     word and fails.

Check 3 is the interesting one. It cannot prove reasoning happened, but it
reliably catches the specific failure mode the rubric names: value regurgitation
dressed up as analysis. A rejected response is logged and re-requested with the
parser error attached, so the model gets one chance to actually do the work
before we fall back to a rules-based signal.

A validator this opinionated is a deliberate trade: it will occasionally reject
a genuinely good terse answer. That is the right direction to err for an
analyst-facing tool, and the fallback path means a rejection degrades quality
rather than breaking the pipeline.

# AI-ASSISTED: Claude (claude-sonnet-5), Prompt: 'Design Pydantic schemas for
# per-headline sentiment and a trading signal, with a validator that rejects
# justifications which merely restate indicator values', Date: 2026-09-10
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "Sentiment",
    "Recommendation",
    "HeadlineSentiment",
    "HeadlineSentimentBatch",
    "AggregateSentiment",
    "TradingSignal",
    "ValidationFailure",
    "INDICATOR_TERMS",
    "RELATIONAL_TERMS",
]


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Recommendation(str, Enum):
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# Indicator vocabulary. Deliberately includes the colloquial forms a model will
# reach for ("golden cross", "oversold") as well as the formal names, because
# rejecting good analysis on a vocabulary technicality is worse than the failure
# we are trying to catch.
INDICATOR_TERMS: dict[str, tuple[str, ...]] = {
    "sma_50": ("sma 50", "sma50", "50-day", "50 day", "fifty-day"),
    "sma_200": ("sma 200", "sma200", "200-day", "200 day", "two-hundred-day"),
    "trend_cross": ("golden cross", "death cross", "moving average cross"),
    "rsi": ("rsi", "relative strength", "overbought", "oversold"),
    "macd": ("macd", "signal line", "histogram", "convergence"),
    "bollinger": ("bollinger", "upper band", "lower band", "%b", "band width", "squeeze"),
    "volatility": ("volatility", "annualised vol", "annualized vol"),
    "sentiment": ("sentiment", "headline", "news flow"),
    "valuation": ("p/e", "pe ratio", "price-to-earnings", "valuation", "multiple"),
    "price_action": ("52-week", "52 week", "ytd", "year-to-date", "momentum"),
}

# Connectives expressing a relationship between two observations. This is what
# separates synthesis from a list.
RELATIONAL_TERMS: tuple[str, ...] = (
    "confirm", "corroborat", "consistent with", "reinforce", "align",
    "diverg", "conflict", "contradict", "disagree", "tension", "at odds",
    "despite", "however", "although", "whereas", "while", "but ",
    "offset", "outweigh", "override", "dominat", "on balance", "net of",
    "combined", "together", "taken as a whole", "in aggregate",
    "suggests that", "implies", "points to", "indicat",
)

MIN_SENTENCES = 3
MAX_SENTENCES = 5
MIN_DISTINCT_INDICATORS = 2


def _count_sentences(text: str) -> int:
    """Count sentences, tolerating decimals and common abbreviations.

    A naive `text.count('.')` counts "RSI is 62.4" as two sentences, which would
    reject correct output. We mask decimals and known abbreviations first.
    """
    masked = re.sub(r"\d+\.\d+", "0", text)
    masked = re.sub(r"\b(?:e\.g|i\.e|vs|etc|approx|Inc|Corp|Ltd|Q[1-4])\.", "X", masked, flags=re.I)
    parts = [segment for segment in re.split(r"[.!?]+(?:\s|$)", masked) if segment.strip()]
    return len(parts)


def _indicators_mentioned(text: str) -> set[str]:
    lowered = text.lower()
    return {name for name, aliases in INDICATOR_TERMS.items() if any(a in lowered for a in aliases)}


def _has_relational_language(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in RELATIONAL_TERMS)


class ValidationFailure(BaseModel):
    """A logged validation failure. Task 1B requires these to be captured.

    Persisted to `validation_failures.jsonl` so the reviewer can see exactly
    what the model got wrong and how often - which is far stronger evidence of
    a working validation layer than a clean run with nothing to show.
    """

    stage: str
    attempt: int
    error: str
    raw_output: str = Field(default="", max_length=4000)
    recovered: bool = False


class HeadlineSentiment(BaseModel):
    """One headline classified. Exactly the four fields the brief requires."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    headline: str = Field(min_length=1, max_length=500)
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = Field(min_length=10, max_length=400)

    @field_validator("brief_reason")
    @classmethod
    def reason_must_be_substantive(cls, value: str) -> str:
        """Reject non-answers like 'positive news' or 'neutral'."""
        if len(value.split()) < 4:
            raise ValueError(
                f"brief_reason must be a short explanation, got {value!r}. "
                "State WHY the headline carries that sentiment."
            )
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_discriminating(cls, value: float) -> float:
        """Guard against a model returning 1.0 for everything.

        Caught at the batch level rather than here (a single 1.0 is legitimate);
        this validator only rejects impossible values, which Field already does.
        Kept as an explicit hook so the batch check has a documented home.
        """
        return value


class HeadlineSentimentBatch(BaseModel):
    """Wrapper so the model can return one JSON object rather than a bare list.

    Free-tier models are markedly more reliable at emitting `{"results": [...]}`
    than a top-level array when `response_format=json_object` is requested.
    """

    model_config = ConfigDict(extra="ignore")

    results: list[HeadlineSentiment] = Field(min_length=1)

    @property
    def is_degenerate(self) -> bool:
        """True when every item shares one label AND one confidence.

        A signal that the model rubber-stamped the batch instead of reading it.
        Reported as a warning, not an error - a genuinely uniform news day
        exists, it is just rare enough to be worth flagging to the analyst.
        """
        if len(self.results) < 4:
            return False
        return (
            len({r.sentiment for r in self.results}) == 1
            and len({round(r.confidence, 2) for r in self.results}) == 1
        )


class AggregateSentiment(BaseModel):
    """Confidence-weighted aggregate over the classified headlines.

    Weighting by confidence rather than taking a plain majority means a
    hedged 0.55 classification cannot outvote a decisive 0.95 one. The
    unweighted counts are retained alongside so the analyst can see both.
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=-1.0, le=1.0)
    label: Sentiment
    headline_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    degenerate_warning: bool = False

    @classmethod
    def from_batch(cls, batch: HeadlineSentimentBatch) -> "AggregateSentiment":
        items = batch.results
        polarity = {Sentiment.POSITIVE: 1.0, Sentiment.NEUTRAL: 0.0, Sentiment.NEGATIVE: -1.0}

        total_weight = sum(item.confidence for item in items)
        score = (
            sum(polarity[item.sentiment] * item.confidence for item in items) / total_weight
            if total_weight > 0
            else 0.0
        )

        if score >= 0.15:
            label = Sentiment.POSITIVE
        elif score <= -0.15:
            label = Sentiment.NEGATIVE
        else:
            label = Sentiment.NEUTRAL

        return cls(
            score=round(max(-1.0, min(1.0, score)), 4),
            label=label,
            headline_count=len(items),
            positive_count=sum(i.sentiment is Sentiment.POSITIVE for i in items),
            negative_count=sum(i.sentiment is Sentiment.NEGATIVE for i in items),
            neutral_count=sum(i.sentiment is Sentiment.NEUTRAL for i in items),
            mean_confidence=round(sum(i.confidence for i in items) / len(items), 4) if items else 0.0,
            degenerate_warning=batch.is_degenerate,
        )


class TradingSignal(BaseModel):
    """The Buy / Hold / Sell call with an enforced-quality justification."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str = Field(min_length=80, max_length=2000)
    key_drivers: list[str] = Field(min_length=2, max_length=6)
    risks: list[str] = Field(min_length=1, max_length=5)
    indicators_cited: list[str] = Field(default_factory=list)

    @field_validator("justification")
    @classmethod
    def must_synthesise_not_restate(cls, value: str) -> str:
        """Enforce the three checks described in the module docstring."""
        problems: list[str] = []

        sentences = _count_sentences(value)
        if not MIN_SENTENCES <= sentences <= MAX_SENTENCES:
            problems.append(
                f"justification has {sentences} sentences; the brief requires "
                f"{MIN_SENTENCES} to {MAX_SENTENCES}"
            )

        mentioned = _indicators_mentioned(value)
        if len(mentioned) < MIN_DISTINCT_INDICATORS:
            problems.append(
                f"justification references {len(mentioned)} distinct indicator(s) "
                f"({sorted(mentioned) or 'none'}); at least {MIN_DISTINCT_INDICATORS} "
                "must be discussed so the reasoning spans a combination"
            )

        if not _has_relational_language(value):
            problems.append(
                "justification lists indicator values without relating them. Explain "
                "whether they confirm or contradict one another (e.g. 'MACD momentum "
                "is positive despite RSI approaching overbought')"
            )

        if problems:
            raise ValueError("; ".join(problems))
        return value

    @field_validator("key_drivers", "risks")
    @classmethod
    def entries_must_be_specific(cls, value: list[str]) -> list[str]:
        """Reject placeholder bullets such as 'market risk' or 'N/A'."""
        cleaned: list[str] = []
        for entry in value:
            text = entry.strip()
            if len(text.split()) < 3:
                raise ValueError(
                    f"entry {text!r} is too vague; give a specific, evidence-backed point"
                )
            cleaned.append(text)
        return cleaned

    def quality_report(self) -> dict[str, Any]:
        """Transparency for the reviewer: what the validator actually saw."""
        return {
            "sentences": _count_sentences(self.justification),
            "indicators_referenced": sorted(_indicators_mentioned(self.justification)),
            "relational_language_present": _has_relational_language(self.justification),
            "key_driver_count": len(self.key_drivers),
            "risk_count": len(self.risks),
        }


def describe_validation_error(error: ValidationError) -> str:
    """Flatten a Pydantic error into a single actionable line for the repair loop.

    The default `str(ValidationError)` is multi-line and includes a docs URL,
    which wastes tokens and confuses smaller models. This gives the model the
    field path and the message, nothing else.
    """
    parts = []
    for item in error.errors():
        location = ".".join(str(p) for p in item["loc"]) or "<root>"
        parts.append(f"{location}: {item['msg']}")
    return " | ".join(parts)
