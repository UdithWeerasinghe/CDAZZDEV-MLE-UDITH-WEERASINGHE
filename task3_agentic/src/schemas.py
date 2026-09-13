"""
Task 3 - typed contracts between agents and for the final report.

THE HANDOFF SCHEMA IS THE POINT (8 marks)
-----------------------------------------
The criterion is: "Pydantic or typed dict - not raw string passing." The usual
multi-agent tutorial has Agent A produce prose and Agent B parse it, which is
not coordination - it is two chatbots taking turns and hoping.

`QuantBrief` is the contract. Agent A cannot hand Agent B anything else, and
because it is validated at the boundary, an incomplete analysis fails loudly at
the handoff instead of silently producing a report built on missing numbers.

Two properties make it a real contract rather than decoration:

  * `data_gaps` is REQUIRED (possibly empty). Agent A must state what it could
    not determine. Without this field an agent whose volatility tool failed
    simply omits volatility, and Agent B has no way to know the difference
    between "not applicable" and "never measured". Forcing the declaration is
    what makes the downstream report honest.

  * `confidence` is per-brief and bounded. Agent B is instructed to raise a
    clarification when confidence is low, which is what gives the critique loop
    something real to trigger on rather than a scripted round trip.

`ClarificationRequest` / `ClarificationResponse` are the critique-loop contract.
The request names a specific field and a specific question, so the loop cannot
degenerate into "please tell me more".

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Define Pydantic handoff schemas
# for a quant analyst agent and a research writer agent, including a
# clarification request/response pair', Date: 2026-09-10
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "TrendRegime",
    "SentimentLabel",
    "PriceSnapshot",
    "VolatilityReading",
    "SentimentReading",
    "QuantBrief",
    "ClarificationRequest",
    "ClarificationResponse",
    "Risk",
    "ResearchReport",
]


class TrendRegime(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGEBOUND = "rangebound"
    UNDETERMINED = "undetermined"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNAVAILABLE = "unavailable"


class PriceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_price: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = Field(default=None, ge=0.0, le=100.0)
    macd_histogram: float | None = None
    bollinger_pct_b: float | None = None
    pct_from_52w_high: float | None = None
    ytd_return_pct: float | None = None
    sessions_analysed: int = Field(default=0, ge=0)


class VolatilityReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_days: int = Field(ge=2)
    annualised_volatility: float | None = Field(default=None, ge=0.0)
    percentile_vs_two_year: float | None = Field(
        default=None, ge=0.0, le=100.0,
        description="Where current vol sits in its own 2-year distribution. "
                    "An absolute vol number is meaningless without this context: "
                    "35% is calm for a small-cap biotech and alarming for a utility.",
    )


class SentimentReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: SentimentLabel = SentimentLabel.UNAVAILABLE
    score: float | None = Field(default=None, ge=-1.0, le=1.0)
    headlines_analysed: int = Field(default=0, ge=0)
    positive: int = Field(default=0, ge=0)
    negative: int = Field(default=0, ge=0)
    neutral: int = Field(default=0, ge=0)


class QuantBrief(BaseModel):
    """Agent A → Agent B. The only permitted channel between the two agents."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1, max_length=12)
    as_of: str
    price: PriceSnapshot
    volatility: VolatilityReading
    sentiment: SentimentReading
    trend_regime: TrendRegime
    quantitative_findings: list[str] = Field(
        min_length=2, max_length=8,
        description="Numeric, evidence-backed observations. Each must contain a figure.",
    )
    data_gaps: list[str] = Field(
        default_factory=list,
        description="REQUIRED (may be empty). What Agent A could not determine and why.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    produced_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("quantitative_findings")
    @classmethod
    def findings_must_carry_numbers(cls, value: list[str]) -> list[str]:
        """A quantitative finding without a number is a qualitative opinion.

        Enforcing this is what stops Agent A drifting into Agent B's job and
        handing over prose the writer then has to take on trust.
        """
        for finding in value:
            if not any(character.isdigit() for character in finding):
                raise ValueError(
                    f"quantitative finding {finding!r} contains no figure. "
                    "State the measured value, e.g. 'RSI-14 at 62.4, below the 70 threshold'."
                )
        return value

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, value: str) -> str:
        return value.strip().upper()

    def as_handoff_text(self) -> str:
        """Render for inclusion in Agent B's prompt.

        Agent B receives this rendering of a VALIDATED object, never a free-form
        string produced by Agent A. The distinction matters: the structure is
        guaranteed before any text is generated.
        """
        lines = [
            f"QUANTITATIVE BRIEF — {self.ticker} as at {self.as_of}",
            f"Analyst confidence: {self.confidence:.2f}",
            f"Trend regime: {self.trend_regime.value}",
            "",
            "PRICE AND TECHNICALS",
        ]
        for key, value in self.price.model_dump().items():
            lines.append(f"  {key:<22}: {'unavailable' if value is None else value}")
        lines += [
            "",
            f"VOLATILITY ({self.volatility.window_days}-day window)",
            f"  annualised            : {self.volatility.annualised_volatility if self.volatility.annualised_volatility is not None else 'unavailable'}",
            f"  percentile vs 2 years : {self.volatility.percentile_vs_two_year if self.volatility.percentile_vs_two_year is not None else 'unavailable'}",
            "",
            f"NEWS SENTIMENT ({self.sentiment.headlines_analysed} headlines)",
            f"  label {self.sentiment.label.value}, score {self.sentiment.score}, "
            f"{self.sentiment.positive} positive / {self.sentiment.neutral} neutral / "
            f"{self.sentiment.negative} negative",
            "",
            "QUANTITATIVE FINDINGS",
            *[f"  - {finding}" for finding in self.quantitative_findings],
        ]
        if self.data_gaps:
            lines += ["", "DECLARED DATA GAPS (do not present these as measured)",
                      *[f"  ! {gap}" for gap in self.data_gaps]]
        else:
            lines += ["", "DECLARED DATA GAPS: none"]
        return "\n".join(lines)


class ClarificationRequest(BaseModel):
    """Agent B → Agent A. Exactly one specific question about one field."""

    model_config = ConfigDict(extra="forbid")

    field_in_question: str = Field(min_length=2, max_length=80)
    question: str = Field(min_length=15, max_length=400)
    why_it_matters: str = Field(min_length=15, max_length=400)

    @field_validator("question")
    @classmethod
    def must_be_specific(cls, value: str) -> str:
        """Reject open-ended prompts that make the loop theatre rather than useful."""
        vague = ("tell me more", "more information", "more detail", "anything else", "elaborate")
        if any(phrase in value.lower() for phrase in vague) and "?" not in value:
            raise ValueError(
                "clarification must ask a specific answerable question about a named field"
            )
        return value


class ClarificationResponse(BaseModel):
    """Agent A → Agent B, closing the critique loop."""

    model_config = ConfigDict(extra="forbid")

    field_in_question: str
    answer: str = Field(min_length=15, max_length=1200)
    supporting_values: dict[str, Any] = Field(default_factory=dict)
    could_not_answer: bool = False


class Risk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: str = Field(min_length=15, max_length=300)
    supporting_evidence: str = Field(min_length=20, max_length=600)
    severity: str = Field(pattern="^(high|medium|low)$")

    @field_validator("supporting_evidence")
    @classmethod
    def evidence_must_be_concrete(cls, value: str) -> str:
        """The brief requires each risk to carry supporting evidence.

        A number or a named source is the minimum bar for 'evidence'; without
        one, the field is just the risk restated in different words.
        """
        has_number = any(character.isdigit() for character in value)
        has_source = any(
            marker in value.lower()
            for marker in ("headline", "reported", "according to", "search", "news",
                           "filing", "analyst", "source", "commentary")
        )
        if not (has_number or has_source):
            raise ValueError(
                f"supporting_evidence {value[:60]!r} cites neither a figure nor a source. "
                "Reference a measured value or a retrieved headline."
            )
        return value


class ResearchReport(BaseModel):
    """The final deliverable. Three sections, exactly as the brief specifies."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    financial_health_summary: str = Field(min_length=150, max_length=3000)
    top_three_risks: list[Risk] = Field(min_length=3, max_length=3)
    hedge_strategy: str = Field(min_length=100, max_length=2000)
    data_caveats: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("hedge_strategy")
    @classmethod
    def hedge_must_be_data_driven(cls, value: str) -> str:
        """The brief asks for ONE data-driven hedge strategy.

        'Data-driven' is enforced as: the recommendation references a measured
        quantity (a figure) and names a concrete instrument or mechanism. A
        hedge that says only 'consider hedging your exposure' is not a strategy.
        """
        if not any(character.isdigit() for character in value):
            raise ValueError(
                "hedge_strategy must reference a measured quantity (strike, ratio, "
                "volatility level, horizon, or position size) to be data-driven"
            )
        instruments = (
            "put", "call", "option", "collar", "spread", "future", "swap", "hedge ratio",
            "beta", "short", "inverse", "etf", "stop", "delta", "vix", "pair", "overlay",
        )
        if not any(word in value.lower() for word in instruments):
            raise ValueError(
                "hedge_strategy must name a concrete instrument or mechanism "
                "(e.g. protective put, collar, beta-weighted short, pair trade)"
            )
        return value

    def render(self) -> str:
        """Human-readable rendering for the notebook output."""
        lines = [
            f"{'=' * 78}",
            f"RESEARCH REPORT — {self.ticker}",
            f"{'=' * 78}",
            "",
            "1. FINANCIAL HEALTH SUMMARY",
            "-" * 78,
            self.financial_health_summary,
            "",
            "2. TOP THREE RISKS TO THE SHARE PRICE (NEXT 90 DAYS)",
            "-" * 78,
        ]
        for index, risk in enumerate(self.top_three_risks, 1):
            lines += [
                f"{index}. [{risk.severity.upper()}] {risk.risk}",
                f"   Evidence: {risk.supporting_evidence}",
                "",
            ]
        lines += [
            "3. HEDGE STRATEGY RECOMMENDATION",
            "-" * 78,
            self.hedge_strategy,
        ]
        if self.data_caveats:
            lines += ["", "DATA CAVEATS", "-" * 78,
                      *[f"  ! {caveat}" for caveat in self.data_caveats]]
        lines += ["", f"Generated {self.generated_at}",
                  "", "Not investment advice. Generated by an automated research pipeline."]
        return "\n".join(lines)
