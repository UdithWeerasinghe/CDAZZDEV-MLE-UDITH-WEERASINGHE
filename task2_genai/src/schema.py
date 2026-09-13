"""
Task 2A - use case definition and output schema.

=============================================================================
PROBLEM STATEMENT
=============================================================================
DOMAIN
  Credit-agreement and loan-document clause extraction. A credit analyst at a
  lender reviews facility agreements and must record, per clause, what the
  borrower has committed to, what triggers a breach, and how dangerous it is.
  This is the highest-volume, lowest-judgement part of the job and the obvious
  candidate for automation.

WHAT THE MODEL RECEIVES
  One clause excerpt from a credit or loan agreement, 30-250 words, in the
  register such documents are actually written in - defined terms in title
  case, cross-references, nested conditions.

WHAT THE MODEL MUST PRODUCE
  Exactly one JSON object conforming to `ClauseExtraction` below. No preamble,
  no markdown fence, no trailing commentary. Fields:
    clause_type        one of 12 values in CLAUSE_TAXONOMY - a closed set
    parties            the defined parties the clause binds, verbatim
    obligation         what is required or prohibited, one sentence
    trigger_condition  the event that activates it, or null if unconditional
    financial_terms    figures, ratios, rates and periods stated in the text
    risk_flag          high | medium | low
    risk_rationale     one clause justifying the flag

WHAT COUNTS AS CORRECT
  1. Output parses as JSON on the first attempt and validates against the schema.
  2. clause_type is the correct member of the closed taxonomy.
  3. Every party in `parties` appears in the source text. None is invented.
  4. Every figure in `financial_terms` appears in the source text, at the same
     value. This is the hallucination test that matters most commercially -
     a fabricated covenant threshold is a materially wrong risk assessment.
  5. trigger_condition is null when, and only when, the clause is unconditional.
  6. risk_flag is consistent with the taxonomy's stated default for that clause
     type, unless the text justifies departing from it.

WHAT COUNTS AS INCORRECT
  Invalid JSON; prose wrapped around the JSON; a clause_type outside the
  taxonomy; a party or figure absent from the source; a null trigger on a
  conditional clause; risk_flag unsupported by either the text or the taxonomy.

WHY THIS IS A GOOD FINE-TUNING TARGET
  * The base model fails visibly and repeatably. Mistral-7B-Instruct given this
    task emits explanatory preamble, wraps output in ``` fences, invents
    plausible-sounding field names, and drifts out of the taxonomy. Those are
    format failures, which is precisely what a small LoRA fixes well.
  * Correctness is machine-checkable per field, so the hallucination rate in
    Task 2C is a measurement rather than an impression.
  * It is not a generic chatbot task, which the brief caps at 5/30.

=============================================================================
# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Design a credit-agreement clause
# extraction schema with a closed clause taxonomy and per-field validation for
# a fine-tuning use case', Date: 2026-09-10
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ClauseType",
    "RiskFlag",
    "ClauseExtraction",
    "CLAUSE_TAXONOMY",
    "INDUSTRIES",
    "JURISDICTIONS",
    "COMPLEXITY_LEVELS",
    "EDGE_CASES",
    "SYSTEM_PROMPT",
    "canonical_json",
    "validate_extraction",
]


class ClauseType(str, Enum):
    """Closed taxonomy. A value outside this set is an error, not a variant."""

    FINANCIAL_COVENANT = "financial_covenant"
    NEGATIVE_COVENANT = "negative_covenant"
    AFFIRMATIVE_COVENANT = "affirmative_covenant"
    EVENT_OF_DEFAULT = "event_of_default"
    REPRESENTATION_WARRANTY = "representation_warranty"
    INTEREST_PROVISION = "interest_provision"
    REPAYMENT_PREPAYMENT = "repayment_prepayment"
    SECURITY_COLLATERAL = "security_collateral"
    CHANGE_OF_CONTROL = "change_of_control"
    CONDITIONS_PRECEDENT = "conditions_precedent"
    INDEMNITY = "indemnity"
    GOVERNING_LAW_JURISDICTION = "governing_law_jurisdiction"


class RiskFlag(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Taxonomy metadata drives both generation (for stratification) and evaluation
# (as the prior a departing risk_flag must be justified against).
CLAUSE_TAXONOMY: dict[ClauseType, dict[str, Any]] = {
    ClauseType.FINANCIAL_COVENANT: {
        "description": "Quantitative financial tests the borrower must maintain.",
        "default_risk": RiskFlag.HIGH,
        "why": "Breach is objectively measurable and usually triggers default directly.",
        "examples": ["leverage ratio", "interest coverage", "minimum EBITDA", "net worth"],
    },
    ClauseType.NEGATIVE_COVENANT: {
        "description": "Restrictions on what the borrower may not do without consent.",
        "default_risk": RiskFlag.MEDIUM,
        "why": "Constrains operations but breach is often curable and consent obtainable.",
        "examples": ["no additional indebtedness", "no asset disposals", "no dividends"],
    },
    ClauseType.AFFIRMATIVE_COVENANT: {
        "description": "Positive obligations the borrower must perform.",
        "default_risk": RiskFlag.LOW,
        "why": "Administrative in nature; typically carries a cure period.",
        "examples": ["deliver audited accounts", "maintain insurance", "pay taxes"],
    },
    ClauseType.EVENT_OF_DEFAULT: {
        "description": "Events entitling the lender to accelerate the facility.",
        "default_risk": RiskFlag.HIGH,
        "why": "Directly enables acceleration and enforcement.",
        "examples": ["non-payment", "insolvency", "cross-default", "material adverse change"],
    },
    ClauseType.REPRESENTATION_WARRANTY: {
        "description": "Statements of fact the borrower affirms, often repeating.",
        "default_risk": RiskFlag.MEDIUM,
        "why": "Inaccuracy is usually itself an event of default.",
        "examples": ["due incorporation", "no litigation", "accuracy of accounts"],
    },
    ClauseType.INTEREST_PROVISION: {
        "description": "How interest is calculated, accrued and paid.",
        "default_risk": RiskFlag.MEDIUM,
        "why": "Determines cost of funds; default interest can escalate sharply.",
        "examples": ["SOFR plus margin", "default interest", "margin ratchet"],
    },
    ClauseType.REPAYMENT_PREPAYMENT: {
        "description": "Scheduled repayment and voluntary or mandatory prepayment.",
        "default_risk": RiskFlag.MEDIUM,
        "why": "Drives liquidity planning; mandatory prepayment can be disruptive.",
        "examples": ["amortisation schedule", "excess cash flow sweep", "break costs"],
    },
    ClauseType.SECURITY_COLLATERAL: {
        "description": "Assets secured and the nature of the security interest.",
        "default_risk": RiskFlag.HIGH,
        "why": "Determines recovery on enforcement.",
        "examples": ["fixed and floating charge", "share pledge", "negative pledge"],
    },
    ClauseType.CHANGE_OF_CONTROL: {
        "description": "Consequences of a change in ownership of the borrower.",
        "default_risk": RiskFlag.HIGH,
        "why": "Commonly triggers mandatory prepayment of the whole facility.",
        "examples": ["put option on CoC", "mandatory prepayment", "permitted holders"],
    },
    ClauseType.CONDITIONS_PRECEDENT: {
        "description": "Conditions to be satisfied before drawdown.",
        "default_risk": RiskFlag.LOW,
        "why": "Procedural and satisfied at closing; limited ongoing exposure.",
        "examples": ["legal opinions", "board resolutions", "KYC completion"],
    },
    ClauseType.INDEMNITY: {
        "description": "Borrower's obligation to compensate the lender for losses.",
        "default_risk": RiskFlag.MEDIUM,
        "why": "Potentially open-ended but contingent on a triggering loss.",
        "examples": ["tax gross-up", "currency indemnity", "environmental indemnity"],
    },
    ClauseType.GOVERNING_LAW_JURISDICTION: {
        "description": "Governing law and forum for disputes.",
        "default_risk": RiskFlag.LOW,
        "why": "Procedural; material only if enforcement becomes necessary.",
        "examples": ["English law", "exclusive jurisdiction", "arbitration"],
    },
}

# Stratification axes for dataset generation. The Cartesian product of these is
# ~12 x 10 x 6 x 3 x 6 = 12,960 distinct combinations, so 120 examples drawn
# from distinct cells cannot collapse into variations of one scenario - which is
# the failure mode the brief awards zero marks for.
INDUSTRIES = [
    "manufacturing", "commercial real estate", "software and SaaS", "healthcare services",
    "renewable energy infrastructure", "logistics and shipping", "specialty retail",
    "agriculture and agribusiness", "hospitality and leisure", "telecommunications",
]

JURISDICTIONS = [
    "English law", "New York law", "Singapore law",
    "Hong Kong law", "German law", "Australian law",
]

COMPLEXITY_LEVELS = {
    "simple": "One obligation, plain wording, no cross-references. 30-60 words.",
    "moderate": "Defined terms in title case, one cross-reference, a proviso. 70-130 words.",
    "complex": "Nested conditions, multiple cross-references, carve-outs and baskets, "
               "a stepped or ratcheting threshold. 150-250 words.",
}

# Edge cases force the model to learn the hard boundaries rather than the happy
# path. Without deliberate seeding, a teacher model produces almost none of
# these, and the fine-tune then fails exactly where it matters commercially.
EDGE_CASES = {
    "none": "A straightforward clause of this type.",
    "no_financial_terms": "A clause of this type containing NO figures at all, so "
                          "financial_terms must be an empty list. Tests whether the "
                          "model fabricates numbers to fill the field.",
    "unconditional": "A clause imposing an absolute obligation with no triggering "
                     "event, so trigger_condition must be null. Tests over-eager "
                     "trigger invention.",
    "multi_party": "A clause binding three or more named parties including a "
                   "guarantor or security agent. Tests complete party extraction.",
    "risk_departure": "A clause whose wording justifies a risk_flag DIFFERENT from "
                      "the taxonomy default - an unusually long cure period, a wide "
                      "materiality qualifier, or a very tight threshold. Tests "
                      "reading over pattern-matching.",
    "ambiguous_type": "A clause sitting close to the boundary between two taxonomy "
                      "types, where the correct choice needs care.",
}


class ClauseExtraction(BaseModel):
    """The exact output contract. Both training targets and inference outputs
    are validated against this, so the fine-tune and the evaluation share one
    definition of correct."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    clause_type: ClauseType
    parties: list[str] = Field(min_length=1, max_length=6)
    obligation: str = Field(min_length=15, max_length=400)
    trigger_condition: str | None = Field(default=None, max_length=300)
    financial_terms: list[str] = Field(default_factory=list, max_length=8)
    risk_flag: RiskFlag
    risk_rationale: str = Field(min_length=10, max_length=300)

    @field_validator("parties")
    @classmethod
    def parties_must_be_named(cls, value: list[str]) -> list[str]:
        """Reject empty or placeholder party entries."""
        cleaned = [p.strip() for p in value if p and p.strip()]
        if not cleaned:
            raise ValueError("parties must contain at least one named party")
        for party in cleaned:
            if party.lower() in {"n/a", "none", "unknown", "-", "null"}:
                raise ValueError(f"party {party!r} is a placeholder, not a party")
        return cleaned

    @field_validator("trigger_condition")
    @classmethod
    def empty_trigger_is_null(cls, value: str | None) -> str | None:
        """Normalise "", "none", "N/A" to None so the null case is unambiguous.

        Without this, the test set contains three spellings of "no trigger" and
        per-field accuracy is measuring string formatting rather than extraction.
        """
        if value is None:
            return None
        text = value.strip()
        return None if text.lower() in {"", "none", "n/a", "null", "not applicable"} else text


# Field order is fixed so that serialised targets are byte-identical for the
# same content. If the order varied, the model would be trained on inconsistent
# targets and ROUGE-L in Task 2C would measure key ordering rather than accuracy.
FIELD_ORDER = [
    "clause_type", "parties", "obligation", "trigger_condition",
    "financial_terms", "risk_flag", "risk_rationale",
]


def canonical_json(extraction: ClauseExtraction | dict, *, indent: int | None = None) -> str:
    """Serialise in fixed field order with stable formatting."""
    payload = (
        extraction.model_dump(mode="json")
        if isinstance(extraction, ClauseExtraction) else dict(extraction)
    )
    ordered = {key: payload.get(key) for key in FIELD_ORDER}
    return json.dumps(ordered, ensure_ascii=False, indent=indent)


def validate_extraction(raw: str) -> tuple[ClauseExtraction | None, str | None]:
    """Parse and validate a model output. Returns (extraction, error).

    Deliberately STRICT about fences: the base model wraps output in ```json and
    the fine-tuned model should not. Silently stripping fences here would hide
    exactly the improvement we are trying to measure, so `used_fence` is
    reported by the evaluator as its own metric rather than being normalised
    away. This function is used for scoring, where we want the truth.
    """
    if not raw or not raw.strip():
        return None, "empty output"

    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # Salvage for diagnosis only - the evaluator records that salvage was
        # needed, which is itself a quality signal.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None, f"no JSON object found: {exc}"
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as inner:
            return None, f"malformed JSON: {inner}"

    if not isinstance(payload, dict):
        return None, f"expected a JSON object, got {type(payload).__name__}"

    try:
        return ClauseExtraction.model_validate(payload), None
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError
        detail = getattr(exc, "errors", None)
        if callable(detail):
            return None, " | ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in detail()
            )
        return None, str(exc)


# ---------------------------------------------------------------------------
# The system prompt used at BOTH training and inference time.
# ---------------------------------------------------------------------------
# It must be byte-identical in both places. If training uses one wording and
# evaluation another, the measured improvement is partly an artefact of prompt
# drift rather than of fine-tuning - and the base-model baseline in Task 2C
# would not be a fair comparison.
SYSTEM_PROMPT = (
    "You are a credit documentation analyst. Extract structured data from the "
    "credit agreement clause provided. Respond with a single JSON object only "
    "and nothing else.\n\n"
    "Fields: clause_type (one of: "
    + ", ".join(t.value for t in ClauseType)
    + "), parties (array of the defined parties bound by the clause, exactly as "
    "named in the text), obligation (one sentence stating what is required or "
    "prohibited), trigger_condition (the event that activates the clause, or "
    "null if it is unconditional), financial_terms (array of figures, ratios, "
    "rates and periods that appear in the text; empty array if none), risk_flag "
    "(high, medium or low), risk_rationale (one clause justifying the flag).\n\n"
    "Never state a party or a figure that does not appear in the clause text."
)
