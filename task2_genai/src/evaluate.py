"""
Task 2C - evaluation and baseline comparison.

WHAT THE BRIEF ASKS FOR, AND WHAT I ADD
---------------------------------------
Required: ROUGE-L base vs fine-tuned on an identical held-out test set as a
table; at least one additional metric (BERTScore F1 or LLM-as-judge); manual
review of ten or more responses labelled correct / partially correct /
hallucinated with a stated hallucination rate; two paragraphs of qualitative
analysis.

All of that is here. So is a fourth family of metrics, because of an honest
problem with the required one:

    ROUGE-L IS A WEAK METRIC FOR SCHEMA-CONSTRAINED OUTPUT.

ROUGE-L measures longest-common-subsequence overlap. Two JSON objects with
identical structure and one wrong `clause_type` share almost every token, so
they score near-identically - while being right and wrong respectively. Worse,
a model that emits the right JSON wrapped in a ```json fence loses ROUGE points
for a formatting artefact, and a model that emits the correct field names with
entirely fabricated values scores well.

So ROUGE-L is reported, because it is asked for and because it does capture
gross format improvement. But the metrics that actually measure what we
fine-tuned for are reported next to it:

  * JSON VALIDITY RATE      - does it parse and validate at all? This is the
                              headline improvement for a format-learning task.
  * PER-FIELD ACCURACY      - exact match on clause_type and risk_flag, set F1
                              on parties and financial_terms.
  * GROUNDED-HALLUCINATION  - the fraction of outputs containing a party or a
    RATE                      figure that does not appear in the input clause.
                              Checkable automatically, and it is the metric a
                              credit risk team would actually care about.

The automated hallucination rate does not replace the manual review the brief
requires - it complements it. The manual labels catch semantic errors the
automated check cannot see (a correct-looking obligation that misreads the
clause); the automated check covers all 12 test cases rather than a sample.

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Build an evaluation suite
# comparing base and fine-tuned models on ROUGE-L, BERTScore, an LLM judge,
# per-field accuracy and automated grounding checks', Date: 2026-09-10
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import ClauseExtraction, canonical_json, validate_extraction  # noqa: E402

logger = logging.getLogger(__name__)

JUDGE_TEMPERATURE = 0.0
MIN_MANUAL_REVIEW = 10      # The brief's minimum.


# ===========================================================================
# ROUGE-L
# ===========================================================================
def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length. O(len(a) * len(b)) space-optimised.

    Implemented directly rather than pulled from `rouge-score` so the metric is
    inspectable and has no dependency that might tokenise differently between
    the reviewer's environment and mine.
    """
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for index, token_b in enumerate(b, start=1):
            current[index] = (
                previous[index - 1] + 1 if token_a == token_b
                else max(previous[index], current[index - 1])
            )
        previous = current
    return previous[len(b)]


def _tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. JSON punctuation is deliberately dropped:
    otherwise ROUGE-L largely measures brace and quote agreement, which every
    model gets right and which tells us nothing."""
    return re.findall(r"[a-z0-9_.:%-]+", (text or "").lower())


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> dict[str, float]:
    """ROUGE-L precision, recall and F-measure.

    beta=1.2 is the value used in the original ROUGE package, which weights
    recall slightly above precision.
    """
    predicted, referenced = _tokenise(prediction), _tokenise(reference)
    if not predicted or not referenced:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(predicted, referenced)
    precision = lcs / len(predicted)
    recall = lcs / len(referenced)
    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f_measure = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f_measure, 4),
    }


# ===========================================================================
# Structural and per-field metrics
# ===========================================================================
def _set_f1(predicted: Sequence[str], expected: Sequence[str]) -> float:
    """Case- and whitespace-insensitive set F1 over list fields."""
    p = {s.strip().lower() for s in predicted if s and s.strip()}
    e = {s.strip().lower() for s in expected if s and s.strip()}
    if not p and not e:
        return 1.0            # Correctly empty is correct.
    if not p or not e:
        return 0.0
    overlap = len(p & e)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(e)
    return round(2 * precision * recall / (precision + recall), 4)


def _digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text or "")


# Matches 3.50, 10,000,000, 45, 2.75 — the numeric forms that appear in credit
# documents. Thousands separators are handled so "USD 10,000,000" and
# "$10000000" compare equal.
_NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _numbers_in(text: str) -> set[float]:
    """Every number in a string, as floats.

    Comparing NUMBERS rather than digit substrings is what makes "3.5:1" match
    a clause that says "3.50:1". A digit-sequence check reads those as "351"
    and "3501" and reports a hallucination that did not happen - which would
    inflate the headline metric of Task 2C with false positives.
    """
    values: set[float] = set()
    for match in _NUMBER_PATTERN.findall(text or ""):
        try:
            values.add(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def _numbers_match(term: str, text_numbers: set[float]) -> bool:
    """True when every number in `term` also occurs in the source text.

    Tolerance is relative (1e-6) so 3.5 == 3.50 while 3.5 != 3.6.
    """
    term_numbers = _numbers_in(term)
    if not term_numbers:
        return False
    return all(
        any(abs(value - candidate) <= max(1e-6, abs(value) * 1e-6)
            for candidate in text_numbers)
        for value in term_numbers
    )


def grounding_check(extraction: ClauseExtraction, clause_text: str) -> dict[str, Any]:
    """Detect parties and figures asserted but absent from the source clause.

    This is the automated hallucination detector. It is deliberately LENIENT on
    formatting - "USD 10,000,000" against "$10,000,000", "3.50:1" against
    "3.5:1" - by comparing digit sequences, because flagging a correct
    extraction over punctuation would inflate the hallucination rate and make
    the metric useless.

    It cannot catch semantic errors (a plausible obligation that misreads the
    clause). Those are what the manual review is for.
    """
    lowered = (clause_text or "").lower()
    text_numbers = _numbers_in(lowered)

    ungrounded_parties = []
    for party in extraction.parties:
        # Match on the distinctive tokens of the name, not the whole string:
        # "Helvern Industrial Holdings Limited" should match a clause that says
        # "Helvern Industrial Holdings Ltd".
        tokens = [t for t in re.findall(r"[a-z]{4,}", party.lower())
                  if t not in {"limited", "holdings", "group", "company", "the"}]
        if party.lower() in lowered:
            continue
        if tokens and any(token in lowered for token in tokens):
            continue
        if not tokens and party.lower() in lowered:
            continue
        ungrounded_parties.append(party)

    ungrounded_terms = []
    for term in extraction.financial_terms:
        if term.lower().strip() in lowered:
            continue
        if _numbers_in(term):
            # Numeric term: every number in it must occur in the clause.
            if _numbers_match(term, text_numbers):
                continue
        else:
            # Non-numeric term such as "SOFR" or "quarterly in arrear".
            words = re.findall(r"[a-z]{3,}", term.lower())
            if words and any(word in lowered for word in words):
                continue
        ungrounded_terms.append(term)

    return {
        "grounded": not (ungrounded_parties or ungrounded_terms),
        "ungrounded_parties": ungrounded_parties,
        "ungrounded_financial_terms": ungrounded_terms,
    }


@dataclass
class PredictionRecord:
    """One model output scored every way."""

    index: int
    clause_text: str
    reference_json: str
    raw_output: str
    parsed: ClauseExtraction | None = None
    parse_error: str | None = None
    used_code_fence: bool = False
    had_preamble: bool = False
    rouge: dict[str, float] = field(default_factory=dict)
    field_scores: dict[str, float] = field(default_factory=dict)
    grounding: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    manual_label: str | None = None


def score_prediction(
    index: int, clause_text: str, reference_json: str, raw_output: str
) -> PredictionRecord:
    """Score one output against its reference. No LLM required."""
    record = PredictionRecord(
        index=index, clause_text=clause_text,
        reference_json=reference_json, raw_output=raw_output or "",
    )

    stripped = record.raw_output.strip()
    record.used_code_fence = "```" in stripped
    record.had_preamble = bool(stripped) and not stripped.startswith(("{", "```"))

    record.rouge = rouge_l(record.raw_output, reference_json)

    parsed, error = validate_extraction(record.raw_output)
    record.parsed, record.parse_error = parsed, error

    if parsed is None:
        record.field_scores = {
            "clause_type": 0.0, "risk_flag": 0.0, "parties_f1": 0.0,
            "financial_terms_f1": 0.0, "trigger_null_match": 0.0, "overall": 0.0,
        }
        record.grounding = {"grounded": False, "ungrounded_parties": [],
                            "ungrounded_financial_terms": [], "unparseable": True}
        return record

    try:
        reference = ClauseExtraction.model_validate(json.loads(reference_json))
    except Exception:  # noqa: BLE001
        logger.warning("Reference %d is itself invalid; field scores skipped.", index)
        return record

    scores = {
        "clause_type": 1.0 if parsed.clause_type is reference.clause_type else 0.0,
        "risk_flag": 1.0 if parsed.risk_flag is reference.risk_flag else 0.0,
        "parties_f1": _set_f1(parsed.parties, reference.parties),
        "financial_terms_f1": _set_f1(parsed.financial_terms, reference.financial_terms),
        # The null case is the one models get wrong most often, so it is scored
        # separately rather than folded into a string comparison.
        "trigger_null_match": 1.0 if (
            (parsed.trigger_condition is None) == (reference.trigger_condition is None)
        ) else 0.0,
    }
    scores["overall"] = round(sum(scores.values()) / len(scores), 4)
    record.field_scores = scores
    record.grounding = grounding_check(parsed, clause_text)
    return record


# ===========================================================================
# BERTScore
# ===========================================================================
def bertscore_f1(predictions: list[str], references: list[str]) -> dict[str, Any]:
    """BERTScore F1, with an explicit unavailable result rather than a crash.

    Uses a small model (distilbert) so it runs on a free Colab T4 alongside
    everything else. If the library is missing, we say so and the pipeline
    continues - the LLM judge is the second metric and either satisfies the
    brief's requirement.
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        return {"available": False,
                "reason": "bert-score not installed (pip install bert-score)"}

    try:
        precision, recall, f1 = bert_score(
            predictions, references, lang="en", model_type="distilbert-base-uncased",
            verbose=False, rescale_with_baseline=False,
        )
        return {
            "available": True,
            "model": "distilbert-base-uncased",
            "precision": round(float(precision.mean()), 4),
            "recall": round(float(recall.mean()), 4),
            "f1": round(float(f1.mean()), 4),
            "per_example_f1": [round(float(v), 4) for v in f1],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


# ===========================================================================
# LLM as judge
# ===========================================================================
JUDGE_SYSTEM = """\
You are a senior credit documentation reviewer scoring an automated extraction \
against the source clause. You are strict, consistent, and you do not reward \
plausible-sounding output that is not supported by the text.

Score each dimension from 0 to 5:

schema_compliance  5 = valid JSON with exactly the seven required fields, no \
extras, no prose, no code fence. 3 = parses but with a structural flaw. \
0 = does not parse.
type_accuracy      5 = clause_type is correct. 2 = a defensible neighbouring \
type. 0 = plainly wrong.
extraction_fidelity 5 = every party and figure appears in the clause and none \
is missed. Deduct heavily for anything asserted but absent from the source - \
that is a hallucination and it is the most serious failure here. Deduct \
moderately for omissions.
reasoning_quality  5 = obligation and risk_rationale accurately reflect the \
clause and the risk_flag follows from them. 0 = generic or contradicted.

Then assign one overall verdict:
  "correct"           - usable as-is by an analyst.
  "partially_correct" - broadly right, with an error an analyst would fix.
  "hallucinated"      - contains a party, figure or condition NOT in the source \
clause, or is unparseable. Grounding failure takes precedence: if anything was \
invented, the verdict is "hallucinated" regardless of how good the rest is.

Return one JSON object and nothing else:
{"schema_compliance": 0-5, "type_accuracy": 0-5, "extraction_fidelity": 0-5,
 "reasoning_quality": 0-5, "verdict": "...", "justification": "<one sentence>",
 "specific_errors": ["..."]}"""

JUDGE_USER = """\
SOURCE CLAUSE
<clause>
{clause_text}
</clause>

REFERENCE EXTRACTION (the ground truth)
{reference}

MODEL OUTPUT TO SCORE (verbatim, including any formatting artefacts)
{prediction}

Score the model output. Note that any party or figure in the model output that \
does not appear in the source clause is a hallucination."""


def judge_prediction(llm: Any, record: PredictionRecord) -> dict[str, Any]:
    """Score one prediction with an LLM judge returning structured JSON."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            clause_text=record.clause_text[:2500],
            reference=record.reference_json,
            prediction=(record.raw_output or "<empty>")[:2500],
        )},
    ]
    try:
        payload = llm.chat_json(messages, temperature=JUDGE_TEMPERATURE, max_tokens=600)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    scores = {}
    for key in ("schema_compliance", "type_accuracy", "extraction_fidelity", "reasoning_quality"):
        try:
            scores[key] = max(0.0, min(5.0, float(payload.get(key, 0))))
        except (TypeError, ValueError):
            scores[key] = 0.0

    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"correct", "partially_correct", "hallucinated"}:
        verdict = "partially_correct"

    return {
        "available": True,
        **scores,
        "mean_score": round(sum(scores.values()) / len(scores), 3),
        "verdict": verdict,
        "justification": str(payload.get("justification", ""))[:400],
        "specific_errors": [str(e)[:200] for e in (payload.get("specific_errors") or [])][:5],
    }


# ===========================================================================
# Aggregation
# ===========================================================================
def aggregate(records: list[PredictionRecord], label: str) -> dict[str, Any]:
    """Roll a set of scored predictions into one comparable summary."""
    n = len(records) or 1
    parsed = [r for r in records if r.parsed is not None]
    judged = [r for r in records if r.judge.get("available")]

    verdicts = Counter(r.judge.get("verdict") for r in judged)
    manual = Counter(r.manual_label for r in records if r.manual_label)

    grounding_failures = [
        r for r in parsed
        if not r.grounding.get("grounded", True)
    ]

    summary = {
        "model": label,
        "n": len(records),
        # --- Required metric ---
        "rouge_l_f1": round(sum(r.rouge.get("f1", 0) for r in records) / n, 4),
        "rouge_l_precision": round(sum(r.rouge.get("precision", 0) for r in records) / n, 4),
        "rouge_l_recall": round(sum(r.rouge.get("recall", 0) for r in records) / n, 4),
        # --- Structural ---
        "json_validity_rate": round(len(parsed) / n, 4),
        "code_fence_rate": round(sum(r.used_code_fence for r in records) / n, 4),
        "preamble_rate": round(sum(r.had_preamble for r in records) / n, 4),
        # --- Per-field (over parsed outputs only; unparsed already score 0) ---
        "clause_type_accuracy": round(
            sum(r.field_scores.get("clause_type", 0) for r in records) / n, 4),
        "risk_flag_accuracy": round(
            sum(r.field_scores.get("risk_flag", 0) for r in records) / n, 4),
        "parties_f1": round(sum(r.field_scores.get("parties_f1", 0) for r in records) / n, 4),
        "financial_terms_f1": round(
            sum(r.field_scores.get("financial_terms_f1", 0) for r in records) / n, 4),
        "trigger_null_accuracy": round(
            sum(r.field_scores.get("trigger_null_match", 0) for r in records) / n, 4),
        "field_accuracy_overall": round(
            sum(r.field_scores.get("overall", 0) for r in records) / n, 4),
        # --- Hallucination, automated ---
        "automated_hallucination_rate": round(
            (len(grounding_failures) + (n - len(parsed))) / n, 4),
        "grounding_failure_count": len(grounding_failures),
        "unparseable_count": n - len(parsed),
    }

    if judged:
        summary["llm_judge"] = {
            "n_judged": len(judged),
            "mean_score_out_of_5": round(
                sum(r.judge.get("mean_score", 0) for r in judged) / len(judged), 3),
            "schema_compliance": round(
                sum(r.judge.get("schema_compliance", 0) for r in judged) / len(judged), 2),
            "type_accuracy": round(
                sum(r.judge.get("type_accuracy", 0) for r in judged) / len(judged), 2),
            "extraction_fidelity": round(
                sum(r.judge.get("extraction_fidelity", 0) for r in judged) / len(judged), 2),
            "reasoning_quality": round(
                sum(r.judge.get("reasoning_quality", 0) for r in judged) / len(judged), 2),
            "verdicts": dict(verdicts),
            "hallucination_rate": round(verdicts.get("hallucinated", 0) / len(judged), 4),
        }

    if manual:
        total = sum(manual.values())
        summary["manual_review"] = {
            "n_reviewed": total,
            "meets_brief_minimum": total >= MIN_MANUAL_REVIEW,
            "correct": manual.get("correct", 0),
            "partially_correct": manual.get("partially_correct", 0),
            "hallucinated": manual.get("hallucinated", 0),
            "hallucination_rate_pct": round(100 * manual.get("hallucinated", 0) / total, 1),
            "correct_rate_pct": round(100 * manual.get("correct", 0) / total, 1),
        }

    return summary


def comparison_table(base: dict[str, Any], tuned: dict[str, Any]) -> str:
    """Render the base vs fine-tuned comparison the brief asks for.

    Every row states the direction of improvement explicitly, because for some
    metrics (code_fence_rate, hallucination_rate) lower is better, and a table
    of bare deltas invites the reader to misread a good result as a bad one.
    """
    rows = [
        ("ROUGE-L F1", "rouge_l_f1", "higher"),
        ("ROUGE-L precision", "rouge_l_precision", "higher"),
        ("ROUGE-L recall", "rouge_l_recall", "higher"),
        ("JSON validity rate", "json_validity_rate", "higher"),
        ("Code-fence rate", "code_fence_rate", "lower"),
        ("Preamble rate", "preamble_rate", "lower"),
        ("clause_type accuracy", "clause_type_accuracy", "higher"),
        ("risk_flag accuracy", "risk_flag_accuracy", "higher"),
        ("parties F1", "parties_f1", "higher"),
        ("financial_terms F1", "financial_terms_f1", "higher"),
        ("trigger null accuracy", "trigger_null_accuracy", "higher"),
        ("Field accuracy (mean)", "field_accuracy_overall", "higher"),
        ("Hallucination rate (auto)", "automated_hallucination_rate", "lower"),
    ]

    lines = [
        "| Metric | Base | Fine-tuned | Δ | Better |",
        "|---|---:|---:|---:|:---:|",
    ]
    for name, key, direction in rows:
        b, t = base.get(key), tuned.get(key)
        if b is None or t is None:
            continue
        delta = t - b
        improved = (delta > 0) if direction == "higher" else (delta < 0)
        marker = "✅" if improved and abs(delta) > 1e-9 else ("—" if abs(delta) < 1e-9 else "❌")
        lines.append(f"| {name} | {b:.4f} | {t:.4f} | {delta:+.4f} | {marker} |")

    for label, key in (("LLM judge (mean /5)", "mean_score_out_of_5"),
                       ("LLM judge hallucination rate", "hallucination_rate")):
        b = base.get("llm_judge", {}).get(key)
        t = tuned.get("llm_judge", {}).get(key)
        if b is None or t is None:
            continue
        delta = t - b
        improved = delta > 0 if "mean" in key else delta < 0
        lines.append(f"| {label} | {b:.4f} | {t:.4f} | {delta:+.4f} | "
                     f"{'✅' if improved else '❌'} |")

    return "\n".join(lines)


def manual_review_sheet(records: list[PredictionRecord], path: str | Path) -> Path:
    """Write a review sheet for the manual labelling the brief requires.

    Pre-populates the automated grounding verdict as a SUGGESTION, clearly
    marked, so the reviewer can work quickly - but the label column is left
    blank. Pre-filling it would make the "manual" review a rubber stamp of the
    automated check and destroy its value as an independent signal.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual review sheet — fine-tuned model",
        "",
        f"Label each of the {len(records)} responses below as `correct`, "
        "`partially_correct`, or `hallucinated`, then record the counts in the "
        "notebook. The brief requires a minimum of ten.",
        "",
        "The automated grounding check is shown as a *suggestion only*. Judge for "
        "yourself — it cannot see semantic errors, such as an obligation that reads "
        "plausibly but misstates what the clause requires.",
        "",
    ]
    for record in records:
        suggestion = (
            "hallucinated (ungrounded values found)"
            if not record.grounding.get("grounded", True)
            else "unparseable" if record.parsed is None else "no grounding failure detected"
        )
        lines += [
            f"## Example {record.index}",
            "",
            "**Source clause**", "", f"> {record.clause_text[:900]}", "",
            "**Reference**", "", "```json", record.reference_json, "```", "",
            "**Model output**", "", "```", (record.raw_output or "<empty>")[:1200], "```", "",
            f"- Automated suggestion: `{suggestion}`",
            f"- Field accuracy: {record.field_scores.get('overall', 0):.2f} · "
            f"ROUGE-L F1: {record.rouge.get('f1', 0):.3f}",
            "- **Your label:** `______________`",
            "- Notes:",
            "", "---", "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
