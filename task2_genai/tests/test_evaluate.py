"""
Verification of the Task 2C evaluation suite.

Simulates realistic base-model and fine-tuned-model outputs and asserts the
metrics move in the right direction and by a sensible amount. Also asserts the
central claim in evaluate.py's docstring - that ROUGE-L is a weak signal for
schema-constrained output - by measuring it, rather than only asserting it in
a comment.

Run:  python task2_genai/tests/test_evaluate.py

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Write tests proving the
# evaluation suite separates base-style from fine-tuned-style outputs and
# quantifying ROUGE-L insensitivity', Date: 2026-09-10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "task2_genai" / "src")]

from evaluate import (  # noqa: E402
    aggregate, comparison_table, grounding_check, manual_review_sheet,
    rouge_l, score_prediction,
)
from schema import ClauseExtraction  # noqa: E402

FAILURES = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  PASS {label:<50} {detail}")
    else:
        FAILURES += 1
        print(f"  FAIL {label:<50} {detail}")


CLAUSE = (
    "The Borrower shall procure that the ratio of Consolidated Total Net Debt to "
    "Consolidated EBITDA in respect of any Relevant Period shall not exceed 3.50:1, "
    "tested on each Quarter Date by reference to the Compliance Certificate delivered "
    "by Helvern Industrial Holdings Limited under Clause 21.2."
)

REFERENCE = json.dumps({
    "clause_type": "financial_covenant",
    "parties": ["Helvern Industrial Holdings Limited"],
    "obligation": "Maintain the ratio of Consolidated Total Net Debt to Consolidated "
                  "EBITDA at or below the stated threshold.",
    "trigger_condition": "tested on each Quarter Date",
    "financial_terms": ["3.50:1"],
    "risk_flag": "high",
    "risk_rationale": "Objectively measurable and breach triggers default directly.",
})

# What Mistral-7B-Instruct actually does with this prompt before fine-tuning:
# a preamble, a markdown fence, and drift out of the field contract.
BASE_OUTPUT = """Sure! Here's the structured extraction of the clause you provided:

```json
{
  "clause_type": "Financial Covenant",
  "parties": ["The Borrower", "Helvern Industrial Holdings Limited"],
  "obligation": "The Borrower must keep leverage below 3.50:1",
  "trigger_condition": "quarterly testing",
  "financial_terms": ["3.50:1", "Clause 21.2"],
  "risk_flag": "High",
  "risk_rationale": "Financial covenants are important",
  "notes": "This is a standard leverage covenant."
}
```

Let me know if you'd like me to extract any other clauses!"""

# What the fine-tuned model should produce: the object, alone.
TUNED_OUTPUT = json.dumps({
    "clause_type": "financial_covenant",
    "parties": ["Helvern Industrial Holdings Limited"],
    "obligation": "Maintain the ratio of Consolidated Total Net Debt to Consolidated "
                  "EBITDA at or below the stated threshold.",
    "trigger_condition": "tested on each Quarter Date",
    "financial_terms": ["3.50:1"],
    "risk_flag": "high",
    "risk_rationale": "Objectively measurable and breach triggers default directly.",
})

# A hallucinating output: perfect structure, invented figure and party.
HALLUCINATED_OUTPUT = json.dumps({
    "clause_type": "financial_covenant",
    "parties": ["Northbank Commercial Finance plc"],
    "obligation": "Maintain the leverage ratio below the covenant level.",
    "trigger_condition": "tested on each Quarter Date",
    "financial_terms": ["3.50:1", "USD 45,000,000", "2.75:1"],
    "risk_flag": "high",
    "risk_rationale": "Breach triggers default.",
})


if __name__ == "__main__":
    print("\nEvaluation suite verification\n" + "=" * 80)

    print("\n1 · ROUGE-L implementation")
    identical = rouge_l("the quick brown fox", "the quick brown fox")
    check("Identical strings score 1.0", identical["f1"] == 1.0, str(identical))
    disjoint = rouge_l("alpha beta", "gamma delta")
    check("Disjoint strings score 0.0", disjoint["f1"] == 0.0, str(disjoint))
    check("Empty prediction scores 0.0", rouge_l("", "something")["f1"] == 0.0)
    partial = rouge_l("the quick fox jumps", "the quick brown fox")
    check("Partial overlap scores between", 0.0 < partial["f1"] < 1.0, str(partial))

    print("\n2 · Scoring a base-model-style output")
    base = score_prediction(1, CLAUSE, REFERENCE, BASE_OUTPUT)
    check("Code fence detected", base.used_code_fence is True)
    check("Preamble detected", base.had_preamble is True)
    check("Rejected by the schema", base.parsed is None,
          f"{(base.parse_error or '')[:60]}")
    check("Field scores all zero when unparseable",
          base.field_scores["overall"] == 0.0)

    print("\n3 · Scoring a fine-tuned-style output")
    tuned = score_prediction(1, CLAUSE, REFERENCE, TUNED_OUTPUT)
    check("No code fence", tuned.used_code_fence is False)
    check("No preamble", tuned.had_preamble is False)
    check("Parses and validates", tuned.parsed is not None)
    check("Field accuracy is perfect", tuned.field_scores["overall"] == 1.0,
          str(tuned.field_scores))
    check("Grounded against the source", tuned.grounding["grounded"] is True)

    print("\n4 · Automated hallucination detection")
    hallucinated = score_prediction(1, CLAUSE, REFERENCE, HALLUCINATED_OUTPUT)
    check("Parses cleanly (structure is fine)", hallucinated.parsed is not None)
    check("Ungrounded party caught",
          "Northbank Commercial Finance plc" in hallucinated.grounding["ungrounded_parties"],
          str(hallucinated.grounding["ungrounded_parties"]))
    check("Ungrounded figures caught",
          len(hallucinated.grounding["ungrounded_financial_terms"]) == 2,
          str(hallucinated.grounding["ungrounded_financial_terms"]))
    check("Real figure NOT falsely flagged",
          "3.50:1" not in hallucinated.grounding["ungrounded_financial_terms"],
          "3.50:1 appears in the clause and must pass")
    check("Overall verdict is ungrounded", hallucinated.grounding["grounded"] is False)

    print("\n5 · Format leniency (must not create false hallucinations)")
    lenient = ClauseExtraction.model_validate({
        **json.loads(REFERENCE),
        "financial_terms": ["3.5:1", "Clause 21.2"],   # Different formatting, same values.
    })
    result = grounding_check(lenient, CLAUSE)
    check("Reformatted figure accepted", result["grounded"] is True,
          "'3.5:1' matched against '3.50:1' on digit sequence")

    print("\n6 · ROUGE-L insensitivity — the claim in the module docstring")
    wrong_type = json.dumps({**json.loads(REFERENCE), "clause_type": "indemnity"})
    wrong_risk = json.dumps({**json.loads(REFERENCE), "risk_flag": "low"})
    r_perfect = rouge_l(TUNED_OUTPUT, REFERENCE)["f1"]
    r_type = rouge_l(wrong_type, REFERENCE)["f1"]
    r_risk = rouge_l(wrong_risk, REFERENCE)["f1"]

    check("Wrong clause_type barely moves ROUGE-L",
          abs(r_perfect - r_type) < 0.05,
          f"perfect {r_perfect:.4f} vs wrong-type {r_type:.4f} "
          f"(Δ {r_perfect - r_type:+.4f})")
    check("Wrong risk_flag barely moves ROUGE-L",
          abs(r_perfect - r_risk) < 0.05,
          f"perfect {r_perfect:.4f} vs wrong-risk {r_risk:.4f} "
          f"(Δ {r_perfect - r_risk:+.4f})")

    type_field = score_prediction(1, CLAUSE, REFERENCE, wrong_type).field_scores["clause_type"]
    check("Field accuracy DOES catch it", type_field == 0.0,
          "clause_type accuracy 0.0 where ROUGE-L saw almost no change — "
          "this is why per-field metrics are reported alongside")

    print("\n7 · Aggregation and the comparison table")
    base_records = [score_prediction(i, CLAUSE, REFERENCE, BASE_OUTPUT) for i in range(12)]
    tuned_records = [score_prediction(i, CLAUSE, REFERENCE, TUNED_OUTPUT) for i in range(10)]
    tuned_records += [score_prediction(i, CLAUSE, REFERENCE, HALLUCINATED_OUTPUT)
                      for i in range(10, 12)]
    for index, record in enumerate(tuned_records):
        record.manual_label = "hallucinated" if index >= 10 else "correct"

    base_summary = aggregate(base_records, "Mistral-7B-Instruct-v0.3 (base)")
    tuned_summary = aggregate(tuned_records, "Fine-tuned")

    check("Base JSON validity is 0%", base_summary["json_validity_rate"] == 0.0,
          f"{base_summary['json_validity_rate']:.0%}")
    check("Tuned JSON validity is 100%", tuned_summary["json_validity_rate"] == 1.0,
          f"{tuned_summary['json_validity_rate']:.0%}")
    check("Base code-fence rate is 100%", base_summary["code_fence_rate"] == 1.0)
    check("Tuned code-fence rate is 0%", tuned_summary["code_fence_rate"] == 0.0)
    check("Field accuracy improves",
          tuned_summary["field_accuracy_overall"] > base_summary["field_accuracy_overall"],
          f"{base_summary['field_accuracy_overall']:.3f} → "
          f"{tuned_summary['field_accuracy_overall']:.3f}")
    check("Hallucination rate reflects the 2 bad outputs",
          abs(tuned_summary["automated_hallucination_rate"] - 2 / 12) < 1e-3,
          f"{tuned_summary['automated_hallucination_rate']:.4f} = 2/12")
    check("Manual review recorded",
          tuned_summary["manual_review"]["n_reviewed"] == 12
          and tuned_summary["manual_review"]["meets_brief_minimum"],
          f"{tuned_summary['manual_review']['hallucination_rate_pct']}% hallucination rate")

    table = comparison_table(base_summary, tuned_summary)
    check("Comparison table renders", table.count("\n") >= 13,
          f"{table.count(chr(10)) + 1} rows")
    check("Table marks improvements", "✅" in table)

    print("\n8 · Manual review sheet")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sheet = manual_review_sheet(tuned_records, Path(tmp) / "manual_review.md")
        content = sheet.read_text()
        check("Sheet written", sheet.exists(), f"{len(content.splitlines())} lines")
        check("Label column left blank for the reviewer",
              content.count("**Your label:**") == 12,
              "not pre-filled — an auto-filled label would make the review a rubber stamp")
        check("Automated suggestion shown as a suggestion",
              "Automated suggestion" in content and "suggestion only" in content)

    print("\n" + "=" * 80)
    print("\nComparison table as it will appear in the notebook:\n")
    print(table)
    print("\n" + "=" * 80)
    print("ALL CHECKS PASSED" if not FAILURES else f"{FAILURES} CHECK(S) FAILED")
    sys.exit(1 if FAILURES else 0)
