"""
Verification that the diversity analyser detects mode collapse.

A diversity report that says PASS on everything is worthless unless it also says
FAIL on something. This builds two synthetic datasets - one genuinely varied, one
deliberately mode-collapsed (the same clause with different numbers, which is
exactly the failure the brief awards zero marks for) - and asserts the analyser
separates them.

Run:  python task2_genai/tests/test_diversity.py

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Write a test proving the
# diversity analyser flags a mode-collapsed dataset and passes a varied one',
# Date: 2026-09-10
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "task2_genai" / "src")]

from diversity import analyse, print_report                      # noqa: E402
from schema import (                                              # noqa: E402
    CLAUSE_TAXONOMY, COMPLEXITY_LEVELS, EDGE_CASES, INDUSTRIES,
    JURISDICTIONS, SYSTEM_PROMPT, ClauseType,
)

FAILURES = 0

# Distinct drafting patterns per clause type, so the "good" dataset varies in
# structure rather than only in the numbers.
PATTERNS: dict[ClauseType, list[str]] = {
    ClauseType.FINANCIAL_COVENANT: [
        "The Borrower shall procure that the ratio of Consolidated Total Net Debt to "
        "Consolidated EBITDA in respect of any Relevant Period shall not exceed {r}:1, "
        "tested on each Quarter Date by reference to the Compliance Certificate "
        "delivered under Clause {c}.",
        "{p} undertakes that Interest Cover shall at no time be less than {r}:1, "
        "calculated in accordance with Schedule {s} and certified semi-annually by "
        "the Chief Financial Officer of {p}.",
    ],
    ClauseType.NEGATIVE_COVENANT: [
        "No Obligor shall, without the prior written consent of the Majority Lenders, "
        "incur, create or permit to subsist any Financial Indebtedness other than "
        "Permitted Indebtedness, save that the aggregate principal amount outstanding "
        "under this exception shall not at any time exceed {m}.",
        "{p} shall not dispose of any asset forming part of the Security Assets except "
        "for Permitted Disposals, provided that the aggregate consideration for all "
        "such disposals in any financial year does not exceed {m}.",
    ],
    ClauseType.AFFIRMATIVE_COVENANT: [
        "{p} shall deliver to the Facility Agent its audited consolidated financial "
        "statements within {d} days after the end of each of its financial years, "
        "prepared in accordance with the Accounting Principles.",
        "The Borrower undertakes to maintain insurances with reputable insurers against "
        "such risks and to such extent as is usual for companies carrying on the same "
        "business, and to supply certificates of currency to the Agent on request.",
    ],
    ClauseType.EVENT_OF_DEFAULT: [
        "It shall be an Event of Default if any Obligor fails to pay on the due date any "
        "amount payable under a Finance Document, unless payment is made within {d} "
        "Business Days of the due date and the failure arose solely from administrative "
        "or technical error.",
        "An Event of Default occurs if any Financial Indebtedness of any member of the "
        "Group is declared due and payable prior to its stated maturity, provided that "
        "no Event of Default arises where the aggregate amount falls below {m}.",
    ],
    ClauseType.REPRESENTATION_WARRANTY: [
        "{p} represents and warrants to each Finance Party that it is duly incorporated "
        "and validly existing under the laws of its jurisdiction of incorporation and "
        "has the power to own its assets and carry on its business as it is being "
        "conducted, such representation being repeated on each Repeat Date.",
        "Each Obligor represents that no litigation, arbitration or administrative "
        "proceeding which, if adversely determined, would reasonably be expected to "
        "have a Material Adverse Effect has been started or threatened against it.",
    ],
    ClauseType.INTEREST_PROVISION: [
        "The rate of interest on each Loan for each Interest Period is the percentage "
        "rate per annum which is the aggregate of the applicable Margin of {pc} per cent "
        "and Term SOFR, payable in arrear on the last day of each Interest Period.",
        "If an Obligor fails to pay any amount payable by it under a Finance Document on "
        "its due date, interest shall accrue on the overdue amount at a rate which is "
        "{pc} per cent per annum higher than the rate which would have been payable.",
    ],
    ClauseType.REPAYMENT_PREPAYMENT: [
        "The Borrower shall repay the Facility in {n} equal quarterly instalments, the "
        "first such instalment falling due on the first Quarter Date after the end of "
        "the Availability Period and the final instalment on the Termination Date.",
        "The Borrower shall apply an amount equal to {pc} per cent of Excess Cash Flow "
        "in prepayment of the Loans within {d} days after the date on which the annual "
        "financial statements are delivered.",
    ],
    ClauseType.SECURITY_COLLATERAL: [
        "As continuing security for the payment of the Secured Obligations, {p} charges "
        "by way of first fixed charge all of its present and future right, title and "
        "interest in and to the Charged Property described in Schedule {s}.",
        "{p} shall procure that each member of the Group which holds assets with a book "
        "value exceeding {m} accedes to the Security Agreement as an Additional Chargor "
        "within {d} days of becoming such a member.",
    ],
    ClauseType.CHANGE_OF_CONTROL: [
        "If any person or group of persons acting in concert other than a Permitted "
        "Holder acquires beneficial ownership of more than {pc} per cent of the issued "
        "share capital of {p}, each Lender may by notice cancel its Commitments and "
        "declare its participation in all outstanding Loans immediately due and payable.",
        "Upon the occurrence of a Change of Control, the Borrower shall promptly notify "
        "the Agent and shall prepay the Facility in full together with accrued interest "
        "within {d} days of such notification.",
    ],
    ClauseType.CONDITIONS_PRECEDENT: [
        "The Lenders shall not be obliged to make the Facility available unless the Agent "
        "has received all of the documents and other evidence listed in Part {s} of "
        "Schedule 2 in form and substance satisfactory to it.",
        "No Utilisation Request may be delivered until the Agent has confirmed receipt of "
        "legal opinions from counsel in each Relevant Jurisdiction and certified copies "
        "of the constitutional documents of each Obligor.",
    ],
    ClauseType.INDEMNITY: [
        "{p} shall, within {d} Business Days of demand, indemnify each Finance Party "
        "against any cost, loss or liability incurred by that Finance Party as a result "
        "of the occurrence of any Event of Default or a failure by an Obligor to pay any "
        "amount due under a Finance Document.",
        "The Borrower shall indemnify each Finance Party against any loss or liability "
        "which that Finance Party incurs as a consequence of any funds being converted "
        "from one currency into another, whether or not such conversion was reasonable.",
    ],
    ClauseType.GOVERNING_LAW_JURISDICTION: [
        "This Agreement and any non-contractual obligations arising out of or in "
        "connection with it are governed by {j}. The courts of that jurisdiction have "
        "exclusive jurisdiction to settle any Dispute.",
        "Each Obligor irrevocably submits to the exclusive jurisdiction of the courts "
        "specified in this Clause and waives any objection on the ground of "
        "inconvenient forum, this submission being for the benefit of the Finance "
        "Parties only.",
    ],
}

PREFIXES = ["Helvern", "Northbank", "Calder", "Vantris", "Brookmere", "Ashgrove", "Merrindale",
            "Southport", "Kelverton", "Ravenlea", "Thornbury", "Wexmoor", "Aldreth", "Pinehaven",
            "Corrindale", "Beckwith", "Stanmere", "Larchfield", "Ostmark", "Quillane"]
SUFFIXES = ["Industrial Holdings Limited", "Commercial Finance plc", "Group Holdings B.V.",
            "Capital Partners LLC", "Infrastructure Trust", "Logistics Pte Ltd",
            "Energy Holdings GmbH", "Retail Group Limited", "Healthcare Services Inc",
            "Telecom Holdings Limited"]


def make_record(clause: str, payload: dict, metadata: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clause},
            {"role": "assistant", "content": json.dumps(payload)},
        ],
        "metadata": metadata,
    }


# Optional provisos, sampled independently, so two examples sharing a base
# template still differ substantially. This mimics real teacher generation, where
# free-form drafting produces combinatorial rather than template variety.
PROVISOS = [
    "For the purposes of this Clause, any amount denominated in a currency other than "
    "the Base Currency shall be converted at the Agent's Spot Rate of Exchange.",
    "No double-counting shall arise where a single item falls within more than one "
    "basket or exception set out in this Agreement.",
    "This provision shall be tested by reference to the most recent Compliance "
    "Certificate delivered to the Agent.",
    "Any breach capable of remedy shall not constitute an Event of Default if remedied "
    "within the applicable grace period.",
    "The Agent may, acting on the instructions of the Majority Lenders, waive compliance "
    "in whole or in part by written notice.",
    "Nothing in this Clause obliges any Finance Party to make enquiry as to the accuracy "
    "of information supplied by an Obligor.",
    "Amounts prepaid may not be redrawn and shall reduce the Commitments rateably.",
    "This obligation survives the repayment in full of the Secured Obligations.",
]

OPENERS = ["", "Notwithstanding any other provision of this Agreement, ",
           "Subject to the Agreed Security Principles, ", "Without prejudice to Clause 18, ",
           "At all times during the Security Period, "]


def build_diverse(n: int = 120, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    types = list(ClauseType)
    edges = [e for e in EDGE_CASES if e != "none"]
    records = []

    for index in range(n):
        clause_type = types[index % len(types)]
        party = f"{rng.choice(PREFIXES)} {rng.choice(SUFFIXES)}"
        jurisdiction = rng.choice(JURISDICTIONS)
        template = rng.choice(PATTERNS[clause_type])

        clause = template.format(
            p=party, j=jurisdiction,
            r=f"{rng.uniform(1.5, 5.0):.2f}",
            m=f"USD {rng.randrange(2, 90) * 500_000:,}",
            d=rng.choice([5, 10, 15, 20, 30, 45, 60, 90, 120]),
            n=rng.choice([8, 12, 16, 20, 24]),
            pc=f"{rng.uniform(0.5, 7.5):.2f}",
            c=f"{rng.randrange(15, 28)}.{rng.randrange(1, 9)}",
            s=rng.randrange(2, 9),
        )
        clause = rng.choice(OPENERS) + clause

        # Vary length to exercise the complexity strata.
        complexity = rng.choice(list(COMPLEXITY_LEVELS))
        if complexity == "complex":
            clause += " " + " ".join(rng.sample(PROVISOS, 3))
        elif complexity == "moderate":
            clause += " " + rng.choice(PROVISOS)

        edge = rng.choice(edges) if index % 3 == 0 else "none"
        payload = {
            "clause_type": clause_type.value,
            "parties": [party] if edge != "multi_party" else [
                party, f"{rng.choice(PREFIXES)} {rng.choice(SUFFIXES)}", "the Security Agent"],
            "obligation": f"Obligation arising under the {clause_type.value.replace('_', ' ')}.",
            "trigger_condition": None if edge == "unconditional" else "on the occurrence of the specified event",
            "financial_terms": [] if edge == "no_financial_terms" else [f"{rng.uniform(1, 6):.2f}:1"],
            "risk_flag": CLAUSE_TAXONOMY[clause_type]["default_risk"].value,
            "risk_rationale": "Consistent with the taxonomy default for this clause type.",
        }
        records.append(make_record(clause, payload, {
            "clause_type": clause_type.value,
            "industry": rng.choice(INDUSTRIES),
            "jurisdiction": jurisdiction,
            "complexity": complexity,
            "edge_case": edge,
        }))
    return records


def build_collapsed(n: int = 120, seed: int = 7) -> list[dict]:
    """The failure mode: one scenario, 120 times, different numbers."""
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        ratio = f"{rng.uniform(3.0, 4.0):.2f}"
        clause = (
            f"The Borrower shall procure that the ratio of Consolidated Total Net Debt to "
            f"Consolidated EBITDA in respect of any Relevant Period shall not exceed "
            f"{ratio}:1, tested on each Quarter Date by reference to the Compliance "
            f"Certificate delivered under Clause 21.2."
        )
        payload = {
            "clause_type": "financial_covenant",
            "parties": ["Acme Holdings Limited"],
            "obligation": "Maintain the leverage ratio below the stated threshold.",
            "trigger_condition": "tested on each Quarter Date",
            "financial_terms": [f"{ratio}:1"],
            "risk_flag": "high",
            "risk_rationale": "Objectively measurable and triggers default directly.",
        }
        records.append(make_record(clause, payload, {
            "clause_type": "financial_covenant", "industry": "manufacturing",
            "jurisdiction": "English law", "complexity": "moderate", "edge_case": "none",
        }))
    return records


def check(label: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  PASS {label:<50} {detail}")
    else:
        FAILURES += 1
        print(f"  FAIL {label:<50} {detail}")


def apply_guard(records: list[dict]) -> tuple[list[dict], int]:
    """Run the real DuplicateGuard over records, as generation does.

    The raw fixture is a crude stand-in built from two templates per clause
    type, so it deliberately contains some near-duplicates. Passing it through
    the actual guard is what generation does before anything reaches the
    dataset, so this exercises the guard and the analyser together rather than
    testing the analyser against an unrealistically clean input.
    """
    from generate_dataset import DuplicateGuard

    guard = DuplicateGuard()
    kept, rejected = [], 0
    for record in records:
        text = next(m["content"] for m in record["messages"] if m["role"] == "user")
        is_duplicate, _, _ = guard.check(text)
        if is_duplicate:
            rejected += 1
            continue
        guard.accept(text)
        kept.append(record)
    return kept, rejected


if __name__ == "__main__":
    print("\nDiversity analyser verification\n" + "=" * 78)

    print("\nA · Diverse dataset, after the DuplicateGuard (as generation produces it)")
    raw = build_diverse(150)
    guarded, rejected = apply_guard(raw)
    print(f"  Guard rejected {rejected} of {len(raw)} fixture examples as near-duplicates; "
          f"{len(guarded)} retained")
    good = analyse(guarded)
    verdict = good["verdict"]
    check("Guard actually rejected something", rejected > 0,
          f"{rejected} rejected — the guard is not a no-op")
    check("Overall verdict is PASS", verdict["overall"] == "PASS",
          f"{verdict['passed']}/{verdict['total']} checks")
    check("Zero near-duplicate pairs",
          good["pairwise_similarity"]["near_duplicate_pairs"] == 0,
          f"mean cosine {good['pairwise_similarity']['mean']}, "
          f"max {good['pairwise_similarity']['max']}")
    check("Clause types near-uniform",
          good["stratification"]["clause_type"]["normalised_entropy"] > 0.95,
          f"entropy {good['stratification']['clause_type']['normalised_entropy']}")
    check("Length genuinely varies",
          good["length_distribution"]["words"]["coefficient_of_variation"] > 0.30,
          f"CV {good['length_distribution']['words']['coefficient_of_variation']}, "
          f"{good['length_distribution']['words']['min']}–"
          f"{good['length_distribution']['words']['max']} words")
    check("Party names mostly unique",
          good["entity_uniqueness"]["uniqueness_ratio"] > 0.60,
          f"ratio {good['entity_uniqueness']['uniqueness_ratio']}")
    check("Edge cases present",
          len(good["stratification"]["edge_case"]["distribution"]) >= 3,
          f"{sorted(good['stratification']['edge_case']['distribution'])}")

    print("\nB · Mode-collapsed dataset (what earns zero marks)")
    collapsed_raw = build_collapsed(120)
    _, collapsed_rejected = apply_guard(collapsed_raw)
    check("Guard rejects nearly all of a collapsed corpus",
          collapsed_rejected >= 118,
          f"{collapsed_rejected}/120 rejected — generation would regenerate all of these")

    bad = analyse(collapsed_raw)
    bad_verdict = bad["verdict"]
    check("Overall verdict is REVIEW, not PASS", bad_verdict["overall"] == "REVIEW",
          f"only {bad_verdict['passed']}/{bad_verdict['total']} checks passed")
    check("Near-duplicates detected",
          bad["pairwise_similarity"]["near_duplicate_pairs"] > 1000,
          f"{bad['pairwise_similarity']['near_duplicate_pairs']} pairs "
          f"({bad['pairwise_similarity']['near_duplicate_pct']}%)")
    check("Mean similarity is high",
          bad["pairwise_similarity"]["mean"] > 0.8,
          f"mean cosine {bad['pairwise_similarity']['mean']} vs "
          f"{good['pairwise_similarity']['mean']} for the diverse set")
    check("Clause type entropy collapses",
          bad["stratification"]["clause_type"]["normalised_entropy"] == 0.0,
          "one type accounts for 100% of the data")
    check("Party reuse detected",
          bad["entity_uniqueness"]["uniqueness_ratio"] < 0.05,
          f"ratio {bad['entity_uniqueness']['uniqueness_ratio']} — one name, 120 times")

    print("\nC · Separation")
    gap = bad["pairwise_similarity"]["mean"] - good["pairwise_similarity"]["mean"]
    check("Analyser separates the two clearly", gap > 0.5,
          f"mean-similarity gap of {gap:.3f}")

    print("\n" + "=" * 78)
    print("\nFull report for the diverse dataset:\n")
    print_report(good)

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED" if not FAILURES else f"{FAILURES} CHECK(S) FAILED")
    sys.exit(1 if FAILURES else 0)
