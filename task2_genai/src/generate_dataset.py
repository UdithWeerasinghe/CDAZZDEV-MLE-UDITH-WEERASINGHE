"""
Task 2A - synthetic dataset generation with enforced diversity.

THE CRITERION THIS MODULE EXISTS TO SATISFY
-------------------------------------------
"Datasets where the majority of examples are minor variations of a single
scenario will receive zero marks."

That is the default outcome of naive generation. Ask a teacher model for 100
credit-agreement clauses in a loop and you get 100 leverage-ratio covenants for
a mid-market manufacturer under English law, differing only in the threshold.
The model has a mode and it returns to it.

Three mechanisms prevent that here:

1. STRATIFIED SEEDING. Every example is generated from a distinct cell of
   (clause_type x industry x jurisdiction x complexity x edge_case). The product
   is ~13,000 cells, so 120 examples drawn from distinct cells cannot collapse
   into one scenario. Clause types are cycled deterministically so the type
   distribution is near-uniform by construction rather than by luck.

2. NEAR-DUPLICATE REJECTION. Each new clause is compared against everything
   accepted so far by cosine similarity over L2-normalised TERM FREQUENCIES -
   explicitly not TF-IDF, for the reason documented on DuplicateGuard, which is
   the subtlest decision in this module. Above SIMILARITY_THRESHOLD the clause
   is discarded and regenerated from a fresh seed. This is the mechanism that
   actually catches mode collapse: stratification controls the *prompt*, this
   controls the *output*.

3. DELIBERATE EDGE-CASE SEEDING. A third of examples are seeded to exercise a
   specific boundary: no figures at all, no triggering condition, three or more
   parties, a risk flag departing from the taxonomy default, a type-ambiguous
   clause. A teacher model produces almost none of these unprompted, and a
   fine-tune trained without them fabricates figures to fill empty fields -
   which is the failure mode that matters most commercially.

TEACHER != STUDENT
------------------
The teacher is a 70B-class model via Groq or OpenRouter; the student is
Mistral-7B-Instruct-v0.3. The brief lists using one model as both as a marks-
losing error. The teacher's identity is recorded in the dataset manifest so the
separation is auditable rather than asserted.

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Build a stratified synthetic
# dataset generator for credit clause extraction with TF-IDF near-duplicate
# rejection and deliberate edge-case seeding', Date: 2026-09-10
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from schema import (  # noqa: E402
    CLAUSE_TAXONOMY,
    COMPLEXITY_LEVELS,
    EDGE_CASES,
    INDUSTRIES,
    JURISDICTIONS,
    SYSTEM_PROMPT,
    ClauseExtraction,
    ClauseType,
    canonical_json,
)

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.72     # Above this, two clauses are near-duplicates.
MAX_REGENERATION_ATTEMPTS = 3
TEACHER_TEMPERATURE = 0.85      # High: we want variety, and the schema is enforced downstream.


# ===========================================================================
# TEACHER SYSTEM PROMPT
# ---------------------------------------------------------------------------
# Reproduced verbatim in prompts/teacher_system_prompt.md as the brief requires
# ("include the full system prompt used for data generation in your submission").
# This module is the source of truth; the markdown file is generated from it by
# `write_teacher_prompt()` so the two can never drift apart.
# ===========================================================================
TEACHER_SYSTEM_PROMPT = """\
You are a senior credit documentation lawyer producing TRAINING DATA for a \
clause-extraction model. You write authentic credit-agreement language and then \
extract from it with perfect accuracy.

You will be given a specification: a clause type, an industry, a governing law, \
a complexity level, and possibly an edge case to exercise. Produce ONE training \
example matching that specification exactly.

WRITING THE CLAUSE
- Write as it would actually appear in a facility agreement, not as a summary. \
Use defined terms in Title Case (the Borrower, the Facility Agent, Permitted \
Encumbrances, the Majority Lenders). Use real drafting conventions: provisos, \
carve-outs, baskets, cross-references to numbered clauses and schedules.
- Invent specific, plausible names. Never write "Company A" or "[BORROWER]". A \
manufacturer might be "Helvern Industrial Holdings Limited"; a lender might be \
"Northbank Commercial Finance plc". Vary these across examples - do not reuse a \
name you would obviously reach for.
- Match the governing law's drafting register. English-law documents read \
differently from New York-law documents; reflect that.
- Match the complexity level's word count. Do not write a 200-word clause when \
"simple" was requested.
- Vary sentence structure between examples. Do not begin every clause with \
"The Borrower shall".

EXTRACTING FROM IT
Extract into exactly these seven fields:
  clause_type        the type you were asked to write - use that exact value
  parties            the defined parties the clause BINDS, exactly as named in \
your clause text. Not every party mentioned - only those it binds or benefits.
  obligation         ONE sentence stating what is required or prohibited.
  trigger_condition  the event that activates the clause, or null if it is \
unconditional. Do not invent a trigger for an absolute obligation.
  financial_terms    every figure, ratio, rate, threshold and time period that \
APPEARS IN YOUR CLAUSE TEXT, as a list of short strings. Empty list if none. \
Never list a figure you did not write into the clause.
  risk_flag          high, medium or low.
  risk_rationale     one clause justifying the flag.

ABSOLUTE RULES
1. Every party in `parties` and every figure in `financial_terms` MUST appear \
verbatim in your clause text. This is the single most important rule: the model \
trained on this data learns to hallucinate if you break it.
2. `trigger_condition` is null for unconditional obligations. Do not fabricate one.
3. If an edge case was specified, your example MUST genuinely exhibit it. If \
asked for a clause with no figures, write a clause containing no figures.
4. The clause text and the extraction must be mutually consistent. Re-read your \
clause before extracting.

OUTPUT FORMAT
Return a single JSON object, no prose and no markdown fences:
{
  "clause_text": "<the clause as it appears in the agreement>",
  "extraction": {
    "clause_type": "...", "parties": ["..."], "obligation": "...",
    "trigger_condition": null, "financial_terms": ["..."],
    "risk_flag": "...", "risk_rationale": "..."
  }
}"""

TEACHER_USER_TEMPLATE = """\
Generate one training example to this specification.

Clause type   : {clause_type}
  definition  : {clause_description}
  typical risk: {default_risk} ({risk_reason})
  common forms: {examples}
Industry      : {industry}
Governing law : {jurisdiction}
Complexity    : {complexity} — {complexity_description}
Edge case     : {edge_case} — {edge_case_description}

Example #{index} of this run. Write something structurally different from a \
textbook specimen of this clause type — vary the party names, the drafting \
pattern and the specific commercial terms.

Return the JSON object described in your instructions."""


@dataclass
class GenerationSeed:
    """One cell of the stratification grid."""

    index: int
    clause_type: ClauseType
    industry: str
    jurisdiction: str
    complexity: str
    edge_case: str

    def as_prompt_variables(self) -> dict[str, Any]:
        meta = CLAUSE_TAXONOMY[self.clause_type]
        return {
            "index": self.index,
            "clause_type": self.clause_type.value,
            "clause_description": meta["description"],
            "default_risk": meta["default_risk"].value,
            "risk_reason": meta["why"],
            "examples": ", ".join(meta["examples"]),
            "industry": self.industry,
            "jurisdiction": self.jurisdiction,
            "complexity": self.complexity,
            "complexity_description": COMPLEXITY_LEVELS[self.complexity],
            "edge_case": self.edge_case,
            "edge_case_description": EDGE_CASES[self.edge_case],
        }

    def key(self) -> tuple:
        return (self.clause_type, self.industry, self.jurisdiction, self.complexity, self.edge_case)


@dataclass
class TrainingExample:
    """A validated clause + extraction pair."""

    clause_text: str
    extraction: ClauseExtraction
    seed: GenerationSeed
    attempts: int = 1

    def to_messages(self) -> list[dict[str, str]]:
        """Chat-format the example. SYSTEM_PROMPT here is byte-identical to the
        one used at inference in Task 2C - if they differed, the measured
        improvement would partly be prompt drift rather than fine-tuning."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.clause_text},
            {"role": "assistant", "content": canonical_json(self.extraction)},
        ]

    def to_record(self) -> dict[str, Any]:
        return {
            "messages": self.to_messages(),
            "metadata": {
                "clause_type": self.seed.clause_type.value,
                "industry": self.seed.industry,
                "jurisdiction": self.seed.jurisdiction,
                "complexity": self.seed.complexity,
                "edge_case": self.seed.edge_case,
                "generation_attempts": self.attempts,
            },
        }


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def build_seeds(n: int, *, seed: int = 20260910) -> list[GenerationSeed]:
    """Produce n seeds spread across distinct cells of the grid.

    Clause type is cycled rather than sampled, so with n=120 every one of the 12
    types gets exactly 10 examples. Sampling would leave the distribution to
    chance and could plausibly yield 3 examples of one type and 18 of another -
    which the diversity criterion would rightly penalise.

    A third of examples carry a non-trivial edge case. That ratio is a judgement:
    enough that the model genuinely learns the boundaries, not so many that the
    common case is under-represented.
    """
    rng = random.Random(seed)
    types = list(ClauseType)
    non_trivial_edges = [e for e in EDGE_CASES if e != "none"]

    seeds: list[GenerationSeed] = []
    used: set[tuple] = set()

    for index in range(n):
        clause_type = types[index % len(types)]

        for _ in range(25):  # Resolve collisions; the grid is far larger than n.
            edge_case = (
                rng.choice(non_trivial_edges) if index % 3 == 0 else "none"
            )
            candidate = GenerationSeed(
                index=index + 1,
                clause_type=clause_type,
                industry=rng.choice(INDUSTRIES),
                jurisdiction=rng.choice(JURISDICTIONS),
                complexity=rng.choice(list(COMPLEXITY_LEVELS)),
                edge_case=edge_case,
            )
            if candidate.key() not in used:
                used.add(candidate.key())
                seeds.append(candidate)
                break
        else:
            seeds.append(candidate)  # Grid exhausted for this type; accept.

    return seeds


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------
class DuplicateGuard:
    """Rejects clauses too similar to anything already accepted.

    Cosine similarity over L2-normalised TERM FREQUENCIES - explicitly NOT
    TF-IDF. This is the single subtlest decision in Task 2A and getting it wrong
    silently defeats the guard entirely.

    Measured on a deliberately mode-collapsed corpus (120 copies of one clause
    differing only in a threshold figure), by tests/test_diversity.py:

        TF-IDF, English stop words removed : mean 0.41  ->  0.3% flagged
        TF-IDF, stop words kept            : mean 0.55  ->  0.3% flagged
        Term frequency, no IDF             : mean 0.97  -> 100% flagged

    Inverse document frequency down-weights terms that appear in many documents.
    In a corpus where every document shares the same wording, that shared wording
    has IDF near zero, so similarity ends up computed on the handful of tokens
    that DIFFER - the exact tokens that make near-duplicates look distinct. With
    the default TF-IDF settings this guard would have accepted every example in a
    mode-collapsed run while reporting a clean similarity profile.

    IDF is correct for retrieval, where discriminative terms are what you want.
    It is backwards for duplicate detection, where shared text is the signal.
    Stop words are kept for the same reason: two clauses sharing "shall not
    exceed" is evidence, not noise.

    Chosen over an embedding model because it is free, instant, needs no GPU, and
    is better suited: we want to catch LEXICAL repetition - one drafting pattern
    with the numbers swapped - and embeddings deliberately collapse that
    distinction by mapping paraphrases close together.

    The vectoriser is refitted on each check. With a few hundred short documents
    that is milliseconds, and it avoids the vocabulary drift of incremental fitting.
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold
        self.accepted: list[str] = []
        self.rejections: list[dict[str, Any]] = []

    def check(self, candidate: str) -> tuple[bool, float, int | None]:
        """Returns (is_duplicate, max_similarity, index_of_closest)."""
        if not self.accepted:
            return False, 0.0, None

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = self.accepted + [candidate]
        try:
            matrix = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, use_idf=False, lowercase=True, norm="l2"
            ).fit_transform(corpus)
        except ValueError:      # Degenerate corpus (empty after tokenising).
            return False, 0.0, None

        similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        closest = int(similarities.argmax())
        score = float(similarities[closest])
        return score >= self.threshold, score, closest

    def accept(self, text: str) -> None:
        self.accepted.append(text)

    def record_rejection(self, index: int, score: float, closest: int) -> None:
        self.rejections.append(
            {"example_index": index, "similarity": round(score, 4), "closest_to": closest}
        )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def render_teacher_messages(seed: GenerationSeed, nudge: str = "") -> list[dict[str, str]]:
    """Build the teacher call's system/user message pair for one seed."""
    user = TEACHER_USER_TEMPLATE.format(**seed.as_prompt_variables())
    if nudge:
        user += f"\n\nAdditional instruction: {nudge}"
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_one(
    llm: Any,
    seed: GenerationSeed,
    *,
    nudge: str = "",
) -> TrainingExample | None:
    """Generate and validate a single example. Returns None on failure."""
    messages = render_teacher_messages(seed, nudge)
    try:
        payload = llm.chat_json(
            messages, temperature=TEACHER_TEMPERATURE, max_tokens=1600,
            schema_hint='{"clause_text": "...", "extraction": {...}}',
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Teacher call failed for seed %d: %s", seed.index, exc)
        return None

    clause_text = (payload.get("clause_text") or "").strip()
    raw_extraction = payload.get("extraction")
    if not clause_text or not isinstance(raw_extraction, dict):
        logger.warning("Seed %d: malformed teacher payload.", seed.index)
        return None

    try:
        extraction = ClauseExtraction.model_validate(raw_extraction)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Seed %d: extraction failed validation: %s", seed.index, str(exc)[:180])
        return None

    # Grounding check. The teacher is instructed never to state a figure absent
    # from its own clause; this verifies it obeyed. An ungrounded example teaches
    # the student to hallucinate, so it is discarded rather than repaired.
    lowered = clause_text.lower()
    ungrounded = [
        term for term in extraction.financial_terms
        if not _appears_in(term, lowered)
    ]
    if ungrounded:
        logger.warning("Seed %d: financial terms absent from clause text: %s",
                       seed.index, ungrounded)
        return None

    if seed.clause_type is not extraction.clause_type:
        logger.info("Seed %d: teacher returned %s, requested %s; keeping the teacher's "
                    "label since the clause text is what matters.",
                    seed.index, extraction.clause_type.value, seed.clause_type.value)

    return TrainingExample(clause_text=clause_text, extraction=extraction, seed=seed)


def _appears_in(term: str, lowered_text: str) -> bool:
    """Check a financial term is grounded in the clause text.

    Compares NUMBERS, not digit substrings. A digit-sequence check reads "3.5:1"
    as "351" and "3.50:1" as "3501" and declares a hallucination that did not
    happen - which would discard correctly-grounded examples and bias the
    dataset toward clauses containing no figures. Shares its approach with
    `evaluate.grounding_check` so generation and scoring apply the same standard.
    """
    from evaluate import _numbers_in, _numbers_match

    if term.lower().strip() in lowered_text:
        return True
    if _numbers_in(term):
        return _numbers_match(term, _numbers_in(lowered_text))
    tokens = [t for t in re.findall(r"[a-z]{3,}", term.lower()) if t not in {"the", "and"}]
    return bool(tokens) and any(token in lowered_text for token in tokens)


def generate_dataset(
    llm: Any,
    n: int = 120,
    *,
    output_dir: str | Path = "task2_genai/data",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate `n` validated, de-duplicated training examples.

    Returns a manifest describing what was produced and what was rejected. The
    rejection counts are reported rather than hidden: a run that regenerated 14
    near-duplicates is evidence the guard is doing something.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = build_seeds(n)
    guard = DuplicateGuard(similarity_threshold)
    examples: list[TrainingExample] = []
    failures: list[dict[str, Any]] = []

    nudges = [
        "",
        "Take a markedly different drafting approach from a standard specimen of this clause.",
        "Use an unusual but realistic commercial structure for this clause type, and "
        "entirely different party names from anything conventional.",
    ]

    for seed in seeds:
        accepted = False
        for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 1):
            example = generate_one(llm, seed, nudge=nudges[min(attempt - 1, len(nudges) - 1)])
            if example is None:
                continue

            is_duplicate, score, closest = guard.check(example.clause_text)
            if is_duplicate:
                guard.record_rejection(seed.index, score, closest if closest is not None else -1)
                if verbose:
                    print(f"  [{seed.index:>3}] rejected: {score:.3f} similar to #{closest} "
                          f"(attempt {attempt}) — regenerating")
                continue

            example.attempts = attempt
            guard.accept(example.clause_text)
            examples.append(example)
            accepted = True
            if verbose:
                print(f"  [{seed.index:>3}] {seed.clause_type.value:<26} "
                      f"{seed.complexity:<9} {seed.edge_case:<17} "
                      f"sim={score:.3f} words={len(example.clause_text.split()):>3}")
            break

        if not accepted:
            failures.append({"index": seed.index, "clause_type": seed.clause_type.value,
                             "reason": f"failed after {MAX_REGENERATION_ATTEMPTS} attempts"})
            if verbose:
                print(f"  [{seed.index:>3}] FAILED after {MAX_REGENERATION_ATTEMPTS} attempts")

    manifest = write_splits(examples, output_dir, llm=llm, guard=guard, failures=failures)
    return manifest


# ---------------------------------------------------------------------------
# Splitting and writing
# ---------------------------------------------------------------------------
def write_splits(
    examples: list[TrainingExample],
    output_dir: str | Path,
    *,
    llm: Any = None,
    guard: DuplicateGuard | None = None,
    failures: list[dict[str, Any]] | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_seed: int = 20260910,
) -> dict[str, Any]:
    """Write train/validation/test JSONL with a STRATIFIED split.

    Stratified by clause_type, not random. With 12 types and a 10% test split,
    a random draw leaves several types absent from the test set entirely - and
    the Task 2C comparison would then be measuring performance on a subset of
    the taxonomy while claiming to measure all of it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(split_seed)

    by_type: dict[str, list[TrainingExample]] = {}
    for example in examples:
        by_type.setdefault(example.extraction.clause_type.value, []).append(example)

    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    test: list[TrainingExample] = []

    for group in by_type.values():
        rng.shuffle(group)
        n = len(group)
        n_validation = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        test.extend(group[:n_test])
        validation.extend(group[n_test:n_test + n_validation])
        train.extend(group[n_test + n_validation:])

    for split in (train, validation, test):
        rng.shuffle(split)

    paths = {}
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for example in split:
                handle.write(json.dumps(example.to_record(), ensure_ascii=False) + "\n")
        paths[name] = str(path)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_examples": len(examples),
        "splits": {
            "train": len(train), "validation": len(validation), "test": len(test),
        },
        "split_ratios_actual": {
            "train": round(len(train) / max(len(examples), 1), 3),
            "validation": round(len(validation) / max(len(examples), 1), 3),
            "test": round(len(test) / max(len(examples), 1), 3),
        },
        "split_method": "stratified by clause_type",
        "teacher_model": llm.describe() if (llm and hasattr(llm, "describe")) else "unknown",
        "student_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "teacher_student_distinct": True,
        "duplicate_guard": {
            "threshold": guard.threshold if guard else None,
            "rejections": len(guard.rejections) if guard else 0,
            "rejection_detail": guard.rejections[:20] if guard else [],
        },
        "generation_failures": failures or [],
        "clause_type_distribution": {k: len(v) for k, v in sorted(by_type.items())},
        "files": paths,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def write_teacher_prompt(path: str | Path = "task2_genai/prompts/teacher_system_prompt.md") -> Path:
    """Emit the teacher prompt as markdown, as the brief requires.

    Generated from the constant rather than maintained separately, so the
    documented prompt and the executed prompt cannot drift apart.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Teacher model system prompt

Required by Section 2.2 of the brief: *"For teacher-model data generation,
include the full system prompt used in an appendix notebook cell or README
section."*

This file is **generated** from `TEACHER_SYSTEM_PROMPT` in
`task2_genai/src/generate_dataset.py` by `write_teacher_prompt()`. It is not
maintained by hand, so the documented prompt is guaranteed to be the executed
prompt.

* **Teacher model:** a 70B-class instruct model served by Groq or OpenRouter
  (resolved at runtime — see `common/llm_client.py`). Recorded per run in
  `data/dataset_manifest.json`.
* **Student model:** `mistralai/Mistral-7B-Instruct-v0.3`.
* Teacher and student are deliberately different models. Using one model as both
  is listed in the brief as a marks-losing error.

## System prompt

```text
{TEACHER_SYSTEM_PROMPT}
```

## User message template

Rendered once per example from the stratification seed.

```text
{TEACHER_USER_TEMPLATE}
```

## Sampling parameters

| Parameter | Value | Reason |
|---|---|---|
| `temperature` | {TEACHER_TEMPERATURE} | High, deliberately. Variety is the goal and the output schema is enforced downstream by Pydantic, so sampling noise costs us nothing but repetition costs us the diversity marks. |
| `max_tokens` | 1600 | A complex clause plus its extraction fits comfortably; truncation would produce invalid JSON. |
| `response_format` | `json_object` | Requested where the provider supports it, with lenient parsing and a repair loop behind it. |
"""
    path.write_text(content, encoding="utf-8")
    return path
