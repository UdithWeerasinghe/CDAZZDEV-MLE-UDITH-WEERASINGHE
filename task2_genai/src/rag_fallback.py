"""
Task 2 bonus - retrieval-augmented fallback for low-confidence extractions.

The brief: "when the fine-tuned model's confidence falls below a defined
threshold (measured by perplexity or LLM self-rating), retrieve relevant context
from a ChromaDB vector store built from your training domain documents and
re-query the model."

CONFIDENCE IS MEASURED BY PERPLEXITY, NOT SELF-RATING
-----------------------------------------------------
Both are offered by the brief. Perplexity is the better choice here and the
reason is worth stating: a fine-tuned model asked to rate its own confidence has
been trained to produce this schema fluently, so it rates almost everything
highly - the fine-tuning that improved the output also destroyed the calibration
of its self-assessment. Perplexity is computed from the logits and cannot be
talked around. It is also free: we already ran the forward pass.

Concretely: token-level perplexity of the generated JSON under the fine-tuned
model itself. High perplexity means the model found its own output surprising,
which on a schema-constrained task reliably indicates an unfamiliar clause type
or unusual drafting - exactly the cases where retrieved examples help.

THE THRESHOLD IS CALIBRATED, NOT GUESSED
----------------------------------------
`calibrate_threshold` runs the model over the validation split and sets the
threshold at a chosen percentile of the observed perplexity distribution. A
hardcoded "perplexity > 3.0" would be meaningless: the scale depends on the
model, the tokeniser and the task. Calibrating on held-out data makes the
threshold mean "unusual relative to what this model normally produces".

THE CORPUS IS THE TRAINING SPLIT
--------------------------------
Retrieval returns the k most similar clause/extraction pairs from the TRAINING
split only. Retrieving from validation or test would leak the answer into the
prediction and invalidate every number in Task 2C.

# AI-ASSISTED: Claude (claude-sonnet-5), Prompt: 'Implement a perplexity-gated
# ChromaDB RAG fallback for a fine-tuned extraction model with a calibrated
# threshold', Date: 2026-09-10
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import SYSTEM_PROMPT, validate_extraction  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "credit_clauses"
DEFAULT_K = 3
DEFAULT_PERCENTILE = 75      # Fall back on the most-surprising quartile.


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------
def sequence_perplexity(model: Any, tokenizer: Any, prompt: str, completion: str) -> float:
    """Perplexity of `completion` given `prompt`, under `model`.

    Prompt tokens are masked to -100 so the loss is computed over the generated
    JSON only. Including the prompt would mostly measure how predictable the
    clause text is, which tells us nothing about the model's confidence in its
    own answer.
    """
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + completion, return_tensors="pt",
                         add_special_tokens=False).input_ids
    full_ids = full_ids.to(model.device)

    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100      # Mask the prompt.

    if (labels != -100).sum() == 0:
        return float("inf")

    with torch.no_grad():
        loss = model(input_ids=full_ids, labels=labels).loss

    return float(torch.exp(loss).item())


def calibrate_threshold(
    perplexities: Sequence[float],
    percentile: int = DEFAULT_PERCENTILE,
) -> dict[str, float]:
    """Set the fallback threshold from an observed perplexity distribution."""
    import numpy as np

    finite = np.array([p for p in perplexities if np.isfinite(p)], dtype="float64")
    if finite.size == 0:
        return {"threshold": float("inf"), "n": 0}

    return {
        "threshold": round(float(np.percentile(finite, percentile)), 4),
        "percentile": percentile,
        "n": int(finite.size),
        "mean": round(float(finite.mean()), 4),
        "median": round(float(np.median(finite)), 4),
        "p25": round(float(np.percentile(finite, 25)), 4),
        "p90": round(float(np.percentile(finite, 90)), 4),
        "max": round(float(finite.max()), 4),
    }


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class ClauseRetriever:
    """ChromaDB store over the training split.

    Uses Chroma's default embedding function (all-MiniLM-L6-v2, downloaded once
    and run locally on CPU). No API key, no per-query cost, and it comfortably
    outperforms lexical matching for "find me a clause that does the same job as
    this one" - which is the retrieval we actually want here, unlike the
    duplicate detection in generate_dataset.py where lexical matching is
    deliberately the right tool.
    """

    def __init__(
        self,
        persist_directory: str | Path = "task2_genai/data/chroma",
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        import chromadb

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def build(self, train_path: str | Path, *, reset: bool = True) -> dict[str, Any]:
        """Index the TRAINING split. Never validation or test - that would leak."""
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:  # noqa: BLE001 - absent collection is fine
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

        documents, metadatas, ids = [], [], []
        for index, line in enumerate(
            Path(train_path).read_text(encoding="utf-8").splitlines()
        ):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = {m["role"]: m["content"] for m in record.get("messages", [])}
            clause, extraction = messages.get("user"), messages.get("assistant")
            if not clause or not extraction:
                continue

            documents.append(clause)
            metadatas.append({
                "extraction": extraction,
                "clause_type": record.get("metadata", {}).get("clause_type", "unknown"),
                "complexity": record.get("metadata", {}).get("complexity", "unknown"),
            })
            ids.append(f"train-{index}")

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        logger.info("Indexed %d training clauses into '%s'.", len(documents), self.collection_name)
        return {
            "indexed": len(documents),
            "collection": self.collection_name,
            "source": str(train_path),
            "note": "Training split only — indexing validation or test would leak "
                    "reference answers into predictions.",
        }

    def retrieve(self, clause: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            logger.warning("Retriever queried but the collection is empty.")
            return []
        result = self.collection.query(
            query_texts=[clause], n_results=min(k, self.collection.count())
        )
        hits = []
        for index in range(len(result["ids"][0])):
            metadata = result["metadatas"][0][index]
            hits.append({
                "clause": result["documents"][0][index],
                "extraction": metadata.get("extraction", ""),
                "clause_type": metadata.get("clause_type"),
                "distance": round(float(result["distances"][0][index]), 4),
                "similarity": round(1.0 - float(result["distances"][0][index]), 4),
            })
        return hits


# ---------------------------------------------------------------------------
# The fallback pipeline
# ---------------------------------------------------------------------------
FEWSHOT_PREFIX = (
    "Here are similar clauses from the reference corpus, with their correct "
    "extractions. Use them to guide the format and the level of detail, but "
    "extract ONLY what appears in the new clause — do not carry over parties or "
    "figures from these examples.\n\n"
)


@dataclass
class FallbackResult:
    """One extraction, recording whether the fallback fired and what it changed."""

    clause: str
    first_pass_output: str
    first_pass_perplexity: float
    threshold: float
    fallback_triggered: bool
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    second_pass_output: str | None = None
    second_pass_perplexity: float | None = None
    final_output: str = ""
    improved: bool | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "perplexity_first_pass": round(self.first_pass_perplexity, 3),
            "threshold": round(self.threshold, 3),
            "fallback_triggered": self.fallback_triggered,
            "retrieved_count": len(self.retrieved),
            "retrieved_types": [h["clause_type"] for h in self.retrieved],
            "perplexity_second_pass": (
                round(self.second_pass_perplexity, 3)
                if self.second_pass_perplexity is not None else None
            ),
            "improved": self.improved,
        }


def extract_with_fallback(
    clause: str,
    *,
    model: Any,
    tokenizer: Any,
    retriever: ClauseRetriever,
    threshold: float,
    k: int = DEFAULT_K,
    max_new_tokens: int = 400,
    generate_fn: Any = None,
) -> FallbackResult:
    """Extract once; if the model was surprised by its own output, retry with context.

    `generate_fn(prompt) -> str` is injectable so the pipeline can be unit-tested
    without a GPU.
    """
    def _default_generate(prompt: str) -> str:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    generate = generate_fn or _default_generate

    def build_prompt(user_content: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )

    # --- Pass 1 -------------------------------------------------------------
    prompt = build_prompt(clause)
    first = generate(prompt).strip()
    perplexity = sequence_perplexity(model, tokenizer, prompt, first)

    result = FallbackResult(
        clause=clause, first_pass_output=first, first_pass_perplexity=perplexity,
        threshold=threshold, fallback_triggered=perplexity > threshold, final_output=first,
    )
    if not result.fallback_triggered:
        return result

    # --- Pass 2, with retrieved context -------------------------------------
    logger.info("Perplexity %.3f exceeds threshold %.3f; retrieving context.",
                perplexity, threshold)
    hits = retriever.retrieve(clause, k=k)
    result.retrieved = hits
    if not hits:
        return result

    context = FEWSHOT_PREFIX + "\n\n".join(
        f"--- Reference clause {i + 1} ({hit['clause_type']}, "
        f"similarity {hit['similarity']:.3f}) ---\n{hit['clause']}\n"
        f"Correct extraction:\n{hit['extraction']}"
        for i, hit in enumerate(hits)
    ) + f"\n\n--- NEW CLAUSE TO EXTRACT ---\n{clause}"

    augmented_prompt = build_prompt(context)
    second = generate(augmented_prompt).strip()

    result.second_pass_output = second
    result.second_pass_perplexity = sequence_perplexity(model, tokenizer, augmented_prompt, second)

    # Prefer the second pass only if it is actually better. "Valid where the
    # first was invalid" is the decisive test; perplexity between two different
    # prompts is not directly comparable, so it is recorded but not used to choose.
    first_valid = validate_extraction(first)[0] is not None
    second_valid = validate_extraction(second)[0] is not None

    if second_valid and not first_valid:
        result.final_output, result.improved = second, True
    elif second_valid and first_valid:
        result.final_output, result.improved = second, None   # Both valid; no clear winner.
    else:
        result.final_output, result.improved = first, False   # Retry did not help.

    return result


def demo_before_after(results: Sequence[FallbackResult]) -> str:
    """Render a concrete before-and-after, which the bonus explicitly asks for."""
    fired = [r for r in results if r.fallback_triggered and r.second_pass_output]
    if not fired:
        return ("The fallback did not trigger on any test example — every extraction fell "
                "below the calibrated perplexity threshold. That is a legitimate outcome "
                "and is itself a result: on this test set the fine-tuned model was never "
                "surprised by its own output.")

    example = next((r for r in fired if r.improved), fired[0])
    lines = [
        f"Fallback fired on {len(fired)} of {len(results)} examples.",
        "",
        "WORKED EXAMPLE",
        "=" * 78,
        "CLAUSE",
        example.clause[:700],
        "",
        f"PASS 1 — no retrieval (perplexity {example.first_pass_perplexity:.3f} > "
        f"threshold {example.threshold:.3f})",
        "-" * 78,
        example.first_pass_output[:700],
        f"  parses and validates: {validate_extraction(example.first_pass_output)[0] is not None}",
        "",
        f"RETRIEVED {len(example.retrieved)} REFERENCE CLAUSES",
        "-" * 78,
        *[f"  {i + 1}. [{h['clause_type']}] similarity {h['similarity']:.3f} — {h['clause'][:110]}…"
          for i, h in enumerate(example.retrieved)],
        "",
        "PASS 2 — with retrieved context",
        "-" * 78,
        (example.second_pass_output or "")[:700],
        f"  parses and validates: "
        f"{validate_extraction(example.second_pass_output or '')[0] is not None}",
        "",
        f"OUTCOME: {'improved' if example.improved else 'no improvement' if example.improved is False else 'both valid'}",
    ]
    return "\n".join(lines)
