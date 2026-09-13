# Task 2 — Domain-Specific Fine-Tuning Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UdithWeerasinghe/CDAZZDEV-MLE-UDITH-WEERASINGHE/blob/main/task2_genai/notebooks/task2_finetuning_pipeline.ipynb)

**Notebook:** [`notebooks/task2_finetuning_pipeline.ipynb`](notebooks/task2_finetuning_pipeline.ipynb) · **needs a T4 GPU** · ~90 min

**Use case:** credit-agreement clause extraction → strict JSON.
**Base model:** `mistralai/Mistral-7B-Instruct-v0.3` (4-bit NF4 QLoRA).
**Teacher:** a 70B-class model on Groq/OpenRouter — recorded per run in `data/dataset_manifest.json`. Teacher ≠ student.

| Criterion | Marks | Where |
|---|---:|---|
| Use case quality | 10 | `src/schema.py` — problem statement + closed 12-type taxonomy |
| Dataset size (100+) + teacher prompt | 5 | `prompts/teacher_system_prompt.md` (generated from code) |
| Dataset diversity | 10 | `src/diversity.py` — verified in `tests/test_diversity.py` |
| Format and 80/10/10 split | 5 | Chat JSONL, **stratified** by clause type |
| QLoRA implementation | 10 | Notebook §2 — NF4 + double quant, fp16 compute |
| Hyperparameter justification | 15 | Notebook §2 — 22-row table, no unexplained defaults |
| Loss monitoring | 10 | Per-epoch train/val loss + curve |
| Model saved | 5 | `merge_and_unload()` in fp16 on CPU, pushed to HF Hub |
| ROUGE-L comparison | 8 | `src/evaluate.py::comparison_table` |
| Additional metric | 7 | BERTScore F1 **and** an LLM judge with a structured rubric |
| Hallucination rate | 7 | Manual review sheet + automated grounding check |
| Qualitative analysis | 8 | Notebook §3.5 |
| **Bonus** — RAG fallback | +5 | `src/rag_fallback.py` — perplexity-gated ChromaDB |

## The bug the tests caught

The duplicate guard originally used TF-IDF cosine similarity. Measured on 120 copies of one
clause differing only in a number:

| Vectoriser | Mean similarity | Pairs flagged |
|---|---:|---:|
| TF-IDF, stop words removed | 0.41 | 0.3% |
| TF-IDF, stop words kept | 0.55 | 0.3% |
| **Term frequency, no IDF** | **0.97** | **100%** |

IDF down-weights terms common to many documents. When every document is near-identical, the
shared text has IDF ≈ 0 and similarity is computed on the tokens that *differ*. The guard
would have accepted a dataset the brief awards **zero marks** for, while reporting a clean
profile. IDF is right for retrieval and backwards for duplicate detection.

Found only because `tests/test_diversity.py` asserts the analyser **fails** on bad input, not
merely that it passes on good input.

## Why ROUGE-L is reported *and* criticised

ROUGE-L is required, so it is here. But for schema-constrained output it is weak: the test
suite measures a **wrong `clause_type` moving ROUGE-L by 0.02** while field accuracy goes
1.0 → 0.0. So JSON validity rate, per-field accuracy and an automated grounding check are
reported alongside it — the last being the number a credit risk team would actually care about.

## Run (no GPU needed)

```bash
python task2_genai/tests/test_diversity.py   # proves mode collapse is detected
python task2_genai/tests/test_evaluate.py    # proves the metrics separate base from tuned
```
