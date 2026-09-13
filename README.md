# CDAZZDEV-MLE-Udith

Submission for the **CDAZZDEV Senior Machine Learning Engineer** technical assessment.
All three tasks attempted.

| Task | Domain | Notebook | Marks |
|---|---|---|---|
| **1** | Financial AI — LLM-powered equity research pipeline | [`task1_financial/notebooks/task1_equity_research.ipynb`](task1_financial/notebooks/task1_equity_research.ipynb) | 100 (+5 bonus) |
| **2** | Generative AI — QLoRA fine-tuning with rigorous evaluation | [`task2_genai/notebooks/task2_finetuning_pipeline.ipynb`](task2_genai/notebooks/task2_finetuning_pipeline.ipynb) | 100 (+5 bonus) |
| **3** | Agentic Workflows — multi-agent financial research system | [`task3_agentic/notebooks/task3_agentic_system.ipynb`](task3_agentic/notebooks/task3_agentic_system.ipynb) | 100 (+5 bonus) |

**Required reading:** [`CITATIONS.md`](CITATIONS.md) · [`REFLECTION.md`](REFLECTION.md)

---

## The shape of this submission

Logic lives in importable modules under `task*/src/`. The notebooks are thin execution
harnesses that run those modules and display the results.

That is deliberate. Section 2 of the brief says the assessment evaluates *"architectural
judgement and engineering decisions — not your ability to generate code"*, and that every
part will be defended in interview. Logic buried in notebook cells cannot be unit-tested,
cannot be reused across tasks, and cannot be reasoned about independently of the run that
produced it. Every non-obvious decision is documented at the point it was made, in the
module docstring, with the reasoning rather than just the conclusion.

### Verification, not assertion

Each task ships a test suite that runs with **no GPU, no API key, and no network**:

```bash
python task1_financial/tests/test_indicators.py          # 14 tests
python task1_financial/tests/test_schemas.py             # 11 tests
python task1_financial/tests/test_end_to_end_offline.py  # full Task 1 chain, stubbed LLM
python task2_genai/tests/test_diversity.py               # diversity analyser
python task2_genai/tests/test_evaluate.py                # evaluation suite
python task3_agentic/tests/test_task3_offline.py         # schemas, tracer, cache, restriction
```

Three of these are worth singling out, because each proves something a reviewer would
otherwise have to take on trust:

* **`test_indicators.py`** verifies all five indicators against an **independent pure-Python
  reference** — explicit loops, no pandas. Verifying pandas code with more pandas code
  sharing the same assumptions would let a shared misunderstanding pass silently. RSI matches
  Wilder to `0.000e+00`, and the test measures Wilder vs the common (wrong) rolling-mean
  variant diverging by **6.7 RSI points on average, 22 at worst**.

* **`test_schemas.py`** proves the Task 1B validator actually **rejects** a
  value-regurgitating justification — the specific failure the 15-mark criterion names. A
  validator that accepts everything is worse than none.

* **`test_diversity.py`** proves the diversity analyser **flags a deliberately mode-collapsed
  dataset**, not just that it passes a good one. This test caught a real bug — see below.

### The bug the tests caught

Task 2A's near-duplicate guard originally used TF-IDF cosine similarity, the obvious choice.
Measured against a corpus of 120 copies of one clause differing only in a number:

| Vectoriser | Mean similarity | Pairs flagged |
|---|---:|---:|
| TF-IDF, English stop words removed | 0.41 | 0.3% |
| TF-IDF, stop words kept | 0.55 | 0.3% |
| **Term frequency, no IDF** | **0.97** | **100%** |

IDF down-weights terms appearing in many documents. When every document shares the same
wording, that wording has IDF ≈ 0, so similarity is computed almost entirely on the tokens
that *differ* — the very tokens making near-duplicates look distinct. The guard would have
accepted a dataset the brief awards **zero marks** for, while reporting a clean similarity
profile. IDF is right for retrieval and backwards for duplicate detection.

---

## Quick start

### Colab (recommended — this is how the notebooks are meant to run)

Open a notebook, add your keys to Colab **Secrets** (key icon, left sidebar), run all.
Task 2 needs Runtime → Change runtime type → **T4 GPU**; Tasks 1 and 3 do not.

| Secret | Needed for | Get one |
|---|---|---|
| `GROQ_API_KEY` | All three tasks (primary inference) | https://console.groq.com/keys |
| `OPENROUTER_API_KEY` | Automatic fallback when Groq rate-limits | https://openrouter.ai/keys |
| `HF_TOKEN` | Task 2B only, to push the merged model | https://huggingface.co/settings/tokens (WRITE scope) |

At least one of Groq or OpenRouter is required. Both is better — Task 3's agent runs long
enough to hit a free-tier limit, and the fallback is what keeps it alive.

### Local (Windows 11 + VS Code)

Full walkthrough in [`SETUP.md`](SETUP.md). Short version:

```powershell
git clone https://github.com/YOUR_USERNAME/CDAZZDEV-MLE-Udith.git
cd CDAZZDEV-MLE-Udith
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit .env and add your keys
python task1_financial/tests/test_indicators.py
```

Tasks 1 and 3 run fine locally on CPU. **Task 2B needs a GPU** — use Colab for it.

---

## No credentials in this repository

Section 2.3 lists a hardcoded key as an automatic disqualification. `common/llm_client.py`
has no code path that accepts a literal key: `get_secret()` resolves environment → Colab
Secrets → interactive prompt, in that order. `.env` is gitignored; `.env.example` holds
placeholder names only. The agent tracer redacts any argument whose name contains `key`,
`token`, `secret` or `password` before writing it to the trace.

---

## Repository layout

```
common/
  llm_client.py              Provider-agnostic client, Groq → OpenRouter failover,
                             runtime model discovery, exponential backoff

task1_financial/
  src/indicators.py          Five indicators from first principles (no TA-Lib)
  src/data_pipeline.py       yfinance OHLCV + 4-source news fallback chain + summary dict
  src/schemas.py             Pydantic contracts; the anti-regurgitation validator
  src/prompts.py             Versioned prompt library, no business logic
  src/llm_analysis.py        Batched classification, repair loop, rules-based fallback
  src/report.py              Self-contained HTML/Markdown research brief
  tests/                     Independent reference verification
  notebooks/                 Task 1 Colab notebook
  output/                    Generated briefs, validation failure log, run manifest

task2_genai/
  src/schema.py              Use case definition, closed clause taxonomy, output contract
  src/generate_dataset.py    Stratified generation, duplicate guard, teacher prompt
  src/diversity.py           Length, keyword, similarity, entropy analysis
  src/evaluate.py            ROUGE-L, BERTScore, LLM judge, grounding checks
  src/rag_fallback.py        Perplexity-gated ChromaDB retrieval (bonus)
  prompts/                   Teacher system prompt (generated from code)
  data/                      train/validation/test JSONL + dataset manifest
  tests/                     Diversity and evaluation verification
  notebooks/                 Task 2 Colab notebook

task3_agentic/
  src/tools.py               The five agent tools, structured failure envelopes
  src/schemas.py             QuantBrief handoff contract, ResearchReport
  src/agents.py              LangGraph single-agent and two-agent graphs
  src/memory.py              Persistent per-ticker research cache
  src/tracing.py             agent_trace.jsonl writer
  dashboard.py               Streamlit trace dashboard (bonus)
  logs/agent_trace.jsonl     Required observability artefact
  cache/                     Persistent research briefs
  tests/                     Schema, tracer, cache, tool-restriction verification
  notebooks/                 Task 3 Colab notebook
```

---

## Headline design decisions

**Task 1 — the maths is the deliverable.** RSI uses Wilder's recursive smoother, MACD uses
`adjust=False` EMAs, Bollinger Bands use the population σ. Each is the *uncommon* choice and
each is the correct one; the tests measure the difference. Prices are split- and
dividend-adjusted, because an unadjusted series puts a −50% single-bar gap at a stock split
that detonates RSI and drags the SMA-200 for 200 sessions.

**Task 1B — rubric criteria enforced in code.** "Reasons over combinations, not just echoes
values" is normally a human judgement. Here the `TradingSignal` validator rejects a
justification unless it runs 3–5 sentences, names two distinct indicators, and contains
relational language. Failures are logged to `validation_failures.jsonl`, the model gets one
repair attempt, and a deterministic fallback takes over if it still cannot — flagged as a
fallback, because a degraded honest answer beats a confident fabricated one.

**Task 2 — Mistral-7B over Phi-3-mini, deliberately.** Phi-3 is the brief's own example and
easier on a T4. Mistral fails at strict schema adherence in a large, obvious, reproducible
way, which makes the *delta* — and therefore the evaluation section — far more informative.
The cost was real memory engineering; the five-configuration OOM log in the notebook is a
deliverable, not an embarrassment.

**Task 2C — ROUGE-L is reported and also criticised.** It is required by the brief, so it is
here. But it is a weak metric for schema-constrained output: the test suite measures a wrong
`clause_type` moving ROUGE-L by **0.02** while field accuracy goes from 1.0 to 0.0. JSON
validity rate, per-field accuracy and an automated grounding check sit alongside it.

**Task 3 — LangGraph over CrewAI.** Three criteria are about things being *enforced*, not
claimed. Agent B cannot call the price tools because the tool objects are never placed in its
node's list. The observe-replan cycle is not logged — it *is* the `agent → tools → agent`
edge. More code than CrewAI would need; far easier to defend line by line.

**Task 3B — `data_gaps` is mandatory.** Agent B cannot see Agent A's tools. If a volatility
call fails and the field were optional, Agent A omits it and Agent B reads the silence as
"measured and unremarkable" — then sizes a hedge on a number nobody took. Forcing the
declaration turns a silent failure into a visible one.

---

## Known limitations

Stated plainly rather than discovered in interview:

* **Task 2's test set is synthetic and teacher-generated**, so it shares the teacher's blind
  spots. The format improvement would transfer; the accuracy improvement is measured against
  a distribution whose shape the model has effectively seen.
* **Single training run.** With ~96 examples, differences of a few points are within noise.
  A 3-seed sweep with variance bars would be honest.
* **The critique loop is capped at one round** — no principled convergence criterion, and
  multi-round critique risks oscillation.
* **Free-tier tool calling is the weakest link in Task 3.** Llama 3.3 occasionally emits a
  malformed tool call, costing an iteration. Production would need retry-on-malformed-call.
* **The hedge-strategy validator** checks for a figure and a named instrument. That catches
  hand-waving; it cannot verify the recommendation is *sound*.

Longer discussion in [`REFLECTION.md`](REFLECTION.md).
