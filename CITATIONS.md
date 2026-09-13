# CITATIONS

Required by Section 2.2 of the assessment brief. Every instance of AI assistance and every
piece of adapted open-source code is recorded here.

---

## 1. AI assistance

**Assistant used:** Claude (`claude-opus-5`), via the Claude web interface (paid tier).
**Dates:** 2026-09-10.

### On the tier

Section 2.1 permits "GitHub Copilot, Cursor, Claude, ChatGPT, Gemini, or any AI assistant
for code generation and debugging". The Cost Policy states "Free-tier tools only — no
personal expenditure required", which I have read as a candidate-protection clause: no part
of this assessment requires anyone to spend money. **Every runtime dependency in this
repository is free** — Groq and OpenRouter free tiers, Colab free T4, Hugging Face free
hosting, ChromaDB local, DuckDuckGo search. Nothing here needs a paid account to run or to
reproduce.

I used a paid Claude subscription I already hold as the coding assistant. If that reading of
the cost policy is wrong, I would rather be told than have it discovered, so I am stating it
plainly rather than leaving it to be inferred.

### Scope of assistance

Claude was used substantially throughout: for architecture discussion, for drafting most of
the implementation code, for writing the test suites, and for drafting documentation. The
architectural decisions recorded in `REFLECTION.md` and in the module docstrings were made
in dialogue and I can defend each of them, including the ones I would now make differently.

Every module carries an inline `# AI-ASSISTED:` comment naming the model, the prompt, and
the date. The full list:

| File | Prompt summary |
|---|---|
| `common/llm_client.py` | Provider-agnostic OpenAI-compatible client, Groq primary with OpenRouter fallback, runtime model discovery, exponential backoff |
| `task1_financial/src/indicators.py` | SMA, EMA, Wilder RSI, MACD, Bollinger Bands from first principles with correct smoothing and ddof semantics |
| `task1_financial/src/data_pipeline.py` | yfinance OHLCV + news ingestion with a multi-source fallback chain and a null-safe summary dictionary |
| `task1_financial/src/schemas.py` | Pydantic schemas for headline sentiment and trading signal, with a validator rejecting value-restatement |
| `task1_financial/src/prompts.py` | Versioned prompt library using `string.Template` with system/user separation |
| `task1_financial/src/llm_analysis.py` | Batched classification with item reconciliation, a bounded Pydantic repair loop, rules-based fallback |
| `task1_financial/src/report.py` | Self-contained HTML brief with an embedded base64 matplotlib chart |
| `task1_financial/tests/*.py` | Independent pure-Python reference implementations; validator and end-to-end tests |
| `task2_genai/src/schema.py` | Credit-clause extraction schema with a closed taxonomy and per-field validation |
| `task2_genai/src/generate_dataset.py` | Stratified synthetic generation with near-duplicate rejection and edge-case seeding |
| `task2_genai/src/diversity.py` | Length distribution, keyword frequency, pairwise similarity, per-axis entropy |
| `task2_genai/src/evaluate.py` | ROUGE-L, BERTScore, LLM judge, per-field accuracy, automated grounding checks |
| `task2_genai/src/rag_fallback.py` | Perplexity-gated ChromaDB retrieval fallback with a calibrated threshold |
| `task2_genai/tests/*.py` | Tests proving the diversity analyser flags mode collapse and quantifying ROUGE-L insensitivity |
| `task3_agentic/src/tracing.py` | JSONL agent tracer recording tool, inputs, truncated output, duration |
| `task3_agentic/src/tools.py` | Five LangChain tools returning structured envelopes with actionable failure suggestions |
| `task3_agentic/src/schemas.py` | Pydantic handoff schemas and clarification request/response pair |
| `task3_agentic/src/memory.py` | Persistent per-ticker JSON research cache with staleness and corruption handling |
| `task3_agentic/src/agents.py` | LangGraph single-agent and two-agent graphs with structurally enforced tool restriction |
| `task3_agentic/dashboard.py` | Streamlit dashboard over `agent_trace.jsonl` |
| Notebooks (all three) | Structure, narration and rubric self-check cells |

### Teacher model used for data generation (Task 2A)

The full system prompt is in **`task2_genai/prompts/teacher_system_prompt.md`**, generated
from the `TEACHER_SYSTEM_PROMPT` constant in `task2_genai/src/generate_dataset.py` so the
documented prompt cannot drift from the executed one. The concrete teacher model is
resolved at runtime and recorded per run in `task2_genai/data/dataset_manifest.json` — a
70B-class instruct model on Groq or OpenRouter, distinct from the fine-tuned student
(`mistralai/Mistral-7B-Instruct-v0.3`).

---

## 2. Adapted open-source code

No source file was copied from an existing repository. The algorithms below are implemented
from their published definitions rather than adapted from an implementation, which is why
`task1_financial/tests/test_indicators.py` verifies each against an independent
pure-Python reference:

| Component | Source of the definition |
|---|---|
| RSI (Wilder's smoothing) | J. Welles Wilder, *New Concepts in Technical Trading Systems* (1978) |
| MACD | Gerald Appel's original formulation; recursive EMA (`adjust=False`) as used by charting platforms |
| Bollinger Bands | John Bollinger's definition; population σ (`ddof=0`) |
| ROUGE-L | Lin, C-Y. (2004), *ROUGE: A Package for Automatic Evaluation of Summaries*. LCS with β=1.2 as in the original package |
| QLoRA configuration | Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs* — NF4, double quantisation, and the finding that targeting all linear layers matters more than rank |
| Normalised Shannon entropy | Standard information-theoretic measure, normalised by log(k) |

---

## 3. Libraries used

All free and open source. Versions pinned in `requirements.txt`.

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jinja2`, `pydantic`, `openai`,
`yfinance`, `requests`, `langgraph`, `langchain-core`, `langchain-openai`, `ddgs`,
`transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`, `chromadb`,
`bert-score`, `huggingface-hub`, `streamlit`.

---

## 4. Data sources

| Source | Use | Access |
|---|---|---|
| Yahoo Finance (via `yfinance`) | OHLCV price history, fundamentals, news | Free, no key |
| Yahoo Finance RSS | News fallback | Free, no key |
| Google News RSS | News fallback | Free, no key |
| NewsAPI | Optional news fallback | Free tier, skipped when no key is set |
| DuckDuckGo (via `ddgs`) | Agent web search | Free, no key |
| Groq | LLM inference (primary) | Free tier |
| OpenRouter | LLM inference (fallback) | Free tier |
| Hugging Face Hub | Base model download and fine-tuned model hosting | Free |

**All clause text in `task2_genai/data/` is synthetic**, generated by the teacher model for
this assessment. No real credit agreement, and no confidential or proprietary document, was
used at any point. Party names in the dataset are invented; any resemblance to a real
entity is coincidental.
