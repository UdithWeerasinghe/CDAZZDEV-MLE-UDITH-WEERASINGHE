# CITATIONS

Required by Section 2.2 of the assessment brief. Every instance of AI assistance and every
piece of adapted open-source code is recorded here.

---

## 1. AI assistance

**Assistant used:** Claude (`claude-sonnet-5`), via the Claude web interface .
**Dates:** 2026-09-10 (build), 2026-09-13 (debugging session, see below).

### On the tier

Section 2.1 permits "GitHub Copilot, Cursor, Claude, ChatGPT, Gemini, or any AI assistant
for code generation and debugging". The Cost Policy states "Free-tier tools only — no
personal expenditure required", which I have read as a candidate-protection clause: no part
of this assessment requires anyone to spend money. **Every runtime dependency in this
repository is free** — Groq and OpenRouter free tiers, Colab free T4, Hugging Face free
hosting, ChromaDB local, DuckDuckGo search. Nothing here needs a paid account to run or to
reproduce.

### Scope of assistance

Claude was used substantially throughout: for architecture discussion, for drafting most of
the implementation code, for writing the test suites, and for drafting documentation. The
architectural decisions recorded in `REFLECTION.md` and in the module docstrings were made
in dialogue and I can defend each of them, including the ones I would now make differently.

Every module carries an inline `# AI-ASSISTED:` comment naming the model, the prompt, and
the date. The full list:

| File                                   | Prompt summary                                                                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `common/llm_client.py`                 | Provider-agnostic OpenAI-compatible client, Groq primary with OpenRouter fallback, runtime model discovery, exponential backoff |
| `task1_financial/src/indicators.py`    | SMA, EMA, Wilder RSI, MACD, Bollinger Bands from first principles with correct smoothing and ddof semantics                     |
| `task1_financial/src/data_pipeline.py` | yfinance OHLCV + news ingestion with a multi-source fallback chain and a null-safe summary dictionary                           |
| `task1_financial/src/schemas.py`       | Pydantic schemas for headline sentiment and trading signal, with a validator rejecting value-restatement                        |
| `task1_financial/src/prompts.py`       | Versioned prompt library using `string.Template` with system/user separation                                                    |
| `task1_financial/src/llm_analysis.py`  | Batched classification with item reconciliation, a bounded Pydantic repair loop, rules-based fallback                           |
| `task1_financial/src/report.py`        | Self-contained HTML brief with an embedded base64 matplotlib chart                                                              |
| `task1_financial/tests/*.py`           | Independent pure-Python reference implementations; validator and end-to-end tests                                               |
| `task2_genai/src/schema.py`            | Credit-clause extraction schema with a closed taxonomy and per-field validation                                                 |
| `task2_genai/src/generate_dataset.py`  | Stratified synthetic generation with near-duplicate rejection and edge-case seeding                                             |
| `task2_genai/src/diversity.py`         | Length distribution, keyword frequency, pairwise similarity, per-axis entropy                                                   |
| `task2_genai/src/evaluate.py`          | ROUGE-L, BERTScore, LLM judge, per-field accuracy, automated grounding checks                                                   |
| `task2_genai/src/rag_fallback.py`      | Perplexity-gated ChromaDB retrieval fallback with a calibrated threshold                                                        |
| `task2_genai/tests/*.py`               | Tests proving the diversity analyser flags mode collapse and quantifying ROUGE-L insensitivity                                  |
| `task3_agentic/src/tracing.py`         | JSONL agent tracer recording tool, inputs, truncated output, duration                                                           |
| `task3_agentic/src/tools.py`           | Five LangChain tools returning structured envelopes with actionable failure suggestions                                         |
| `task3_agentic/src/schemas.py`         | Pydantic handoff schemas and clarification request/response pair                                                                |
| `task3_agentic/src/memory.py`          | Persistent per-ticker JSON research cache with staleness and corruption handling                                                |
| `task3_agentic/src/agents.py`          | LangGraph single-agent and two-agent graphs with structurally enforced tool restriction                                         |
| `task3_agentic/dashboard.py`           | Streamlit dashboard over `agent_trace.jsonl`                                                                                    |
| Notebooks (all three)                  | Structure, narration and rubric self-check cells                                                                                |

### Second session — execution and debugging (2026-09-13)

**Assistant used:** Claude (`claude-sonnet-5`) via Claude Code (CLI), on Windows 11.

The first session wrote the code; this one ran it end to end for the first time and fixed
what that surfaced. Every change below was diagnosed from an actual failing run, and each
is documented at the point of the fix:

| File                                                    | Defect found by running it                                                                                                                                                                                                                  | Fix                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task3_agentic/src/agents.py`                           | `NameError: add_messages` / `NameError: State` — LangGraph >= 1.0 resolves state annotations with `get_type_hints()` against module globals, but `from __future__ import annotations` makes them strings and both names were function-local | Import `add_messages` at module level; drop the unresolvable `State` parameter annotations                                                                                                                                                                                 |
| `task3_agentic/src/tools.py`                            | `llm_sentiment` always failed: `cannot import name 'AggregateSentiment' from 'schemas'` — Task 1 and Task 3 both define `schemas`, and Task 3's directory precedes Task 1's on `sys.path`                                                   | Load Task 1's module by file path under a distinct `sys.modules` key                                                                                                                                                                                                       |
| `task3_agentic/src/agents.py`                           | Provider 400, `'messages' : minimum number of items is 1` — a failed handoff left `b_messages` empty and Agent B invoked the model with no transcript                                                                                       | Seed a degraded brief so Agent B continues on qualitative sources and declares the gap                                                                                                                                                                                     |
| `common/llm_client.py`                                  | Groq returns HTTP **413** (not 429) when a request exceeds the free tier's 8,000 tokens/minute; 413 was not retryable, so it escaped instead of failing over                                                                                | Add 413 to `RETRYABLE_STATUS`                                                                                                                                                                                                                                              |
| `common/llm_client.py`                                  | Model discovery fell back to `aion-labs/aion-2.0`, a **paid** model, violating the brief's free-tier cost policy; all four preferred OpenRouter models had been delisted                                                                    | Add a `free_only_suffix` constraint and refresh the OpenRouter lists to free, tool-capable models                                                                                                                                                                          |
| `task3_agentic/src/agents.py`                           | Tool observations were replayed at 6,000 characters each, exhausting the per-minute token budget within a few iterations                                                                                                                    | Cap replayed observations at 2,000 characters; the full output still reaches `agent_trace.jsonl`                                                                                                                                                                           |
| `task1_financial/src/data_pipeline.py`                  | For NVDA the yfinance feed returned ten headlines of which seven were about other companies, making per-headline sentiment meaningless                                                                                                      | Reorder the source chain so Google News RSS (a ticker-scoped query) is tried first                                                                                                                                                                                         |
| Task 3 notebook                                         | A reused `calls_before` variable silently broke the short-term-memory rubric check; one failing tool aborted the five-tool smoke cell                                                                                                       | Rename the variables, bind tools to the memory probe, make the smoke cell report per-tool failure                                                                                                                                                                          |
| Task 2 notebook                                         | `MANUAL_LABELS` shipped placeholder labels that would have been reported as a real hallucination rate                                                                                                                                       | Add a `LABELS_REVIEWED` gate that refuses to present the figure as genuine until the review is actually done                                                                                                                                                               |
| `task3_agentic/src/schemas.py`, `agents.py`, `tools.py` | Agent A could not produce a valid `QuantBrief` in three attempts, losing the structured handoff and the critique loop. It was being asked to transcribe ~20 nested numeric fields out of tool output under `extra="forbid"`                 | Split the handoff: `QuantJudgement` asks the model only for regime, findings, gaps and confidence; `build_quant_brief` fills the figures in from `ToolContext.results`, the payloads the tools actually returned                                                           |
| `task3_agentic/src/tools.py`                            | One terse `brief_reason` failed `HeadlineSentimentBatch` validation for the entire batch, discarding eleven good classifications with the bad one                                                                                           | Validate per headline, keep what passes, and report `headlines_rejected` rather than hiding the drop                                                                                                                                                                       |
| `task3_agentic/src/agents.py`                           | The agents never had the failover the README claimed: `build_chat_model` returned a bare single-provider `ChatOpenAI`, so Groq's HTTP 413 (8,000 tokens/minute) and 429 (200,000 tokens/day) each ended the run outright                    | `FailoverChatModel` builds one model per configured provider and chains them with `.with_fallbacks()`, binding tools to each first because `RunnableWithFallbacks` has no `bind_tools`                                                                                     |
| `task3_agentic/src/agents.py`                           | The transcript grows monotonically through an agent loop, so a long run eventually exceeds the per-minute ceiling regardless of behaviour                                                                                                   | `_trim_to_budget` keeps the system prompt and the most recent turns; `max_tokens` cut 2048 -> 1024, since Groq counts reserved output against the same budget                                                                                                              |
| `common/llm_client.py`                                  | OpenRouter model ranking was by size, not by whether the model answers: `nemotron-3-ultra-550b:free` timed out at 90s, gemma returned 429, inkling 403                                                                                      | Ranked by measured tool-calling latency on this account (`nemotron-3-super-120b-a12b:free`, 2.3s)                                                                                                                                                                          |
| `common/llm_client.py`                                  | Both configured free tiers were exhausted inside one working session - Groq at 200,000 tokens/day, OpenRouter at 50 free-model requests/day - leaving the pipeline with nowhere to go                                                       | Added Google AI Studio (Gemini) as a third provider: free, no credit card, 1,500 requests/day, function calling, OpenAI-compatible. Cerebras was evaluated and rejected because its docs now require a verified payment method, and the brief forbids personal expenditure |
| `common/llm_client.py`, `task3_agentic/src/agents.py`   | Gemini's OpenAI-compatible `/models` returns fully-qualified ids (`models/gemini-2.5-flash`), which would never match a preference list written in bare names                                                                               | Normalise `models/x` to `x` during discovery in both resolvers                                                                                                                                                                                                             |
| `common/llm_client.py`, `task3_agentic/src/agents.py` | Gemini's OpenAI-compatible endpoint drops Gemini 3's `thought_signature`, so the FIRST agent tool call succeeded and the replay of it answered 400. A single-turn probe cannot see this; only replaying a real agent turn does | Added `Provider.agent_client` and reached Gemini through `langchain-google-genai` (native API), which carries the field. The same two-turn exchange then completed in 3.6-5.2s |
| `task3_agentic/src/agents.py` | `ChatGoogleGenerativeAI` returns `content` as a LIST of blocks where `ChatOpenAI` returns a string: `.strip()` raised `AttributeError`, and `str(content)` silently produced a Python repr that could never parse as the JSON a structured call expects | One `message_text()` helper behind every `.content` read, unit-checked against six shapes including thinking-block skipping |
| `task3_agentic/src/agents.py` | Gemini's native API validates turn order (`400 - function call turn must come immediately after a user turn or a function response turn`). The prompt trimmer selected a window purely by size and could cut through a tool exchange, orphaning it. OpenAI-compatible providers tolerated the same transcript | `_repair_turn_order()` walks the trimmed window forward to the first genuine user turn |
| `common/llm_client.py` | Gemini's free tier is capped **per model per day**, not at the headline project figure: `gemini-3.6-flash` and `gemini-3.5-flash` allow **20 requests/day** each, fewer than one Task 3 run needs. Google publishes this nowhere - it appears only in AI Studio or inside a 429's `QuotaFailure` | Read the quotas out of the API, then ranked the preference lists by measured headroom so the `-flash-lite` and `-preview` variants lead and the flagships sit last |
| Task 2 notebook | `SFTConfig.__init__() got an unexpected keyword argument 'warmup_ratio'` on Colab. TRL v1.13 / transformers v5 renamed `warmup_ratio` to `warmup_steps` (a float below 1 is still read as a ratio) and `max_seq_length` to `max_length` | Build the config from the documented names, remap to whatever the installed signature accepts, print every remap, and assert the settings the justification table argues for so a rename cannot silently revert one to a default |
| Task 2 notebook | Transformers v5 deprecated `torch_dtype` on `from_pretrained` in favour of `dtype`, across three model-load sites | Resolve the spelling once through a `DTYPE_KW()` helper rather than in call sites that would drift apart |
| Task 2 notebook | `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` at the first optimiser step. `fp16=True` drives `torch.amp.GradScaler`, whose unscale kernel has no BFloat16 implementation; Mistral's checkpoint is bf16 and transformers v5 honours a checkpoint's own dtype, so adapters inherited bf16 despite fp16 being requested | Probe `torch.cuda.is_bf16_supported()` once and drive the bnb compute dtype, the model load and the trainer's AMP mode from that single decision; force trainable parameters to fp32 (what QLoRA does anyway) and assert no non-fp32 trainable tensor survives |
| Task 2 notebook | A git merge between the local branch and a "Created using Colab" commit left `<<<<<<< HEAD` conflict markers **committed** inside the .ipynb, so the notebook was no longer valid JSON and would not open | Resolved to the Colab side, which carried both the compatibility fixes and 11 executed cells, rather than the local side which had identical code and no outputs |
| Task 2 notebook | Every Colab run re-generated the dataset from the teacher (~137 calls including duplicate regeneration), exhausting all three free tiers. It also produced DIFFERENT splits from the committed `dataset_manifest.json` and diversity report, so the fine-tune trained on data the published analysis did not describe | Load the committed splits by default behind an explicit `REGENERATE` flag; generation is opt-in, and the artefacts stay coherent with the evaluation |
| `common/llm_client.py` | Free-tier quota is metered **per model per day**, but `LLMClient` resolved one model per provider - so `gemini-3.5-flash-lite` hitting its 500/day cap retired the whole Gemini provider while each sibling model still had an untouched 500 | `_resolve_models()` returns a ranked list and `_active` holds one entry per model, taking the chain from 3 links to 7-8 and roughly quadrupling the daily Gemini budget |
| Task 2 notebook | `Your session crashed after using all available RAM` at the merge step, ~67% through loading. Mistral-7B in fp16 is 7.24e9 x 2 = **14.5 GB** against Colab free tier's **~12.7 GB** of system RAM; the T4's 15 GB VRAM is no better once the CUDA context is counted. An arithmetic problem, not a garbage-collection one | Keep `merge_and_unload()` as the first path since the brief names it, and fall back to a shard-wise merge straight from the safetensors when the model cannot be held: peak memory becomes one shard (~5 GB) and the weights are identical, `W <- W + (B @ A) * alpha/r`. The adapter-key parser is unit-tested against PEFT's naming variants |
| Task READMEs | All three Colab badges pointed at `YOUR_USERNAME/CDAZZDEV-MLE-Udith` - wrong account and wrong repository - so every reviewer-facing "Open in Colab" link would 404 | Pointed at the real repository; removed every remaining placeholder from the docs |
| `REFLECTION.md` | 609 words against the brief's stated 600-word maximum | Trimmed to 586 and corrected the stated count |

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

| Component                  | Source of the definition                                                                                                                                                    |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RSI (Wilder's smoothing)   | J. Welles Wilder, _New Concepts in Technical Trading Systems_ (1978)                                                                                                        |
| MACD                       | Gerald Appel's original formulation; recursive EMA (`adjust=False`) as used by charting platforms                                                                           |
| Bollinger Bands            | John Bollinger's definition; population σ (`ddof=0`)                                                                                                                        |
| ROUGE-L                    | Lin, C-Y. (2004), _ROUGE: A Package for Automatic Evaluation of Summaries_. LCS with β=1.2 as in the original package                                                       |
| QLoRA configuration        | Dettmers et al. (2023), _QLoRA: Efficient Finetuning of Quantized LLMs_ — NF4, double quantisation, and the finding that targeting all linear layers matters more than rank |
| Normalised Shannon entropy | Standard information-theoretic measure, normalised by log(k)                                                                                                                |

---

## 3. Libraries used

All free and open source. Versions pinned in `requirements.txt`.

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jinja2`, `pydantic`, `openai`,
`yfinance`, `requests`, `langgraph`, `langchain-core`, `langchain-openai`, `ddgs`,
`transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`, `chromadb`,
`bert-score`, `huggingface-hub`, `streamlit`.

---

## 4. Data sources

| Source                         | Use                                              | Access                                |
| ------------------------------ | ------------------------------------------------ | ------------------------------------- |
| Yahoo Finance (via `yfinance`) | OHLCV price history, fundamentals; news fallback | Free, no key                          |
| Yahoo Finance RSS              | News fallback                                    | Free, no key                          |
| Google News RSS                | News retrieval (primary; ticker-scoped query)    | Free, no key                          |
| NewsAPI                        | Optional news fallback                           | Free tier, skipped when no key is set |
| DuckDuckGo (via `ddgs`)        | Agent web search                                 | Free, no key                          |
| Groq                           | LLM inference (primary)                          | Free tier                             |
| OpenRouter                     | LLM inference (fallback)                         | Free tier                             |
| Hugging Face Hub               | Base model download and fine-tuned model hosting | Free                                  |

**All clause text in `task2_genai/data/` is synthetic**, generated by the teacher model for
this assessment. No real credit agreement, and no confidential or proprietary document, was
used at any point. Party names in the dataset are invented; any resemblance to a real
entity is coincidental.
