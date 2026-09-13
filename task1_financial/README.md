# Task 1 — LLM-Powered Equity Research Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UdithWeerasinghe/CDAZZDEV-MLE-UDITH-WEERASINGHE/blob/main/task1_financial/notebooks/task1_equity_research.ipynb)

**Notebook:** [`notebooks/task1_equity_research.ipynb`](notebooks/task1_equity_research.ipynb) · CPU only, ~10 min · needs `GROQ_API_KEY` or `OPENROUTER_API_KEY`

| Criterion | Marks | Where |
|---|---:|---|
| OHLCV fetch, 2y+, no hardcoded dates | 10 | `src/data_pipeline.py::fetch_ohlcv` |
| Five indicators from first principles | 25 | `src/indicators.py` — verified in `tests/test_indicators.py` |
| 10+ headlines from a working free source | 10 | `src/data_pipeline.py::fetch_news` — 4-source chain |
| Summary dictionary complete | 10 | `src/data_pipeline.py::build_summary` |
| Robustness | 5 | Null-safe throughout; named constants; no magic numbers |
| Per-headline JSON + aggregation | 10 | `src/llm_analysis.py::classify_headlines` |
| Signal reasoning over combinations | 15 | `src/schemas.py::TradingSignal` validator |
| Structured output validation | 10 | Pydantic + repair loop + `validation_failures.jsonl` |
| Prompt engineering | 5 | `src/prompts.py` — versioned, zero business logic |
| **Bonus** — rendered brief | +5 | `src/report.py` |

## Three correctness decisions worth defending

1. **RSI uses Wilder's smoothing, not `rolling(14).mean()`.** The rolling-mean version is
   *Cutler's* RSI — a different indicator. `tests/test_indicators.py` measures them diverging
   by **6.7 points on average, 22 at worst**, and never re-converging: Wilder has infinite
   memory, the rolling window does not.
2. **MACD uses `adjust=False` EMAs.** pandas defaults to `adjust=True`, which re-weights early
   observations and puts the signal-line crossovers — the thing you actually trade — in the
   wrong places.
3. **Bollinger Bands use the population σ (`ddof=0`).** pandas defaults to `ddof=1`. At n=20
   the correction is √(20/19), so a 2σ band computed with the default is **2.6% too wide** —
   enough to change whether a close counts as a band touch.

Each is verified against an **independent pure-Python reference** — explicit loops, no pandas.
Verifying pandas with more pandas lets a shared misunderstanding pass silently.

## The anti-regurgitation validator

The 15-mark criterion is *"reasons over indicator combinations, not just echoes values"* —
normally a human judgement. `TradingSignal.justification` must pass three machine checks:
3–5 sentences, at least two distinct named indicators, and relational language (*confirms,
diverges, despite, outweighs*). A model listing "RSI is 62, MACD is 1.4" contains no such
word and is rejected, handed its own validator error, and asked again. If it still cannot
synthesise, a deterministic fallback takes over — **flagged as a fallback**, because a
degraded honest answer beats a confident fabricated one.

`tests/test_schemas.py` proves the rejection actually happens.

## Run

```bash
python task1_financial/tests/test_indicators.py          # 14/14
python task1_financial/tests/test_schemas.py             # 11/11
python task1_financial/tests/test_end_to_end_offline.py  # full chain, stubbed LLM, no network
```
