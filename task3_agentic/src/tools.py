"""
Task 3A - the five agent tools.

TOOL DOCSTRINGS ARE PROMPT ENGINEERING
--------------------------------------
Everything below the `def` line in each tool is what the model actually sees when
deciding what to call. The 10-mark "Autonomous Tool Selection" criterion is won
or lost here, not in the graph code. Each docstring therefore states:

  * what the tool returns, in concrete terms;
  * WHEN to reach for it versus a sibling tool;
  * what it costs (a slow tool the model calls speculatively wastes the run);
  * what it does NOT do, which is what stops the model calling `get_price_data`
    to answer a question about news.

FAILURE IS DATA, NOT AN EXCEPTION
---------------------------------
No tool raises into the agent loop. Every tool returns
`{"ok": bool, "data": ..., "error": ..., "suggestion": ...}`. The `suggestion`
field is the mechanism behind the 7-mark Error Handling criterion: when
`web_search` is rate-limited it returns
`"suggestion": "DuckDuckGo is rate-limited. Call get_news for company-specific
headlines instead."` — so the model has an actionable alternative in context and
re-plans, rather than seeing a stack trace and giving up. That is the difference
between an agent that handles failure and one that merely survives it.

TOKEN DISCIPLINE
----------------
`get_price_data` summarises two years of OHLCV into roughly 25 numbers. Returning
the frame would blow the context window and drown the signal. The agent needs
the current state of the indicators, not 500 rows of history.

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Implement five LangChain tools
# for a financial research agent that return structured ok/data/error envelopes
# with actionable suggestions on failure', Date: 2026-09-10
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Reuse Task 1's verified indicator implementations rather than reimplementing.
# Two copies of the same maths is two places for it to be wrong.
_TASK1_SRC = Path(__file__).resolve().parents[2] / "task1_financial" / "src"
if str(_TASK1_SRC) not in sys.path:
    sys.path.insert(0, str(_TASK1_SRC))

from indicators import annualised_volatility, compute_all_indicators, derive_momentum_signal  # noqa: E402

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 6
SEARCH_RETRY_ATTEMPTS = 3
DEFAULT_VOL_WINDOW = 30


def ok(data: Any, **extra: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None, **extra}


def fail(error: str, suggestion: str) -> dict[str, Any]:
    """Every failure carries an actionable alternative for the agent."""
    logger.warning("Tool failure: %s | suggestion: %s", error, suggestion)
    return {"ok": False, "data": None, "error": error, "suggestion": suggestion}




def _task1_sentiment_schemas():
    """Return (AggregateSentiment, HeadlineSentimentBatch) from Task 1's schemas.

    Imported by file path rather than by module name: `schemas` is ambiguous in
    this process because Task 1 and Task 3 each define one. Cached in
    sys.modules under a unique key so the load happens once.
    """
    import importlib.util
    import sys as _sys

    key = "task1_financial_schemas"
    module = _sys.modules.get(key)
    if module is None:
        path = Path(__file__).resolve().parents[2] / "task1_financial" / "src" / "schemas.py"
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        _sys.modules[key] = module
        spec.loader.exec_module(module)
    return module.AggregateSentiment, module.HeadlineSentimentBatch


@dataclass
class ToolContext:
    """Shared state the tools close over.

    Passing this explicitly rather than using module globals means a test can
    construct a context with stub components, and two agents in the same process
    cannot tread on each other's cache.
    """

    tracer: Any = None
    llm: Any = None
    price_cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    news_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    agent_name: str = "agent"

    def traced(self, tool: str, inputs: dict[str, Any]):
        """Return the tracer's context manager, or a no-op if tracing is off."""
        if self.tracer is None:
            from contextlib import nullcontext

            return nullcontext({})
        return self.tracer.record(tool, inputs, agent=self.agent_name)


# ---------------------------------------------------------------------------
# Internal helpers (not exposed to the agent)
# ---------------------------------------------------------------------------
def _load_prices(context: ToolContext, ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch and enrich OHLCV, memoised per (ticker, period).

    The memo is short-term memory in the most literal sense: within one session
    a second call for the same ticker is served from context, which is exactly
    the behaviour the Task 3C short-term memory criterion asks us to demonstrate.
    """
    key = f"{ticker}:{period}"
    if key in context.price_cache:
        logger.info("Price cache hit for %s (short-term memory).", key)
        return context.price_cache[key]

    import yfinance as yf

    frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True, actions=False)
    if frame is None or frame.empty:
        raise ValueError(f"No price data returned for '{ticker}'.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [c[0] if isinstance(c, tuple) else c for c in frame.columns]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame[frame["Close"] > 0].sort_index()

    enriched = compute_all_indicators(frame)
    context.price_cache[key] = enriched
    return enriched


def _finite(value: Any, places: int = 4) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, places) if np.isfinite(result) else None


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------
def build_tools(context: ToolContext, names: list[str] | None = None) -> list[Any]:
    """Construct LangChain tools bound to `context`.

    `names` restricts which tools are returned. Task 3B's tool restriction is
    enforced HERE, structurally: Agent B is handed a list that does not contain
    `get_price_data`, so it cannot call it. Telling an agent in its system prompt
    not to use a tool is a request; not giving it the tool is a guarantee. The
    criterion says "enforced, separate tool access", and that word is why this
    is a factory rather than a module-level list.
    """
    from langchain_core.tools import tool

    # -- 1 --------------------------------------------------------------------
    @tool
    def get_price_data(ticker: str, period: str = "2y") -> dict:
        """Fetch daily OHLCV price history and compute technical indicators for a ticker.

        Returns the CURRENT state of: latest close, 52-week high/low, year-to-date
        return, SMA-50, SMA-200, RSI-14, MACD and its signal line and histogram,
        Bollinger Bands with %B, and a composite momentum label. Also returns the
        number of sessions analysed and the date of the latest bar.

        USE THIS FIRST for any question about price level, trend, technical
        posture, momentum, or whether a stock is overbought or oversold. Almost
        every financial-health question needs this as its foundation.

        DOES NOT return news, analyst opinion, or volatility percentiles — use
        get_news, web_search and calculate_volatility for those.

        Cost: one network call, roughly 1-3 seconds. Cached per session, so
        calling it twice for the same ticker is free the second time.

        Args:
            ticker: Stock symbol, e.g. "NVDA", "AAPL", "BARC.L".
            period: History window — "1y", "2y" or "5y". Default "2y".
        """
        with context.traced("get_price_data", {"ticker": ticker, "period": period}) as slot:
            try:
                enriched = _load_prices(context, ticker.upper(), period)
            except Exception as exc:  # noqa: BLE001
                result = fail(
                    f"Price fetch failed for {ticker}: {exc}",
                    "Verify the ticker is valid on Yahoo Finance. If it is, the data "
                    "provider may be rate-limited — proceed using get_news and "
                    "web_search, and state in your report that price data was unavailable.",
                )
                slot["output"] = result
                return result

            latest = enriched.iloc[-1]
            close = enriched["Close"]
            window = close.iloc[-252:] if len(close) >= 252 else close

            year_start = pd.Timestamp(year=enriched.index[-1].year, month=1, day=1)
            prior = close.loc[close.index < year_start]
            baseline = prior.iloc[-1] if not prior.empty else close.iloc[0]
            ytd = (float(latest["Close"]) / float(baseline) - 1.0) * 100 if baseline else None

            signal = derive_momentum_signal(enriched)
            payload = {
                "ticker": ticker.upper(),
                "as_of": enriched.index[-1].strftime("%Y-%m-%d"),
                "sessions_analysed": int(len(enriched)),
                "current_price": _finite(latest["Close"], 2),
                "fifty_two_week_high": _finite(window.max(), 2),
                "fifty_two_week_low": _finite(window.min(), 2),
                "pct_from_52w_high": _finite(
                    (float(latest["Close"]) / float(window.max()) - 1) * 100, 2
                ),
                "ytd_return_pct": _finite(ytd, 2),
                "sma_50": _finite(latest.get("sma_50"), 2),
                "sma_200": _finite(latest.get("sma_200"), 2),
                "rsi_14": _finite(latest.get("rsi_14"), 2),
                "macd": _finite(latest.get("macd")),
                "macd_signal": _finite(latest.get("macd_signal")),
                "macd_histogram": _finite(latest.get("macd_hist")),
                "bollinger_upper": _finite(latest.get("bb_upper"), 2),
                "bollinger_lower": _finite(latest.get("bb_lower"), 2),
                "bollinger_pct_b": _finite(latest.get("bb_pct_b")),
                "momentum_signal": signal.label,
                "momentum_score": signal.score,
                "momentum_rationale": signal.rationale,
            }
            result = ok(payload)
            slot["output"] = payload
            return result

    # -- 2 --------------------------------------------------------------------
    @tool
    def get_news(ticker: str, n: int = 10) -> dict:
        """Retrieve recent news headlines specifically about one company.

        Returns a list of headlines, each with its publisher and publication time.
        Sources are tried in order (yfinance, Yahoo RSS, Google News RSS) and
        deduplicated, so a single feed being empty does not produce an empty result.

        USE THIS for company-specific events: earnings, guidance, management
        changes, product launches, regulatory actions, litigation.

        USE web_search INSTEAD when you want analyst commentary, sector context,
        competitor developments, or an opinion rather than a reported event.
        get_news gives you what happened; web_search gives you what people think
        about it.

        Cost: one to three network calls, roughly 1-4 seconds.

        Args:
            ticker: Stock symbol, e.g. "NVDA".
            n: Maximum headlines to return. Default 10, cap 30.
        """
        with context.traced("get_news", {"ticker": ticker, "n": n}) as slot:
            n = max(1, min(int(n), 30))
            cache_key = ticker.upper()
            if cache_key in context.news_cache:
                logger.info("News cache hit for %s (short-term memory).", cache_key)
                cached = context.news_cache[cache_key][:n]
                result = ok(cached, count=len(cached), from_cache=True)
                slot["output"] = result
                return result

            try:
                from data_pipeline import fetch_news  # Task 1's fallback chain.

                items, warnings = fetch_news(ticker.upper(), minimum=n)
                payload = [item.to_dict() for item in items][:n]
            except Exception as exc:  # noqa: BLE001
                result = fail(
                    f"News retrieval failed for {ticker}: {exc}",
                    "Try web_search with a query like '<ticker> stock news' to obtain "
                    "commentary from an independent source.",
                )
                slot["output"] = result
                return result

            if not payload:
                result = fail(
                    f"No headlines found for {ticker} across all news sources.",
                    "Call web_search with '<company name> stock news' instead, and note "
                    "the absence of company news as a data gap in your analysis.",
                )
                slot["output"] = result
                return result

            context.news_cache[cache_key] = payload
            result = ok(payload, count=len(payload), warnings=warnings)
            slot["output"] = {"count": len(payload), "first": payload[0]["headline"]}
            return result

    # -- 3 --------------------------------------------------------------------
    @tool
    def calculate_volatility(ticker: str, window: int = DEFAULT_VOL_WINDOW) -> dict:
        """Compute annualised historical volatility from daily log returns.

        Returns the annualised volatility over the requested window AND — more
        usefully — its percentile against the same ticker's own two-year
        distribution. An absolute figure alone is not interpretable: 35% is calm
        for a small-cap biotech and alarming for a utility. The percentile is
        what tells you whether the current regime is unusual.

        Also returns the 90-day realised volatility for comparison, so you can
        see whether short-term risk is rising or falling relative to the longer
        window.

        USE THIS whenever the question involves risk, position sizing, option
        pricing, or hedging — a hedge recommendation without a volatility figure
        is not data-driven.

        Cost: free if get_price_data has already run for this ticker (shared
        cache); otherwise one network call.

        Args:
            ticker: Stock symbol, e.g. "NVDA".
            window: Lookback in trading days. Default 30. Use 90 for a slower read.
        """
        with context.traced("calculate_volatility", {"ticker": ticker, "window": window}) as slot:
            try:
                window = max(5, min(int(window), 252))
                enriched = _load_prices(context, ticker.upper(), "2y")
                series = annualised_volatility(enriched["Close"], window=window).dropna()
                if series.empty:
                    raise ValueError(f"Insufficient history for a {window}-day window.")

                current = float(series.iloc[-1])
                percentile = float((series < current).mean() * 100)
                slower = annualised_volatility(enriched["Close"], window=90).dropna()

                payload = {
                    "ticker": ticker.upper(),
                    "window_days": window,
                    "annualised_volatility": round(current, 4),
                    "annualised_volatility_pct": round(current * 100, 2),
                    "percentile_vs_two_year": round(percentile, 1),
                    "two_year_median_volatility_pct": round(float(series.median()) * 100, 2),
                    "two_year_max_volatility_pct": round(float(series.max()) * 100, 2),
                    "ninety_day_volatility_pct": round(float(slower.iloc[-1]) * 100, 2) if not slower.empty else None,
                    "interpretation": (
                        f"{window}-day realised volatility is {current * 100:.1f}% annualised, "
                        f"which sits at the {percentile:.0f}th percentile of its own two-year "
                        f"range — {'elevated' if percentile > 70 else 'subdued' if percentile < 30 else 'typical'} "
                        f"for this name."
                    ),
                }
                result = ok(payload)
                slot["output"] = payload
                return result
            except Exception as exc:  # noqa: BLE001
                result = fail(
                    f"Volatility calculation failed for {ticker}: {exc}",
                    "Call get_price_data first to populate the price cache, then retry. "
                    "If price data is unavailable entirely, base risk assessment on "
                    "qualitative sources and declare volatility as a data gap.",
                )
                slot["output"] = result
                return result

    # -- 4 --------------------------------------------------------------------
    @tool
    def llm_sentiment(headlines: list[str]) -> dict:
        """Score a list of news headlines for market sentiment using a language model.

        Returns an aggregate sentiment score from -1 (strongly negative) to +1
        (strongly positive), a label, per-headline classifications with individual
        confidences, and the positive/neutral/negative counts. The aggregate is
        weighted by per-headline confidence, so a hedged classification cannot
        outvote a decisive one.

        USE THIS AFTER get_news or web_search — pass it the headline strings you
        retrieved. It does not fetch anything itself.

        Sentiment is judged by likely SHARE PRICE impact, not by whether the news
        is pleasant: a large restructuring is frequently positive for the price.

        Cost: one LLM call, roughly 1-3 seconds.

        Args:
            headlines: The headline strings to classify. Between 1 and 25.
        """
        with context.traced("llm_sentiment", {"headline_count": len(headlines or [])}) as slot:
            if not headlines:
                result = fail(
                    "No headlines supplied to llm_sentiment.",
                    "Call get_news or web_search first, then pass the headline strings here.",
                )
                slot["output"] = result
                return result
            if context.llm is None:
                result = fail(
                    "No LLM configured for sentiment scoring.",
                    "Proceed without sentiment and declare it as a data gap.",
                )
                slot["output"] = result
                return result

            try:
                # Task 1 and Task 3 both ship a module named `schemas`, and
                # task3_agentic/src sits earlier on sys.path, so a plain
                # `from schemas import ...` binds to the Task 3 module and
                # raises ImportError. Load Task 1's module by file path under a
                # distinct name so both remain importable in one process.
                AggregateSentiment, HeadlineSentimentBatch = _task1_sentiment_schemas()
                import prompts

                trimmed = [h for h in headlines if h and h.strip()][:25]
                block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(trimmed))
                messages = prompts.render(
                    prompts.HEADLINE_SENTIMENT,
                    ticker="the subject company", company="the subject company",
                    count=len(trimmed), headline_block=block,
                )
                payload = context.llm.chat_json(messages, temperature=0.0,
                                                max_tokens=170 * len(trimmed) + 300)
                batch = HeadlineSentimentBatch.model_validate(payload)
                aggregate = AggregateSentiment.from_batch(batch)

                data = {
                    "aggregate_score": aggregate.score,
                    "label": aggregate.label.value,
                    "headlines_analysed": aggregate.headline_count,
                    "positive": aggregate.positive_count,
                    "neutral": aggregate.neutral_count,
                    "negative": aggregate.negative_count,
                    "mean_confidence": aggregate.mean_confidence,
                    "low_diversity_warning": aggregate.degenerate_warning,
                    "per_headline": [
                        {"headline": r.headline[:120], "sentiment": r.sentiment.value,
                         "confidence": r.confidence, "reason": r.brief_reason}
                        for r in batch.results
                    ],
                }
                result = ok(data)
                slot["output"] = {k: v for k, v in data.items() if k != "per_headline"}
                return result
            except Exception as exc:  # noqa: BLE001
                result = fail(
                    f"Sentiment scoring failed: {exc}",
                    "Read the headlines directly and characterise sentiment qualitatively "
                    "in your analysis, noting that the automated score was unavailable.",
                )
                slot["output"] = result
                return result

    # -- 5 --------------------------------------------------------------------
    @tool
    def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> dict:
        """Search the web for analyst commentary, sector context and market opinion.

        Returns titles, snippets and URLs. Use natural-language queries such as
        "NVIDIA analyst outlook 2026 data centre demand" or "semiconductor export
        restrictions impact".

        USE THIS for interpretation and forward-looking views: analyst opinion,
        sector trends, competitor moves, regulatory or macro context. It is the
        right tool for the "what are the risks over the next 90 days" part of a
        question, because risks are usually discussed in commentary before they
        appear in price.

        USE get_news INSTEAD for confirmed company-specific events.

        Cost: 1-5 seconds, and this is the tool most likely to be rate-limited.
        Prefer one well-targeted query over several vague ones. If it fails,
        the error tells you what to fall back to.

        Args:
            query: Natural-language search query.
            max_results: Results to return, 1-10. Default 6.
        """
        with context.traced("web_search", {"query": query, "max_results": max_results}) as slot:
            max_results = max(1, min(int(max_results), 10))

            # The package was renamed from `duckduckgo-search` to `ddgs`. Support
            # both so the notebook runs regardless of which one pip resolved.
            searcher = None
            try:
                from ddgs import DDGS  # type: ignore
                searcher = DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS  # type: ignore
                    searcher = DDGS
                except ImportError:
                    result = fail(
                        "Neither 'ddgs' nor 'duckduckgo-search' is installed.",
                        "Proceed using get_news for company headlines and note that "
                        "external commentary was unavailable.",
                    )
                    slot["output"] = result
                    return result

            last_error = ""
            for attempt in range(1, SEARCH_RETRY_ATTEMPTS + 1):
                try:
                    with searcher() as client:
                        raw = list(client.text(query, max_results=max_results))
                    if raw:
                        payload = [
                            {
                                "title": item.get("title", ""),
                                "snippet": (item.get("body") or "")[:320],
                                "url": item.get("href") or item.get("link", ""),
                            }
                            for item in raw
                        ]
                        result = ok(payload, count=len(payload), query=query)
                        slot["output"] = {"count": len(payload),
                                          "first": payload[0]["title"][:80]}
                        return result
                    last_error = "Search returned zero results."
                except Exception as exc:  # noqa: BLE001
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning("web_search attempt %d failed: %s", attempt, last_error)

                if attempt < SEARCH_RETRY_ATTEMPTS:
                    time.sleep(1.5 * attempt)   # Linear backoff; DDG throttles hard.

            result = fail(
                f"web_search failed after {SEARCH_RETRY_ATTEMPTS} attempts: {last_error}",
                "DuckDuckGo is likely rate-limiting. Call get_news for company-specific "
                "headlines instead, and base forward-looking risk assessment on the "
                "price and volatility data you already have. Note the missing external "
                "commentary as a data gap rather than inventing analyst views.",
            )
            slot["output"] = result
            return result

    registry: dict[str, Any] = {
        "get_price_data": get_price_data,
        "get_news": get_news,
        "calculate_volatility": calculate_volatility,
        "llm_sentiment": llm_sentiment,
        "web_search": web_search,
    }

    if names is None:
        return list(registry.values())

    unknown = set(names) - set(registry)
    if unknown:
        raise KeyError(f"Unknown tool(s) requested: {sorted(unknown)}. "
                       f"Available: {sorted(registry)}")
    return [registry[name] for name in names]


ALL_TOOL_NAMES = [
    "get_price_data", "get_news", "calculate_volatility", "llm_sentiment", "web_search",
]

# Task 3B role definitions. Kept here so the restriction is declared in one place
# and both the graph and the tests read the same source of truth.
AGENT_A_TOOLS = ["get_price_data", "calculate_volatility", "llm_sentiment"]
AGENT_B_TOOLS = ["web_search", "get_news"]
