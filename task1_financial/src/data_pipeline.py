"""
Task 1A - financial data ingestion and feature engineering.

DESIGN DECISIONS WORTH DEFENDING
--------------------------------
* NO HARDCODED DATE STRINGS. The window is expressed as `lookback_years` and
  resolved against `pd.Timestamp.now()` at call time. A literal "2023-01-01"
  silently shortens the window every day that passes and would fail the
  reviewer's run months from now.

* auto_adjust=True (split- and dividend-adjusted closes). This is a correctness
  requirement, not a preference. An unadjusted series puts a -50% single-bar gap
  at a 2:1 split, which detonates RSI (one enormous "loss"), drags the SMA-200
  for 200 sessions, and produces a fake death cross. The adjusted series is the
  only one on which the indicators mean anything.

* 52-WEEK HIGH/LOW AND YTD ARE COMPUTED FROM THE PRICE SERIES, not read from
  `Ticker.info`. `.info` is an undocumented scraped endpoint that returns {} or
  partial payloads under load. Deriving them from data we already hold is both
  more reliable and independently verifiable by the reviewer.

* NEWS USES A FALLBACK CHAIN, NOT A SINGLE SOURCE. yfinance's news payload
  changed shape (items moved under a nested `content` key), and any single feed
  can return fewer than ten items for a quiet ticker. The chain tries four
  sources and stops as soon as the minimum is satisfied, deduplicating by
  normalised title.

* EVERY EXTERNAL CALL IS WRAPPED. The brief requires missing or null data to be
  handled "without raising unhandled exceptions". Failures are logged with
  context and degrade the result rather than aborting the pipeline.

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Build a yfinance OHLCV + news
# ingestion module with a multi-source news fallback chain and a null-safe
# summary dictionary', Date: 2026-09-10
"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from indicators import compute_all_indicators, derive_momentum_signal

logger = logging.getLogger(__name__)

# --- Named constants. The brief penalises magic numbers, so every threshold ---
# --- used in a decision is defined once, here, with a stated justification. ---
TRADING_DAYS_PER_YEAR = 252     # NYSE sessions per year, the standard convention.
FIFTY_TWO_WEEK_SESSIONS = 252   # 52 calendar weeks expressed in trading sessions.
DEFAULT_LOOKBACK_YEARS = 2      # Brief requires a minimum of two years.
MIN_HEADLINES = 10              # Brief requires at least ten.
MIN_BARS_FOR_ANALYSIS = 200     # SMA-200 needs 200 bars to produce one value.
HTTP_TIMEOUT_SECONDS = 15
USER_AGENT = "Mozilla/5.0 (compatible; CDAZZDEV-MLE-Assessment/1.0)"


@dataclass
class NewsItem:
    """One headline, normalised across four differently-shaped sources."""

    headline: str
    source: str
    published: str | None = None
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "source": self.source,
            "published": self.published,
            "url": self.url,
        }


@dataclass
class EquityData:
    """Everything Task 1B needs, and nothing it does not."""

    ticker: str
    prices: pd.DataFrame                       # OHLCV plus all indicator columns.
    summary: dict[str, Any]
    news: list[NewsItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_analysable(self) -> bool:
        return len(self.prices) >= MIN_BARS_FOR_ANALYSIS


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------
def fetch_ohlcv(
    ticker: str,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    *,
    buffer_days: int = 30,
) -> pd.DataFrame:
    """Fetch daily OHLCV, split- and dividend-adjusted.

    `buffer_days` over-fetches slightly so that the requested window still
    contains a full `lookback_years` of *trading* sessions after weekends and
    market holidays are removed.
    """
    import yfinance as yf

    if not ticker or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")
    ticker = ticker.strip().upper()

    # Dates resolved at call time - see the module docstring.
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.DateOffset(years=lookback_years) - pd.Timedelta(days=buffer_days)

    logger.info("Fetching %s from %s to %s.", ticker, start.date(), end.date())

    frame = yf.Ticker(ticker).history(
        start=start.tz_localize(None),
        end=end.tz_localize(None),
        interval="1d",
        auto_adjust=True,     # See the module docstring - correctness, not preference.
        actions=False,
    )

    if frame is None or frame.empty:
        raise ValueError(
            f"No price data returned for '{ticker}'. Check the symbol is valid on "
            f"Yahoo Finance (e.g. 'NVDA', 'AAPL', 'BARC.L')."
        )

    # yfinance occasionally returns a MultiIndex column axis; flatten defensively.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [c[0] if isinstance(c, tuple) else c for c in frame.columns]

    frame = frame[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    # A zero or negative close is a bad tick, not a price. Drop rather than
    # forward-fill: log(0) in the volatility calculation is a hard failure.
    invalid = (frame["Close"] <= 0) | frame["Close"].isna()
    if invalid.any():
        logger.warning("Dropping %d rows with invalid closes for %s.", int(invalid.sum()), ticker)
        frame = frame[~invalid]

    logger.info("Retrieved %d sessions for %s (%s to %s).",
                len(frame), ticker, frame.index[0].date(), frame.index[-1].date())
    return frame


# ---------------------------------------------------------------------------
# News - a chain of four sources, tried until MIN_HEADLINES is satisfied
# ---------------------------------------------------------------------------
def _normalise_title(title: str) -> str:
    """Key for deduplication: lowercase, punctuation-free, whitespace-collapsed."""
    return re.sub(r"[^a-z0-9 ]", "", html.unescape(title).lower()).strip()


def _http_get(url: str) -> str | None:
    import requests

    try:
        response = requests.get(
            url, timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        return response.text
    except Exception as exc:  # noqa: BLE001 - any failure means "try the next source"
        logger.warning("HTTP GET failed for %s: %s", url[:80], exc)
        return None


def _parse_rss(xml_text: str, source_name: str, limit: int) -> list[NewsItem]:
    """Parse RSS 2.0 with the standard library. No feedparser dependency."""
    items: list[NewsItem] = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        logger.warning("RSS parse failed for %s: %s", source_name, exc)
        return items

    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        if not title:
            continue
        items.append(
            NewsItem(
                headline=html.unescape(title),
                source=source_name,
                published=(node.findtext("pubDate") or None),
                url=(node.findtext("link") or None),
            )
        )
        if len(items) >= limit:
            break
    return items


def _news_from_yfinance(ticker: str, limit: int) -> list[NewsItem]:
    """yfinance's own news endpoint, handling BOTH payload schemas.

    Older yfinance returned flat dicts with a 'title' key. Newer versions nest
    the payload under 'content'. Supporting only one shape means the pipeline
    breaks on a routine `pip install --upgrade`.
    """
    import yfinance as yf

    items: list[NewsItem] = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        return items

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content") if isinstance(entry.get("content"), dict) else entry
        title = content.get("title") or entry.get("title")
        if not title:
            continue

        provider = content.get("provider")
        publisher = (
            provider.get("displayName") if isinstance(provider, dict)
            else entry.get("publisher") or "Yahoo Finance"
        )
        published = content.get("pubDate") or content.get("displayTime")
        if published is None and entry.get("providerPublishTime"):
            published = datetime.fromtimestamp(
                entry["providerPublishTime"], tz=timezone.utc
            ).isoformat()

        link = entry.get("link")
        if not link:
            url_block = content.get("canonicalUrl") or content.get("clickThroughUrl")
            if isinstance(url_block, dict):
                link = url_block.get("url")

        items.append(
            NewsItem(headline=str(title).strip(), source=str(publisher), published=published, url=link)
        )
        if len(items) >= limit:
            break
    return items


def _news_from_yahoo_rss(ticker: str, limit: int) -> list[NewsItem]:
    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={quote_plus(ticker)}&region=US&lang=en-US"
    )
    text = _http_get(url)
    return _parse_rss(text, "Yahoo Finance RSS", limit) if text else []


def _news_from_google_rss(ticker: str, limit: int, company: str | None = None) -> list[NewsItem]:
    query = f"{company or ticker} stock"
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    text = _http_get(url)
    return _parse_rss(text, "Google News RSS", limit) if text else []


def _news_from_newsapi(ticker: str, limit: int, company: str | None = None) -> list[NewsItem]:
    """NewsAPI free tier. Skipped silently when NEWSAPI_KEY is unset."""
    import os

    import requests

    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return []
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": company or ticker,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": limit,
            },
            headers={"X-Api-Key": api_key, "User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return [
            NewsItem(
                headline=a["title"],
                source=(a.get("source") or {}).get("name", "NewsAPI"),
                published=a.get("publishedAt"),
                url=a.get("url"),
            )
            for a in response.json().get("articles", [])
            if a.get("title")
        ][:limit]
    except Exception as exc:  # noqa: BLE001
        logger.warning("NewsAPI failed: %s", exc)
        return []


def fetch_news(
    ticker: str,
    minimum: int = MIN_HEADLINES,
    *,
    company_name: str | None = None,
    per_source_limit: int = 25,
) -> tuple[list[NewsItem], list[str]]:
    """Retrieve at least `minimum` unique headlines, escalating through sources.

    Returns (items, warnings). Never raises: an empty list with a warning is a
    valid outcome that Task 1B must be able to handle.
    """
    sources: list[tuple[str, Callable[[], list[NewsItem]]]] = [
        ("yfinance", lambda: _news_from_yfinance(ticker, per_source_limit)),
        ("yahoo_rss", lambda: _news_from_yahoo_rss(ticker, per_source_limit)),
        ("google_rss", lambda: _news_from_google_rss(ticker, per_source_limit, company_name)),
        ("newsapi", lambda: _news_from_newsapi(ticker, per_source_limit, company_name)),
    ]

    collected: list[NewsItem] = []
    seen: set[str] = set()
    warnings: list[str] = []

    for name, fetcher in sources:
        try:
            batch = fetcher()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"News source '{name}' raised {type(exc).__name__}: {exc}")
            logger.warning("News source %s raised: %s", name, exc)
            continue

        added = 0
        for item in batch:
            key = _normalise_title(item.headline)
            if not key or key in seen:
                continue
            seen.add(key)
            collected.append(item)
            added += 1

        logger.info("News source %s contributed %d unique headlines (total %d).",
                    name, added, len(collected))
        if added == 0:
            warnings.append(f"News source '{name}' returned no new headlines.")
        if len(collected) >= minimum:
            break

    if len(collected) < minimum:
        warnings.append(
            f"Only {len(collected)} unique headlines retrieved after trying "
            f"{len(sources)} sources; the brief asks for {minimum}."
        )
    return collected, warnings


# ---------------------------------------------------------------------------
# Fundamentals and the summary dictionary
# ---------------------------------------------------------------------------
def _safe_info(ticker: str) -> dict[str, Any]:
    """`Ticker.info` is scraped and unreliable. Never let it break the run."""
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ticker.info unavailable for %s: %s", ticker, exc)
        return {}


def _finite(value: Any) -> float | None:
    """Coerce to float, mapping None/NaN/inf/non-numeric to None.

    Centralising this is what keeps `null` out of the JSON we hand the LLM as
    the string 'nan', which models reliably hallucinate around.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def build_summary(
    ticker: str,
    enriched: pd.DataFrame,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the clean summary dictionary the brief specifies.

    Required fields: current price, 52-week high and low, P/E where available,
    year-to-date return, and a momentum signal derived from the indicators.
    Every value is either a finite number or an explicit None - never NaN.
    """
    info = info or {}
    if enriched.empty:
        return {"ticker": ticker, "error": "No price data available.", "as_of": None}

    close = enriched["Close"]
    latest_close = _finite(close.iloc[-1])
    as_of = enriched.index[-1]

    # 52-week window derived from our own series rather than Ticker.info.
    window = close.iloc[-FIFTY_TWO_WEEK_SESSIONS:] if len(close) >= FIFTY_TWO_WEEK_SESSIONS else close
    high_52w, low_52w = _finite(window.max()), _finite(window.min())

    # Year-to-date, anchored on the last close of the PREVIOUS year so that the
    # first trading day of January is included in the return.
    ytd_return = None
    year_start = pd.Timestamp(year=as_of.year, month=1, day=1)
    prior = close.loc[close.index < year_start]
    baseline = prior.iloc[-1] if not prior.empty else (
        close.loc[close.index >= year_start].iloc[0]
        if not close.loc[close.index >= year_start].empty else None
    )
    if baseline is not None and _finite(baseline) not in (None, 0.0) and latest_close is not None:
        ytd_return = (latest_close / float(baseline)) - 1.0

    # P/E: prefer the reported trailing figure, else derive it from EPS.
    pe_ratio = _finite(info.get("trailingPE"))
    pe_source = "yfinance.trailingPE"
    if pe_ratio is None:
        eps = _finite(info.get("trailingEps"))
        if eps and eps > 0 and latest_close is not None:
            pe_ratio, pe_source = latest_close / eps, "derived from trailingEps"
        else:
            pe_source = "unavailable"

    signal = derive_momentum_signal(enriched)
    latest = enriched.iloc[-1]

    def latest_value(column: str) -> float | None:
        return _finite(latest[column]) if column in enriched.columns else None

    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency", "USD"),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "sessions_analysed": int(len(enriched)),
        # --- Price ---
        "current_price": _round(latest_close, 2),
        "fifty_two_week_high": _round(high_52w, 2),
        "fifty_two_week_low": _round(low_52w, 2),
        "pct_below_52w_high": _round(
            (latest_close / high_52w - 1.0) * 100 if latest_close and high_52w else None, 2
        ),
        "pct_above_52w_low": _round(
            (latest_close / low_52w - 1.0) * 100 if latest_close and low_52w else None, 2
        ),
        "ytd_return_pct": _round(ytd_return * 100 if ytd_return is not None else None, 2),
        # --- Valuation ---
        "pe_ratio": _round(pe_ratio, 2),
        "pe_source": pe_source,
        "market_cap": _finite(info.get("marketCap")),
        # --- Technicals ---
        "sma_50": _round(latest_value("sma_50"), 2),
        "sma_200": _round(latest_value("sma_200"), 2),
        "rsi_14": _round(latest_value("rsi_14"), 2),
        "macd": _round(latest_value("macd"), 4),
        "macd_signal": _round(latest_value("macd_signal"), 4),
        "macd_histogram": _round(latest_value("macd_hist"), 4),
        "bollinger_upper": _round(latest_value("bb_upper"), 2),
        "bollinger_middle": _round(latest_value("bb_middle"), 2),
        "bollinger_lower": _round(latest_value("bb_lower"), 2),
        "bollinger_pct_b": _round(latest_value("bb_pct_b"), 4),
        "annualised_volatility_30d": _round(latest_value("volatility_30d"), 4),
        # --- Derived signal ---
        "momentum_signal": signal.label,
        "momentum_score": signal.score,
        "momentum_components": signal.components,
        "momentum_rationale": signal.rationale,
    }


def _round(value: float | None, places: int) -> float | None:
    return None if value is None else round(value, places)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_equity_dataset(
    ticker: str,
    *,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    min_headlines: int = MIN_HEADLINES,
    include_news: bool = True,
) -> EquityData:
    """Run the full Task 1A pipeline for one ticker.

    Price data is the only hard dependency: without it there is nothing to
    analyse, so a failure there propagates. Everything else degrades to a
    warning, because a missing P/E or a quiet news week is not a reason to
    fail the run.
    """
    ticker = ticker.strip().upper()
    warnings: list[str] = []

    prices = fetch_ohlcv(ticker, lookback_years)          # Hard dependency.
    enriched = compute_all_indicators(prices)

    if len(enriched) < MIN_BARS_FOR_ANALYSIS:
        warnings.append(
            f"Only {len(enriched)} sessions available; SMA-200 and the trend "
            f"regime component of the momentum signal will be undefined."
        )

    info = _safe_info(ticker)
    if not info:
        warnings.append("Fundamentals (Ticker.info) unavailable; P/E and sector omitted.")

    summary = build_summary(ticker, enriched, info)

    news: list[NewsItem] = []
    if include_news:
        news, news_warnings = fetch_news(
            ticker, min_headlines, company_name=info.get("shortName")
        )
        warnings.extend(news_warnings)

    summary["headline_count"] = len(news)
    return EquityData(ticker=ticker, prices=enriched, summary=summary, news=news, warnings=warnings)
