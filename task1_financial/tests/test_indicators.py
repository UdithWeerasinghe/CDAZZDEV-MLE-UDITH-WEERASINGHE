"""
Verification of every indicator against an INDEPENDENT reference implementation.

The references below are written in plain Python - explicit loops over lists, no
pandas, no numpy vectorisation. That matters: if I verified the pandas code with
more pandas code sharing the same assumptions, a shared misunderstanding (ddof,
adjust, seeding) would pass silently. Two independent derivations of the same
textbook formula agreeing to 1e-9 is real evidence.

Run:  python -m pytest task1_financial/tests/test_indicators.py -v
      python task1_financial/tests/test_indicators.py        (no pytest needed)

# AI-ASSISTED: Claude (claude-sonnet-5), Prompt: 'Write independent pure-Python
# reference implementations of Wilder RSI, EMA/MACD and Bollinger Bands and
# assert the pandas implementations match', Date: 2026-09-10
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicators import (  # noqa: E402
    annualised_volatility,
    bollinger_bands,
    compute_all_indicators,
    derive_momentum_signal,
    ema,
    macd,
    rsi,
    sma,
)

TOL = 1e-9


# ---------------------------------------------------------------------------
# Independent reference implementations (plain Python, textbook definitions)
# ---------------------------------------------------------------------------
def ref_sma(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(math.nan)
        else:
            out.append(sum(values[i - window + 1 : i + 1]) / window)
    return out


def ref_ema_recursive(values: list[float], span: int) -> list[float]:
    """EMA in the adjust=False recursive form, seeded at the span-th bar.

    pandas with min_periods=span emits NaN before the seed and seeds the
    recursion from the first observation. To match, we run the recursion from
    index 0 and only expose values from index span-1 onwards.
    """
    alpha = 2.0 / (span + 1.0)
    out: list[float] = []
    running = values[0]
    for i, value in enumerate(values):
        running = value if i == 0 else alpha * value + (1.0 - alpha) * running
        out.append(math.nan if i + 1 < span else running)
    return out


def ref_wilder_rsi(closes: list[float], period: int = 14) -> list[float]:
    """Wilder (1978) RSI, straight from the definition.

    Seed:  avg_gain = mean of the first `period` gains, likewise avg_loss.
    Step:  avg = (prev_avg * (period - 1) + current) / period
    """
    out: list[float] = [math.nan] * len(closes)
    if len(closes) <= period:
        return out

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def to_rsi(gain: float, loss: float) -> float:
        if gain == 0.0 and loss == 0.0:
            return 50.0
        if loss == 0.0:
            return 100.0
        if gain == 0.0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    out[period] = to_rsi(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = to_rsi(avg_gain, avg_loss)
    return out


def ref_cutler_rsi(closes: list[float], period: int = 14) -> list[float]:
    """Cutler's RSI - the WRONG one, using a simple rolling mean.

    Present only to prove the two are materially different, which is the
    justification for implementing Wilder's smoothing the hard way.
    """
    out: list[float] = [math.nan] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    for i in range(period, len(closes)):
        window_gain = sum(gains[i - period : i]) / period
        window_loss = sum(losses[i - period : i]) / period
        if window_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + window_gain / window_loss)
    return out


def ref_bollinger(values: list[float], window: int, num_std: float):
    """Bollinger Bands using the POPULATION standard deviation."""
    upper, middle, lower = [], [], []
    for i in range(len(values)):
        if i + 1 < window:
            upper.append(math.nan); middle.append(math.nan); lower.append(math.nan)
            continue
        chunk = values[i - window + 1 : i + 1]
        mean = sum(chunk) / window
        variance = sum((x - mean) ** 2 for x in chunk) / window     # ddof = 0
        sigma = math.sqrt(variance)
        middle.append(mean)
        upper.append(mean + num_std * sigma)
        lower.append(mean - num_std * sigma)
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def synthetic_prices(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Geometric random walk with drift, a volatility regime shift and gaps.

    The regime shift and the injected NaNs matter: a series that is too tame
    will not exercise the divide-by-zero and warm-up branches.
    """
    rng = np.random.default_rng(seed)
    volatility = np.where(np.arange(n) < n // 2, 0.011, 0.026)
    shocks = rng.normal(0.0004, 1.0, n) * volatility
    close = 100.0 * np.exp(np.cumsum(shocks))

    index = pd.bdate_range("2023-01-02", periods=n)
    frame = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 9_000_000, n),
        },
        index=index,
    )
    return frame


def assert_series_match(actual: pd.Series, expected: list[float], label: str) -> None:
    actual_values = actual.to_numpy(dtype="float64")
    expected_values = np.asarray(expected, dtype="float64")
    assert actual_values.shape == expected_values.shape, f"{label}: shape mismatch"

    actual_nan = np.isnan(actual_values)
    expected_nan = np.isnan(expected_values)
    mismatched = int((actual_nan != expected_nan).sum())
    assert mismatched == 0, f"{label}: NaN placement differs at {mismatched} positions"

    both = ~actual_nan
    if both.any():
        worst = float(np.max(np.abs(actual_values[both] - expected_values[both])))
        assert worst < TOL, f"{label}: max absolute deviation {worst:.3e} exceeds {TOL:.0e}"
        print(f"  PASS {label:<34} n={int(both.sum()):>4}  max_dev={worst:.3e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_sma_matches_reference() -> None:
    frame = synthetic_prices()
    closes = frame["Close"].tolist()
    for window in (50, 200):
        assert_series_match(sma(frame["Close"], window), ref_sma(closes, window), f"SMA-{window}")


def test_ema_matches_recursive_reference() -> None:
    frame = synthetic_prices()
    closes = frame["Close"].tolist()
    for span in (12, 26, 9):
        assert_series_match(ema(frame["Close"], span), ref_ema_recursive(closes, span), f"EMA-{span}")


def test_rsi_matches_wilder_reference() -> None:
    frame = synthetic_prices()
    closes = frame["Close"].tolist()
    assert_series_match(rsi(frame["Close"], 14), ref_wilder_rsi(closes, 14), "RSI-14 (Wilder)")


def test_rsi_is_not_cutler() -> None:
    """Prove Wilder and Cutler genuinely diverge, justifying the extra work."""
    frame = synthetic_prices()
    closes = frame["Close"].tolist()
    wilder = np.array(ref_wilder_rsi(closes, 14), dtype="float64")
    cutler = np.array(ref_cutler_rsi(closes, 14), dtype="float64")
    both = ~np.isnan(wilder) & ~np.isnan(cutler)
    max_gap = float(np.max(np.abs(wilder[both] - cutler[both])))
    mean_gap = float(np.mean(np.abs(wilder[both] - cutler[both])))
    assert max_gap > 1.0, "Expected Wilder and Cutler RSI to differ materially"
    print(f"  PASS {'Wilder vs Cutler divergence':<34} mean={mean_gap:.2f} pts, max={max_gap:.2f} pts")


def test_rsi_analytic_edge_cases() -> None:
    """Monotone and flat series have known closed-form RSI values."""
    rising = pd.Series(np.arange(1, 61, dtype="float64"))
    assert abs(rsi(rising, 14).iloc[-1] - 100.0) < TOL, "Unbroken advance must give RSI 100"

    falling = pd.Series(np.arange(60, 0, -1, dtype="float64"))
    assert abs(rsi(falling, 14).iloc[-1] - 0.0) < TOL, "Unbroken decline must give RSI 0"

    flat = pd.Series(np.full(60, 42.0))
    assert abs(rsi(flat, 14).iloc[-1] - 50.0) < TOL, "Flat series must give RSI 50"

    mixed = synthetic_prices()["Close"]
    values = rsi(mixed, 14).dropna()
    assert values.between(0.0, 100.0).all(), "RSI escaped the [0, 100] range"
    assert np.isfinite(values).all(), "RSI produced non-finite values"
    print(f"  PASS {'RSI edge cases (100/0/50/bounds)':<34} range=[{values.min():.1f}, {values.max():.1f}]")


def test_macd_components_and_identity() -> None:
    frame = synthetic_prices()
    closes = frame["Close"].tolist()
    result = macd(frame["Close"], 12, 26, 9)

    fast = np.array(ref_ema_recursive(closes, 12), dtype="float64")
    slow = np.array(ref_ema_recursive(closes, 26), dtype="float64")
    assert_series_match(result["macd"], (fast - slow).tolist(), "MACD line")

    identity = (result["macd"] - result["macd_signal"] - result["macd_hist"]).abs().max()
    assert identity < TOL, "macd_hist must equal macd - macd_signal exactly"
    print(f"  PASS {'MACD histogram identity':<34} max_dev={identity:.3e}")

    assert result["macd_signal"].first_valid_index() > result["macd"].first_valid_index(), (
        "Signal line must warm up later than the MACD line"
    )


def test_bollinger_uses_population_std() -> None:
    frame = synthetic_prices()
    closes = frame["Close"].tolist()
    bands = bollinger_bands(frame["Close"], 20, 2.0)

    upper, middle, lower = ref_bollinger(closes, 20, 2.0)
    assert_series_match(bands["bb_upper"], upper, "Bollinger upper (ddof=0)")
    assert_series_match(bands["bb_middle"], middle, "Bollinger middle")
    assert_series_match(bands["bb_lower"], lower, "Bollinger lower (ddof=0)")

    # Demonstrate the ddof error the module docstring warns about.
    wrong = frame["Close"].rolling(20, min_periods=20).std(ddof=1)
    right = bands["bb_std"]
    ratio = float((wrong / right).dropna().mean())
    assert abs(ratio - math.sqrt(20 / 19)) < 1e-6
    print(f"  PASS {'ddof=1 would inflate sigma by':<34} {(ratio - 1) * 100:.2f}%")

    pct_b = bands["bb_pct_b"].dropna()
    assert np.isfinite(pct_b).all(), "%B produced non-finite values"


def test_warmup_nans_are_not_filled() -> None:
    """A fabricated 200-day SMA on day 3 would be a silent lie."""
    frame = synthetic_prices()
    enriched = compute_all_indicators(frame)
    expectations = {
        "sma_50": 49, "sma_200": 199, "rsi_14": 14,
        "macd": 25, "bb_middle": 19, "volatility_30d": 30,
    }
    for column, first_valid_position in expectations.items():
        series = enriched[column]
        leading = int(series.isna().to_numpy().argmin())
        assert leading == first_valid_position, (
            f"{column}: first valid value at position {leading}, expected {first_valid_position}"
        )
    print(f"  PASS {'Warm-up NaN placement':<34} {len(expectations)} columns correct")


def test_short_history_degrades_gracefully() -> None:
    """90 bars cannot support a 200-day SMA. Nothing may raise."""
    frame = synthetic_prices(n=90)
    enriched = compute_all_indicators(frame)
    assert enriched["sma_200"].isna().all(), "SMA-200 must be entirely NaN on 90 bars"
    assert enriched["rsi_14"].notna().any(), "RSI-14 should still be computable on 90 bars"

    signal = derive_momentum_signal(enriched)
    assert "trend_regime" not in signal.components, "Undefined indicators must not vote"
    assert -1.0 <= signal.score <= 1.0
    assert any("warm-up" in line for line in signal.rationale), "Exclusion must be disclosed"
    print(f"  PASS {'Short history degradation':<34} score={signal.score:+.3f} ({signal.label})")


def test_missing_data_does_not_raise() -> None:
    """Holidays, halts and bad ticks arrive as NaN. None may propagate a crash."""
    frame = synthetic_prices()
    frame.loc[frame.index[100:106], "Close"] = np.nan
    frame.loc[frame.index[300], "Close"] = np.nan

    enriched = compute_all_indicators(frame)
    signal = derive_momentum_signal(enriched)

    for column in ("rsi_14", "macd", "bb_middle"):
        tail = enriched[column].iloc[-50:]
        assert tail.notna().all(), f"{column} failed to recover after the gap"
        assert np.isfinite(tail).all(), f"{column} produced inf after the gap"
    assert signal.label in {"strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"}
    print(f"  PASS {'NaN gap handling':<34} recovered, signal={signal.label}")


def test_momentum_signal_direction_is_sane() -> None:
    """A strong uptrend must not score bearish, and vice versa."""
    n = 400
    index = pd.bdate_range("2023-01-02", periods=n)
    up = pd.DataFrame({"Close": np.linspace(100, 260, n)}, index=index)
    down = pd.DataFrame({"Close": np.linspace(260, 100, n)}, index=index)

    up_signal = derive_momentum_signal(compute_all_indicators(up))
    down_signal = derive_momentum_signal(compute_all_indicators(down))

    assert up_signal.score > 0.3, f"Uptrend scored {up_signal.score}"
    assert down_signal.score < -0.3, f"Downtrend scored {down_signal.score}"
    assert len(up_signal.rationale) >= 4, "Rationale must explain each component"
    print(f"  PASS {'Momentum direction':<34} up={up_signal.score:+.3f}, down={down_signal.score:+.3f}")


def test_volatility_is_annualised_correctly() -> None:
    """A series with known daily sigma must annualise by sqrt(252)."""
    rng = np.random.default_rng(7)
    daily_sigma = 0.02
    n = 800
    returns = rng.normal(0.0, daily_sigma, n)
    close = pd.Series(100 * np.exp(np.cumsum(returns)), index=pd.bdate_range("2022-01-03", periods=n))

    result = annualised_volatility(close, window=250).dropna()
    expected = daily_sigma * math.sqrt(252)
    observed = float(result.mean())
    assert abs(observed - expected) < 0.02, f"Expected ~{expected:.3f}, observed {observed:.3f}"
    print(f"  PASS {'Annualised volatility':<34} expected={expected:.3f}, observed={observed:.3f}")


def test_compute_all_does_not_mutate_input() -> None:
    """Notebook cells must be idempotent under re-execution."""
    frame = synthetic_prices(n=300)
    before = list(frame.columns)
    compute_all_indicators(frame)
    assert list(frame.columns) == before, "compute_all_indicators mutated the caller's frame"
    print(f"  PASS {'Input frame not mutated':<34} {len(before)} columns unchanged")


def test_invalid_parameters_rejected() -> None:
    close = synthetic_prices(n=120)["Close"]
    for bad in (0, -5, 2.5):
        try:
            sma(close, bad)                      # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"sma accepted invalid window {bad!r}")

    try:
        macd(close, fast=26, slow=12)
        raise AssertionError("macd accepted fast >= slow")
    except ValueError:
        pass

    try:
        compute_all_indicators(pd.DataFrame({"Price": [1, 2, 3]}))
        raise AssertionError("compute_all_indicators accepted a frame with no Close column")
    except KeyError:
        pass
    print(f"  PASS {'Invalid parameter rejection':<34} 5 cases")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"\nRunning {len(tests)} verification tests against independent references\n" + "=" * 74)
    failures = 0
    for test in tests:
        print(f"\n{test.__name__}")
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {exc}")
    print("\n" + "=" * 74)
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
