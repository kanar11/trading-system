"""Trend-following moving averages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    if window <= 0:
        raise ValueError("window must be > 0")
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (pandas span convention)."""
    if span <= 0:
        raise ValueError("span must be > 0")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly-weighted moving average (most recent bar weighted most heavily)."""
    if window <= 0:
        raise ValueError("window must be > 0")
    weights = np.arange(1, window + 1, dtype=float)
    weight_sum = weights.sum()

    def _wma_window(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weight_sum)

    return series.rolling(window, min_periods=window).apply(_wma_window, raw=True)


def vwma(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Volume-weighted moving average over a rolling window."""
    if window <= 0:
        raise ValueError("window must be > 0")
    pv = price * volume
    return (
        pv.rolling(window, min_periods=window).sum()
        / volume.rolling(window, min_periods=window).sum()
    )


def hma(series: pd.Series, window: int) -> pd.Series:
    """Hull Moving Average — a low-lag, smooth trend filter.

    HMA(n) = WMA( 2 * WMA(n / 2) - WMA(n), round(sqrt(n)) ).
    """
    if window <= 1:
        raise ValueError("window must be >= 2")
    half = max(window // 2, 1)
    sqrt_window = max(int(round(window**0.5)), 1)
    raw = 2 * wma(series, half) - wma(series, window)
    return wma(raw, sqrt_window)


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """Aroon up / down / oscillator.

    Aroon Up measures how recently the rolling ``period``-bar high occurred
    (100 = the high is the current bar, 0 = it was ``period`` bars ago); Aroon
    Down does the same for the low. The oscillator is up - down, in [-100, 100].

    Returns a DataFrame with columns: ``up``, ``down``, ``oscillator``.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    def _bars_since_high(x: np.ndarray) -> float:
        return float(len(x) - 1 - np.argmax(x))

    def _bars_since_low(x: np.ndarray) -> float:
        return float(len(x) - 1 - np.argmin(x))

    win = period + 1
    high_age = high.rolling(win, min_periods=win).apply(_bars_since_high, raw=True)
    low_age = low.rolling(win, min_periods=win).apply(_bars_since_low, raw=True)
    up = 100 * (period - high_age) / period
    down = 100 * (period - low_age) / period
    return pd.DataFrame({"up": up, "down": down, "oscillator": up - down})


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Vortex Indicator (Botes & Siepman, 2010).

    Captures the strength of up vs down trend movement:
        VI+ = sum(|high - prev_low|)  / sum(true_range)
        VI- = sum(|low  - prev_high|) / sum(true_range)
    over ``period`` bars. VI+ crossing above VI- signals an emerging up-trend.

    Returns a DataFrame with ``vi_plus`` and ``vi_minus`` columns.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}.")

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    tr_sum = true_range.rolling(period, min_periods=period).sum().replace(0, np.nan)
    vi_plus = vm_plus.rolling(period, min_periods=period).sum() / tr_sum
    vi_minus = vm_minus.rolling(period, min_periods=period).sum() / tr_sum
    return pd.DataFrame({"vi_plus": vi_plus, "vi_minus": vi_minus})


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Average Directional Index with the +DI / -DI directional lines.

    Wilder's trend-strength system: directional movement (+DM/-DM, the
    larger one-sided range extension) is smoothed and normalised by ATR
    into the directional indicators, whose normalised spread smooths into
    ADX. Readings above ~25 mark a strong trend; the DI crossover gives
    its direction. Uses Wilder's smoothing (``ewm(alpha=1/period)``),
    matching the private implementation the regime detector has used all
    along — this is the public, reusable form.

    Returns a DataFrame with columns:
        * ``adx``      — trend strength in [0, 100].
        * ``di_plus``  — positive directional indicator.
        * ``di_minus`` — negative directional indicator.

    Raises:
        ValueError: If ``period`` < 1.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}.")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # true range inline: trend.py must not import from volatility.py (cycle)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
    # DM <= TR bar by bar, so the DIs are mathematically bounded by 100;
    # the clip only removes smoothing float dust above the bound
    di_plus = (100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr).clip(
        upper=100.0
    )
    di_minus = (100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr).clip(
        upper=100.0
    )

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    strength = dx.ewm(alpha=1 / period, min_periods=period).mean().clip(upper=100.0)
    return pd.DataFrame({"adx": strength, "di_plus": di_plus, "di_minus": di_minus})


def kama(
    series: pd.Series,
    er_period: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """Kaufman Adaptive Moving Average (Kaufman, 1995).

    An EMA whose smoothing constant adapts to the Efficiency Ratio
    ``ER = |net change| / sum(|bar changes|)`` over ``er_period`` bars:
    in a clean trend (ER → 1) it moves as fast as an EMA(``fast``), in
    choppy noise (ER → 0) it flattens toward an EMA(``slow``)::

        SC = (ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))²
        KAMA_t = KAMA_{t-1} + SC_t * (price_t - KAMA_{t-1})

    The recursion is seeded with the price at the first bar with a valid
    ER; earlier bars are NaN. A perfectly flat window (zero volatility)
    gets ER = 0, i.e. the slowest smoothing.

    Raises:
        ValueError: If ``er_period`` < 1 or not ``1 <= fast < slow``.
    """
    if er_period < 1:
        raise ValueError(f"er_period must be >= 1, got {er_period}.")
    if not 1 <= fast < slow:
        raise ValueError(f"need 1 <= fast < slow, got fast={fast}, slow={slow}.")

    prices = series.to_numpy(dtype=float)
    n = len(prices)
    out = np.full(n, np.nan)
    if n > er_period:
        change = np.abs(series.diff(er_period).to_numpy(dtype=float))
        volatility = series.diff().abs().rolling(er_period, min_periods=er_period).sum()
        vol = volatility.to_numpy(dtype=float)
        er = np.divide(change, vol, out=np.zeros(n), where=vol > 0)

        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        out[er_period] = prices[er_period]
        for i in range(er_period + 1, n):
            out[i] = out[i - 1] + sc[i] * (prices[i] - out[i - 1])
    return pd.Series(out, index=series.index)


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> pd.DataFrame:
    """Parabolic Stop-and-Reverse (Wilder, 1978).

    A trailing stop that accelerates toward price: while a trend runs, the
    SAR moves toward it by an acceleration factor ``af`` that starts at
    ``af_step``, grows by ``af_step`` on every new extreme, and caps at
    ``af_max``; the SAR is also never placed inside the previous two bars'
    range. When price crosses the SAR the position stops *and reverses*:
    the SAR jumps to the old extreme point and the trend flips.

    Returns a DataFrame with columns:
        * ``sar``   — the stop level (below price in up-trends, above in
          down-trends). The first bar is NaN (no prior trend).
        * ``trend`` — +1 (up) / -1 (down); 0 on the seed bar.

    Raises:
        ValueError: If not ``0 < af_step <= af_max``.
    """
    if not 0 < af_step <= af_max:
        raise ValueError(f"need 0 < af_step <= af_max, got {af_step} and {af_max}.")

    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    n = len(highs)
    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    if n < 2:
        return pd.DataFrame({"sar": sar, "trend": trend}, index=high.index)

    # seed from the first two bars: rising highs start an up-trend
    up = highs[1] >= highs[0]
    sar[1] = lows[0] if up else highs[0]
    extreme = highs[1] if up else lows[1]
    af = af_step
    trend[1] = 1 if up else -1

    for i in range(2, n):
        next_sar = sar[i - 1] + af * (extreme - sar[i - 1])
        if up:
            # never place the SAR inside the previous two bars' lows
            next_sar = min(next_sar, lows[i - 1], lows[i - 2])
            if lows[i] < next_sar:  # stopped out -> reverse down
                up = False
                next_sar = extreme
                extreme = lows[i]
                af = af_step
            elif highs[i] > extreme:  # new extreme -> accelerate
                extreme = highs[i]
                af = min(af + af_step, af_max)
        else:
            next_sar = max(next_sar, highs[i - 1], highs[i - 2])
            if highs[i] > next_sar:  # stopped out -> reverse up
                up = True
                next_sar = extreme
                extreme = highs[i]
                af = af_step
            elif lows[i] < extreme:
                extreme = lows[i]
                af = min(af + af_step, af_max)
        sar[i] = next_sar
        trend[i] = 1 if up else -1

    return pd.DataFrame({"sar": sar, "trend": trend}, index=high.index)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    conversion: int = 9,
    base: int = 26,
    span_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Ichimoku Kinko Hyo cloud components.

    Each line is a midpoint of rolling highs/lows; the leading spans are shifted
    forward by ``displacement`` and the lagging span backward, matching the
    standard causal representation on the input index.

    Returns a DataFrame with columns:
        * ``tenkan``    — conversion line, midpoint over ``conversion`` bars.
        * ``kijun``     — base line, midpoint over ``base`` bars.
        * ``senkou_a``  — leading span A, (tenkan + kijun) / 2 shifted forward.
        * ``senkou_b``  — leading span B, midpoint over ``span_b`` shifted forward.
        * ``chikou``    — lagging span, close shifted back by ``displacement``.

    Raises:
        ValueError: If a window is < 1 or ``displacement`` < 0.
    """
    if min(conversion, base, span_b) < 1:
        raise ValueError("conversion/base/span_b must be >= 1")
    if displacement < 0:
        raise ValueError(f"displacement must be >= 0, got {displacement}")

    def _midpoint(window: int) -> pd.Series:
        highest = high.rolling(window, min_periods=window).max()
        lowest = low.rolling(window, min_periods=window).min()
        return (highest + lowest) / 2

    tenkan = _midpoint(conversion)
    kijun = _midpoint(base)
    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": ((tenkan + kijun) / 2).shift(displacement),
            "senkou_b": _midpoint(span_b).shift(displacement),
            "chikou": close.shift(-displacement),
        }
    )
