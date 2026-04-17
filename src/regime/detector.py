"""Market regime detection module.

Classifies each day into one of three regimes — trending, mean-reverting,
or undefined — using a combination of:
1. ADX (Average Directional Index) for trend strength
2. Rolling Hurst exponent estimate for mean-reversion detection
3. Volatility regime (high vs low) as a secondary filter

The detected regime can be used to automatically select the best strategy:
- Trending regime  → momentum strategy
- Mean-reverting   → mean reversion strategy
- Undefined        → flat / reduced position size
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeType(str, Enum):
    """Market regime classification."""

    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    UNDEFINED = "undefined"


@dataclass
class RegimeConfig:
    """Configuration for regime detection.

    Attributes:
        adx_period: Lookback period for ADX calculation.
        adx_trending_threshold: ADX above this → trending regime.
        adx_weak_threshold: ADX below this → possible mean-reversion.
        hurst_window: Rolling window for Hurst exponent estimation.
        hurst_trending_threshold: Hurst above this → trending.
        hurst_mr_threshold: Hurst below this → mean-reverting.
        vol_window: Window for volatility regime detection.
        vol_percentile: Percentile threshold for high-vol classification.
        smoothing_window: Smooth regime labels over this many days to
            avoid whipsawing. Set to 1 to disable.
    """

    adx_period: int = 14
    adx_trending_threshold: float = 25.0
    adx_weak_threshold: float = 20.0
    hurst_window: int = 100
    hurst_trending_threshold: float = 0.55
    hurst_mr_threshold: float = 0.45
    vol_window: int = 20
    vol_percentile: float = 75.0
    smoothing_window: int = 5


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate the Average Directional Index (ADX).

    ADX measures trend strength regardless of direction.
    Values above ~25 indicate a strong trend.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period.

    Returns:
        ADX series.
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    return adx


def _rolling_hurst(series: pd.Series, window: int = 100) -> pd.Series:
    """Estimate the Hurst exponent using a rolling R/S analysis.

    H > 0.5 → trending (persistent)
    H = 0.5 → random walk
    H < 0.5 → mean-reverting (anti-persistent)

    This is a simplified rescaled-range estimate suitable for
    regime classification, not a rigorous statistical test.

    Args:
        series: Price or return series.
        window: Rolling window size.

    Returns:
        Rolling Hurst exponent estimate.
    """
    log_returns = np.log(series / series.shift(1))

    def _hurst_rs(x):
        x = x.dropna()
        if len(x) < 20:
            return np.nan

        mean_r = x.mean()
        deviations = (x - mean_r).cumsum()
        r = deviations.max() - deviations.min()
        s = x.std()

        if s == 0 or r == 0:
            return np.nan

        return np.log(r / s) / np.log(len(x))

    hurst = log_returns.rolling(window).apply(_hurst_rs, raw=False)
    return hurst


def detect_regime(
    df: pd.DataFrame,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """Detect market regime for each row in the DataFrame.

    Requires columns: 'high', 'low', 'close'. If 'high' and 'low' are
    not present, falls back to using only close-based indicators.

    Args:
        df: DataFrame with price data.
        config: Regime detection configuration. Uses defaults if None.

    Returns:
        DataFrame with added columns:
        - 'adx': Average Directional Index (if high/low available)
        - 'hurst': Rolling Hurst exponent estimate
        - 'vol_regime': 'high' or 'low' volatility
        - 'regime': RegimeType classification
    """
    if config is None:
        config = RegimeConfig()

    df = df.copy()
    has_hl = "high" in df.columns and "low" in df.columns

    # --- ADX (trend strength) ---
    if has_hl:
        df["adx"] = _adx(df["high"], df["low"], df["close"], period=config.adx_period)
    else:
        # fallback: use close-based proxy for trend strength
        # (absolute return over period, normalised)
        logger.info("No high/low columns — using close-based trend proxy instead of ADX.")
        ret = df["close"].pct_change(config.adx_period).abs() * 100
        df["adx"] = ret.rolling(config.adx_period).mean()

    # --- Hurst exponent ---
    df["hurst"] = _rolling_hurst(df["close"], window=config.hurst_window)

    # --- Volatility regime ---
    daily_vol = df["close"].pct_change().rolling(config.vol_window).std() * np.sqrt(252)
    vol_threshold = daily_vol.expanding().quantile(config.vol_percentile / 100)
    df["vol_regime"] = np.where(daily_vol > vol_threshold, "high", "low")

    # --- Classify regime ---
    regimes = []
    for i in range(len(df)):
        adx_val = df["adx"].iloc[i]
        hurst_val = df["hurst"].iloc[i]

        if pd.isna(adx_val) or pd.isna(hurst_val):
            regimes.append(RegimeType.UNDEFINED)
            continue

        # primary: ADX + Hurst agreement
        is_trending = (
            adx_val >= config.adx_trending_threshold
            and hurst_val >= config.hurst_trending_threshold
        )
        is_mean_reverting = (
            adx_val <= config.adx_weak_threshold
            and hurst_val <= config.hurst_mr_threshold
        )

        if is_trending:
            regimes.append(RegimeType.TRENDING)
        elif is_mean_reverting:
            regimes.append(RegimeType.MEAN_REVERTING)
        else:
            regimes.append(RegimeType.UNDEFINED)

    df["regime"] = regimes

    # --- optional smoothing (majority vote) ---
    if config.smoothing_window > 1:
        df["regime"] = _smooth_regime(df["regime"], config.smoothing_window)

    return df


def _smooth_regime(
    regime_series: pd.Series,
    window: int,
) -> pd.Series:
    """Smooth regime labels using a rolling majority vote.

    Prevents rapid switching between regimes.

    Args:
        regime_series: Series of RegimeType values.
        window: Rolling window for majority vote.

    Returns:
        Smoothed regime series.
    """
    # encode as integers for rolling
    mapping = {
        RegimeType.TRENDING: 2,
        RegimeType.MEAN_REVERTING: 0,
        RegimeType.UNDEFINED: 1,
    }
    reverse = {v: k for k, v in mapping.items()}

    encoded = regime_series.map(mapping).astype(float)

    def _majority(x):
        x = x.dropna()
        if len(x) == 0:
            return 1  # undefined
        counts = np.bincount(x.astype(int), minlength=3)
        return counts.argmax()

    smoothed = encoded.rolling(window, center=False, min_periods=1).apply(
        _majority, raw=False
    )
    return smoothed.map(reverse)


def adaptive_strategy(
    df: pd.DataFrame,
    momentum_fn,
    mean_reversion_fn,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """Apply regime-adaptive strategy selection.

    Detects the market regime, then applies the momentum strategy during
    trending periods and the mean reversion strategy during mean-reverting
    periods. Positions are flattened during undefined regimes.

    Args:
        df: DataFrame with price data ('close' required, 'high'/'low' optional).
        momentum_fn: Callable that takes df and returns df with 'signal'.
        mean_reversion_fn: Callable that takes df and returns df with 'signal'.
        config: Regime detection configuration.

    Returns:
        DataFrame with 'signal' column set by the active strategy per regime,
        plus all regime indicator columns.
    """
    # detect regime
    regime_df = detect_regime(df, config=config)

    # run both strategies on the full data
    mom_df = momentum_fn(df)
    mr_df = mean_reversion_fn(df)

    # select signal based on regime
    signals = np.zeros(len(regime_df))
    for i in range(len(regime_df)):
        regime = regime_df["regime"].iloc[i]
        if regime == RegimeType.TRENDING:
            signals[i] = mom_df["signal"].iloc[i]
        elif regime == RegimeType.MEAN_REVERTING:
            signals[i] = mr_df["signal"].iloc[i]
        else:
            signals[i] = 0  # flat during undefined regime

    regime_df["signal"] = signals.astype(int)

    # copy indicator columns from the strategies for debugging
    for col in ["returns", "sma200"]:
        if col in mom_df.columns:
            regime_df[f"mom_{col}"] = mom_df[col]
    for col in ["bb_middle", "bb_upper", "bb_lower", "rsi", "percent_b"]:
        if col in mr_df.columns:
            regime_df[col] = mr_df[col]

    logger.info(
        "Regime distribution: %s",
        regime_df["regime"].value_counts().to_dict(),
    )

    return regime_df
