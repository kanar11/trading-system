"""Parquet-backed cache for downloaded OHLCV data.

Avoid hammering Yahoo (or any other provider) on repeat backtests by
persisting downloaded frames as parquet files under
``~/.trading_system_cache/`` (configurable). Cache key combines the
ticker + start + end + adjustment flag.

The cache is dependency-light: pandas writes parquet via pyarrow or
fastparquet. If neither is available, the cache transparently falls
back to CSV — slower and bigger but always works.

Typical usage::

    from src.data.cache import CachedLoader
    from src.data.loader import load_yahoo_ohlcv

    loader = CachedLoader(load_yahoo_ohlcv)
    df = loader("SPY", start="2010-01-01")     # downloads + caches
    df = loader("SPY", start="2010-01-01")     # served from cache
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".trading_system_cache"


def _make_key(ticker: str, **kwargs: Any) -> str:
    """Deterministic cache key from the call signature."""
    bits = [f"{ticker.upper()}"]
    for k in sorted(kwargs):
        v = kwargs[k]
        bits.append(f"{k}={v}")
    digest = hashlib.sha1("|".join(bits).encode()).hexdigest()[:16]
    return f"{ticker.upper()}_{digest}"


def _try_write(df: pd.DataFrame, path: Path) -> Path:
    """Write df to parquet if possible, otherwise fall back to CSV."""
    try:
        df.to_parquet(path.with_suffix(".parquet"))
        return path.with_suffix(".parquet")
    except (ImportError, ValueError) as exc:
        logger.info("parquet unavailable (%s); falling back to CSV.", exc)
        df.to_csv(path.with_suffix(".csv"))
        return path.with_suffix(".csv")


def _try_read(stem: Path) -> pd.DataFrame | None:
    """Return cached frame if present (parquet preferred), else None."""
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except ImportError:
            pass
    if csv.exists():
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        return df
    return None


@dataclass
class CachedLoader:
    """Caching wrapper around a base loader callable.

    Attributes:
        loader: Any callable with signature ``loader(ticker, **kwargs) -> DataFrame``.
        cache_dir: Directory to read/write cache files in.
        enabled: Set False to bypass the cache entirely (useful for tests).
    """

    loader: Callable[..., pd.DataFrame]
    cache_dir: Path = DEFAULT_CACHE_DIR
    enabled: bool = True

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir).expanduser()
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, ticker: str, **kwargs: Any) -> pd.DataFrame:
        if not self.enabled:
            return self.loader(ticker, **kwargs)

        key = _make_key(ticker, **kwargs)
        stem = self.cache_dir / key
        cached = _try_read(stem)
        if cached is not None and not cached.empty:
            logger.info("Cache HIT: %s", stem.name)
            return cached

        logger.info("Cache MISS: %s — calling loader.", stem.name)
        df = self.loader(ticker, **kwargs)
        if df is not None and not df.empty:
            _try_write(df, stem)
        return df

    def clear(self, ticker: str | None = None) -> int:
        """Delete cached files. If ``ticker`` is given, only its files."""
        n = 0
        pattern = f"{ticker.upper()}_*" if ticker else "*"
        for p in self.cache_dir.glob(pattern):
            if p.suffix in (".parquet", ".csv"):
                p.unlink()
                n += 1
        return n
