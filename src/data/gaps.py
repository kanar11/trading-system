"""Bar-frequency inference and missing-session detection.

:mod:`src.data.quality` finds bad *values*; this module finds bars that
should exist but *don't* — the feed dropped a session, a vendor export
truncated a week, an intraday file lost an hour. Everything is measured
against an explicit expected grid (a pandas offset alias, ``"B"`` for
daily equity data by default) between the index's first and last bar, so
weekends and anything else outside the grid are never false positives.

Direct-import module::

    from src.data.gaps import gap_report, infer_bar_frequency, missing_sessions
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validated(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"index must be a DatetimeIndex, got {type(index).__name__}.")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted in increasing order.")
    if index.has_duplicates:
        raise ValueError("index must not contain duplicate timestamps.")
    return index


def infer_bar_frequency(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Modal spacing between consecutive bars.

    The most common bar-to-bar gap — robust to occasional missing bars
    and weekend jumps, unlike the mean.

    Raises:
        TypeError: If ``index`` is not a DatetimeIndex.
        ValueError: If it is unsorted, duplicated, or has fewer than 2 bars.
    """
    index = _validated(index)
    if len(index) < 2:
        raise ValueError(f"need at least 2 bars to infer a frequency, got {len(index)}.")
    diffs = pd.Series(index).diff().dropna()
    return pd.Timedelta(diffs.mode().iloc[0])


def missing_sessions(index: pd.DatetimeIndex, freq: str = "B") -> pd.DatetimeIndex:
    """Expected-but-absent timestamps on the ``freq`` grid.

    Args:
        index: Observed bar timestamps (sorted, unique).
        freq: Pandas offset alias of the expected grid (``"B"`` business
            days, ``"30min"``, ...). The grid spans the index's first to
            last bar, so nothing outside the observed range is flagged.

    Returns:
        DatetimeIndex of grid points with no bar (empty when complete).

    Raises:
        TypeError / ValueError: As for :func:`infer_bar_frequency`
            (empty input returns an empty index).
    """
    index = _validated(index)
    if len(index) == 0:
        return pd.DatetimeIndex([])
    expected = pd.date_range(index[0], index[-1], freq=freq)
    return expected.difference(index)


def gap_report(index: pd.DatetimeIndex, freq: str = "B") -> pd.DataFrame:
    """Group missing sessions into contiguous gap episodes.

    Returns:
        DataFrame with one row per gap: ``start`` and ``end`` (first and
        last missing grid point) and ``n_missing`` (grid points lost),
        longest gaps first. Empty when the series is complete.

    Raises:
        TypeError / ValueError: As for :func:`missing_sessions`.
    """
    missing = missing_sessions(index, freq=freq)
    if len(missing) == 0:
        return pd.DataFrame(columns=["start", "end", "n_missing"])

    grid = pd.date_range(index[0], index[-1], freq=freq)
    positions = grid.get_indexer(missing)
    episode = np.concatenate([[0], np.cumsum(np.diff(positions) != 1)])

    rows = []
    for episode_id in np.unique(episode):
        chunk = missing[episode == episode_id]
        rows.append({"start": chunk[0], "end": chunk[-1], "n_missing": len(chunk)})
    table = pd.DataFrame(rows)
    return table.sort_values("n_missing", ascending=False).reset_index(drop=True)
