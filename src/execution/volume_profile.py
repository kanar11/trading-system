"""Intraday volume-profile generators.

The child-order schedulers (:func:`src.execution.schedule.vwap_schedule`,
:func:`~src.execution.schedule.pov_schedule`) need an *expected volume per
bucket* to slice against — but real intraday volume is not flat. It is the
well-documented **U shape** (or J): a burst at the open as overnight orders
clear, a midday lull, and a surge into the close on index/rebalance flow.
Slicing a VWAP order against a flat profile would over-trade the quiet
midday and under-participate at the liquid open and close.

:func:`intraday_volume_profile` produces that expected shape as normalised
weights summing to 1, ready to pass straight to the schedulers. The
``"u"`` shape is a floored parabola (elevated ends, ``depth`` sets how
shallow the midday trough is); ``"flat"`` reproduces the uniform profile
for A/B comparison. Pure numpy.
"""

from __future__ import annotations

import numpy as np


def intraday_volume_profile(
    n_buckets: int,
    shape: str = "u",
    depth: float = 0.4,
) -> np.ndarray:
    """Expected intraday volume weights per bucket (sum to 1).

    Args:
        n_buckets: Number of time buckets over the session (>= 1).
        shape: ``"u"`` for the elevated-open/close U profile, ``"flat"``
            for a uniform profile.
        depth: For the U shape, the midday trough height relative to the
            ends, in (0, 1] — 1.0 is flat, small values dig a deep midday
            lull. Ignored for ``"flat"``.

    Returns:
        Array of ``n_buckets`` non-negative weights summing to 1.

    Raises:
        ValueError: If ``n_buckets`` < 1, ``shape`` is unknown, or
            ``depth`` is outside (0, 1].
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}.")
    if shape not in ("u", "flat"):
        raise ValueError(f"shape must be 'u' or 'flat', got {shape!r}.")

    if shape == "flat" or n_buckets == 1:
        # bind to a typed local: numpy stubs otherwise infer Any here on the
        # py3.11 toolchain -> mypy [no-any-return]
        uniform: np.ndarray = np.full(n_buckets, 1.0 / n_buckets)
        return uniform

    if not 0.0 < depth <= 1.0:
        raise ValueError(f"depth must be in (0, 1], got {depth}.")

    # x in [0, 1]; a floored parabola: 1 at the ends, `depth` at the middle
    x = np.linspace(0.0, 1.0, n_buckets)
    raw = depth + (1.0 - depth) * 4.0 * (x - 0.5) ** 2
    weights: np.ndarray = raw / raw.sum()
    return weights
