"""Child-order scheduling: TWAP, VWAP and POV slicing.

A parent order is rarely sent to the market in one piece; execution algos
slice it across the trading window. The three industry workhorses:

* **TWAP** — equal quantities per time bucket; minimises timing decisions
  when no volume information is available.
* **VWAP** — quantities proportional to the expected volume profile, so the
  order trades along with the market and its average price tracks VWAP
  (complements the post-trade :func:`src.execution.tca.vwap_slippage`).
* **POV** (percent-of-volume) — trade a fixed participation rate of each
  bucket's expected volume until the parent is filled; by construction it
  may *not* complete inside the window if volume is thin — the shortfall is
  the caller's signal to extend or cross the spread.

These schedulers plan quantities only; costs of the resulting trajectory
can be priced with :mod:`src.execution.impact` (Almgren-Chriss). Pure
numpy; ``vwap_schedule`` and ``pov_schedule`` preserve a pandas index when
the volume profile is a Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_total(total_quantity: float) -> None:
    if total_quantity < 0:
        raise ValueError(f"total_quantity must be >= 0, got {total_quantity}.")


def _profile_to_array(volume_profile: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    profile = np.asarray(volume_profile, dtype=float)
    if profile.ndim != 1 or len(profile) == 0:
        raise ValueError("volume_profile must be a non-empty 1-D sequence.")
    if np.isnan(profile).any() or (profile < 0).any():
        raise ValueError("volume_profile must be non-negative and NaN-free.")
    return profile


def _wrap_like(
    values: np.ndarray,
    template: pd.Series | np.ndarray | list[float],
) -> pd.Series | np.ndarray:
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name="child_quantity")
    return values


def twap_schedule(total_quantity: float, n_slices: int) -> np.ndarray:
    """Equal-quantity time slicing.

    The last slice absorbs the floating-point remainder of the division,
    so the algebraic sum of the children equals the parent quantity.

    Args:
        total_quantity: Parent order size (>= 0, sign-free).
        n_slices: Number of time buckets (>= 1).

    Returns:
        Array of ``n_slices`` child quantities.

    Raises:
        ValueError: If ``total_quantity`` < 0 or ``n_slices`` < 1.
    """
    _validate_total(total_quantity)
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}.")
    per_slice = total_quantity / n_slices
    slices = np.full(n_slices, per_slice)
    slices[-1] = total_quantity - per_slice * (n_slices - 1)  # absorb float dust
    return slices


def vwap_schedule(
    total_quantity: float,
    volume_profile: pd.Series | np.ndarray | list[float],
) -> pd.Series | np.ndarray:
    """Slice a parent order proportionally to an expected volume profile.

    Args:
        total_quantity: Parent order size (>= 0, sign-free).
        volume_profile: Expected volume per bucket (non-negative, at least
            one bucket positive).

    Returns:
        Child quantities per bucket (Series when the profile is a Series,
        ndarray otherwise), summing to the total.

    Raises:
        ValueError: If the total is negative or the profile is empty,
            negative, NaN or all-zero.
    """
    _validate_total(total_quantity)
    profile = _profile_to_array(volume_profile)
    volume_sum = float(profile.sum())
    if volume_sum <= 0:
        raise ValueError("volume_profile must contain at least one positive bucket.")
    children = total_quantity * profile / volume_sum
    return _wrap_like(children, volume_profile)


def pov_schedule(
    total_quantity: float,
    volume_profile: pd.Series | np.ndarray | list[float],
    participation: float = 0.1,
) -> pd.Series | np.ndarray:
    """Percent-of-volume slicing at a fixed participation rate.

    Each bucket trades ``participation * expected_volume``, capped by the
    remaining parent quantity; once the parent is filled the remaining
    buckets are zero. If the window's volume is too thin the schedule ends
    unfilled — compare ``sum(schedule)`` to ``total_quantity``.

    Args:
        total_quantity: Parent order size (>= 0, sign-free).
        volume_profile: Expected volume per bucket (non-negative, NaN-free).
        participation: Target participation rate in (0, 1].

    Returns:
        Child quantities per bucket (Series when the profile is a Series,
        ndarray otherwise); the sum never exceeds ``total_quantity``.

    Raises:
        ValueError: If the total is negative, the profile is invalid, or
            ``participation`` is outside (0, 1].
    """
    _validate_total(total_quantity)
    profile = _profile_to_array(volume_profile)
    if not 0.0 < participation <= 1.0:
        raise ValueError(f"participation must be in (0, 1], got {participation}.")

    children = np.zeros(len(profile))
    remaining = total_quantity
    for i, bucket_volume in enumerate(profile):
        if remaining <= 0:
            break
        children[i] = min(participation * bucket_volume, remaining)
        remaining -= children[i]
    return _wrap_like(children, volume_profile)
