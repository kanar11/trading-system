"""Probability of Backtest Overfitting (PBO) via CSCV.

Combinatorially-symmetric cross-validation (Bailey, Borwein, López de Prado &
Zhu, 2017): given a panel of per-period returns for N candidate strategy
configurations, estimate how likely it is that the configuration which looks
best in-sample is merely overfit — i.e. fails to beat the median out-of-sample.

The series is cut into S equal time blocks; every way of choosing S/2 blocks as
the in-sample (IS) set — with the complement as out-of-sample (OOS) — is
evaluated. For each split we pick the IS-best config, find its OOS performance
rank, map that rank to a logit, and define PBO as the fraction of splits where
the IS-best config lands in the bottom half OOS (logit <= 0). Pure numpy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PBOResult:
    """Outcome of a CSCV overfitting analysis.

    Attributes:
        pbo: Probability of backtest overfitting in [0, 1]. ~0.5 means the
            selection has no real edge (pure overfitting risk); ~0 means the
            IS-best config reliably generalises; ~1 means systematic overfit.
        logits: Per-split logit of the IS-best config's OOS relative rank.
        is_best_oos_relative_rank: Per-split OOS relative rank (omega in (0, 1))
            of the IS-best config.
        n_combinations: Number of IS/OOS splits evaluated, i.e. C(S, S/2).
        n_strategies: Number of candidate configurations (N).
    """

    pbo: float
    logits: np.ndarray
    is_best_oos_relative_rank: np.ndarray
    n_combinations: int
    n_strategies: int


def _sharpe(matrix: np.ndarray) -> np.ndarray:
    """Per-column (per-config) Sharpe of a (rows, N) return block.

    Scale is irrelevant — only the cross-config ranking matters for CSCV — so
    this returns the plain mean / std (no annualisation). Constant columns map
    to 0.
    """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    safe_std = np.where(std > 0, std, 1.0)
    out: np.ndarray = np.where(std > 0, mean / safe_std, 0.0)
    return out


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_blocks: int = 10,
) -> PBOResult:
    """Estimate the probability of backtest overfitting via CSCV.

    Args:
        returns_matrix: (T, N) array of per-period returns; column j is the
            return series of candidate configuration j.
        n_blocks: Number of equal time blocks S (must be even and >= 2). The
            number of IS/OOS splits evaluated is C(S, S/2).

    Returns:
        A :class:`PBOResult`.

    Raises:
        ValueError: If the matrix is not 2-D, N < 2, ``n_blocks`` is odd or < 2,
            or there are fewer than ``2 * n_blocks`` observations.
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T observations x N configs).")
    n_obs, n_strategies = matrix.shape
    if n_strategies < 2:
        raise ValueError(f"Need at least 2 configurations, got {n_strategies}.")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even and >= 2, got {n_blocks}.")
    if n_obs < 2 * n_blocks:
        raise ValueError(f"Need at least {2 * n_blocks} observations, got {n_obs}.")

    block_rows = np.array_split(np.arange(n_obs), n_blocks)
    all_blocks = range(n_blocks)

    logits: list[float] = []
    omegas: list[float] = []
    for is_blocks in combinations(all_blocks, n_blocks // 2):
        is_set = set(is_blocks)
        is_rows = np.concatenate([block_rows[b] for b in is_blocks])
        oos_rows = np.concatenate([block_rows[b] for b in all_blocks if b not in is_set])

        is_perf = _sharpe(matrix[is_rows])
        oos_perf = _sharpe(matrix[oos_rows])

        n_star = int(np.argmax(is_perf))
        # 0-based ascending OOS rank of the IS-best config (ties -> strictly-less)
        rank = int(np.sum(oos_perf < oos_perf[n_star]))
        omega = (rank + 1) / (n_strategies + 1)
        omegas.append(omega)
        logits.append(float(np.log(omega / (1.0 - omega))))

    logit_arr = np.asarray(logits)
    pbo = float(np.mean(logit_arr <= 0.0))
    return PBOResult(
        pbo=pbo,
        logits=logit_arr,
        is_best_oos_relative_rank=np.asarray(omegas),
        n_combinations=int(logit_arr.size),
        n_strategies=n_strategies,
    )
