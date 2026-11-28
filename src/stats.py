"""Resampling functions for comparing two groups (permutation test and bootstrap)."""
from typing import Callable

import numpy as np


def difference_in_means(x: np.ndarray, y: np.ndarray) -> float:
    """Return mean(x) - mean(y)."""
    return float(np.mean(x) - np.mean(y))


def _check_groups(group_a: np.ndarray, group_b: np.ndarray) -> None:
    """Raise a clear error if either group is empty."""
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError("Both groups must contain at least one observation.")


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray, np.ndarray], float] = difference_in_means,
) -> tuple[float, float, np.ndarray]:
    """Two-sided permutation test for two independent groups.

    Pool both groups, randomly reassign the observations to two groups of the
    original sizes, and recompute the statistic. Repeating this builds the
    distribution of the statistic expected if group labels were arbitrary.

    Returns (observed statistic, Monte Carlo p-value, null distribution).
    The p-value uses the (1 + count) / (n_perm + 1) correction, so it is
    never exactly zero.
    """
    _check_groups(group_a, group_b)
    observed = statistic(group_a, group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    null = np.empty(n_perm)
    for i in range(n_perm):
        permuted = rng.permutation(combined)
        null[i] = statistic(permuted[:n_a], permuted[n_a:])

    p_value = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
    return observed, float(p_value), null


def bootstrap_two_sample(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray, np.ndarray], float] = difference_in_means,
) -> np.ndarray:
    """Bootstrap distribution of a two-group statistic.

    Each replicate resamples every group with replacement (keeping its size)
    and recomputes the statistic. Returns the array of replicates.
    """
    _check_groups(group_a, group_b)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        boot_a = rng.choice(group_a, size=len(group_a), replace=True)
        boot_b = rng.choice(group_b, size=len(group_b), replace=True)
        reps[i] = statistic(boot_a, boot_b)
    return reps


def percentile_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval: the middle (1 - alpha) share of the values."""
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)