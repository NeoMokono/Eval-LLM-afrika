"""Bootstrap confidence intervals from per-example sufficient statistics.

Every metric here is a function of column sums over examples (accuracy = correct / n, chrF++ =
f(summed n-gram counts), perplexity = exp(nll / tokens), ...). So a resample never rescoring
text: draw multinomial example weights, take weighted column sums, apply the metric.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Aggregate = Callable[[np.ndarray], float]  # column sums (k,) -> metric value


def bootstrap_ci(
    stats: np.ndarray,
    aggregate: Aggregate,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval of `aggregate` over resamples of the rows of `stats` (n, k)."""
    stats = np.asarray(stats, dtype=np.float64)
    if stats.ndim == 1:
        stats = stats[:, None]
    n = stats.shape[0]
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_resamples)  # (B, n)
    sums = weights @ stats  # (B, k)
    values = np.array([aggregate(row) for row in sums])
    tail = (1.0 - confidence) / 2.0
    low, high = np.nanquantile(values, [tail, 1.0 - tail])
    return (float(low), float(high))
