"""Token-level perplexity from per-document [nll, n_tokens]."""

from __future__ import annotations

import math

import numpy as np


def perplexity(sums: np.ndarray) -> float:
    nll, tokens = float(sums[0]), float(sums[1])
    return math.exp(nll / tokens) if tokens else float("nan")
