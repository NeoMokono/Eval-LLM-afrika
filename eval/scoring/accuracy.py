"""Accuracy and macro-F1, expressed as functions of per-example sufficient statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ratio(sums: np.ndarray) -> float:
    """sums = [numerator, denominator]."""
    return float(sums[0] / sums[1]) if sums[1] else float("nan")


def accuracy_stats(correct: Sequence[bool]) -> np.ndarray:
    """(n, 2) rows of [correct, 1]; `ratio` of the column sums is the accuracy."""
    flags = np.asarray(correct, dtype=np.float64)
    return np.stack([flags, np.ones_like(flags)], axis=1)


def macro_f1_stats(gold: Sequence[int], pred: Sequence[int], n_classes: int) -> np.ndarray:
    """(n, 3 * n_classes) one-hot contributions to [tp_c | fp_c | fn_c].

    A prediction outside 0..n_classes-1 (unparseable) is a false negative for the gold class
    and a true positive for none.
    """
    out = np.zeros((len(gold), 3 * n_classes), dtype=np.float64)
    for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
        if p == g:
            out[i, g] = 1.0
        else:
            out[i, 2 * n_classes + g] = 1.0
            if 0 <= p < n_classes:
                out[i, n_classes + p] = 1.0
    return out


def macro_f1(sums: np.ndarray) -> float:
    """Unweighted mean of per-class F1 (a class with no gold and no predictions scores 0)."""
    tp, fp, fn = np.split(np.asarray(sums, dtype=np.float64), 3)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())
