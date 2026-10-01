"""Confusion matrices and prediction-distribution summaries for classification runs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

UNPARSEABLE_LABEL = "(unparseable)"


def confusion(gold: Sequence[int], pred: Sequence[int], labels: Sequence[str]) -> np.ndarray:
    """Rows = gold class, columns = predicted class; a final column counts unparseable
    predictions (index -1) when there are any."""
    n = len(labels)
    has_unparseable = any(p < 0 or p >= n for p in pred)
    mapped = [p if 0 <= p < n else n for p in pred]
    matrix = confusion_matrix(list(gold), mapped, labels=list(range(n + 1)))
    return matrix[:n, : n + 1] if has_unparseable else matrix[:n, :n]


def prediction_shares(pred: Sequence[int], n_classes: int) -> list[float]:
    """Fraction of predictions falling on each class (unparseable ones are left out of the
    numerator but kept in the denominator)."""
    if not pred:
        return [0.0] * n_classes
    counts = np.bincount([p for p in pred if 0 <= p < n_classes], minlength=n_classes)
    return [float(c) / len(pred) for c in counts]


def to_markdown(matrix: np.ndarray, labels: Sequence[str]) -> str:
    columns = list(labels) + ([UNPARSEABLE_LABEL] if matrix.shape[1] > len(labels) else [])
    lines = ["| gold \\ pred | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    for label, row in zip(labels, matrix, strict=True):
        lines.append(f"| {label} | " + " | ".join(str(int(v)) for v in row) + " |")
    return "\n".join(lines)
