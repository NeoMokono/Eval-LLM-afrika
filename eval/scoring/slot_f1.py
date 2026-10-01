"""Span F1 for INJONGO slot filling.

Gold and predictions are strings of `LABEL: value $$ LABEL: value`. Scoring follows the
CoNLL-style span F1 that lm-evaluation-harness uses for the same format (MasakhaNER): a span
counts when label and normalised value both match; F1 is micro-averaged over all spans.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

_PUNCTUATION = re.compile(r"""[!"$%&'()*+,\-./:;<=>?\[\\\]^_`{|}~•@]+""")
_SEPARATOR = re.compile(r"\s*\$+\s*")


def _normalize(text: str) -> str:
    text = _PUNCTUATION.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def parse_spans(text: str) -> Counter[tuple[str, str]]:
    """Multiset of (label, normalised value). Empty or 'none' means no spans; fragments
    without a `LABEL: value` shape are ignored."""
    spans: Counter[tuple[str, str]] = Counter()
    stripped = text.strip()
    if not stripped or stripped.lower().strip(".") == "none":
        return spans
    for part in _SEPARATOR.split(stripped.replace("\n", " $$ ")):
        label, sep, value = part.partition(":")
        label, value = label.strip().upper(), _normalize(value)
        if sep and label and value and re.fullmatch(r"[A-Z_]+", label):
            spans[(label, value)] += 1
    return spans


def slot_stats(outputs: Sequence[str], references: Sequence[str]) -> np.ndarray:
    """(n, 3) rows of [true positives, false positives, false negatives]."""
    rows = []
    for output, reference in zip(outputs, references, strict=True):
        pred, gold = parse_spans(output), parse_spans(reference)
        tp = sum((pred & gold).values())
        rows.append([tp, sum(pred.values()) - tp, sum(gold.values()) - tp])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def slot_f1(sums: np.ndarray) -> float:
    tp, fp, fn = (float(x) for x in sums)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0
