"""chrF++ via sacrebleu (character 6-grams + word bigrams, `word_order=2`), on a 0-100 scale."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sacrebleu.metrics import CHRF

_METRIC = CHRF(word_order=2)


def chrf_stats(hypotheses: Sequence[str], references: Sequence[str]) -> np.ndarray:
    """(n, k) per-sentence n-gram statistics; their column sums give the corpus score."""
    if not hypotheses:
        return np.zeros((0, 1))
    return np.asarray(
        _METRIC._extract_corpus_statistics(list(hypotheses), [list(references)]), dtype=np.float64
    )


def chrf_from_sums(sums: np.ndarray) -> float:
    return float(_METRIC._compute_score_from_stats([round(float(x)) for x in sums]).score)


def signature() -> str:
    return "chrF2++|nc:6|nw:2|case:mixed|eff:yes|space:no|sacrebleu"
