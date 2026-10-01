"""Metric implementations, bootstrap CIs and confusion matrices.

`score_language` turns one language's prediction rows into the primary metric, its 95%
bootstrap interval, any extra metrics and the count of unparseable outputs.

Scales: accuracy, exact match, macro-F1 and slot F1 are stored in [0, 1]; chrF++ in [0, 100];
perplexity as is. `to_points` puts the score metrics on a common 0-100 scale for reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval.registry import BenchmarkSpec, Bootstrap
from eval.scoring import accuracy, chrf, exact_match, perplexity, slot_f1
from eval.scoring.bootstrap import Aggregate, bootstrap_ci

UNIT_INTERVAL_METRICS = frozenset({"accuracy", "exact_match", "macro_f1", "slot_f1"})


def to_points(metric: str, value: float) -> float:
    """Score on a 0-100 scale (perplexity is not a score and is returned unchanged)."""
    return value * 100.0 if metric in UNIT_INTERVAL_METRICS else value


@dataclass(frozen=True)
class Interval:
    value: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class LanguageScore:
    primary: Interval
    n_examples: int
    n_unparseable: int = 0
    extra: dict[str, Interval] = field(default_factory=dict)


def _interval(stats: np.ndarray, aggregate: Aggregate, bootstrap: Bootstrap) -> Interval:
    low, high = bootstrap_ci(
        stats,
        aggregate,
        n_resamples=bootstrap.n_resamples,
        seed=bootstrap.seed,
        confidence=bootstrap.confidence,
    )
    return Interval(aggregate(np.asarray(stats, dtype=np.float64).sum(axis=0)), low, high)


def score_language(
    benchmark: BenchmarkSpec, rows: list[dict[str, Any]], bootstrap: Bootstrap
) -> LanguageScore:
    """Score one language. For exact match this also writes `correct`/`parse_ok` into `rows`."""
    if not rows:
        raise ValueError(f"{benchmark.id}: no predictions to score")
    n = len(rows)
    metric = benchmark.metric

    if metric == "accuracy":
        primary = _interval(
            accuracy.accuracy_stats([r["correct"] for r in rows]), accuracy.ratio, bootstrap
        )
        extra = {}
        if "macro_f1" in benchmark.extra_metrics:
            n_classes = len(rows[0]["scores"]) or (max(r["gold"] for r in rows) + 1)
            stats = accuracy.macro_f1_stats(
                [r["gold"] for r in rows], [r["pred"] for r in rows], n_classes
            )
            extra["macro_f1"] = _interval(stats, accuracy.macro_f1, bootstrap)
        unparseable = sum(not r["parse_ok"] for r in rows)
        return LanguageScore(primary, n, unparseable, extra)

    if metric == "exact_match":
        for r in rows:
            r["correct"], r["parse_ok"] = exact_match.number_match(r["output"], r["reference"])
        stats = accuracy.accuracy_stats([r["correct"] for r in rows])
        unparseable = sum(not r["parse_ok"] for r in rows)
        return LanguageScore(_interval(stats, accuracy.ratio, bootstrap), n, unparseable)

    if metric == "chrf++":
        stats = chrf.chrf_stats([r["output"] for r in rows], [r["reference"] for r in rows])
        empty = sum(not r["output"].strip() for r in rows)
        return LanguageScore(_interval(stats, chrf.chrf_from_sums, bootstrap), n, empty)

    if metric == "slot_f1":
        stats = slot_f1.slot_stats([r["output"] for r in rows], [r["reference"] for r in rows])
        return LanguageScore(_interval(stats, slot_f1.slot_f1, bootstrap), n)

    if metric == "perplexity":
        stats = np.asarray([[r["nll"], r["n_tokens"]] for r in rows], dtype=np.float64)
        return LanguageScore(_interval(stats, perplexity.perplexity, bootstrap), n)

    raise ValueError(f"no scorer for metric {metric!r}")
