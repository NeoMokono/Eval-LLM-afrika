"""Deterministic few-shot selection."""

from __future__ import annotations

import random
from collections.abc import Sequence

from eval.backends.base import Turn
from eval.tasks import Example, ShotPolicy

FEWSHOT_SEED = 1234


class FewshotError(Exception):
    pass


def prepare_pool(examples: Sequence[Example], policy: ShotPolicy) -> list[Example]:
    """Order the few-shot pool once per (benchmark, language).

    `same_group` keeps dataset order (MMLU's dev split is the canonical 5 per subject);
    the other policies use a fixed-seed shuffle so the shots are not just the first rows.
    """
    pool = list(examples)
    if policy != "same_group":
        random.Random(FEWSHOT_SEED).shuffle(pool)
    return pool


def choose_shots(
    example: Example, pool: Sequence[Example], k: int, policy: ShotPolicy
) -> tuple[Turn, ...]:
    """The `k` few-shot turns for one example. Never includes the example itself."""
    if k == 0:
        return ()
    candidates = [e for e in pool if e.example_id != example.example_id or e.query != example.query]
    if policy == "same_group":
        same = [e for e in candidates if e.group == example.group]
        candidates = same + [e for e in candidates if e.group != example.group]
    elif policy == "other_group":
        candidates = [e for e in candidates if e.group != example.group]
    if len(candidates) < k:
        raise FewshotError(f"only {len(candidates)} few-shot candidates available, need {k}")
    return tuple(Turn(e.query, e.target) for e in candidates[:k])
