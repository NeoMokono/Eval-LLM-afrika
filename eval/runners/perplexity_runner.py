"""Token-level perplexity runner (English retention): one row per document."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eval.backends.base import ModelBackend
from eval.tasks import Example


def predict(examples: Sequence[Example], backend: ModelBackend) -> list[dict[str, Any]]:
    results = backend.perplexity([example.query for example in examples])
    return [
        {
            "example_id": example.example_id,
            "language": example.language,
            "nll": result.nll,
            "n_tokens": result.n_tokens,
        }
        for example, result in zip(examples, results, strict=True)
    ]
