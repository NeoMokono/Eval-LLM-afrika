"""Dataset access. Runners never call `datasets` directly: they ask a loader for the rows of
one (benchmark, subset, split), which lets tests substitute in-memory rows."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Protocol

from eval.registry import BenchmarkSpec

Row = dict[str, Any]


class DataError(Exception):
    pass


class DataLoader(Protocol):
    def rows(self, benchmark: BenchmarkSpec, subset: str, split: str) -> list[Row]:
        """Rows of `split` for one subset (an HF config name, or a `language_column` value)."""
        ...


class HFDataLoader:
    """Loads from the HF Hub at the exact revision the planner fingerprinted."""

    def __init__(self, revisions: Mapping[str, str] | None = None) -> None:
        self._revisions = dict(revisions or {})  # benchmark id -> resolved commit SHA
        self._memo: dict[tuple[str, str, str], list[Row]] = {}

    def _load(self, benchmark: BenchmarkSpec, name: str, split: str) -> list[Row]:
        key = (benchmark.id, name, split)
        if key not in self._memo:
            from datasets import load_dataset

            if benchmark.dataset.startswith("local:"):
                raise DataError(f"{benchmark.id}: local datasets are not supported yet")
            revision = self._revisions.get(benchmark.id, benchmark.revision)
            try:
                dataset = load_dataset(
                    benchmark.dataset,
                    name,
                    split=split,
                    revision=revision,
                    token=os.environ.get("HF_TOKEN"),
                )
            except Exception as exc:
                hint = (
                    " (gated dataset: set HF_TOKEN in the environment or in .env)"
                    if not os.environ.get("HF_TOKEN")
                    else ""
                )
                raise DataError(
                    f"{benchmark.id}: could not load {benchmark.dataset} [{name}/{split}] "
                    f"at {revision}{hint}: {exc}"
                ) from exc
            self._memo[key] = [dict(row) for row in dataset]
        return self._memo[key]

    def rows(self, benchmark: BenchmarkSpec, subset: str, split: str) -> list[Row]:
        if benchmark.language_column is None:
            return self._load(benchmark, benchmark.config or subset, split)
        assert benchmark.config is not None  # enforced by the schema
        rows = self._load(benchmark, benchmark.config, split)
        return [r for r in rows if r[benchmark.language_column] == subset]


class InMemoryLoader:
    """Rows supplied directly, keyed by (benchmark id, subset, split). For tests."""

    def __init__(self, data: Mapping[tuple[str, str, str], list[Row]]) -> None:
        self._data = dict(data)
        self.requests: list[tuple[str, str, str]] = []

    def rows(self, benchmark: BenchmarkSpec, subset: str, split: str) -> list[Row]:
        key = (benchmark.id, subset, split)
        self.requests.append(key)
        if key not in self._data:
            raise DataError(f"no in-memory rows for {key}")
        return list(self._data[key])
