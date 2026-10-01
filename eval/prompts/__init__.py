"""Versioned prompt templates. The template id and a hash of its text are part of every
fingerprint, so editing a template's wording reruns exactly the benchmarks that use it.

Phase 2 ships placeholders so planning works end to end; Phase 3 replaces them with real
templates built on verified dataset columns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from eval.registry import BenchmarkSpec


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def template_for(benchmark: BenchmarkSpec) -> PromptTemplate:
    return PromptTemplate(id=f"{benchmark.id}-placeholder-v0", text="TODO(phase3)")
