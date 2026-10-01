"""Deterministic, scriptable backend for development and tests (no model, no GPU)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from eval.backends.base import LoglikelihoodRequest, ModelBackend, Prompt, TokenNLL
from eval.registry import Decoding


def _unit(text: str) -> float:
    """A stable pseudo-random number in [0, 1) derived from `text`."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0x1_0000_0000


class MockBackend(ModelBackend):
    """By default every answer is a deterministic function of the prompt text.

    `choose(prompt, continuations) -> index` scripts multiple-choice answers,
    `respond(prompt) -> text` scripts generations, and `nll_per_token` sets perplexity.
    """

    def __init__(
        self,
        *,
        choose: Callable[[Prompt, Sequence[str]], int] | None = None,
        respond: Callable[[Prompt], str] | None = None,
        nll_per_token: float = 2.0,
    ) -> None:
        super().__init__()
        self._choose = choose
        self._respond = respond
        self._nll_per_token = nll_per_token

    def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[list[float]]:
        self.calls["loglikelihood"] += len(requests)
        out = []
        for r in requests:
            if self._choose is None:
                out.append([-10.0 * _unit(r.prompt.raw() + "\x00" + c) for c in r.continuations])
            else:
                picked = self._choose(r.prompt, r.continuations)
                out.append([-0.1 if i == picked else -5.0 for i in range(len(r.continuations))])
        return out

    def generate(self, prompts: Sequence[Prompt], decoding: Decoding) -> list[str]:
        self.calls["generate"] += len(prompts)
        if self._respond is not None:
            return [self._respond(p) for p in prompts]
        return [f"mock-{int(_unit(p.raw()) * 1e6):06d}" for p in prompts]

    def perplexity(self, texts: Sequence[str]) -> list[TokenNLL]:
        self.calls["perplexity"] += len(texts)
        return [TokenNLL(self._nll_per_token * len(t.split()), len(t.split())) for t in texts]
