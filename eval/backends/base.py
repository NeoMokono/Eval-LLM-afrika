"""The model interface every runner talks to: log-likelihood, generation and perplexity.

A `Prompt` is the query plus few-shot turns. The backend decides how it becomes text: through
the model's chat template (few-shots as alternating user/model turns) or as raw text
(lm-eval style: turns joined by blank lines, answer after a single space).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from eval.registry import Decoding


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


@dataclass(frozen=True)
class Prompt:
    query: str
    shots: tuple[Turn, ...] = ()

    def messages(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for turn in self.shots:
            out.append({"role": "user", "content": turn.user})
            out.append({"role": "assistant", "content": turn.assistant})
        out.append({"role": "user", "content": self.query})
        return out

    def raw(self) -> str:
        parts = [f"{t.user} {t.assistant}" for t in self.shots]
        return "\n\n".join([*parts, self.query])


@dataclass(frozen=True)
class LoglikelihoodRequest:
    """Score each continuation given the same prompt."""

    prompt: Prompt
    continuations: tuple[str, ...]


@dataclass(frozen=True)
class TokenNLL:
    """Summed negative log-likelihood (nats) over `n_tokens` scored tokens of one text."""

    nll: float
    n_tokens: int


class ModelBackend(ABC):
    """`calls` counts examples sent to the model per method; tests use it to prove that cached
    units trigger zero inference."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    @property
    def n_inferences(self) -> int:
        return sum(self.calls.values())

    @abstractmethod
    def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[list[float]]:
        """For each request, the log-probability of each continuation."""

    @abstractmethod
    def generate(self, prompts: Sequence[Prompt], decoding: Decoding) -> list[str]:
        """One completion per prompt (new text only)."""

    @abstractmethod
    def perplexity(self, texts: Sequence[str]) -> list[TokenNLL]:
        """Token-level NLL of each raw text (no chat template)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release model memory."""
