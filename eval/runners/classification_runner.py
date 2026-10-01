"""Multiple-choice runner: log-likelihood over the options (default) or generate-then-parse."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from eval.backends.base import LoglikelihoodRequest, ModelBackend, Prompt
from eval.registry import Decoding
from eval.tasks import Example

UNPARSEABLE = -1


def parse_choice(output: str, options: Sequence[str]) -> int:
    """Index of the option the model's free-text answer names, or UNPARSEABLE.

    Accepts an answer that is exactly an option, starts with one ("B. because ..."), or
    mentions exactly one option. Anything ambiguous is unparseable (and scored as wrong).
    """
    text = output.strip().casefold()
    folded = [o.casefold() for o in options]
    cleaned = text.strip(" \t\n.:;,!\"'()[]*")
    if cleaned in folded:
        return folded.index(cleaned)
    starts = [i for i, o in enumerate(folded) if re.match(rf"\W*{re.escape(o)}(?![\w/])", text)]
    if starts:
        return max(starts, key=lambda i: len(folded[i]))  # "science/technology" over "science"
    mentioned = [
        i for i, o in enumerate(folded) if re.search(rf"(?<![\w/]){re.escape(o)}(?![\w/])", text)
    ]
    return mentioned[0] if len(mentioned) == 1 else UNPARSEABLE


def _row(example: Example, pred: int, **extra: Any) -> dict[str, Any]:
    assert example.gold is not None
    return {
        "example_id": example.example_id,
        "language": example.language,
        "gold": example.gold,
        "pred": pred,
        "gold_label": example.options[example.gold],
        "pred_label": example.options[pred] if pred != UNPARSEABLE else "",
        "correct": pred == example.gold,
        **extra,
    }


def predict_loglikelihood(
    examples: Sequence[Example], prompts: Sequence[Prompt], backend: ModelBackend
) -> list[dict[str, Any]]:
    requests = [
        LoglikelihoodRequest(prompt, example.options)
        for example, prompt in zip(examples, prompts, strict=True)
    ]
    rows = []
    for example, scores in zip(examples, backend.loglikelihood(requests), strict=True):
        pred = max(range(len(scores)), key=scores.__getitem__)
        rows.append(
            _row(example, pred, scores=[float(s) for s in scores], parse_ok=True, output="")
        )
    return rows


def predict_generate(
    examples: Sequence[Example],
    prompts: Sequence[Prompt],
    backend: ModelBackend,
    decoding: Decoding,
) -> list[dict[str, Any]]:
    rows = []
    for example, output in zip(examples, backend.generate(prompts, decoding), strict=True):
        pred = parse_choice(output, example.options)
        rows.append(_row(example, pred, scores=[], parse_ok=pred != UNPARSEABLE, output=output))
    return rows
