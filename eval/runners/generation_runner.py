"""Free-generation runner (AfriMGSM, FLORES+, AfriScience MT, INJONGO slots).

Stores the raw output next to the reference; answer extraction and metrics live in scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eval.backends.base import ModelBackend, Prompt
from eval.registry import Decoding
from eval.tasks import Example


def predict(
    examples: Sequence[Example],
    prompts: Sequence[Prompt],
    backend: ModelBackend,
    decoding: Decoding,
) -> list[dict[str, Any]]:
    outputs = backend.generate(prompts, decoding)
    return [
        {
            "example_id": example.example_id,
            "language": example.language,
            "output": output,
            "reference": example.target,
        }
        for example, output in zip(examples, outputs, strict=True)
    ]
