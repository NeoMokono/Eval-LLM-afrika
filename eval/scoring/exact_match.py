"""Exact match on the final number of a generated answer (AfriMGSM).

Extraction follows lm-evaluation-harness's `flexible-extract` filter for afrimgsm: take the
last number in the output. Outputs with no number are unparseable and count as wrong.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def _to_float(text: str) -> float | None:
    cleaned = text.replace("$", "").replace(",", "").rstrip(".")
    if cleaned in {"", "-", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_number(output: str) -> float | None:
    """The last parseable number in `output`, or None."""
    for match in reversed(list(_NUMBER.finditer(output))):
        value = _to_float(match.group(0))
        if value is not None:
            return value
    return None


def number_match(output: str, reference: str) -> tuple[bool, bool]:
    """(correct, parse_ok) for one generated answer against the reference number."""
    predicted = extract_number(output)
    if predicted is None:
        return (False, False)
    target = _to_float(reference)
    if target is None:
        raise ValueError(f"reference {reference!r} is not a number")
    return (abs(predicted - target) < 1e-6, True)
