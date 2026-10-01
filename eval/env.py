"""Minimal `.env` support so HF_TOKEN can live in an untracked file instead of shell history."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> list[str]:
    """Set variables from `path` that are not already in the environment.

    Understands `KEY=value`, `export KEY=value`, comments and surrounding quotes. Returns the
    names it set (never the values).
    """
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
