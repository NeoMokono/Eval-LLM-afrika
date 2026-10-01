"""Crash-safe file writes: write a temp file in the same directory, fsync, then rename."""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Replace `path` with `text` so readers see either the old or the new file, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive advisory lock on `<path>.lock`, so concurrent processes serialise updates."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
