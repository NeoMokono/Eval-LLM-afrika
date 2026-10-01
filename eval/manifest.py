"""results/manifest.json: the index of every (model, benchmark, shots) evaluation ever attempted.

Key: (model_id, checkpoint_hash, benchmark_id, benchmark_version, config_fingerprint).
Only entries with status `complete` are cache hits. An entry is marked `running` *before*
inference starts, so a crash leaves a `running` entry behind, never a fake cache hit. Every
update takes a file lock, re-reads the file, applies the change and writes it atomically.
Old entries are kept, so reverting a config change is served from cache again.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, ValidationError

from eval.atomic import atomic_write_text, file_lock
from eval.fingerprint import Fingerprint

SCHEMA_VERSION = 1
Status = Literal["running", "complete", "failed"]


class ManifestError(Exception):
    pass


class ManifestKey(NamedTuple):
    model_id: str
    checkpoint_hash: str
    benchmark_id: str
    benchmark_version: str
    config_fingerprint: str

    @classmethod
    def of(cls, model_id: str, benchmark_id: str, fp: Fingerprint) -> ManifestKey:
        return cls(
            model_id, fp.checkpoint_hash, benchmark_id, fp.benchmark_version, fp.config_fingerprint
        )


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    checkpoint_hash: str
    benchmark_id: str
    benchmark_version: str
    config_fingerprint: str
    shots: int
    fingerprint: str
    components: dict[str, Any]
    status: Status
    run_id: str
    harness_version: str
    started_at: str
    completed_at: str | None = None
    record_path: str | None = None
    previous_record_path: str | None = None
    rerun_reason: str | None = None
    error: str | None = None

    @property
    def key(self) -> ManifestKey:
        return ManifestKey(
            self.model_id,
            self.checkpoint_hash,
            self.benchmark_id,
            self.benchmark_version,
            self.config_fingerprint,
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path, entries: list[ManifestEntry] | None = None) -> None:
        self.path = path
        self.entries: list[ManifestEntry] = entries or []

    # ------------------------------------------------------------------ reading

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.exists():
            return cls(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("schema_version") != SCHEMA_VERSION:
                raise ManifestError(
                    f"{path}: schema_version {doc.get('schema_version')!r}, "
                    f"expected {SCHEMA_VERSION}"
                )
            entries = [ManifestEntry.model_validate(e) for e in doc.get("entries", [])]
        except (json.JSONDecodeError, ValidationError, AttributeError) as exc:
            # Never treat a damaged manifest as empty: that would silently rerun everything.
            raise ManifestError(f"{path} is unreadable ({exc}); fix or restore it") from exc
        return cls(path, entries)

    def find(self, key: ManifestKey) -> ManifestEntry | None:
        return next((e for e in self.entries if e.key == key), None)

    def complete(self, key: ManifestKey) -> ManifestEntry | None:
        entry = self.find(key)
        return entry if entry is not None and entry.status == "complete" else None

    def latest_complete(self, model_id: str, benchmark_id: str, shots: int) -> ManifestEntry | None:
        done = [
            e
            for e in self.entries
            if (e.model_id, e.benchmark_id, e.shots, e.status)
            == (model_id, benchmark_id, shots, "complete")
        ]
        return max(done, key=lambda e: e.completed_at or "", default=None)

    # ------------------------------------------------------------------ writing

    def _mutate(self, change: Callable[[Manifest], ManifestEntry]) -> ManifestEntry:
        with file_lock(self.path):
            fresh = Manifest.load(self.path)
            entry = change(fresh)
            fresh._save()
            self.entries = fresh.entries
        return entry

    def _save(self) -> None:
        entries = sorted(
            (e.model_dump() for e in self.entries),
            key=lambda e: (e["model_id"], e["benchmark_id"], e["shots"], e["started_at"]),
        )
        doc = {"schema_version": SCHEMA_VERSION, "entries": entries}
        atomic_write_text(self.path, json.dumps(doc, indent=2, sort_keys=True) + "\n")

    def mark_running(
        self,
        *,
        model_id: str,
        benchmark_id: str,
        shots: int,
        fingerprint: Fingerprint,
        run_id: str,
        harness_version: str,
        rerun_reason: str | None = None,
    ) -> ManifestEntry:
        key = ManifestKey.of(model_id, benchmark_id, fingerprint)

        def change(m: Manifest) -> ManifestEntry:
            previous = m.find(key)
            entry = ManifestEntry(
                **key._asdict(),
                shots=shots,
                fingerprint=fingerprint.value,
                components=fingerprint.components,
                status="running",
                run_id=run_id,
                harness_version=harness_version,
                started_at=_now(),
                previous_record_path=previous.record_path if previous else None,
                rerun_reason=rerun_reason,
            )
            m.entries = [e for e in m.entries if e.key != key] + [entry]
            return entry

        return self._mutate(change)

    def _finish(self, key: ManifestKey, **updates: Any) -> ManifestEntry:
        def change(m: Manifest) -> ManifestEntry:
            entry = m.find(key)
            if entry is None or entry.status != "running":
                raise ManifestError(f"no running manifest entry for {key}")
            done = entry.model_copy(update={**updates, "completed_at": _now()})
            m.entries = [done if e.key == key else e for e in m.entries]
            return done

        return self._mutate(change)

    def mark_complete(self, key: ManifestKey, record_path: str) -> ManifestEntry:
        return self._finish(key, status="complete", record_path=record_path, error=None)

    def mark_failed(self, key: ManifestKey, error: str) -> ManifestEntry:
        return self._finish(key, status="failed", error=error)
