"""Manifest lifecycle: only `complete` is a cache hit, and writes are atomic."""

import json
import os
from pathlib import Path

import pytest

from eval.fingerprint import compute_fingerprint
from eval.manifest import Manifest, ManifestError, ManifestKey
from eval.prompts import PromptTemplate
from eval.registry import BenchmarkSpec, Prompting
from tests.conftest import SHA, classification_bench


def _fp(shots: int = 0, harness: str = "0.1.0"):
    return compute_fingerprint(
        checkpoint_hash="c" * 64,
        benchmark=BenchmarkSpec.model_validate(classification_bench()),
        revision=SHA,
        template=PromptTemplate("t", "text"),
        prompting=Prompting(shots=[0, 5]),
        shots=shots,
        harness=harness,
    )


def _start(manifest: Manifest, model: str = "rung2", shots: int = 0, **over) -> ManifestKey:
    fp = over.pop("fingerprint", _fp(shots))
    entry = manifest.mark_running(
        model_id=model,
        benchmark_id="afrixnli",
        shots=shots,
        fingerprint=fp,
        run_id="run-1",
        harness_version="0.1.0",
        **over,
    )
    return entry.key


def test_missing_manifest_is_empty(tmp_path: Path) -> None:
    assert Manifest.load(tmp_path / "manifest.json").entries == []


def test_running_entry_is_written_before_work_and_is_not_a_cache_hit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = Manifest.load(path)
    key = _start(manifest)
    on_disk = Manifest.load(path)  # what a new process would see after a crash
    assert on_disk.find(key).status == "running"
    assert on_disk.complete(key) is None


def test_complete_entry_is_a_cache_hit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = Manifest.load(path)
    key = _start(manifest)
    manifest.mark_complete(key, "runs/run-1/rung2/afrixnli/0shot")
    hit = Manifest.load(path).complete(key)
    assert hit is not None
    assert hit.record_path == "runs/run-1/rung2/afrixnli/0shot"
    assert hit.completed_at is not None


def test_failed_entry_is_not_a_cache_hit(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path / "manifest.json")
    key = _start(manifest)
    manifest.mark_failed(key, "CUDA out of memory")
    reloaded = Manifest.load(manifest.path)
    assert reloaded.complete(key) is None
    assert reloaded.find(key).error == "CUDA out of memory"


def test_finishing_requires_a_running_entry(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path / "manifest.json")
    key = ManifestKey.of("rung2", "afrixnli", _fp())
    with pytest.raises(ManifestError, match="no running manifest entry"):
        manifest.mark_complete(key, "x")


def test_key_separates_shots_and_fingerprints(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path / "manifest.json")
    zero, five = _start(manifest, shots=0), _start(manifest, shots=5)
    assert zero != five
    manifest.mark_complete(zero, "p0")
    assert manifest.complete(zero) is not None
    assert manifest.complete(five) is None
    assert manifest.latest_complete("rung2", "afrixnli", 0).record_path == "p0"
    assert manifest.latest_complete("rung2", "afrixnli", 5) is None


def test_forced_rerun_records_reason_and_previous_result(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path / "manifest.json")
    key = _start(manifest)
    manifest.mark_complete(key, "old/path")
    _start(manifest, rerun_reason="re-ran decontamination")
    entry = Manifest.load(manifest.path).find(key)
    assert entry.status == "running"
    assert entry.rerun_reason == "re-ran decontamination"
    assert entry.previous_record_path == "old/path"
    assert len(manifest.entries) == 1


def test_manifest_file_is_plain_json_with_the_spec_key(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path / "manifest.json")
    _start(manifest)
    doc = json.loads(manifest.path.read_text())
    assert doc["schema_version"] == 1
    assert {
        "model_id",
        "checkpoint_hash",
        "benchmark_id",
        "benchmark_version",
        "config_fingerprint",
        "status",
        "rerun_reason",
        "started_at",
    } <= doc["entries"][0].keys()


def test_write_is_atomic_when_the_process_dies_mid_save(tmp_path: Path, monkeypatch) -> None:
    manifest = Manifest.load(tmp_path / "manifest.json")
    key = _start(manifest)
    manifest.mark_complete(key, "p")
    before = manifest.path.read_text()

    def crash(src, dst):
        raise OSError("disk pulled")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="disk pulled"):
        _start(manifest, model="other")
    monkeypatch.undo()

    assert manifest.path.read_text() == before  # old file intact, still valid
    assert Manifest.load(manifest.path).complete(key) is not None
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_damaged_manifest_is_an_error_not_an_empty_cache(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": 1, "entries": [')
    with pytest.raises(ManifestError, match="unreadable"):
        Manifest.load(path)
    path.write_text('{"schema_version": 99, "entries": []}')
    with pytest.raises(ManifestError, match="schema_version"):
        Manifest.load(path)


def test_two_handles_do_not_lose_each_others_updates(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    a, b = Manifest.load(path), Manifest.load(path)
    key_a = _start(a, model="model-a")
    key_b = _start(b, model="model-b")  # b's in-memory copy is stale; it must re-read
    a.mark_complete(key_a, "pa")
    final = Manifest.load(path)
    assert final.complete(key_a) is not None
    assert final.find(key_b).status == "running"
