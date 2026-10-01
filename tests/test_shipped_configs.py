"""The configs committed under configs/ must always load and follow the agreed policy."""

import re
from pathlib import Path

import pytest

from eval.registry import Bucket, TaskType, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def config():
    # Adapter-parent checks read local checkpoints, which only exist on the training box.
    return load_config(CONFIGS / "run.yaml", check_adapters=False)


def test_shipped_configs_load(config) -> None:
    assert config.models.base.id == "gemma-4-e2b-it"
    assert config.selected_models[0].id == "gemma-4-e2b-it"


def test_adapter_parents_match_when_checkpoints_are_present(config) -> None:
    from eval.registry import check_adapter_parents

    local = [m for m in config.models.all_models if m.checkpoint_ref.scheme == "local"]
    if not any(Path(m.checkpoint).exists() for m in local):
        pytest.skip("local checkpoints not on this machine")
    check_adapter_parents(config.models)


def test_hf_benchmarks_are_pinned_to_commit_shas(config) -> None:
    for b in config.benchmarks.benchmarks:
        if not b.dataset.startswith("local:"):
            assert re.fullmatch(r"[0-9a-f]{40}", b.revision or ""), b.id


def test_retention_benchmarks_are_english(config) -> None:
    for b in config.benchmarks.benchmarks:
        if b.bucket is Bucket.RETENTION:
            assert b.languages == ["eng"], b.id


def test_classification_benchmarks_carry_chance(config) -> None:
    for b in config.benchmarks.benchmarks:
        if b.type is TaskType.CLASSIFICATION:
            assert b.chance_level is not None and 0 < b.chance_level < 1, b.id


def test_scaffolds_are_disabled(config) -> None:
    assert not config.benchmarks.get("afrimcqa").enabled


def test_agreed_thresholds(config) -> None:
    t = config.run.thresholds
    assert t.noise_threshold == 1.0
    assert t.retention_tolerance == 3.0
