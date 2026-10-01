"""models.yaml schema and the role rules (invariant 1)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.registry import (
    CheckpointRef,
    ConfigError,
    ModelsConfig,
    Role,
    check_adapter_parents,
    load_config,
)
from tests.conftest import base_model, models_doc, plain_model, run_doc, target_model


def _models(doc: dict) -> ModelsConfig:
    return ModelsConfig.model_validate(doc)


def _error(doc: dict) -> str:
    with pytest.raises(ValidationError) as exc:
        _models(doc)
    return str(exc.value)


# --- invariant 1: base required, plain baseline optional, never conflated --------------------


def test_base_only_is_valid_and_has_no_plain_baseline() -> None:
    cfg = _models(models_doc(plain=False))
    assert cfg.base.id == "gemma-base"
    assert cfg.plain_baseline is None
    assert cfg.has_plain_baseline is False


def test_plain_baseline_is_never_base() -> None:
    cfg = _models(models_doc(plain=True))
    assert cfg.plain_baseline is not None
    assert cfg.plain_baseline.id == "plain"
    assert cfg.plain_baseline.id != cfg.base.id


def test_missing_base_is_rejected() -> None:
    doc = {"reference_models": [plain_model(adapter_of=None)], "target_models": []}
    assert "exactly one model with role: base" in _error(doc)


def test_two_bases_are_rejected() -> None:
    doc = {"reference_models": [base_model(), base_model(id="other", checkpoint="/m/other")]}
    assert "exactly one model with role: base" in _error(doc)


def test_two_plain_baselines_are_rejected() -> None:
    doc = models_doc()
    doc["reference_models"].append(plain_model(id="plain2", checkpoint="/models/plain2"))
    assert "at most one model may have role: plain_baseline" in _error(doc)


def test_plain_baseline_aliasing_base_checkpoint_is_rejected() -> None:
    doc = models_doc()
    doc["reference_models"][1] = plain_model(checkpoint=base_model()["checkpoint"])
    assert "same checkpoint as base" in _error(doc)


def test_extra_references_are_neither_base_nor_plain_baseline() -> None:
    doc = models_doc(plain=False)
    doc["reference_models"].append(
        {"id": "gemma-pt", "role": "reference", "checkpoint": "hf://google/gemma-4-E2B"}
    )
    cfg = _models(doc)
    assert [m.id for m in cfg.extra_references] == ["gemma-pt"]
    assert cfg.plain_baseline is None
    assert cfg.base.id == "gemma-base"


@pytest.mark.parametrize(
    ("section", "model"),
    [
        ("target_models", base_model(id="b2")),
        ("reference_models", target_model(id="t2")),
        ("external_models", target_model(id="t3")),
    ],
)
def test_role_must_match_section(section: str, model: dict) -> None:
    doc = models_doc()
    doc.setdefault(section, []).append(model)
    assert "sits under" in _error(doc)


def test_duplicate_ids_are_rejected() -> None:
    doc = models_doc()
    doc["target_models"].append(target_model(checkpoint="/models/other"))
    assert "duplicate model ids: rung2" in _error(doc)


# --- adapters must name the base as their parent ("clarifying the base") ----------------------


def test_adapter_of_unknown_model_is_rejected() -> None:
    doc = models_doc()
    doc["target_models"][0] = target_model(adapter_of="nope")
    assert "adapter_of 'nope' is not a model id" in _error(doc)


def test_target_adapter_of_non_base_reference_is_rejected() -> None:
    doc = models_doc(plain=False)
    doc["reference_models"].append(
        {"id": "gemma-pt", "role": "reference", "checkpoint": "hf://google/gemma-4-E2B"}
    )
    doc["target_models"][0] = target_model(adapter_of="gemma-pt")
    assert "base must be the checkpoint" in _error(doc)


def _adapter_dir(tmp_path: Path, parent: str) -> Path:
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": parent}))
    return d


def test_adapter_parent_matching_base_passes(tmp_path: Path) -> None:
    doc = models_doc(plain=False)
    path = _adapter_dir(tmp_path, "google/gemma-4-E2B-it")
    doc["target_models"][0] = target_model(checkpoint=str(path))
    check_adapter_parents(_models(doc))


def test_adapter_parent_mismatch_is_rejected(tmp_path: Path) -> None:
    doc = models_doc(plain=False)
    path = _adapter_dir(tmp_path, "google/gemma-2-2b")
    doc["target_models"][0] = target_model(checkpoint=str(path))
    with pytest.raises(ConfigError, match="trained from 'google/gemma-2-2b'"):
        check_adapter_parents(_models(doc))


def test_adapter_without_adapter_of_is_rejected(tmp_path: Path) -> None:
    doc = models_doc(plain=False)
    path = _adapter_dir(tmp_path, "google/gemma-4-E2B-it")
    doc["target_models"][0] = target_model(checkpoint=str(path), adapter_of=None)
    with pytest.raises(ConfigError, match="set adapter_of"):
        check_adapter_parents(_models(doc))


# --- checkpoints and sft_data ----------------------------------------------------------------


def test_checkpoint_uri_parsing() -> None:
    assert CheckpointRef.parse("hf://google/gemma-4-E2B-it@abc123") == CheckpointRef(
        scheme="hf", location="google/gemma-4-E2B-it", revision="abc123"
    )
    assert CheckpointRef.parse("s3://bucket/run/ckpt").scheme == "s3"
    assert CheckpointRef.parse("file:///models/x").location == "/models/x"


@pytest.mark.parametrize("uri", ["relative/path", "hf://no-org", "s3://", "s3://bucket"])
def test_bad_checkpoint_uris_are_rejected(uri: str) -> None:
    doc = models_doc()
    doc["target_models"][0] = target_model(checkpoint=uri)
    _error(doc)


def test_sft_data_on_base_is_rejected() -> None:
    doc = models_doc()
    doc["reference_models"][0] = base_model(sft_data={"version": "v1", "path": "/data"})
    assert "upstream checkpoint" in _error(doc)


def test_sft_data_languages_are_normalised_and_deduplicated() -> None:
    doc = models_doc()
    doc["target_models"][0] = target_model(
        sft_data={"version": "v1", "path": "/d", "languages": ["zul_Latn", "swh"]}
    )
    assert _models(doc).get("rung2").sft_data.languages == ["zul", "swa"]
    doc["target_models"][0]["sft_data"]["languages"] = ["swa", "swh"]
    assert "both normalise to 'swa'" in _error(doc)


def test_role_enum_values() -> None:
    assert {r.value for r in Role} == {
        "base",
        "plain_baseline",
        "reference",
        "target",
        "external",
    }


# --- definition of done: deleting plain_baseline needs no other config change -----------------


def test_deleting_plain_baseline_needs_no_other_change(write_configs) -> None:
    run = run_doc()
    with_plain = load_config(write_configs(models=models_doc(plain=True), run=run))
    without = load_config(write_configs(models=models_doc(plain=False), run=run))
    assert with_plain.models.has_plain_baseline
    assert not without.models.has_plain_baseline
    assert [m.id for m in without.selected_models] == ["gemma-base", "rung2"]
