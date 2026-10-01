"""run.yaml schema, cross-file validation and readable loading errors."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.registry import ConfigError, Decoding, Prompting, RunConfig, load_config
from tests.conftest import benchmarks_doc, classification_bench, models_doc, run_doc

# --- invariant 3 (schema part): a forced rerun needs a reason ------------------------------


@pytest.mark.parametrize("reason", ["", "   "])
def test_force_rerun_without_reason_is_rejected(reason: str) -> None:
    doc = run_doc(force_rerun=[{"model": "rung2", "benchmark": "afrixnli", "reason": reason}])
    with pytest.raises(ValidationError, match="needs a non-empty reason"):
        RunConfig.model_validate(doc)


def test_force_rerun_missing_reason_key_is_rejected() -> None:
    doc = run_doc(force_rerun=[{"model": "rung2", "benchmark": "afrixnli"}])
    with pytest.raises(ValidationError, match="reason"):
        RunConfig.model_validate(doc)


def test_force_rerun_reason_is_kept() -> None:
    doc = run_doc(force_rerun=[{"model": "rung2", "benchmark": "afrixnli", "reason": " redo "}])
    assert RunConfig.model_validate(doc).force_rerun[0].reason == "redo"


def test_force_rerun_duplicates_are_rejected() -> None:
    entry = {"model": "rung2", "benchmark": "afrixnli", "reason": "x"}
    with pytest.raises(ValidationError, match="more than once"):
        RunConfig.model_validate(run_doc(force_rerun=[entry, entry]))


# --- run.yaml fields ------------------------------------------------------------------------


def test_harness_version_cannot_be_typed_by_hand() -> None:
    with pytest.raises(ValidationError, match="read from the installed"):
        RunConfig.model_validate(run_doc(harness_version="0.3.1"))


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunConfig.model_validate(run_doc(force_rerun_al=True))


def test_decoding_shorthand_from_spec() -> None:
    assert Prompting.model_validate({"decoding": "greedy"}).decoding == Decoding()


def test_greedy_rejects_sampling_params() -> None:
    with pytest.raises(ValidationError, match="greedy decoding takes no"):
        Decoding.model_validate({"strategy": "greedy", "temperature": 0.7})


def test_sampling_needs_temperature() -> None:
    with pytest.raises(ValidationError, match="needs a temperature"):
        Decoding.model_validate({"strategy": "sample"})


def test_headline_shots_must_be_run() -> None:
    with pytest.raises(ValidationError, match="headline_shots=5"):
        Prompting.model_validate({"shots": [0], "headline_shots": 5})


def test_threshold_defaults() -> None:
    t = RunConfig.model_validate(run_doc()).thresholds
    assert (t.noise_threshold, t.retention_tolerance, t.near_chance_margin) == (1.0, None, 2.0)
    assert t.perplexity_tolerance_pct is None


# --- cross-file validation ---------------------------------------------------------------


def test_reference_models_are_always_selected(write_configs) -> None:
    cfg = load_config(write_configs())
    assert [m.id for m in cfg.selected_models] == ["gemma-base", "plain", "rung2"]


def test_listing_a_reference_model_does_not_duplicate_it(write_configs) -> None:
    cfg = load_config(write_configs(run=run_doc(models=["gemma-base", "rung2"])))
    assert [m.id for m in cfg.selected_models] == ["gemma-base", "plain", "rung2"]


def test_unknown_model_is_reported_with_known_ids(write_configs) -> None:
    with pytest.raises(ConfigError, match=r"\['nope'\] are not in models.yaml.*known: "):
        load_config(write_configs(run=run_doc(models=["nope"])))


def test_unknown_benchmark_is_rejected(write_configs) -> None:
    with pytest.raises(ConfigError, match=r"not in benchmarks\.yaml"):
        load_config(write_configs(run=run_doc(benchmarks=["belebele"])))


def test_disabled_benchmark_cannot_be_selected(write_configs) -> None:
    benches = {"benchmarks": [classification_bench(enabled=False)]}
    with pytest.raises(ConfigError, match="disabled benchmarks"):
        load_config(write_configs(benchmarks=benches))


def test_force_rerun_must_target_this_run(write_configs) -> None:
    run = run_doc(force_rerun=[{"model": "rung2", "benchmark": "flores_plus", "reason": "x"}])
    with pytest.raises(ConfigError, match="not part of this run"):
        load_config(write_configs(run=run))


def test_force_rerun_of_reference_model_is_allowed(write_configs) -> None:
    run = run_doc(force_rerun=[{"model": "gemma-base", "benchmark": "afrixnli", "reason": "x"}])
    assert load_config(write_configs(run=run)).run.force_rerun[0].model == "gemma-base"


# --- readable loading errors ----------------------------------------------------------------


def test_errors_name_the_file_and_field(write_configs) -> None:
    models = models_doc()
    models["target_models"][0]["checkpoint"] = "relative/path"
    with pytest.raises(ConfigError) as exc:
        load_config(write_configs(models=models))
    message = str(exc.value)
    assert "models.yaml" in message
    assert "target_models.0.checkpoint" in message
    assert "absolute paths" in message


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_config(tmp_path / "run.yaml")


def test_duplicate_yaml_keys_are_rejected(write_configs) -> None:
    path = write_configs()
    path.write_text(path.read_text() + "run_id: again\n")
    with pytest.raises(ConfigError, match="duplicate keys"):
        load_config(path)


def test_yaml_merge_keys_are_supported(write_configs, tmp_path: Path) -> None:
    path = write_configs()
    (tmp_path / "models.yaml").write_text(
        """
reference_models:
  - {id: gemma-base, role: base, checkpoint: "hf://google/gemma-4-E2B-it"}
target_models:
  - &t {id: rung2, role: target, checkpoint: /m/a, adapter_of: gemma-base}
  - <<: *t
    id: rung3
    checkpoint: /m/b
"""
    )
    cfg = load_config(path)
    assert cfg.models.get("rung3").adapter_of == "gemma-base"


def test_shared_benchmarks_doc_fixture_is_valid(write_configs) -> None:
    assert len(load_config(write_configs(benchmarks=benchmarks_doc())).benchmarks.benchmarks) == 2
