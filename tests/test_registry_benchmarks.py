"""benchmarks.yaml schema: chance levels (invariant 6), languages, pinned versions."""

import pytest
from pydantic import ValidationError

from eval.registry import BenchmarksConfig, BenchmarkSpec
from tests.conftest import SHA, classification_bench, translation_bench


def _error(**over) -> str:
    with pytest.raises(ValidationError) as exc:
        BenchmarkSpec.model_validate(classification_bench(**over))
    return str(exc.value)


# --- invariant 6: chance level travels with every score ------------------------------------


def test_classification_requires_chance_level() -> None:
    assert "requires chance_level" in _error(chance_level=None)


@pytest.mark.parametrize("chance", [1.0, 1.5, -0.1])
def test_chance_level_range(chance: float) -> None:
    assert "chance_level must be in [0, 1)" in _error(chance_level=chance)


def test_exact_match_requires_chance_level_and_allows_zero() -> None:
    spec = classification_bench(type="generation", metric="exact_match", chance_level=0.0)
    assert BenchmarkSpec.model_validate(spec).chance_level == 0.0
    spec["chance_level"] = None
    with pytest.raises(ValidationError, match="requires chance_level"):
        BenchmarkSpec.model_validate(spec)


def test_chrf_rejects_a_chance_level() -> None:
    with pytest.raises(ValidationError, match="has no chance level"):
        BenchmarkSpec.model_validate(translation_bench(chance_level=0.1))


def test_metric_must_fit_type() -> None:
    assert "classification benchmarks use metric accuracy" in _error(metric="chrf++")


def test_extra_metrics_only_for_classification() -> None:
    with pytest.raises(ValidationError, match="only apply to classification"):
        BenchmarkSpec.model_validate(translation_bench(extra_metrics=["macro_f1"]))


# --- languages ---------------------------------------------------------------------------


def test_languages_are_normalised() -> None:
    spec = BenchmarkSpec.model_validate(classification_bench(languages=["zul_Latn", "swh"]))
    assert spec.languages == ["zul", "swa"]


def test_same_language_twice_is_rejected() -> None:
    assert "both normalise to 'swa'" in _error(languages=["swa", "swh_Latn"])


def test_unknown_language_is_rejected() -> None:
    assert "not in eval/languages.py" in _error(languages=["zul", "kik"])


def test_subsets_resolve_dataset_names() -> None:
    spec = BenchmarkSpec.model_validate(
        classification_bench(languages=["swa", "zul"], subsets={"swh": "swh_Latn"})
    )
    assert spec.subset_for("swa") == "swh_Latn"
    assert spec.subset_for("zul") == "zul"
    with pytest.raises(KeyError):
        spec.subset_for("xho")


def test_subset_for_unlisted_language_is_rejected() -> None:
    assert "not in `languages`" in _error(subsets={"sot": "sot_Latn"})


def test_subset_pointing_at_another_language_is_rejected() -> None:
    assert "is 'xho', but is listed under 'zul'" in _error(subsets={"zul": "xho_Latn"})


def test_two_languages_sharing_a_subset_is_rejected() -> None:
    assert "same subset" in _error(subsets={"zul": "all", "xho": "all"})


def test_language_column_needs_config() -> None:
    assert "set config too" in _error(language_column="lang_pair")


# --- versions and splits -----------------------------------------------------------------


def test_version_shorthand_from_spec_is_accepted() -> None:
    spec = classification_bench()
    del spec["dataset"], spec["revision"]
    spec["version"] = f"masakhane/afrixnli@{SHA}"
    parsed = BenchmarkSpec.model_validate(spec)
    assert (parsed.dataset, parsed.revision) == ("masakhane/afrixnli", SHA)
    assert parsed.version == f"masakhane/afrixnli@{SHA}"


def test_hf_dataset_needs_a_revision() -> None:
    assert "need a revision" in _error(revision=None)


def test_local_dataset_needs_absolute_path() -> None:
    assert "absolute path" in _error(dataset="local:relative.jsonl", revision=None)


def test_fewshot_split_cannot_be_the_scored_split() -> None:
    assert "no test leakage" in _error(fewshot_split="test")


# --- registry-level checks ----------------------------------------------------------------


def test_duplicate_benchmark_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate benchmark ids"):
        BenchmarksConfig.model_validate(
            {"benchmarks": [classification_bench(), classification_bench()]}
        )


def test_capability_cannot_mix_metrics() -> None:
    mixed = translation_bench(id="other", capability="nli")
    with pytest.raises(ValidationError, match="mixes metrics"):
        BenchmarksConfig.model_validate({"benchmarks": [classification_bench(), mixed]})
