"""Checkpoint identity and fingerprint: what does and does not invalidate the cache."""

import os
from pathlib import Path

import pytest

from eval import fingerprint as fp_module
from eval.fingerprint import (
    FingerprintError,
    HfHubResolver,
    IdentityCache,
    canonical_json,
    changed_fields,
    checkpoint_identity,
    compute_fingerprint,
    effective_shots,
    local_identity,
)
from eval.prompts import PromptTemplate
from eval.registry import BenchmarkSpec, ModelsConfig, Prompting
from tests.conftest import (
    SHA,
    FakeResolver,
    base_model,
    classification_bench,
    make_checkpoint,
    target_model,
    translation_bench,
)

TEMPLATE = PromptTemplate(id="nli-v1", text="Premise: {premise}")


def _fp(bench: dict | None = None, prompting: dict | None = None, **over):
    args = {
        "checkpoint_hash": "c" * 64,
        "benchmark": BenchmarkSpec.model_validate(bench or classification_bench()),
        "revision": SHA,
        "template": TEMPLATE,
        "prompting": Prompting.model_validate(prompting or {}),
        "shots": 0,
        "harness": "0.1.0",
    }
    return compute_fingerprint(**{**args, **over})


# --- the fingerprint ------------------------------------------------------------------------


def test_fingerprint_is_deterministic() -> None:
    assert _fp().value == _fp().value
    assert len(_fp().value) == 64


def test_canonical_json_ignores_key_order() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"checkpoint_hash": "d" * 64}, "checkpoint"),
        ({"revision": "1" * 40}, "benchmark_version"),
        ({"template": PromptTemplate("nli-v1", "Premise:  {premise}")}, "prompt.template_sha256"),
        ({"template": PromptTemplate("nli-v2", TEMPLATE.text)}, "prompt.template_id"),
        ({"shots": 5}, "shots"),
        ({"harness": "0.2.0"}, "harness_version"),
        ({"prompting": Prompting(chat_template=False)}, "prompt.chat_template"),
        ({"prompting": Prompting(mc_scoring="generate")}, "prompt.mc_scoring"),
        (
            {
                "benchmark": BenchmarkSpec.model_validate(
                    classification_bench(split="validation", fewshot_split=None)
                )
            },
            "benchmark.split",
        ),
        (
            {
                "benchmark": BenchmarkSpec.model_validate(
                    classification_bench(languages=["zul", "xho", "sot"])
                )
            },
            "benchmark.subsets",
        ),
    ],
)
def test_each_component_changes_the_fingerprint(change: dict, field: str) -> None:
    before, after = _fp(), _fp(**change)
    assert before.value != after.value
    assert field in changed_fields(before.components, after.components)


def test_only_checkpoint_and_data_version_sit_outside_config_fingerprint() -> None:
    base = _fp()
    assert _fp(checkpoint_hash="d" * 64).config_fingerprint == base.config_fingerprint
    assert _fp(revision="1" * 40).config_fingerprint == base.config_fingerprint
    assert _fp(shots=5).config_fingerprint != base.config_fingerprint


@pytest.mark.parametrize(
    "over", [{"chance_level": 0.5}, {"bucket": "seen_task"}, {"license": "mit"}, {"notes": "x"}]
)
def test_report_only_fields_do_not_change_the_fingerprint(over: dict) -> None:
    assert _fp(classification_bench(**over)).value == _fp().value


def test_decoding_only_matters_when_the_task_generates() -> None:
    longer = {"decoding": {"strategy": "greedy", "max_new_tokens": 999}}
    assert _fp(prompting=longer).value == _fp().value  # log-likelihood scoring: no decoding
    assert _fp(translation_bench(), longer).value != _fp(translation_bench()).value
    generate = {"mc_scoring": "generate"}
    assert _fp(prompting={**generate, **longer}).value != _fp(prompting=generate).value


def test_mc_scoring_does_not_affect_generation_benchmarks() -> None:
    assert (
        _fp(translation_bench(), {"mc_scoring": "generate"}).value == _fp(translation_bench()).value
    )


def test_benchmark_max_new_tokens_overrides_run_default() -> None:
    fp = _fp(translation_bench(max_new_tokens=512))
    assert fp.components["decoding"]["max_new_tokens"] == 512
    assert _fp(translation_bench()).components["decoding"]["max_new_tokens"] == 256


def test_perplexity_runs_once_and_ignores_prompt_settings() -> None:
    ppl = {
        "id": "ppl", "dataset": "Salesforce/wikitext", "revision": SHA, "type": "perplexity",
        "capability": "english_lm", "metric": "perplexity", "bucket": "retention",
        "languages": ["eng"], "split": "test",
    }  # fmt: skip
    spec = BenchmarkSpec.model_validate(ppl)
    assert effective_shots(spec, Prompting(shots=[0, 5], headline_shots=0)) == [0]
    assert _fp(ppl, {"chat_template": False}).value == _fp(ppl).value
    assert _fp(ppl, template=PromptTemplate("x", "y")).value == _fp(ppl).value


# --- local checkpoint identity ----------------------------------------------------------------


def test_local_identity_is_stable(tmp_path: Path) -> None:
    ckpt = make_checkpoint(tmp_path / "m")
    assert local_identity(ckpt) == local_identity(ckpt)


def test_weight_file_change_changes_identity(tmp_path: Path) -> None:
    ckpt = make_checkpoint(tmp_path / "m")
    before = local_identity(ckpt)
    weights = ckpt / "adapter_model.safetensors"
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    touched = local_identity(ckpt)
    assert touched != before
    weights.write_bytes(b"weights-v2-longer")
    assert local_identity(ckpt) not in {before, touched}


def test_config_content_change_changes_identity(tmp_path: Path) -> None:
    ckpt = make_checkpoint(tmp_path / "m")
    before = local_identity(ckpt)
    (ckpt / "tokenizer.json").write_text('{"added": 1}')
    assert local_identity(ckpt) != before


def test_training_artefacts_outside_the_model_dir_are_ignored(tmp_path: Path) -> None:
    ckpt = make_checkpoint(tmp_path / "step_100" / "model")
    before = local_identity(ckpt)
    (ckpt.parent / "losses.json").write_text("{}")
    (ckpt / "optim").mkdir()
    (ckpt / "optim" / "state.pt").write_bytes(b"x")
    (ckpt / ".hidden").write_text("x")
    assert local_identity(ckpt) == before


def test_missing_or_weightless_checkpoint_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FingerprintError, match="does not exist"):
        local_identity(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "config.json").write_text("{}")
    with pytest.raises(FingerprintError, match="no weight files"):
        local_identity(tmp_path / "empty")


def test_identity_cache_avoids_rehashing(tmp_path: Path, monkeypatch) -> None:
    ckpt = make_checkpoint(tmp_path / "m")
    hashed: list[Path] = []
    real = fp_module._sha256_file
    monkeypatch.setattr(fp_module, "_sha256_file", lambda p: hashed.append(p) or real(p))

    first = local_identity(ckpt, IdentityCache(tmp_path / "cache.json"))
    n = len(hashed)
    assert n > 0
    again = local_identity(ckpt, IdentityCache(tmp_path / "cache.json"))  # reloaded from disk
    assert again == first
    assert len(hashed) == n

    (ckpt / "tokenizer.json").write_text('{"changed": true}')
    assert local_identity(ckpt, IdentityCache(tmp_path / "cache.json")) != first
    assert len(hashed) > n


# --- checkpoint identity across schemes ------------------------------------------------------


def _models(tmp_path: Path, **target_over) -> ModelsConfig:
    ckpt = make_checkpoint(tmp_path / "rung2", parent="google/gemma-4-E2B-it")
    return ModelsConfig.model_validate(
        {
            "reference_models": [base_model()],
            "target_models": [target_model(checkpoint=str(ckpt), **target_over)],
        }
    )


def test_hf_checkpoint_resolves_default_branch_to_a_commit(tmp_path: Path) -> None:
    models = _models(tmp_path)
    resolver = FakeResolver({"google/gemma-4-E2B-it": "b" * 40})
    identity = checkpoint_identity(models.base, models, resolver=resolver)
    assert identity.resolved_revision == "b" * 40
    assert resolver.calls == [("model", "google/gemma-4-E2B-it", "main")]


def test_upstream_commit_change_changes_base_and_adapter_identity(tmp_path: Path) -> None:
    models = _models(tmp_path)
    old, new = (FakeResolver({"google/gemma-4-E2B-it": c * 40}) for c in "bc")
    for model_id in ("gemma-base", "rung2"):
        m = models.get(model_id)
        a = checkpoint_identity(m, models, resolver=old)
        b = checkpoint_identity(m, models, resolver=new)
        assert a.hash != b.hash, model_id
    adapter = checkpoint_identity(models.get("rung2"), models, resolver=old)
    assert adapter.parent is not None and adapter.parent.resolved_revision == "b" * 40


def test_training_data_version_is_not_part_of_checkpoint_identity(tmp_path: Path) -> None:
    v1 = _models(tmp_path, sft_data={"version": "v1", "path": "/data"})
    v2 = _models(tmp_path, sft_data={"version": "v2", "path": "/data"})
    resolver = FakeResolver()
    a = checkpoint_identity(v1.get("rung2"), v1, resolver=resolver)
    b = checkpoint_identity(v2.get("rung2"), v2, resolver=resolver)
    assert a.hash == b.hash


def test_pinned_sha_needs_no_network() -> None:
    assert HfHubResolver().resolve("dataset", "any/repo", SHA) == SHA


def test_s3_checkpoints_fail_clearly(tmp_path: Path) -> None:
    models = ModelsConfig.model_validate(
        {
            "reference_models": [base_model()],
            "target_models": [target_model(checkpoint="s3://bucket/ckpt", adapter_of=None)],
        }
    )
    with pytest.raises(FingerprintError, match="not implemented yet"):
        checkpoint_identity(models.get("rung2"), models, resolver=FakeResolver())


def test_limit_is_part_of_the_fingerprint() -> None:
    limited = _fp(limit=50)
    assert limited.value != _fp().value
    assert "limit" in changed_fields(_fp().components, limited.components)
