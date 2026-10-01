"""Rerun decisions (invariants 2 and 3): the four plan outcomes and nothing else."""

import os

import pytest

from eval import planner
from eval.manifest import Manifest
from eval.planner import DecisionKind, Plan, build_plan, format_plan
from eval.registry import ForceRerun, load_config
from tests.conftest import FakeResolver, benchmarks_doc, run_doc, translation_bench

BENCHES = ["afrixnli", "flores_plus"]


def _plan(workspace, resolver=None, cli_force=(), **write) -> Plan:
    config = load_config(workspace.write(**write))
    manifest = Manifest.load(workspace.results / "manifest.json")
    return build_plan(config, manifest, resolver=resolver or FakeResolver(), cli_force=cli_force)


def _execute(workspace, plan: Plan, *, finish: str = "complete") -> int:
    """Stand-in for the runner: mark every planned unit as started, then finished."""
    manifest = Manifest.load(workspace.results / "manifest.json")
    for item in plan.to_run:
        manifest.mark_running(
            model_id=item.model.id,
            benchmark_id=item.benchmark.id,
            shots=item.shots,
            fingerprint=item.fingerprint,
            run_id=plan.run_id,
            harness_version=plan.harness_version,
            rerun_reason=item.rerun_reason,
        )
        if finish == "complete":
            manifest.mark_complete(item.key, f"runs/{item.model.id}/{item.benchmark.id}")
        elif finish == "failed":
            manifest.mark_failed(item.key, "boom")
    return len(plan.to_run)


def _labels(plan: Plan) -> dict[tuple[str, str, int], str]:
    return {(i.model.id, i.benchmark.id, i.shots): i.decision.label for i in plan.items}


# --- outcome 1: RUN (new) ----------------------------------------------------------------


def test_fresh_repo_plans_everything_as_new(workspace) -> None:
    plan = _plan(workspace)
    assert len(plan.items) == 3 * 2  # base, plain, rung2 x two benchmarks
    assert {i.decision.label for i in plan.items} == {"RUN (new)"}


# --- outcome 2: CACHED (invariant 2) ------------------------------------------------------


def test_after_a_run_everything_is_cached(workspace) -> None:
    _execute(workspace, _plan(workspace))
    plan = _plan(workspace)
    assert {i.decision.label for i in plan.items} == {"CACHED"}
    assert plan.to_run == []
    assert all(i.cached is not None and i.cached.record_path for i in plan.items)


def test_adding_a_target_only_runs_that_target(workspace) -> None:
    _execute(workspace, _plan(workspace))
    plan = _plan(
        workspace,
        models=workspace.models(targets=("rung2", "rung3")),
        run=run_doc(models=["rung2", "rung3"], benchmarks=BENCHES),
    )
    assert {i.model.id for i in plan.to_run} == {"rung3"}
    assert {i.model.id for i in plan.cached} == {"gemma-base", "plain", "rung2"}


def test_crashed_run_is_not_a_cache_hit(workspace) -> None:
    _execute(workspace, _plan(workspace), finish="crash")  # entries left as `running`
    plan = _plan(workspace)
    assert plan.cached == []
    assert {i.decision.label for i in plan.items} == {"RUN (new; previous attempt running)"}


def test_failed_run_is_not_a_cache_hit(workspace) -> None:
    _execute(workspace, _plan(workspace), finish="failed")
    plan = _plan(workspace)
    assert {i.decision.label for i in plan.items} == {"RUN (new; previous attempt failed)"}


# --- outcome 3: RUN (fingerprint changed: <field>) -----------------------------------------


def test_harness_version_change_reruns_everything(workspace, monkeypatch) -> None:
    _execute(workspace, _plan(workspace))
    monkeypatch.setattr(planner, "harness_version", lambda: "9.9.9")
    plan = _plan(workspace)
    assert {i.decision.label for i in plan.items} == {"RUN (fingerprint changed: harness_version)"}


def test_decoding_change_only_reruns_generation(workspace) -> None:
    _execute(workspace, _plan(workspace))
    run = run_doc(
        benchmarks=BENCHES,
        prompting={"decoding": {"strategy": "greedy", "max_new_tokens": 64}},
    )
    labels = _labels(_plan(workspace, run=run))
    assert labels[("rung2", "afrixnli", 0)] == "CACHED"
    assert (
        labels[("rung2", "flores_plus", 0)] == "RUN (fingerprint changed: decoding.max_new_tokens)"
    )


def test_benchmark_revision_change_reruns_that_benchmark(workspace) -> None:
    _execute(workspace, _plan(workspace))
    benches = benchmarks_doc()
    benches["benchmarks"][1] = translation_bench(revision="1" * 40)
    labels = _labels(_plan(workspace, benchmarks=benches))
    assert labels[("gemma-base", "afrixnli", 0)] == "CACHED"
    assert (
        labels[("gemma-base", "flores_plus", 0)] == "RUN (fingerprint changed: benchmark_version)"
    )


def test_modified_checkpoint_reruns_only_that_model(workspace) -> None:
    _execute(workspace, _plan(workspace))
    weights = workspace.checkpoints["rung2"] / "adapter_model.safetensors"
    weights.write_bytes(b"retrained weights")
    plan = _plan(workspace)
    assert {i.model.id for i in plan.to_run} == {"rung2"}
    assert {i.decision.label for i in plan.to_run} == {"RUN (fingerprint changed: checkpoint)"}


def test_upstream_base_update_reruns_base_and_its_adapters(workspace) -> None:
    _execute(workspace, _plan(workspace))
    plan = _plan(workspace, resolver=FakeResolver({"google/gemma-4-E2B-it": "f" * 40}))
    assert plan.cached == []
    assert {i.decision.label for i in plan.items} == {"RUN (fingerprint changed: checkpoint)"}


def test_adding_shots_keeps_existing_shot_results(workspace) -> None:
    _execute(workspace, _plan(workspace))
    run = run_doc(benchmarks=BENCHES, prompting={"shots": [0, 5]})
    labels = _labels(_plan(workspace, run=run))
    assert labels[("rung2", "afrixnli", 0)] == "CACHED"
    assert labels[("rung2", "afrixnli", 5)] == "RUN (new)"


def test_reverting_a_change_is_served_from_cache_again(workspace) -> None:
    _execute(workspace, _plan(workspace))
    changed = run_doc(benchmarks=BENCHES, prompting={"chat_template": False})
    _execute(workspace, _plan(workspace, run=changed))
    assert _plan(workspace).to_run == []  # the original config's results are still indexed


def test_sft_data_version_change_alone_does_not_rerun(workspace) -> None:
    _execute(workspace, _plan(workspace))
    models = workspace.models()
    models["target_models"][0]["sft_data"] = {"version": "v2", "path": "/data/v2"}
    assert _plan(workspace, models=models).to_run == []


# --- outcome 4: RUN (forced: <reason>) (invariant 3) ----------------------------------------


def test_force_rerun_in_run_yaml(workspace) -> None:
    _execute(workspace, _plan(workspace))
    force = [{"model": "rung2", "benchmark": "afrixnli", "reason": "re-ran decontamination"}]
    plan = _plan(workspace, run=run_doc(benchmarks=BENCHES, force_rerun=force))
    labels = _labels(plan)
    assert labels.pop(("rung2", "afrixnli", 0)) == "RUN (forced: re-ran decontamination)"
    assert set(labels.values()) == {"CACHED"}


def test_force_rerun_all(workspace) -> None:
    _execute(workspace, _plan(workspace))
    plan = _plan(workspace, run=run_doc(benchmarks=BENCHES, force_rerun_all=True))
    assert {i.decision.label for i in plan.items} == {"RUN (forced: force_rerun_all)"}


def test_cli_force_overrides_run_yaml_reason(workspace) -> None:
    _execute(workspace, _plan(workspace))
    force = [{"model": "rung2", "benchmark": "afrixnli", "reason": "from yaml"}]
    cli = [ForceRerun(model="rung2", benchmark="afrixnli", reason="from cli")]
    plan = _plan(workspace, cli_force=cli, run=run_doc(benchmarks=BENCHES, force_rerun=force))
    assert _labels(plan)[("rung2", "afrixnli", 0)] == "RUN (forced: from cli)"


def test_forced_reason_reaches_the_manifest(workspace) -> None:
    _execute(workspace, _plan(workspace))
    force = [{"model": "rung2", "benchmark": "afrixnli", "reason": "double-check"}]
    plan = _plan(workspace, run=run_doc(benchmarks=BENCHES, force_rerun=force))
    assert _execute(workspace, plan) == 1
    entries = Manifest.load(workspace.results / "manifest.json").entries
    assert [e.rerun_reason for e in entries if e.rerun_reason] == ["double-check"]
    assert _plan(workspace).to_run == []  # without the override it is cached again


# --- the four outcomes are the only ones --------------------------------------------------


def test_decision_kinds_are_exactly_the_four_in_the_spec() -> None:
    assert {k.value for k in DecisionKind} == {"cached", "new", "changed", "forced"}


@pytest.mark.parametrize("plain", [True, False])
def test_plan_output_states_plain_baseline_status(workspace, plain: bool) -> None:
    text = format_plan(_plan(workspace, models=workspace.models(plain=plain)))
    assert ("plain_baseline: present" in text) is plain
    assert ("plain_baseline: none" in text) is not plain
    assert "Nothing was run." in text


def test_planning_does_not_touch_the_manifest(workspace) -> None:
    _plan(workspace)
    assert not (workspace.results / "manifest.json").exists()
    assert not os.path.exists(workspace.results / "manifest.json.lock")
