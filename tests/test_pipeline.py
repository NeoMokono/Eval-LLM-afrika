"""End to end on the mock backend: cached units are never re-inferred (invariant 2), results
are stored per language (invariant 5) and carry chance level and contamination status
(invariants 6 and 7)."""

import pandas as pd
import pytest

import cli
from eval.backends.mock import MockBackend
from eval.data import InMemoryLoader
from eval.manifest import Manifest
from eval.pipeline import execute, execution_order
from eval.planner import build_plan
from eval.registry import load_config
from eval.storage.results_store import load_predictions, load_unit
from tests import rows
from tests.conftest import FakeResolver, run_doc

BENCHES = ["afrixnli", "flores_plus"]


def _loader() -> InMemoryLoader:
    data = {}
    for lang in ("zul", "xho"):
        data[("afrixnli", lang, "test")] = [rows.xnli(i, label=i % 3) for i in range(9)]
        data[("afrixnli", lang, "validation")] = [rows.xnli(50 + i, label=i % 3) for i in range(12)]
    for subset in ("eng_Latn", "zul_Latn"):
        for split in ("devtest", "dev"):
            data[("flores_plus", subset, split)] = [
                rows.flores(i, f"{subset} {i}") for i in range(6)
            ]
    return InMemoryLoader(data)


class Harness:
    """Runs the pipeline against a workspace, recording every backend it creates."""

    def __init__(self, workspace) -> None:
        self.workspace = workspace
        self.backends: dict[str, list[MockBackend]] = {}

    def run(self, *, fail_on: str | None = None, **write):
        config = load_config(self.workspace.write(**write))
        manifest = Manifest.load(config.results_path / "manifest.json")
        plan = build_plan(config, manifest, resolver=FakeResolver())

        def factory(item):
            def respond(prompt):
                if fail_on == item.model.id:
                    raise RuntimeError("CUDA out of memory")
                return "zul_Latn 0"

            backend = MockBackend(choose=lambda prompt, conts: 0, respond=respond)
            self.backends.setdefault(item.model.id, []).append(backend)
            return backend

        summary = execute(
            config, plan, manifest, backend_factory=factory, loader=_loader(), log=lambda m: None
        )
        return config, plan, summary

    def inferences(self) -> dict[str, int]:
        return {m: sum(b.n_inferences for b in bs) for m, bs in self.backends.items()}


def test_second_run_performs_zero_inferences(workspace) -> None:
    harness = Harness(workspace)
    _, plan, first = harness.run()
    assert len(first.completed) == len(plan.items) == 6
    assert first.n_inferences == 3 * (18 + 6)  # 3 models x (2 langs x 9 NLI + 6 translations)

    harness.backends.clear()
    _, plan, second = harness.run()
    assert plan.to_run == [] and len(plan.cached) == 6
    assert second.n_inferences == 0
    assert harness.backends == {}  # no model was even loaded


def test_adding_a_target_only_infers_that_target(workspace) -> None:
    harness = Harness(workspace)
    harness.run()
    harness.backends.clear()
    _, plan, summary = harness.run(
        models=workspace.models(targets=("rung2", "rung3")),
        run=run_doc(models=["rung2", "rung3"], benchmarks=BENCHES),
    )
    assert harness.inferences() == {"rung3": 24}
    assert {i.model.id for i in summary.completed} == {"rung3"}
    assert {i.model.id for i in plan.cached} == {"gemma-base", "plain", "rung2"}


def test_records_are_per_language_with_chance_ci_and_contamination(workspace) -> None:
    config, _, _ = Harness(workspace).run()
    manifest = Manifest.load(config.results_path / "manifest.json")
    by_unit = {
        (e.model_id, e.benchmark_id): load_unit(config.results_path, e.record_path)
        for e in manifest.entries
    }
    nli = by_unit[("rung2", "afrixnli")]
    assert sorted(r.language for r in nli) == ["xho", "zul"]  # one record per language
    for record in nli:
        assert record.metric_name == "accuracy"
        assert record.metric_value == pytest.approx(1 / 3)  # always answers option 0
        assert record.ci_low <= record.metric_value <= record.ci_high
        assert record.chance_level == pytest.approx(1 / 3)
        assert record.n_examples == 9 and record.shots == 0
        assert record.bucket == "held_out"
        assert record.model_role == "target"
    # Fine-tuned models are unverified until a decontamination check passes; upstream
    # checkpoints have no SFT data of ours to check.
    assert {r.decontamination_status for r in nli} == {"unverified"}
    assert {r.decontamination_status for r in by_unit[("plain", "afrixnli")]} == {"unverified"}
    assert {r.decontamination_status for r in by_unit[("gemma-base", "afrixnli")]} == {
        "not_applicable"
    }
    mt = by_unit[("rung2", "flores_plus")]
    assert [(r.language, r.metric_name, r.chance_level) for r in mt] == [("zul", "chrf++", None)]
    assert 50 < mt[0].metric_value < 100  # one fixed output against six near-identical references
    assert mt[0].bucket == "seen_task"


def test_no_averaged_rows_are_stored(workspace) -> None:
    config, _, _ = Harness(workspace).run()
    for path in config.results_path.rglob("records.parquet"):
        frame = pd.read_parquet(path)
        assert set(frame["language"]) <= {"zul", "xho"}
        assert not frame["language"].isin(["all", "avg", "mean", "macro"]).any()


def test_predictions_and_confusion_matrices_are_written_per_language(workspace) -> None:
    config, _, _ = Harness(workspace).run()
    unit = config.results_path / "runs/test-run/rung2/afrixnli/0shot"
    record = next(
        r
        for r in load_unit(config.results_path, "runs/test-run/rung2/afrixnli/0shot")
        if r.language == "zul"
    )
    assert (
        record.raw_predictions_path == "runs/test-run/rung2/afrixnli/0shot/zul/predictions.parquet"
    )
    predictions = load_predictions(config.results_path, record)
    assert len(predictions) == 9
    assert {"example_id", "gold", "pred", "correct", "scores", "shots"} <= set(predictions.columns)
    table = (unit / "zul" / "confusion.md").read_text()
    assert "| True | 3 | 0 | 0 |" in table and "| False | 3 | 0 | 0 |" in table
    assert not (
        config.results_path / "runs/test-run/rung2/flores_plus/0shot/zul/confusion.md"
    ).exists()


def test_failed_unit_is_recorded_and_the_run_continues(workspace) -> None:
    harness = Harness(workspace)
    config, _, summary = harness.run(fail_on="plain")
    assert [(i.model.id, i.benchmark.id) for i, _ in summary.failed] == [("plain", "flores_plus")]
    assert len(summary.completed) == 5
    manifest = Manifest.load(config.results_path / "manifest.json")
    failed = [e for e in manifest.entries if e.status == "failed"]
    assert len(failed) == 1 and "CUDA out of memory" in failed[0].error

    harness.backends.clear()
    _, plan, retry = harness.run()  # only the failed unit is retried
    assert [(i.model.id, i.benchmark.id) for i in plan.to_run] == [("plain", "flores_plus")]
    assert plan.to_run[0].decision.label == "RUN (new; previous attempt failed)"
    assert harness.inferences() == {"plain": 6} and retry.failed == []


def test_interrupted_unit_stays_running_and_is_rerun(workspace) -> None:
    config = load_config(workspace.write())
    manifest = Manifest.load(config.results_path / "manifest.json")
    plan = build_plan(config, manifest, resolver=FakeResolver())

    def factory(item):
        def choose(prompt, conts):
            raise KeyboardInterrupt

        return MockBackend(choose=choose)

    with pytest.raises(KeyboardInterrupt):
        execute(
            config, plan, manifest, backend_factory=factory, loader=_loader(), log=lambda m: None
        )
    statuses = [e.status for e in Manifest.load(manifest.path).entries]
    assert statuses == ["running"]
    replanned = build_plan(config, Manifest.load(manifest.path), resolver=FakeResolver())
    assert replanned.cached == []


def test_lower_shot_counts_run_first_for_every_model(workspace) -> None:
    config = load_config(
        workspace.write(run=run_doc(benchmarks=BENCHES, prompting={"shots": [0, 5]}))
    )
    plan = build_plan(
        config, Manifest.load(config.results_path / "m.json"), resolver=FakeResolver()
    )
    order = [(i.shots, i.model.id) for i in execution_order(plan)]
    assert order == sorted(
        order, key=lambda t: (t[0], ["gemma-base", "plain", "rung2"].index(t[1]))
    )
    assert order[0] == (0, "gemma-base") and order[-1] == (5, "rung2")


def test_cli_run_then_run_again_is_fully_cached(workspace, capsys, monkeypatch) -> None:
    created: list[MockBackend] = []

    def factory(config):
        def load(item):
            created.append(MockBackend(choose=lambda p, c: 0, respond=lambda p: "x"))
            return created[-1]

        return load

    monkeypatch.setattr(cli, "_resolver", FakeResolver)
    monkeypatch.setattr(cli, "_backend_factory", factory)
    monkeypatch.setattr(cli, "_data_loader", lambda plan: _loader())
    path = str(workspace.write())
    assert cli.main(["run", "--config", path]) == 0
    assert "6 unit(s) completed, 0 failed, 0 served from cache" in capsys.readouterr().out
    assert len(created) == 3

    created.clear()
    assert cli.main(["run", "--config", path]) == 0
    assert "Everything is cached; no inference needed." in capsys.readouterr().out
    assert created == []
    assert cli.main(["plan", "--config", path]) == 0
    assert "0 to run (0 new, 0 changed, 0 forced), 6 cached" in capsys.readouterr().out
