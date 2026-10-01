"""`afrigemma-eval plan`: prints decisions, runs nothing, fails readably."""

import pytest

import cli
from tests.conftest import FakeResolver


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(cli, "_resolver", FakeResolver)


def test_plan_lists_every_unit_and_runs_nothing(workspace, capsys) -> None:
    assert cli.main(["plan", "--config", str(workspace.write())]) == 0
    out = capsys.readouterr().out
    assert out.count("RUN (new)") == 6
    assert "6 to run (6 new, 0 changed, 0 forced), 0 cached. Nothing was run." in out
    assert not (workspace.results / "manifest.json").exists()


def test_plan_with_cli_force(workspace, capsys) -> None:
    args = ["plan", "--config", str(workspace.write())]
    assert cli.main([*args, "--force-rerun", "rung2:afrixnli", "--reason", "sanity check"]) == 0
    assert "RUN (forced: sanity check)" in capsys.readouterr().out


def test_force_without_reason_is_rejected(workspace, capsys) -> None:
    args = ["plan", "--config", str(workspace.write()), "--force-rerun", "rung2:afrixnli"]
    assert cli.main(args) == 1
    assert "needs --reason" in capsys.readouterr().err
    assert cli.main([*args, "--reason", "  "]) == 1


@pytest.mark.parametrize("pair", ["rung2", "nope:afrixnli", "rung2:nope", ":afrixnli"])
def test_bad_force_targets_are_rejected(workspace, capsys, pair: str) -> None:
    args = ["plan", "--config", str(workspace.write()), "--force-rerun", pair, "--reason", "x"]
    assert cli.main(args) == 1
    assert "error:" in capsys.readouterr().err


def test_config_errors_exit_cleanly(tmp_path, capsys) -> None:
    assert cli.main(["plan", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "file not found" in capsys.readouterr().err


def test_report_is_not_available_yet(workspace, capsys) -> None:
    assert cli.main(["report", "--config", str(workspace.write())]) == 2
    assert "Phase 6" in capsys.readouterr().err
