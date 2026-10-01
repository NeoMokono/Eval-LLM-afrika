"""Invariant 4: training loss never selects a model.

Statically scans everything that can rank or report (eval/, report/, cli.py) for wandb imports,
loss/learning-curve identifiers and the loss-picked checkpoint artefacts training leaves behind
(losses.json, the LOWEST_VAL alias in aliases.json).
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = [ROOT / "eval", ROOT / "report", ROOT / "cli.py"]

FORBIDDEN_MODULES = {"wandb"}
FORBIDDEN = re.compile(
    r"train_loss|eval_loss|val_loss|valid_loss|loss_curve|learning_curve"
    r"|losses\.json|aliases\.json|lowest_val",
    re.IGNORECASE,
)


def violations(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        found += [f"imports {n}" for n in names if n.split(".")[0] in FORBIDDEN_MODULES]

        text = None
        if isinstance(node, ast.Name):
            text = node.id
        elif isinstance(node, ast.Attribute):
            text = node.attr
        elif isinstance(node, ast.arg):
            text = node.arg
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            text = node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        if text and FORBIDDEN.search(text):
            found.append(f"line {getattr(node, 'lineno', '?')}: {text[:60]!r}")
    return found


def _python_files() -> list[Path]:
    files: list[Path] = []
    for p in SCANNED:
        files += [p] if p.is_file() else sorted(p.rglob("*.py"))
    return files


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_training_loss_or_wandb(path: Path) -> None:
    assert violations(path.read_text(encoding="utf-8")) == []


def test_templates_do_not_mention_loss() -> None:
    for path in (ROOT / "report" / "templates").rglob("*"):
        if path.is_file():
            assert not FORBIDDEN.search(path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize(
    "snippet",
    [
        "import wandb",
        "from wandb.sdk import init",
        "best = min(rows, key=lambda r: r.train_loss)",
        "x = record['eval_loss']",
        "def plot_loss_curve(): ...",
        "json.load(open(ckpt / 'losses.json'))",
        "alias = aliases['LOWEST_VAL']",
    ],
)
def test_scanner_catches_violations(snippet: str) -> None:
    assert violations(snippet), snippet


def test_scanner_allows_benchmark_metrics() -> None:
    assert violations("best = max(rows, key=lambda r: r.metric_value)") == []
