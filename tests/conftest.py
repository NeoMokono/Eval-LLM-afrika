"""Shared builders for minimal, valid config dicts that tests then break one field at a time."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

SHA = "0" * 40


def base_model(**over: Any) -> dict[str, Any]:
    return {"id": "gemma-base", "role": "base", "checkpoint": "hf://google/gemma-4-E2B-it", **over}


def plain_model(**over: Any) -> dict[str, Any]:
    return {
        "id": "plain",
        "role": "plain_baseline",
        "checkpoint": "/models/plain",
        "adapter_of": "gemma-base",
        **over,
    }


def target_model(**over: Any) -> dict[str, Any]:
    return {
        "id": "rung2",
        "role": "target",
        "checkpoint": "/models/rung2",
        "adapter_of": "gemma-base",
        **over,
    }


def models_doc(*, plain: bool = True) -> dict[str, Any]:
    refs = [base_model()] + ([plain_model()] if plain else [])
    return {"reference_models": refs, "target_models": [target_model()]}


def classification_bench(**over: Any) -> dict[str, Any]:
    return {
        "id": "afrixnli",
        "dataset": "masakhane/afrixnli",
        "revision": SHA,
        "type": "classification",
        "capability": "nli",
        "metric": "accuracy",
        "chance_level": 1 / 3,
        "bucket": "held_out",
        "languages": ["zul", "xho"],
        "split": "test",
        "fewshot_split": "validation",
        **over,
    }


def translation_bench(**over: Any) -> dict[str, Any]:
    return {
        "id": "flores_plus",
        "dataset": "openlanguagedata/flores_plus",
        "revision": SHA,
        "type": "generation",
        "capability": "translation",
        "metric": "chrf++",
        "bucket": "seen_task",
        "languages": ["zul"],
        "subsets": {"zul": "zul_Latn"},
        "split": "devtest",
        "fewshot_split": "dev",
        **over,
    }


def benchmarks_doc() -> dict[str, Any]:
    return {"benchmarks": [classification_bench(), translation_bench()]}


def run_doc(**over: Any) -> dict[str, Any]:
    return {"run_id": "test-run", "models": ["rung2"], "benchmarks": ["afrixnli"], **over}


@pytest.fixture
def write_configs(tmp_path: Path):
    """Write the three YAML files (defaults overridable) and return the run.yaml path."""

    def _write(
        *,
        models: dict[str, Any] | None = None,
        benchmarks: dict[str, Any] | None = None,
        run: dict[str, Any] | None = None,
    ) -> Path:
        docs = {
            "models.yaml": models if models is not None else models_doc(),
            "benchmarks.yaml": benchmarks if benchmarks is not None else benchmarks_doc(),
            "run.yaml": run if run is not None else run_doc(),
        }
        for name, doc in docs.items():
            (tmp_path / name).write_text(yaml.safe_dump(copy.deepcopy(doc), sort_keys=False))
        return tmp_path / "run.yaml"

    return _write


# --------------------------------------------------------------------------- Phase 2 helpers


class FakeResolver:
    """Offline stand-in for the HF Hub: maps (repo, ref) to a commit SHA and counts calls."""

    def __init__(self, commits: dict[str, str] | None = None) -> None:
        self.commits = commits or {}
        self.calls: list[tuple[str, str, str]] = []

    def resolve(self, kind: str, repo: str, revision: str) -> str:
        self.calls.append((kind, repo, revision))
        if len(revision) == 40:
            return revision
        return self.commits.get(repo, "a" * 40)


def make_checkpoint(
    path: Path, *, weights: bytes = b"weights-v1", parent: str | None = None
) -> Path:
    """A fake local checkpoint directory: a config, a tokenizer file and one weight file."""
    path.mkdir(parents=True, exist_ok=True)
    config = {"base_model_name_or_path": parent} if parent else {"model_type": "tiny"}
    name = "adapter_config.json" if parent else "config.json"
    (path / name).write_text(json.dumps(config))
    (path / "tokenizer.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


@pytest.fixture
def workspace(tmp_path: Path):
    """A complete on-disk setup (configs + fake checkpoints). `write(...)` rewrites the configs
    with overrides and returns the run.yaml path; results land in tmp_path/results."""
    ckpts = tmp_path / "ckpts"
    paths = {
        "plain": make_checkpoint(ckpts / "plain", parent="google/gemma-4-E2B-it"),
        "rung2": make_checkpoint(ckpts / "rung2", parent="google/gemma-4-E2B-it"),
        "rung3": make_checkpoint(ckpts / "rung3", weights=b"w3", parent="google/gemma-4-E2B-it"),
    }

    class Workspace:
        root = tmp_path
        checkpoints = paths
        results = tmp_path / "results"

        def models(self, *, plain: bool = True, targets: tuple[str, ...] = ("rung2",)) -> dict:
            refs = [base_model()]
            if plain:
                refs.append(plain_model(checkpoint=str(paths["plain"])))
            return {
                "reference_models": refs,
                "target_models": [target_model(id=t, checkpoint=str(paths[t])) for t in targets],
            }

        def write(
            self,
            *,
            models: dict | None = None,
            benchmarks: dict | None = None,
            run: dict | None = None,
        ) -> Path:
            cfg = tmp_path / "configs"
            cfg.mkdir(exist_ok=True)
            docs = {
                "models.yaml": models if models is not None else self.models(),
                "benchmarks.yaml": benchmarks if benchmarks is not None else benchmarks_doc(),
                "run.yaml": run
                if run is not None
                else run_doc(benchmarks=["afrixnli", "flores_plus"]),
            }
            for name, doc in docs.items():
                (cfg / name).write_text(yaml.safe_dump(copy.deepcopy(doc), sort_keys=False))
            return cfg / "run.yaml"

    return Workspace()
