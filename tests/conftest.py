"""Shared builders for minimal, valid config dicts that tests then break one field at a time."""

from __future__ import annotations

import copy
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
