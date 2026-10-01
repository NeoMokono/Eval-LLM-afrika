"""Parquet results store: one aggregate record per (model, benchmark, language, shots), plus
per-example predictions.

Layout under results_dir (model and shot count are part of the path so units never collide):

    runs/<run_id>/<model_id>/<benchmark_id>/<N>shot/records.parquet
    runs/<run_id>/<model_id>/<benchmark_id>/<N>shot/<language>/predictions.parquet
    runs/<run_id>/<model_id>/<benchmark_id>/<N>shot/<language>/confusion.md

Scores are stored per language and never pre-averaged (invariant 5); anything about training
(loss curves, GPU hours) stays out of this store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

DecontaminationStatus = Literal["verified", "unverified", "not_applicable"]


class ResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    model_id: str
    model_role: str
    checkpoint_hash: str
    benchmark_id: str
    benchmark_version: str
    capability: str
    language: str
    shots: int
    config_fingerprint: str
    bucket: str
    decontamination_status: DecontaminationStatus
    sft_data_version: str | None
    metric_name: str
    metric_value: float
    ci_low: float
    ci_high: float
    n_examples: int
    n_unparseable: int
    chance_level: float | None
    extra_metrics: dict[str, dict[str, float]]
    seed: int
    limit: int | None
    raw_predictions_path: str
    completed_at: str
    harness_version: str


def unit_path(run_id: str, model_id: str, benchmark_id: str, shots: int) -> str:
    """The unit's directory, relative to results_dir (this is the manifest's record_path)."""
    return f"runs/{run_id}/{model_id}/{benchmark_id}/{shots}shot"


def predictions_path(unit: str, language: str) -> str:
    return f"{unit}/{language}/predictions.parquet"


def write_unit(
    results_dir: Path,
    unit: str,
    records: list[ResultRecord],
    predictions: dict[str, list[dict[str, Any]]],
    confusions: dict[str, str] | None = None,
) -> None:
    root = results_dir / unit
    for language, rows in predictions.items():
        target = results_dir / predictions_path(unit, language)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(target, index=False)
    for language, markdown in (confusions or {}).items():
        (root / language / "confusion.md").write_text(markdown + "\n", encoding="utf-8")
    frame = pd.DataFrame([r.model_dump() for r in records])
    frame["extra_metrics"] = frame["extra_metrics"].map(json.dumps)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "records.parquet", index=False)


def load_unit(results_dir: Path, unit: str) -> list[ResultRecord]:
    frame = pd.read_parquet(results_dir / unit / "records.parquet")
    records = []
    for row in frame.to_dict(orient="records"):
        row["extra_metrics"] = json.loads(row["extra_metrics"])
        for key in ("sft_data_version", "chance_level", "limit"):
            if pd.isna(row[key]):
                row[key] = None
        if row["limit"] is not None:
            row["limit"] = int(row["limit"])
        records.append(ResultRecord.model_validate(row))
    return records


def load_predictions(results_dir: Path, record: ResultRecord) -> pd.DataFrame:
    return pd.read_parquet(results_dir / record.raw_predictions_path)
