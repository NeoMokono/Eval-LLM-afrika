"""Run the units a plan marked as RUN: infer, score per language, store, update the manifest.

Units are ordered by shot count, then model, so the headline (lowest shot count) numbers for
every model exist before the more expensive few-shot passes start. Each model is loaded once
per consecutive group. A unit that raises is marked `failed` and the run moves on; cached
units are never touched (invariant 2).
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import groupby

from eval.backends.base import ModelBackend
from eval.data import DataLoader
from eval.manifest import Manifest
from eval.planner import Plan, PlanItem
from eval.registry import EvalConfig, Role, TaskType
from eval.runners import Predictions, run_unit
from eval.scoring import confusion, score_language
from eval.storage.results_store import ResultRecord, predictions_path, unit_path, write_unit
from eval.tasks import get_task

BackendFactory = Callable[[PlanItem], ModelBackend]


@dataclass
class RunSummary:
    completed: list[PlanItem] = field(default_factory=list)
    failed: list[tuple[PlanItem, str]] = field(default_factory=list)
    n_inferences: int = 0


def execution_order(plan: Plan) -> list[PlanItem]:
    model_order = {m: i for i, m in enumerate(dict.fromkeys(i.model.id for i in plan.items))}
    return sorted(plan.to_run, key=lambda i: (i.shots, model_order[i.model.id]))


def build_records(
    config: EvalConfig, plan: Plan, item: PlanItem, unit: str, predictions: Predictions
) -> tuple[list[ResultRecord], dict[str, str]]:
    bench = item.benchmark
    sft = item.model.sft_data
    fine_tuned = item.model.role in {Role.TARGET, Role.PLAIN_BASELINE}
    completed_at = datetime.now(UTC).isoformat(timespec="seconds")
    records: list[ResultRecord] = []
    confusions: dict[str, str] = {}
    for language, rows in predictions.items():
        score = score_language(bench, rows, config.run.bootstrap)
        records.append(
            ResultRecord(
                run_id=plan.run_id,
                model_id=item.model.id,
                model_role=item.model.role.value,
                checkpoint_hash=item.fingerprint.checkpoint_hash,
                benchmark_id=bench.id,
                benchmark_version=item.fingerprint.benchmark_version,
                capability=bench.capability,
                language=language,
                shots=item.shots,
                config_fingerprint=item.fingerprint.config_fingerprint,
                bucket=bench.bucket.value,
                # Upstream models have no SFT data of ours to check. Our fine-tuned models
                # stay unverified until the overlap check passes for their training-data
                # version, including when no training data is declared at all.
                decontamination_status="unverified" if fine_tuned else "not_applicable",
                sft_data_version=sft.version if sft else None,
                metric_name=bench.metric,
                metric_value=score.primary.value,
                ci_low=score.primary.ci_low,
                ci_high=score.primary.ci_high,
                n_examples=score.n_examples,
                n_unparseable=score.n_unparseable,
                chance_level=bench.chance_level,
                extra_metrics={
                    name: {"value": iv.value, "ci_low": iv.ci_low, "ci_high": iv.ci_high}
                    for name, iv in score.extra.items()
                },
                seed=config.run.bootstrap.seed,
                limit=config.run.limit,
                raw_predictions_path=predictions_path(unit, language),
                completed_at=completed_at,
                harness_version=plan.harness_version,
            )
        )
        if bench.type is TaskType.CLASSIFICATION:
            labels = get_task(bench.id).template.options
            matrix = confusion.confusion(
                [r["gold"] for r in rows], [r["pred"] for r in rows], labels
            )
            confusions[language] = confusion.to_markdown(matrix, labels)
    return records, confusions


def execute(
    config: EvalConfig,
    plan: Plan,
    manifest: Manifest,
    *,
    backend_factory: BackendFactory,
    loader: DataLoader,
    log: Callable[[str], None] = print,
) -> RunSummary:
    summary = RunSummary()
    todo = execution_order(plan)
    done = 0
    for (shots, model_id), group in groupby(todo, key=lambda i: (i.shots, i.model.id)):
        units = list(group)
        log(f"[{model_id}] loading model for {len(units)} unit(s) at {shots}-shot")
        backend = backend_factory(units[0])
        try:
            for item in units:
                done += 1
                tag = f"[{done}/{len(todo)}] {model_id} / {item.benchmark.id} / {shots}-shot"
                manifest.mark_running(
                    model_id=model_id,
                    benchmark_id=item.benchmark.id,
                    shots=shots,
                    fingerprint=item.fingerprint,
                    run_id=plan.run_id,
                    harness_version=plan.harness_version,
                    rerun_reason=item.rerun_reason,
                )
                started = time.monotonic()
                try:
                    predictions = run_unit(
                        item.benchmark,
                        backend=backend,
                        loader=loader,
                        prompting=config.run.prompting,
                        shots=shots,
                        limit=config.run.limit,
                    )
                    unit = unit_path(plan.run_id, model_id, item.benchmark.id, shots)
                    records, confusions = build_records(config, plan, item, unit, predictions)
                    write_unit(config.results_path, unit, records, predictions, confusions)
                    manifest.mark_complete(item.key, unit)
                except Exception as exc:
                    manifest.mark_failed(item.key, f"{type(exc).__name__}: {exc}")
                    summary.failed.append((item, traceback.format_exc()))
                    log(f"{tag}: FAILED after {time.monotonic() - started:.0f}s: {exc}")
                    continue
                summary.completed.append(item)
                n = sum(len(rows) for rows in predictions.values())
                log(
                    f"{tag}: done in {time.monotonic() - started:.0f}s "
                    f"({len(predictions)} languages, {n} examples)"
                )
        finally:
            summary.n_inferences += backend.n_inferences
            backend.close()
    return summary
