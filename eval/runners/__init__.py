"""Benchmark runners: turn task examples + a model backend into per-example predictions.

`run_unit` is the single entry point for one (model, benchmark, shots) unit; it returns the
predictions per language and performs no scoring.
"""

from __future__ import annotations

from typing import Any

from eval.backends.base import ModelBackend, Prompt
from eval.data import DataLoader
from eval.fewshot import choose_shots, prepare_pool
from eval.registry import BenchmarkSpec, Prompting, TaskType
from eval.runners import classification_runner, generation_runner, perplexity_runner
from eval.tasks import Example, Task, TaskError, get_task

Predictions = dict[str, list[dict[str, Any]]]  # language -> one row per example


def _prompts(
    task: Task,
    benchmark: BenchmarkSpec,
    loader: DataLoader,
    language: str,
    scored: list[Example],
    all_test: list[Example],
    shots: int,
) -> list[Prompt]:
    if shots == 0:
        return [Prompt(e.query) for e in scored]
    if benchmark.fewshot_split is None:
        if task.shot_policy != "other_group":
            raise TaskError(
                f"{benchmark.id}: no fewshot_split, and its task cannot safely take shots from "
                "the scored split"
            )
        source = all_test
    else:
        source = task.examples(loader, benchmark, language, benchmark.fewshot_split)
    pool = prepare_pool(source, task.shot_policy)
    return [Prompt(e.query, choose_shots(e, pool, shots, task.shot_policy)) for e in scored]


def run_unit(
    benchmark: BenchmarkSpec,
    *,
    backend: ModelBackend,
    loader: DataLoader,
    prompting: Prompting,
    shots: int,
    limit: int | None = None,
) -> Predictions:
    task = get_task(benchmark.id)
    decoding = prompting.decoding.model_copy(
        update={"max_new_tokens": benchmark.max_new_tokens or prompting.decoding.max_new_tokens}
    )
    predictions: Predictions = {}
    for language in benchmark.languages:
        all_test = task.examples(loader, benchmark, language, benchmark.split)
        scored = all_test[:limit] if limit else all_test
        if benchmark.type is TaskType.PERPLEXITY:
            rows = perplexity_runner.predict(scored, backend)
        else:
            prompts = _prompts(task, benchmark, loader, language, scored, all_test, shots)
            if benchmark.type is TaskType.GENERATION:
                rows = generation_runner.predict(scored, prompts, backend, decoding)
            elif prompting.mc_scoring == "generate":
                rows = classification_runner.predict_generate(scored, prompts, backend, decoding)
            else:
                rows = classification_runner.predict_loglikelihood(scored, prompts, backend)
        for row in rows:
            row["shots"] = shots
        predictions[language] = rows
    return predictions
