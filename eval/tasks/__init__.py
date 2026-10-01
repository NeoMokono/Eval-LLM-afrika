"""Benchmark tasks: turn verified dataset rows into prompts, answer options and gold answers.

Column names and label encodings below were checked against each dataset (datasets-server /
`datasets`) on 2026-10-01 at the revisions pinned in configs/benchmarks.yaml.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Literal

from eval.data import DataLoader, Row
from eval.languages import prompt_name
from eval.prompts import INJONGO_INTENTS, LETTERS, SIB_CATEGORIES, TEMPLATES, PromptTemplate
from eval.registry import BenchmarkSpec

ShotPolicy = Literal["fixed", "same_group", "other_group"]


class TaskError(Exception):
    pass


@dataclass(frozen=True)
class Example:
    example_id: str
    language: str
    query: str  # the rendered user prompt
    target: str  # gold answer as text: the few-shot answer and, for generation, the reference
    options: tuple[str, ...] = ()  # continuations scored for multiple choice
    gold: int | None = None  # index into options
    group: str | None = None  # subject / passage, used to pick or exclude few-shot examples


class Task:
    """One benchmark. Subclasses define how a dataset row becomes an Example."""

    shot_policy: ShotPolicy = "fixed"

    def __init__(self, benchmark_id: str) -> None:
        self.benchmark_id = benchmark_id
        self.template: PromptTemplate = TEMPLATES[benchmark_id]

    def load(
        self, loader: DataLoader, benchmark: BenchmarkSpec, language: str, split: str
    ) -> list[Row]:
        return loader.rows(benchmark, benchmark.subset_for(language), split)

    def example(self, row: Row, index: int, language: str) -> Example:
        raise NotImplementedError

    def examples(
        self, loader: DataLoader, benchmark: BenchmarkSpec, language: str, split: str
    ) -> list[Example]:
        rows = self.load(loader, benchmark, language, split)
        if not rows:
            raise TaskError(f"{benchmark.id}: no rows for {language} in split {split!r}")
        return [self.example(row, i, language) for i, row in enumerate(rows)]

    def _choice(
        self, example_id: str, language: str, query: str, gold: int, group: str | None = None
    ) -> Example:
        options = self.template.options
        return Example(example_id, language, query, options[gold], options, gold, group)


class AfriXNLI(Task):
    def example(self, row: Row, index: int, language: str) -> Example:
        query = self.template.format(premise=row["premise"], hypothesis=row["hypothesis"])
        return self._choice(str(index), language, query, int(row["label"]))


class MMLU(Task):
    """AfriMMLU (choices: stringified list, answer: letter) and cais/mmlu (list, class index)."""

    shot_policy: ShotPolicy = "same_group"

    def example(self, row: Row, index: int, language: str) -> Example:
        choices = row["choices"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        if len(choices) != 4:
            raise TaskError(f"{self.benchmark_id} row {index}: expected 4 choices, got {choices!r}")
        answer = row["answer"]
        gold = LETTERS.index(answer) if isinstance(answer, str) else int(answer)
        a, b, c, d = (str(x) for x in choices)
        query = self.template.format(question=str(row["question"]).strip(), a=a, b=b, c=c, d=d)
        return self._choice(str(index), language, query, gold, group=row["subject"])


class Belebele(Task):
    # The dataset has only a test split, so shots come from test questions about a *different*
    # passage (several questions share one passage).
    shot_policy: ShotPolicy = "other_group"

    def example(self, row: Row, index: int, language: str) -> Example:
        query = self.template.format(
            passage=row["flores_passage"],
            question=str(row["question"]).strip(),
            a=row["mc_answer1"],
            b=row["mc_answer2"],
            c=row["mc_answer3"],
            d=row["mc_answer4"],
        )
        example_id = f"{row['link']}#{row['question_number']}"
        return self._choice(
            example_id, language, query, int(row["correct_answer_num"]) - 1, row["link"]
        )


class SIB200(Task):
    def example(self, row: Row, index: int, language: str) -> Example:
        gold = SIB_CATEGORIES.index(row["category"])
        return self._choice(
            str(row["index_id"]), language, self.template.format(text=row["text"]), gold
        )


class InjongoIntent(Task):
    def example(self, row: Row, index: int, language: str) -> Example:
        gold = INJONGO_INTENTS.index(row["intent"])
        example_id = str(row.get("example_id") or index)
        return self._choice(example_id, language, self.template.format(text=row["text"]), gold)


class InjongoSlots(Task):
    def example(self, row: Row, index: int, language: str) -> Example:
        example_id = str(row.get("example_id") or index)
        target = row["target"] or "none"
        return Example(example_id, language, self.template.format(text=row["text"]), target)


class AfriMGSM(Task):
    def example(self, row: Row, index: int, language: str) -> Example:
        query = self.template.format(question=row["question"])
        return Example(str(index), language, query, str(row["answer_number"]))


class FloresPlus(Task):
    """eng -> xx. Source sentences come from the `eng_Latn` subset, aligned on `id`."""

    source_subset = "eng_Latn"

    def load(
        self, loader: DataLoader, benchmark: BenchmarkSpec, language: str, split: str
    ) -> list[Row]:
        source = {r["id"]: r["text"] for r in loader.rows(benchmark, self.source_subset, split)}
        target = loader.rows(benchmark, benchmark.subset_for(language), split)
        missing = [r["id"] for r in target if r["id"] not in source]
        if missing:
            raise TaskError(f"{benchmark.id}/{language}: no English source for ids {missing[:5]}")
        return [{"id": r["id"], "source": source[r["id"]], "target": r["text"]} for r in target]

    def example(self, row: Row, index: int, language: str) -> Example:
        query = self.template.format(language=prompt_name(language), text=row["source"])
        return Example(str(row["id"]), language, query, row["target"])


class AfriScienceMT(Task):
    def example(self, row: Row, index: int, language: str) -> Example:
        query = self.template.format(language=prompt_name(language), text=row["source"])
        example_id = f"{row['paper_id']}-{row['sentence_id']}"
        return Example(example_id, language, query, row["target"])


_WIKITEXT_HEADING = re.compile(r"^ = [^=].*[^=] = \n?$")


class WikiTextPerplexity(Task):
    """Raw WikiText rows are lines; they are regrouped into articles (one example each)."""

    def load(
        self, loader: DataLoader, benchmark: BenchmarkSpec, language: str, split: str
    ) -> list[Row]:
        articles: list[list[str]] = []
        for row in loader.rows(benchmark, benchmark.subset_for(language), split):
            line = row["text"]
            if _WIKITEXT_HEADING.match(line) or not articles:
                articles.append([])
            articles[-1].append(line)
        return [{"text": "".join(lines).strip()} for lines in articles if "".join(lines).strip()]

    def example(self, row: Row, index: int, language: str) -> Example:
        return Example(str(index), language, row["text"], "")


_TASKS: dict[str, type[Task]] = {
    "afrixnli": AfriXNLI,
    "afrimmlu": MMLU,
    "english_mmlu": MMLU,
    "belebele": Belebele,
    "sib200": SIB200,
    "injongo": InjongoIntent,
    "injongo_slots": InjongoSlots,
    "afrimgsm": AfriMGSM,
    "flores_plus": FloresPlus,
    "afriscience_mt": AfriScienceMT,
    "english_perplexity": WikiTextPerplexity,
}


def get_task(benchmark_id: str) -> Task:
    if benchmark_id not in _TASKS:
        raise TaskError(
            f"benchmark {benchmark_id!r} has no task implementation "
            "(afrimcqa is a scaffold: it needs image input)"
        )
    return _TASKS[benchmark_id](benchmark_id)
