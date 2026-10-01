"""Rows shaped exactly like the real datasets (column names and value types as verified on the
HF Hub), for exercising tasks and runners without network access."""

from __future__ import annotations

from typing import Any

from eval.data import InMemoryLoader
from eval.registry import BenchmarkSpec, load_config
from tests.test_shipped_configs import CONFIGS

Row = dict[str, Any]


def xnli(i: int, label: int = 0) -> Row:
    return {"premise": f"premise {i}", "hypothesis": f"hypothesis {i}", "label": label}


def afrimmlu(i: int, answer: str = "C", subject: str = "global_facts") -> Row:
    return {
        "subject": subject,
        "question": f"question {i}?",
        "choices": f"['w{i}', 'x{i}', 'y{i}', 'z{i}']",
        "answer": answer,
    }


def mmlu(i: int, answer: int = 1, subject: str = "abstract_algebra") -> Row:
    return {
        "question": f"question {i}?",
        "subject": subject,
        "choices": ["0", "4", "2", "6"],
        "answer": answer,
    }


def mgsm(i: int, number: int = 18) -> Row:
    return {
        "question": f"word problem {i}",
        "answer": None,
        "answer_number": number,
        "equation_solution": None,
    }


def sib(i: int, category: str = "sports") -> Row:
    return {"index_id": 1000 + i, "category": category, "text": f"sentence {i}"}


def belebele(i: int, passage: int, correct: str = "1") -> Row:
    return {
        "link": f"https://example.org/passage/{passage}",
        "question_number": i,
        "flores_passage": f"passage text {passage}",
        "question": f"question {i} about passage {passage}? ",
        "mc_answer1": "one",
        "mc_answer2": "two",
        "mc_answer3": "three",
        "mc_answer4": "four",
        "correct_answer_num": correct,
        "dialect": "zul_Latn",
    }


def injongo(i: int, intent: str = "alarm", target: str = "TIME: 7 ekuseni") -> Row:
    return {
        "intent": intent,
        "text": f"utterance {i}",
        "spans": [],
        "target": target,
        "example_id": f"test-{i:08d}",
    }


def flores(i: int, text: str) -> Row:
    return {"id": i, "iso_639_3": "xxx", "text": text, "split": "devtest"}


def science(i: int, pair: str) -> Row:
    return {
        "paper_id": 4,
        "sentence_id": i,
        "domain": "health",
        "lang_pair": pair,
        "source_lang": "eng",
        "target_lang": pair.split("-")[1],
        "source": f"english sentence {i}",
        "target": f"{pair} translation {i}",
    }


WIKITEXT = [
    {"text": ""},
    {"text": " = Robert Boulter = \n"},
    {"text": ""},
    {"text": " Robert Boulter is an English actor . \n"},
    {"text": " = = Career = = \n"},
    {"text": " He had a guest role . \n"},
    {"text": " = Du Fu = \n"},
    {"text": " Du Fu was a Chinese poet . \n"},
]


def shipped_benchmark(benchmark_id: str) -> BenchmarkSpec:
    return load_config(CONFIGS / "run.yaml", check_adapters=False).benchmarks.get(benchmark_id)


def one_language(benchmark_id: str, language: str = "zul", **over: Any) -> BenchmarkSpec:
    """The shipped benchmark definition, narrowed to a single language."""
    spec = shipped_benchmark(benchmark_id)
    subsets = {k: v for k, v in spec.subsets.items() if k == language}
    return spec.model_copy(update={"languages": [language], "subsets": subsets, **over})


def loader_for(
    benchmark: BenchmarkSpec, splits: dict[str, list[Row]], language: str = "zul"
) -> InMemoryLoader:
    subset = benchmark.subset_for(language)
    return InMemoryLoader({(benchmark.id, subset, split): rows for split, rows in splits.items()})
