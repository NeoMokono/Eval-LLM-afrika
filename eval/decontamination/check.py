"""Training-data vs benchmark overlap check (v1: normalised 13-gram overlap).

For one SFT training set and a list of benchmarks, find test examples whose text shares a
normalised 13-token window with any training record. Test texts shorter than 13 tokens are
matched as a whole (down to `MIN_TOKENS`); anything shorter cannot be checked and is counted.

A status file is written per (benchmark, training-data version). A language is `verified`
when no test example overlaps. Records pick their decontamination status up from these files
at report time, so changing a model's `sft_data.version` makes its results `unverified` until
the check is rerun, without touching the model fingerprint or forcing re-inference.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.atomic import atomic_write_text
from eval.data import DataLoader
from eval.registry import BenchmarkSpec, SFTData
from eval.tasks import get_task

NGRAM = 13
MIN_TOKENS = 8
MAX_EXAMPLES_LISTED = 5
_NOT_WORD = re.compile(r"[^\w\s]", re.UNICODE)


class DecontaminationError(Exception):
    pass


def normalize_tokens(text: str) -> list[str]:
    """Case-folded, NFKC-normalised, punctuation-free whitespace tokens."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return _NOT_WORD.sub(" ", text).split()


def _windows(tokens: Sequence[str], size: int) -> Iterator[str]:
    for start in range(len(tokens) - size + 1):
        yield " ".join(tokens[start : start + size])


def slug(version: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", version).strip("_")


# ------------------------------------------------------------------ training data


def _field_values(record: Any, path: str) -> Iterator[str]:
    """Values at a dotted path such as `text` or `messages[].content`."""
    head, _, rest = path.partition(".")
    is_list = head.endswith("[]")
    value = record.get(head.removesuffix("[]")) if isinstance(record, dict) else None
    if value is None:
        return
    for item in value if is_list else [value]:
        if rest:
            yield from _field_values(item, rest)
        elif isinstance(item, str):
            yield item


def training_files(sft: SFTData) -> list[Path]:
    root = Path(sft.path)
    if root.is_file():
        return [root]
    files = sorted(p for p in root.rglob("*") if p.suffix in {".jsonl", ".parquet"})
    if not files:
        raise DecontaminationError(f"no .jsonl or .parquet training files under {root}")
    return files


def iter_training_texts(sft: SFTData) -> Iterator[tuple[str, int, str]]:
    """(file name, record number, text) for every configured text field of every record."""
    for path in training_files(sft):
        if path.suffix == ".parquet":
            import pandas as pd

            records: Iterable[Any] = pd.read_parquet(path).to_dict(orient="records")
        else:
            records = (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
        for number, record in enumerate(records):
            for field_path in sft.text_fields:
                for text in _field_values(record, field_path):
                    yield (path.name, number, text)


# ---------------------------------------------------------------------- the check


@dataclass
class LanguageResult:
    n_examples: int = 0
    n_too_short: int = 0  # no text of at least MIN_TOKENS tokens: cannot be checked
    contaminated: set[int] = field(default_factory=set)
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "verified" if not self.contaminated else "unverified"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_examples": self.n_examples,
            "n_checked": self.n_examples - self.n_too_short,
            "n_too_short": self.n_too_short,
            "n_contaminated": len(self.contaminated),
            "samples": self.samples,
        }


Owner = tuple[str, str, int]  # (benchmark id, language, example index)


def build_index(
    benchmarks: Sequence[BenchmarkSpec], loader: DataLoader
) -> tuple[dict[int, dict[str, list[Owner]]], dict[tuple[str, str], LanguageResult]]:
    """Index every checkable test-text window: window length -> window -> owning examples."""
    index: dict[int, dict[str, list[Owner]]] = {}
    results: dict[tuple[str, str], LanguageResult] = {}
    for bench in benchmarks:
        task = get_task(bench.id)
        for language in bench.languages:
            result = results[(bench.id, language)] = LanguageResult()
            for number, row in enumerate(task.load(loader, bench, language, bench.split)):
                result.n_examples += 1
                checkable = False
                for text in task.texts(row):
                    tokens = normalize_tokens(text)
                    if len(tokens) < MIN_TOKENS:
                        continue
                    checkable = True
                    size = min(len(tokens), NGRAM)
                    for window in _windows(tokens, size):
                        index.setdefault(size, {}).setdefault(window, []).append(
                            (bench.id, language, number)
                        )
                if not checkable:
                    result.n_too_short += 1
    return index, results


def scan_training_data(
    sft: SFTData,
    index: dict[int, dict[str, list[Owner]]],
    results: dict[tuple[str, str], LanguageResult],
) -> int:
    """Mark every indexed example that shares a window with a training text. Returns the
    number of training texts scanned."""
    scanned = 0
    for file_name, number, text in iter_training_texts(sft):
        scanned += 1
        tokens = normalize_tokens(text)
        for size, windows in index.items():
            if len(tokens) < size:
                continue
            for window in _windows(tokens, size):
                owners = windows.get(window)
                if owners is None:
                    continue
                for bench_id, language, example in owners:
                    result = results[(bench_id, language)]
                    if example not in result.contaminated:
                        result.contaminated.add(example)
                        if len(result.samples) < MAX_EXAMPLES_LISTED:
                            result.samples.append(
                                {
                                    "example_index": example,
                                    "training_file": file_name,
                                    "training_record": number,
                                    "window": window,
                                }
                            )
    return scanned


def status_path(results_dir: Path, sft_version: str, benchmark_id: str) -> Path:
    return results_dir / "decontamination" / slug(sft_version) / f"{benchmark_id}.json"


def run_check(
    sft: SFTData,
    benchmarks: Sequence[BenchmarkSpec],
    revisions: dict[str, str],
    loader: DataLoader,
    results_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Check `benchmarks` against one training set and write a status file per benchmark."""
    index, results = build_index(benchmarks, loader)
    scanned = scan_training_data(sft, index, results)
    written: dict[str, dict[str, Any]] = {}
    for bench in benchmarks:
        document = {
            "benchmark_id": bench.id,
            "benchmark_version": f"{bench.dataset}@{revisions[bench.id]}",
            "split": bench.split,
            "sft_data_version": sft.version,
            "sft_data_path": sft.path,
            "method": (
                f"normalised {NGRAM}-gram overlap (whole text for {MIN_TOKENS}-{NGRAM - 1} tokens)"
            ),
            "training_texts_scanned": scanned,
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "languages": {lang: results[(bench.id, lang)].to_json() for lang in bench.languages},
        }
        path = status_path(results_dir, sft.version, bench.id)
        atomic_write_text(path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        written[bench.id] = document
    return written


# ------------------------------------------------------------- status propagation


def load_status(
    results_dir: Path, sft_version: str | None, benchmark_id: str, benchmark_version: str
) -> dict[str, str]:
    """language -> 'verified' | 'unverified' from the status file for this training-data
    version, or {} when no check exists for it (or it was run on another benchmark version)."""
    if sft_version is None:
        return {}
    path = status_path(results_dir, sft_version, benchmark_id)
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("sft_data_version") != sft_version:
        return {}
    if document.get("benchmark_version") != benchmark_version:
        return {}
    return {lang: info["status"] for lang, info in document["languages"].items()}


def current_status(
    results_dir: Path,
    *,
    model_role: str,
    sft_version: str | None,
    benchmark_id: str,
    benchmark_version: str,
    language: str,
) -> str:
    """The decontamination status a record should carry right now."""
    if model_role in {"base", "reference", "external"}:
        return "not_applicable"
    statuses = load_status(results_dir, sft_version, benchmark_id, benchmark_version)
    return statuses.get(language, "unverified")
