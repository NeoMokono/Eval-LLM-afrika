"""HFDataLoader: subset/config resolution, row filtering, pinned revisions, clear errors."""

import sys
import types

import pytest

from eval.data import DataError, HFDataLoader
from tests import rows


@pytest.fixture
def fake_datasets(monkeypatch):
    """Replace `datasets.load_dataset` and record how it was called."""
    calls: list[dict] = []
    tables: dict[tuple[str, str, str], list[dict]] = {}

    def load_dataset(path, name, *, split, revision, token):
        calls.append({"path": path, "name": name, "split": split, "revision": revision})
        if (path, name, split) not in tables:
            raise FileNotFoundError(f"no config {name!r}")
        return tables[(path, name, split)]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return calls, tables


def test_subset_is_the_hf_config_and_revision_is_pinned(fake_datasets) -> None:
    calls, tables = fake_datasets
    bench = rows.one_language("belebele")
    tables[("facebook/belebele", "zul_Latn", "test")] = [rows.belebele(1, 1)]
    loader = HFDataLoader()
    assert len(loader.rows(bench, bench.subset_for("zul"), "test")) == 1
    assert calls == [
        {
            "path": "facebook/belebele",
            "name": "zul_Latn",
            "split": "test",
            "revision": bench.revision,
        }
    ]
    loader.rows(bench, "zul_Latn", "test")
    assert len(calls) == 1  # memoised


def test_planner_resolved_revision_overrides_the_configured_ref(fake_datasets) -> None:
    calls, tables = fake_datasets
    bench = rows.one_language("belebele")
    tables[("facebook/belebele", "zul_Latn", "test")] = [rows.belebele(1, 1)]
    HFDataLoader({"belebele": "f" * 40}).rows(bench, "zul_Latn", "test")
    assert calls[0]["revision"] == "f" * 40


def test_language_column_filters_rows_of_the_shared_config(fake_datasets) -> None:
    calls, tables = fake_datasets
    bench = rows.shipped_benchmark("afriscience_mt")
    tables[("dsfsi/afriscience_mt", "corpus", "test")] = [
        rows.science(1, "eng-zul"),
        rows.science(2, "eng-nso"),
        rows.science(3, "eng-zul"),
    ]
    loader = HFDataLoader()
    zul = loader.rows(bench, bench.subset_for("zul"), "test")
    nso = loader.rows(bench, bench.subset_for("nso"), "test")
    assert [r["sentence_id"] for r in zul] == [1, 3]
    assert [r["sentence_id"] for r in nso] == [2]
    assert [c["name"] for c in calls] == ["corpus"]  # loaded once, filtered per language


def test_fixed_config_is_used_for_single_config_benchmarks(fake_datasets) -> None:
    calls, tables = fake_datasets
    bench = rows.shipped_benchmark("english_mmlu")
    tables[("cais/mmlu", "all", "test")] = [rows.mmlu(1)]
    HFDataLoader().rows(bench, bench.subset_for("eng"), "test")
    assert calls[0]["name"] == "all"


def test_load_failure_names_the_dataset_and_hints_at_the_token(fake_datasets, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    bench = rows.one_language("flores_plus")
    with pytest.raises(DataError) as exc:
        HFDataLoader().rows(bench, "zul_Latn", "devtest")
    message = str(exc.value)
    assert "openlanguagedata/flores_plus [zul_Latn/devtest]" in message
    assert "set HF_TOKEN" in message
