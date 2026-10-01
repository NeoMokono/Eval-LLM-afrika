"""Decontamination: a leaked test example is found, status files drive record status, and a
new training-data version makes results unverified until the check is rerun (invariant 7)."""

import json
from pathlib import Path

import pytest

from eval.decontamination.check import (
    DecontaminationError,
    current_status,
    iter_training_texts,
    load_status,
    normalize_tokens,
    run_check,
    slug,
    status_path,
)
from eval.registry import SFTData
from tests import rows

LEAK = (
    "Umfula omkhulu ugeleza ngokushesha uma imvula ina kakhulu "
    "ezintabeni ngesikhathi sasehlobo njalo"
)
CLEAN = (
    "Izingane zidlala ibhola ngemuva kwesikole nsuku zonke "
    "lapho ilanga selishona kancane kancane emini"
)
SHORT = "Sawubona mngane"
MEDIUM = "Lesi yisigwebo esinamagama ayisishiyagalolunye kuphela nje"  # 6 tokens: too short


def _sft(tmp_path: Path, records: list[dict], version: str = "mix-v1") -> SFTData:
    data = tmp_path / f"train-{version}"
    data.mkdir(exist_ok=True)
    with (data / "shard-00000.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return SFTData(version=version, path=str(data))


def _chat(user: str, assistant: str = "ok") -> dict:
    return {
        "language": "zul",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _sib_setup(texts: list[str]):
    bench = rows.one_language("sib200")
    test = [rows.sib(i) | {"text": t} for i, t in enumerate(texts)]
    return bench, rows.loader_for(bench, {"test": test})


def _check(tmp_path: Path, sft: SFTData, texts: list[str]) -> dict:
    bench, loader = _sib_setup(texts)
    out = run_check(sft, [bench], {bench.id: bench.revision}, loader, tmp_path / "results")
    return out[bench.id]["languages"]["zul"]


def test_normalisation_ignores_case_punctuation_and_spacing() -> None:
    assert normalize_tokens("Sawubona,  MHLABA!\n(yebo)") == ["sawubona", "mhlaba", "yebo"]
    assert normalize_tokens("ﬁne") == ["fine"]  # NFKC folds the ligature


def test_synthetic_leak_is_detected(tmp_path: Path) -> None:
    leaked_in_prompt = f"Classify this text.\n\n{LEAK.upper()}!!"  # different case/punctuation
    sft = _sft(
        tmp_path,
        [
            _chat("Unrelated training text about something else entirely here"),
            _chat(leaked_in_prompt),
        ],
    )
    result = _check(tmp_path, sft, [CLEAN, LEAK])
    assert result["status"] == "unverified"
    assert (result["n_examples"], result["n_contaminated"]) == (2, 1)
    sample = result["samples"][0]
    assert sample["example_index"] == 1
    assert (sample["training_file"], sample["training_record"]) == ("shard-00000.jsonl", 1)


def test_leak_in_the_assistant_turn_is_detected(tmp_path: Path) -> None:
    sft = _sft(tmp_path, [_chat("Translate this", assistant=LEAK)])
    assert _check(tmp_path, sft, [LEAK])["n_contaminated"] == 1


def test_clean_training_data_is_verified(tmp_path: Path) -> None:
    sft = _sft(tmp_path, [_chat("Unrelated training text about something else entirely here")])
    result = _check(tmp_path, sft, [CLEAN, LEAK])
    assert result["status"] == "verified"
    assert result["n_contaminated"] == 0 and result["n_checked"] == 2


def test_partial_overlap_shorter_than_the_window_is_not_a_leak(tmp_path: Path) -> None:
    first_ten_words = " ".join(LEAK.split()[:10])
    sft = _sft(tmp_path, [_chat(first_ten_words + " kodwa bese kuphela ngenye indlela ehlukile")])
    assert _check(tmp_path, sft, [LEAK])["status"] == "verified"


def test_short_test_text_is_matched_as_a_whole(tmp_path: Path) -> None:
    nine_words = "Ngicela ungitshele isikhathi sesitimela esiya eThekwini kusasa ekuseni kakhulu"
    assert len(normalize_tokens(nine_words)) == 9
    sft = _sft(tmp_path, [_chat(f"Umbuzo: {nine_words}. Phendula.")])
    assert _check(tmp_path, sft, [nine_words])["n_contaminated"] == 1


def test_uncheckably_short_text_is_counted_not_passed_silently(tmp_path: Path) -> None:
    sft = _sft(tmp_path, [_chat(SHORT)])
    result = _check(tmp_path, sft, [SHORT, MEDIUM, CLEAN])
    assert result["n_too_short"] == 2 and result["n_checked"] == 1
    assert result["n_contaminated"] == 0


def test_text_fields_are_configurable(tmp_path: Path) -> None:
    sft = _sft(tmp_path, [{"text": LEAK, "meta": {"notes": [{"body": CLEAN}]}}])
    plain = sft.model_copy(update={"text_fields": ["text", "meta.notes[].body"]})
    assert [t for _, _, t in iter_training_texts(plain)] == [LEAK, CLEAN]
    assert list(iter_training_texts(sft)) == []  # default messages[].content finds nothing


def test_missing_training_data_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    sft = SFTData(version="v", path=str(tmp_path / "empty"))
    with pytest.raises(DecontaminationError, match=r"no \.jsonl or \.parquet"):
        list(iter_training_texts(sft))


# --- propagation into record status ------------------------------------------------------------


def _status(
    results: Path, version: str | None, role: str = "target", bench_version: str | None = None
) -> str:
    bench = rows.one_language("sib200")
    return current_status(
        results,
        model_role=role,
        sft_version=version,
        benchmark_id=bench.id,
        benchmark_version=bench_version or bench.version,
        language="zul",
    )


def test_status_file_drives_record_status(tmp_path: Path) -> None:
    results = tmp_path / "results"
    assert _status(results, "mix-v1") == "unverified"  # no check yet
    sft = _sft(tmp_path, [_chat("Unrelated training text about something else entirely here")])
    _check(tmp_path, sft, [CLEAN])
    assert status_path(results, "mix-v1", "sib200").is_file()
    assert _status(results, "mix-v1") == "verified"


def test_new_training_data_version_is_unverified_until_rechecked(tmp_path: Path) -> None:
    results = tmp_path / "results"
    clean = [_chat("Unrelated training text about something else entirely here")]
    _check(tmp_path, _sft(tmp_path, clean, "mix-v1"), [LEAK])
    assert _status(results, "mix-v1") == "verified"
    assert _status(results, "mix-v2") == "unverified"  # version changed, no check for it
    _check(tmp_path, _sft(tmp_path, [*clean, _chat(LEAK)], "mix-v2"), [LEAK])
    assert _status(results, "mix-v2") == "unverified"  # checked, and it leaks
    assert _status(results, "mix-v1") == "verified"  # the old version's result is untouched


def test_check_on_another_benchmark_version_does_not_count(tmp_path: Path) -> None:
    results = tmp_path / "results"
    sft = _sft(tmp_path, [_chat("Unrelated training text about something else entirely here")])
    _check(tmp_path, sft, [CLEAN])
    assert _status(results, "mix-v1", bench_version="Davlan/sib200@" + "9" * 40) == "unverified"
    assert load_status(results, "mix-v1", "sib200", "Davlan/sib200@" + "9" * 40) == {}


def test_upstream_models_are_not_applicable_and_undeclared_data_is_unverified(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    for role in ("base", "reference", "external"):
        assert _status(results, None, role=role) == "not_applicable"
    assert _status(results, None, role="target") == "unverified"
    assert _status(results, None, role="plain_baseline") == "unverified"


def test_slug_is_filesystem_safe() -> None:
    assert slug("top40-gold-v1-core@sha256:c0878d") == "top40-gold-v1-core_sha256_c0878d"
