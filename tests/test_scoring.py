"""Each metric against hand-computed values, plus bootstrap intervals and confusion matrices."""

import math

import numpy as np
import pytest
from sacrebleu.metrics import CHRF
from sklearn.metrics import f1_score

from eval.registry import Bootstrap
from eval.scoring import (
    accuracy,
    chrf,
    confusion,
    exact_match,
    perplexity,
    score_language,
    slot_f1,
    to_points,
)
from eval.scoring.bootstrap import bootstrap_ci
from tests import rows

BOOT = Bootstrap(n_resamples=200, seed=0)


def _sum(stats: np.ndarray) -> np.ndarray:
    return stats.sum(axis=0)


# --- accuracy and macro-F1 ---------------------------------------------------------------------


def test_accuracy() -> None:
    stats = accuracy.accuracy_stats([True, False, True, True])
    assert accuracy.ratio(_sum(stats)) == 0.75


def test_macro_f1_by_hand() -> None:
    gold = [0, 0, 1, 1, 2]
    pred = [0, 1, 1, 1, -1]  # last one unparseable
    # class 0: tp 1, fp 0, fn 1 -> 2/3; class 1: tp 2, fp 1, fn 0 -> 4/5; class 2: tp 0 -> 0
    expected = (2 / 3 + 4 / 5 + 0) / 3
    got = accuracy.macro_f1(_sum(accuracy.macro_f1_stats(gold, pred, 3)))
    assert got == pytest.approx(expected)
    assert got == pytest.approx(f1_score(gold, pred, labels=[0, 1, 2], average="macro"))


def test_macro_f1_perfect_and_empty_class() -> None:
    assert accuracy.macro_f1(_sum(accuracy.macro_f1_stats([0, 1], [0, 1], 2))) == 1.0
    # class 2 never appears: it contributes an F1 of 0 to the mean of three classes
    assert accuracy.macro_f1(_sum(accuracy.macro_f1_stats([0, 1], [0, 1], 3))) == pytest.approx(
        2 / 3
    )


# --- exact match with number parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("18", 18.0),
        ("The answer is 18.", 18.0),
        ("Step 1: 3 + 4 = 7. Step 2: 7 * 2 = 14. So she has 14 apples.", 14.0),
        ("$1,234.50", 1234.5),
        ("It costs 1,000 dollars", 1000.0),
        ("-5", -5.0),
        ("18.0", 18.0),
        ("Impendulo ithi 11.", 11.0),
        ("no number here", None),
        ("", None),
        ("...", None),
    ],
)
def test_extract_number(output: str, expected: float | None) -> None:
    assert exact_match.extract_number(output) == expected


def test_number_match_counts_unparseable_as_wrong() -> None:
    assert exact_match.number_match("The answer is 18.", "18") == (True, True)
    assert exact_match.number_match("18.0", "18") == (True, True)
    assert exact_match.number_match("17", "18") == (False, True)
    assert exact_match.number_match("I am not sure", "18") == (False, False)


# --- chrF++ -----------------------------------------------------------------------------------


def test_chrf_matches_sacrebleu_with_word_order_2() -> None:
    hyps = ["the cat sat", "hello world", "a b c d e"]
    refs = ["the cat sat down", "hello there world", "a b c d f"]
    expected = CHRF(word_order=2).corpus_score(hyps, [refs]).score
    assert chrf.chrf_from_sums(_sum(chrf.chrf_stats(hyps, refs))) == pytest.approx(expected)
    assert expected != pytest.approx(CHRF(word_order=0).corpus_score(hyps, [refs]).score)


def test_chrf_bounds() -> None:
    same = ["ngiyabonga kakhulu", "sawubona"]
    assert chrf.chrf_from_sums(_sum(chrf.chrf_stats(same, same))) == pytest.approx(100.0)
    assert chrf.chrf_from_sums(_sum(chrf.chrf_stats(["xyz"], ["abc"]))) == 0.0


# --- slot F1 ------------------------------------------------------------------------------------


def test_parse_spans() -> None:
    spans = slot_f1.parse_spans("COUNTRY: Mali $$ MONEY: R20 $$ MONEY: R20")
    assert spans == {("COUNTRY", "mali"): 1, ("MONEY", "r20"): 2}
    assert (
        slot_f1.parse_spans("none") == {} == slot_f1.parse_spans("") == slot_f1.parse_spans("None.")
    )
    assert slot_f1.parse_spans("time: 7 ekuseni.\nDATE: Monday") == {
        ("TIME", "7 ekuseni"): 1,
        ("DATE", "monday"): 1,
    }
    assert slot_f1.parse_spans("Here are the slots I found") == {}


def test_slot_f1_by_hand() -> None:
    outputs = ["COUNTRY: Mali $$ MONEY: R20", "none", "TIME: 7 ekuseni"]
    golds = ["COUNTRY: Mali $$ MONEY: R500", "", "TIME: 7 ekuseni $$ DATE: namuhla"]
    stats = slot_f1.slot_stats(outputs, golds)
    assert stats.tolist() == [[1, 1, 1], [0, 0, 0], [1, 0, 1]]
    # tp 2, fp 1, fn 2 -> precision 2/3, recall 2/4 -> F1 = 4 / 7
    assert slot_f1.slot_f1(_sum(stats)) == pytest.approx(4 / 7)
    assert slot_f1.slot_f1(np.zeros(3)) == 0.0


# --- perplexity ---------------------------------------------------------------------------------


def test_perplexity_is_token_weighted() -> None:
    stats = np.array([[2.0 * 10, 10], [4.0 * 30, 30]])
    assert perplexity.perplexity(_sum(stats)) == pytest.approx(math.exp(140 / 40))


# --- bootstrap ----------------------------------------------------------------------------------


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    stats = accuracy.accuracy_stats([True] * 70 + [False] * 30)
    first = bootstrap_ci(stats, accuracy.ratio, n_resamples=1000, seed=0)
    assert first == bootstrap_ci(stats, accuracy.ratio, n_resamples=1000, seed=0)
    assert first != bootstrap_ci(stats, accuracy.ratio, n_resamples=1000, seed=1)
    low, high = first
    assert low < 0.70 < high
    # normal approximation: 0.70 +- 1.96 * sqrt(0.7 * 0.3 / 100) = [0.610, 0.790]
    assert low == pytest.approx(0.610, abs=0.03) and high == pytest.approx(0.790, abs=0.03)


def test_bootstrap_degenerate_cases() -> None:
    assert bootstrap_ci(accuracy.accuracy_stats([True] * 20), accuracy.ratio) == (1.0, 1.0)
    low, high = bootstrap_ci(np.zeros((0, 2)), accuracy.ratio)
    assert math.isnan(low) and math.isnan(high)


def test_bootstrap_interval_narrows_with_more_data() -> None:
    def width(n: int) -> float:
        stats = accuracy.accuracy_stats(([True] * 6 + [False] * 4) * (n // 10))
        low, high = bootstrap_ci(stats, accuracy.ratio, n_resamples=500, seed=0)
        return high - low

    assert width(1000) < width(100) / 2


# --- confusion matrices -------------------------------------------------------------------------


def test_confusion_matrix_with_unparseable_column() -> None:
    matrix = confusion.confusion([0, 0, 1], [0, 1, -1], ["True", "False"])
    assert matrix.tolist() == [[1, 1, 0], [0, 0, 1]]
    markdown = confusion.to_markdown(matrix, ["True", "False"])
    assert markdown.splitlines()[0] == "| gold \\ pred | True | False | (unparseable) |"
    assert markdown.splitlines()[2] == "| True | 1 | 1 | 0 |"
    assert confusion.confusion([0, 1], [0, 1], ["a", "b"]).tolist() == [[1, 0], [0, 1]]


def test_prediction_shares() -> None:
    assert confusion.prediction_shares([0, 0, 0, 1], 4) == [0.75, 0.25, 0.0, 0.0]
    assert confusion.prediction_shares([0, -1], 2) == [0.5, 0.0]


# --- score_language -----------------------------------------------------------------------------


def _mc_rows(correct: list[bool]) -> list[dict]:
    return [
        {"gold": 0, "pred": 0 if c else 1, "correct": c, "parse_ok": True, "scores": [0.0] * 7}
        for c in correct
    ]


def test_score_classification_with_extra_macro_f1() -> None:
    bench = rows.one_language("sib200")
    score = score_language(bench, _mc_rows([True, True, True, False]), BOOT)
    assert score.primary.value == 0.75
    assert score.primary.ci_low <= 0.75 <= score.primary.ci_high
    assert score.n_examples == 4 and score.n_unparseable == 0
    # only class 0 has gold; class 1 has one false positive; 5 classes are empty
    assert score.extra["macro_f1"].value == pytest.approx((2 * 3 / (2 * 3 + 0 + 1)) / 7)


def test_score_counts_unparseable() -> None:
    bench = rows.one_language("afrixnli")
    data = _mc_rows([True, False])
    data[1].update(pred=-1, parse_ok=False)
    assert score_language(bench, data, BOOT).n_unparseable == 1


def test_score_exact_match_annotates_rows() -> None:
    bench = rows.one_language("afrimgsm")
    data = [
        {"output": "So the answer is 18.", "reference": "18"},
        {"output": "no idea", "reference": "5"},
    ]
    score = score_language(bench, data, BOOT)
    assert score.primary.value == 0.5 and score.n_unparseable == 1
    assert [(r["correct"], r["parse_ok"]) for r in data] == [(True, True), (False, False)]


def test_score_chrf_slot_f1_and_perplexity() -> None:
    mt = [{"output": "sawubona mhlaba", "reference": "sawubona mhlaba"}]
    assert score_language(
        rows.one_language("flores_plus"), mt, BOOT
    ).primary.value == pytest.approx(100)
    slots = [
        {"output": "TIME: 7", "reference": "TIME: 7"},
        {"output": "none", "reference": "DATE: x"},
    ]
    assert score_language(
        rows.one_language("injongo_slots"), slots, BOOT
    ).primary.value == pytest.approx(2 / 3)
    ppl = [{"nll": 30.0, "n_tokens": 10}, {"nll": 10.0, "n_tokens": 10}]
    got = score_language(rows.shipped_benchmark("english_perplexity"), ppl, BOOT)
    assert got.primary.value == pytest.approx(math.exp(2.0))


def test_to_points() -> None:
    assert to_points("accuracy", 0.341) == pytest.approx(34.1)
    assert to_points("chrf++", 41.2) == 41.2
    assert to_points("perplexity", 12.5) == 12.5
