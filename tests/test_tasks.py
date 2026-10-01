"""Tasks: verified dataset rows -> prompts, options and gold answers."""

import pytest

from eval.data import DataError, InMemoryLoader
from eval.prompts import INJONGO_INTENTS, SIB_CATEGORIES, TEMPLATES, PromptTemplate, template_for
from eval.registry import TaskType, load_config
from eval.tasks import TaskError, get_task
from tests import rows
from tests.test_shipped_configs import CONFIGS


def _example(benchmark_id: str, row: dict, language: str = "zul"):
    return get_task(benchmark_id).example(row, 0, language)


def test_afrixnli_prompt_and_label_mapping() -> None:
    ex = _example("afrixnli", rows.xnli(1, label=2))
    assert ex.query == "premise 1\nQuestion: hypothesis 1 True, False, or Neither?\nAnswer:"
    assert ex.options == ("True", "Neither", "False")
    assert (ex.gold, ex.target) == (2, "False")  # 2 = contradiction


def test_afrimmlu_parses_stringified_choices_and_letter_answer() -> None:
    ex = _example("afrimmlu", rows.afrimmlu(3, answer="C"))
    assert ex.query == "question 3?\nA. w3\nB. x3\nC. y3\nD. z3\nAnswer:"
    assert (ex.options, ex.gold, ex.target) == (("A", "B", "C", "D"), 2, "C")
    assert ex.group == "global_facts"


def test_english_mmlu_uses_the_same_prompt_with_list_choices() -> None:
    ex = _example("english_mmlu", rows.mmlu(3, answer=1), "eng")
    assert ex.query == "question 3?\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:"
    assert (ex.gold, ex.target, ex.group) == (1, "B", "abstract_algebra")
    assert TEMPLATES["afrimmlu"] is TEMPLATES["english_mmlu"]


def test_mmlu_rejects_malformed_choices() -> None:
    row = rows.mmlu(1) | {"choices": ["only", "three", "choices"]}
    with pytest.raises(TaskError, match="expected 4 choices"):
        _example("english_mmlu", row, "eng")


def test_belebele_gold_is_zero_based_and_grouped_by_passage() -> None:
    ex = _example("belebele", rows.belebele(2, passage=7, correct="3"))
    assert ex.query.startswith("P: passage text 7\nQ: question 2 about passage 7?\nA: one\n")
    assert ex.query.endswith("Please choose the correct answer from the options above:")
    assert (ex.gold, ex.target) == (2, "C")
    assert ex.group == "https://example.org/passage/7"
    assert ex.example_id == "https://example.org/passage/7#2"


def test_sib200_categories() -> None:
    ex = _example("sib200", rows.sib(1, "science/technology"))
    assert "what category does the text: 'sentence 1' belong to" in ex.query
    assert (ex.gold, ex.target, ex.example_id) == (0, "science/technology", "1001")
    with pytest.raises(ValueError):
        _example("sib200", rows.sib(1, "weather"))


def test_injongo_intent_and_slots() -> None:
    intent = _example("injongo", rows.injongo(1, intent="weather"))
    assert intent.options == INJONGO_INTENTS
    assert intent.target == "weather"
    assert "Given the text: 'utterance 1'" in intent.query
    slots = _example("injongo_slots", rows.injongo(1, target="COUNTRY: Mali $$ MONEY: R20"))
    assert slots.target == "COUNTRY: Mali $$ MONEY: R20"
    assert slots.query.endswith("Text: utterance 1")
    assert _example("injongo_slots", rows.injongo(1, target="")).target == "none"


def test_afrimgsm_reference_is_the_answer_number() -> None:
    ex = _example("afrimgsm", rows.mgsm(1, number=18))
    assert (ex.query, ex.target) == ("Question: word problem 1\nAnswer:", "18")


def test_flores_pairs_english_source_with_target_by_id() -> None:
    bench = rows.one_language("flores_plus", "zul")
    loader = InMemoryLoader(
        {
            (bench.id, "eng_Latn", "devtest"): [rows.flores(1, "one"), rows.flores(0, "zero")],
            (bench.id, "zul_Latn", "devtest"): [rows.flores(0, "iqanda"), rows.flores(1, "kunye")],
        }
    )
    examples = get_task(bench.id).examples(loader, bench, "zul", "devtest")
    assert [(e.example_id, e.target) for e in examples] == [("0", "iqanda"), ("1", "kunye")]
    assert examples[0].query == (
        "Translate the following text from English to Zulu. Provide only the translation.\n\nzero"
    )


def test_flores_missing_source_is_an_error() -> None:
    bench = rows.one_language("flores_plus", "zul")
    loader = InMemoryLoader(
        {
            (bench.id, "eng_Latn", "devtest"): [rows.flores(0, "zero")],
            (bench.id, "zul_Latn", "devtest"): [rows.flores(9, "x")],
        }
    )
    with pytest.raises(TaskError, match="no English source"):
        get_task(bench.id).examples(loader, bench, "zul", "devtest")


def test_afriscience_uses_the_language_pair_subset() -> None:
    bench = rows.one_language("afriscience_mt", "zul")
    assert bench.subset_for("zul") == "eng-zul"
    loader = rows.loader_for(bench, {"test": [rows.science(2, "eng-zul")]})
    (ex,) = get_task(bench.id).examples(loader, bench, "zul", "test")
    assert (ex.example_id, ex.target) == ("4-2", "eng-zul translation 2")
    assert "to Zulu." in ex.query and ex.query.endswith("english sentence 2")


def test_wikitext_lines_are_grouped_into_articles() -> None:
    bench = rows.shipped_benchmark("english_perplexity")
    loader = rows.loader_for(bench, {"test": rows.WIKITEXT}, "eng")
    examples = get_task(bench.id).examples(loader, bench, "eng", "test")
    assert len(examples) == 2
    assert examples[0].query.startswith("= Robert Boulter =")
    assert "= = Career = =" in examples[0].query  # sub-headings stay inside the article
    assert examples[1].query.startswith("= Du Fu =")


def test_empty_split_is_an_error() -> None:
    bench = rows.one_language("afrixnli")
    with pytest.raises(TaskError, match="no rows"):
        get_task(bench.id).examples(rows.loader_for(bench, {"test": []}), bench, "zul", "test")
    with pytest.raises(DataError):
        get_task(bench.id).examples(InMemoryLoader({}), bench, "zul", "test")


# --- registry-level consistency -------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIGS / "run.yaml", check_adapters=False)


def test_every_enabled_benchmark_has_a_task_and_template(config) -> None:
    for bench in config.benchmarks.benchmarks:
        if bench.enabled:
            assert get_task(bench.id).template is template_for(bench)


def test_scaffold_benchmark_has_no_task() -> None:
    with pytest.raises(TaskError, match="scaffold"):
        get_task("afrimcqa")


def test_chance_level_matches_the_number_of_options(config) -> None:
    for bench in config.benchmarks.benchmarks:
        if bench.enabled and bench.type is TaskType.CLASSIFICATION:
            options = template_for(bench).options
            assert bench.chance_level == pytest.approx(1 / len(options), abs=1e-6), bench.id


def test_label_sets_have_the_verified_sizes() -> None:
    assert len(SIB_CATEGORIES) == len(set(SIB_CATEGORIES)) == 7
    assert len(INJONGO_INTENTS) == len(set(INJONGO_INTENTS)) == 40


def test_template_hash_covers_text_and_options() -> None:
    base = PromptTemplate("t", "text {x}", ("A", "B"))
    assert base.sha256 == PromptTemplate("other-id", "text {x}", ("A", "B")).sha256
    assert base.sha256 != PromptTemplate("t", "text  {x}", ("A", "B")).sha256
    assert base.sha256 != PromptTemplate("t", "text {x}", ("A", "C")).sha256
