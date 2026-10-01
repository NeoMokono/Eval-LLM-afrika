"""Runners on the mock backend: predictions, few-shot prompts, parsing, limits."""

import pytest

from eval.backends.base import LoglikelihoodRequest, Prompt, Turn
from eval.backends.mock import MockBackend
from eval.fewshot import FewshotError, choose_shots, prepare_pool
from eval.registry import Decoding, Prompting
from eval.runners import run_unit
from eval.runners.classification_runner import UNPARSEABLE, parse_choice
from eval.tasks import Example, TaskError, get_task
from tests import rows

ZERO = Prompting()
FIVE = Prompting(shots=[0, 5])


def _xnli_setup(n: int = 6):
    bench = rows.one_language("afrixnli")
    test = [rows.xnli(i, label=i % 3) for i in range(n)]
    val = [rows.xnli(100 + i, label=i % 3) for i in range(20)]
    return bench, rows.loader_for(bench, {"test": test, "validation": val})


# --- the backend contract --------------------------------------------------------------------


def test_prompt_renders_as_chat_messages_or_raw_text() -> None:
    prompt = Prompt("Q2", (Turn("Q1", "A1"),))
    assert prompt.messages() == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    assert prompt.raw() == "Q1 A1\n\nQ2"


def test_mock_backend_is_deterministic_and_counts_calls() -> None:
    request = LoglikelihoodRequest(Prompt("q"), ("a", "b", "c"))
    one, two = MockBackend(), MockBackend()
    assert one.loglikelihood([request]) == two.loglikelihood([request])
    assert one.generate([Prompt("q")], Decoding()) == two.generate([Prompt("q")], Decoding())
    one.perplexity(["a b c"])
    assert one.calls == {"loglikelihood": 1, "generate": 1, "perplexity": 1}
    assert one.n_inferences == 3


# --- classification: log-likelihood ----------------------------------------------------------


def test_loglikelihood_scoring_picks_the_highest_scoring_option() -> None:
    bench, loader = _xnli_setup()
    always_first = MockBackend(choose=lambda prompt, conts: 0)
    preds = run_unit(bench, backend=always_first, loader=loader, prompting=ZERO, shots=0)["zul"]
    assert [p["pred"] for p in preds] == [0] * 6
    assert [p["correct"] for p in preds] == [True, False, False, True, False, False]
    first = preds[0]
    assert first["pred_label"] == "True" and first["gold_label"] == "True"
    assert len(first["scores"]) == 3 and first["parse_ok"] is True
    assert first["shots"] == 0 and first["language"] == "zul"
    assert always_first.calls == {"loglikelihood": 6}


def test_few_shot_prompts_carry_k_solved_examples_from_the_fewshot_split() -> None:
    bench, loader = _xnli_setup()
    seen: list[Prompt] = []
    backend = MockBackend(choose=lambda prompt, conts: seen.append(prompt) or 0)
    run_unit(bench, backend=backend, loader=loader, prompting=FIVE, shots=5)
    assert all(len(p.shots) == 5 for p in seen)
    assert len({p.shots for p in seen}) == 1  # fixed shots: identical for every example
    shot_queries = {t.user for t in seen[0].shots}
    assert all(
        "premise 1" in q and len(q.split("premise ")[1].split("\n")[0]) == 3 for q in shot_queries
    )
    assert {t.assistant for t in seen[0].shots} <= {"True", "Neither", "False"}
    assert ("afrixnli", "zul", "validation") in loader.requests


def test_zero_shot_never_loads_the_fewshot_split() -> None:
    bench, loader = _xnli_setup()
    run_unit(bench, backend=MockBackend(), loader=loader, prompting=ZERO, shots=0)
    assert loader.requests == [("afrixnli", "zul", "test")]


def test_limit_caps_examples_per_language() -> None:
    bench, loader = _xnli_setup(n=6)
    preds = run_unit(bench, backend=MockBackend(), loader=loader, prompting=ZERO, shots=0, limit=2)
    assert len(preds["zul"]) == 2


# --- classification: generate-then-parse ------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "options", "expected"),
    [
        ("B", ("A", "B", "C", "D"), 1),
        (" b. ", ("A", "B", "C", "D"), 1),
        ("B. Because the passage says so", ("A", "B", "C", "D"), 1),
        ("The answer is C", ("A", "B", "C", "D"), 2),
        ("**D**", ("A", "B", "C", "D"), 3),
        ("Neither", ("True", "Neither", "False"), 1),
        ("False. The premise contradicts it.", ("True", "Neither", "False"), 2),
        ("science/technology", ("science/technology", "travel"), 0),
        ("This text is about travel.", ("science/technology", "travel"), 1),
        ("book_flight", ("book_flight", "book_hotel"), 0),
        ("", ("A", "B", "C", "D"), UNPARSEABLE),
        ("I cannot tell", ("A", "B", "C", "D"), UNPARSEABLE),
        ("Either A or B", ("A", "B", "C", "D"), UNPARSEABLE),
        ("Banana", ("A", "B", "C", "D"), UNPARSEABLE),
    ],
)
def test_parse_choice(output: str, options: tuple[str, ...], expected: int) -> None:
    assert parse_choice(output, options) == expected


def test_generate_then_parse_counts_unparseable_as_wrong() -> None:
    bench, loader = _xnli_setup(n=3)  # gold labels 0, 1, 2
    answers = iter(["True", "no idea", "False"])
    backend = MockBackend(respond=lambda prompt: next(answers))
    prompting = Prompting(mc_scoring="generate")
    preds = run_unit(bench, backend=backend, loader=loader, prompting=prompting, shots=0)["zul"]
    assert [p["parse_ok"] for p in preds] == [True, False, True]
    assert [p["correct"] for p in preds] == [True, False, True]
    assert preds[1]["pred"] == UNPARSEABLE and preds[1]["pred_label"] == ""
    assert preds[1]["output"] == "no idea"
    assert backend.calls == {"generate": 3}


# --- generation and perplexity -----------------------------------------------------------------


def test_generation_runner_stores_output_and_reference() -> None:
    bench = rows.one_language("afrimgsm")
    loader = rows.loader_for(bench, {"test": [rows.mgsm(1, 18), rows.mgsm(2, 5)]})
    backend = MockBackend(respond=lambda prompt: "The answer is 18.")
    preds = run_unit(bench, backend=backend, loader=loader, prompting=ZERO, shots=0)["zul"]
    assert [(p["output"], p["reference"]) for p in preds] == [
        ("The answer is 18.", "18"),
        ("The answer is 18.", "5"),
    ]


def test_benchmark_max_new_tokens_reaches_the_backend() -> None:
    class Recording(MockBackend):
        def generate(self, prompts, decoding):
            self.seen = decoding
            return super().generate(prompts, decoding)

    bench = rows.one_language("afrimgsm")
    assert bench.max_new_tokens == 512
    backend = Recording()
    loader = rows.loader_for(bench, {"test": [rows.mgsm(1)]})
    run_unit(bench, backend=backend, loader=loader, prompting=ZERO, shots=0)
    assert backend.seen.max_new_tokens == 512


def test_perplexity_runner_reports_nll_and_token_counts() -> None:
    bench = rows.shipped_benchmark("english_perplexity")
    loader = rows.loader_for(bench, {"test": rows.WIKITEXT}, "eng")
    backend = MockBackend(nll_per_token=2.0)
    preds = run_unit(bench, backend=backend, loader=loader, prompting=ZERO, shots=0)["eng"]
    assert len(preds) == 2
    assert all(p["nll"] == 2.0 * p["n_tokens"] > 0 for p in preds)
    assert backend.calls == {"perplexity": 2}


# --- few-shot selection ----------------------------------------------------------------------


def _ex(i: int, group: str | None = None) -> Example:
    return Example(str(i), "zul", f"q{i}", f"a{i}", group=group)


def test_fixed_shots_are_deterministic_and_shuffled() -> None:
    pool = prepare_pool([_ex(i) for i in range(50)], "fixed")
    again = prepare_pool([_ex(i) for i in range(50)], "fixed")
    assert pool == again
    assert pool != [_ex(i) for i in range(50)]
    shots = choose_shots(_ex(999), pool, 5, "fixed")
    assert shots == tuple(Turn(e.query, e.target) for e in pool[:5])
    assert choose_shots(_ex(999), pool, 0, "fixed") == ()


def test_same_group_shots_come_from_the_same_subject_in_dataset_order() -> None:
    dev = [_ex(i, group="law" if i % 2 else "maths") for i in range(10)]
    pool = prepare_pool(dev, "same_group")
    shots = choose_shots(_ex(100, group="law"), pool, 5, "same_group")
    assert [t.user for t in shots] == ["q1", "q3", "q5", "q7", "q9"]


def test_other_group_shots_never_share_the_passage_or_the_item() -> None:
    bench = rows.one_language("belebele")
    test = [rows.belebele(q, passage=p) for p in range(8) for q in (1, 2)]
    loader = rows.loader_for(bench, {"test": test})
    seen: list[Prompt] = []
    backend = MockBackend(choose=lambda prompt, conts: seen.append(prompt) or 0)
    run_unit(bench, backend=backend, loader=loader, prompting=FIVE, shots=5)
    assert len(seen) == 16
    for prompt in seen:
        passage = prompt.query.split("\n")[0]
        assert len(prompt.shots) == 5
        assert all(turn.user.split("\n")[0] != passage for turn in prompt.shots)


def test_too_few_candidates_is_an_error() -> None:
    with pytest.raises(FewshotError, match="need 5"):
        choose_shots(_ex(0), [_ex(i) for i in range(3)], 5, "fixed")


def test_few_shot_without_a_fewshot_split_is_refused_unless_safe() -> None:
    bench = rows.one_language("afrixnli", fewshot_split=None)
    loader = rows.loader_for(bench, {"test": [rows.xnli(i) for i in range(10)]})
    with pytest.raises(TaskError, match="cannot safely take shots"):
        run_unit(bench, backend=MockBackend(), loader=loader, prompting=FIVE, shots=5)
    assert get_task("belebele").shot_policy == "other_group"
