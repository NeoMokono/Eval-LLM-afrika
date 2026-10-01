"""Smoke test of the real Transformers backend on a tiny random Gemma-4 model.

`peft-internal-testing/tiny-random-gemma4-E2B` has the same architecture
(Gemma4ForConditionalGeneration) and tokenizer vocabulary as google/gemma-4-E2B-it, at 9 MB.
Skipped when torch/peft are not installed or the model cannot be fetched.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from eval.backends.base import LoglikelihoodRequest, Prompt, Turn  # noqa: E402
from eval.backends.hf import BackendError, HFBackend, _load_full_model  # noqa: E402
from eval.fingerprint import checkpoint_identity  # noqa: E402
from eval.registry import Decoding, ModelsConfig, Prompting  # noqa: E402
from eval.runners import run_unit  # noqa: E402
from tests import rows  # noqa: E402
from tests.conftest import FakeResolver  # noqa: E402

pytestmark = pytest.mark.smoke

REPO = "peft-internal-testing/tiny-random-gemma4-E2B"
SHA = "9dd5f84066264c0b82fe221bea8153c17308d09d"
# The tiny model ships no chat template; this mirrors Gemma-4's turn markers.
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|turn>"
    "{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n{{ m['content'] }}<turn|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC = Prompt("Is the sky blue?\nA. yes\nB. no\nAnswer:", (Turn("Is grass green?\nAnswer:", "A"),))
REQUESTS = [
    LoglikelihoodRequest(MC, ("A", "B")),  # single-token options: one forward pass
    LoglikelihoodRequest(Prompt("Topic of: rain falls"), ("science/technology", "sports")),
    LoglikelihoodRequest(Prompt("x " * 40), ("A", "B", "C", "D")),
]


@pytest.fixture(scope="module")
def tiny():
    try:
        import transformers

        model = _load_full_model(REPO, SHA, torch.float32)
        tokenizer = transformers.AutoTokenizer.from_pretrained(REPO, revision=SHA)
    except Exception as exc:  # offline, hub down, ...
        pytest.skip(f"tiny Gemma-4 model unavailable: {exc}")
    tokenizer.chat_template = CHAT_TEMPLATE
    return model.to(DEVICE), tokenizer


def _naive(backend: HFBackend, prompt: Prompt, continuation: str) -> float:
    """Reference: full logits for one unpadded sequence, no batching, no position slicing."""
    text, ctx = backend.render(prompt)
    cont = backend._continuation_ids(text, ctx, continuation)
    ids = torch.tensor([ctx + cont], device=backend.device)
    with torch.inference_mode():
        logprobs = torch.log_softmax(backend.model(input_ids=ids).logits[0].float(), -1)
    return sum(logprobs[len(ctx) - 1 + j, tok].item() for j, tok in enumerate(cont))


@pytest.mark.parametrize("chat", [True, False])
def test_loglikelihood_matches_naive_full_logits(tiny, chat: bool) -> None:
    backend = HFBackend(*tiny, chat_template=chat)
    got = backend.loglikelihood(REQUESTS)
    for request, scores in zip(REQUESTS, got, strict=True):
        expected = [_naive(backend, request.prompt, c) for c in request.continuations]
        assert scores == pytest.approx(expected, abs=1e-4)
    assert backend.calls == {"loglikelihood": 3}


def test_chat_rendering_uses_turns_and_ends_at_the_model_turn(tiny) -> None:
    text, ids = HFBackend(*tiny, chat_template=True).render(MC)
    assert text.startswith("<bos><|turn>user\nIs grass green?")
    assert "<|turn>model\nA<turn|>" in text
    assert text.endswith("<|turn>model\n")
    assert ids[0] == tiny[1].bos_token_id and ids.count(tiny[1].bos_token_id) == 1
    raw_text, raw_ids = HFBackend(*tiny, chat_template=False).render(MC)
    assert raw_text == MC.raw()
    assert raw_ids[0] == tiny[1].bos_token_id


def test_batching_and_padding_do_not_change_scores(tiny) -> None:
    together = HFBackend(*tiny).loglikelihood(REQUESTS)
    one_by_one = HFBackend(*tiny, max_batch_size=1)
    separate = [one_by_one.loglikelihood([r])[0] for r in REQUESTS]
    for a, b in zip(together, separate, strict=True):
        assert a == pytest.approx(b, abs=1e-4)


def test_greedy_generation_is_deterministic_and_bounded(tiny) -> None:
    backend = HFBackend(*tiny)
    prompts = [MC, Prompt("hello"), Prompt("a much longer prompt " * 10)]
    first = backend.generate(prompts, Decoding(max_new_tokens=6))
    assert first == backend.generate(prompts, Decoding(max_new_tokens=6))
    assert len(first) == 3 and all(isinstance(t, str) for t in first)
    assert all(len(tiny[1].encode(t, add_special_tokens=False)) <= 8 for t in first)
    alone = HFBackend(*tiny, max_batch_size=1).generate(prompts, Decoding(max_new_tokens=6))
    assert alone == first  # left padding does not change the output


def test_strided_perplexity_scores_every_token_once(tiny) -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 120
    n_ids = len(tiny[1].encode(text, add_special_tokens=True))
    strided = HFBackend(*tiny, ppl_window=256, ppl_stride=128).perplexity([text, "short text"])
    whole = HFBackend(*tiny, ppl_window=8192, ppl_stride=8192).perplexity([text])[0]
    assert strided[0].n_tokens == whole.n_tokens == n_ids - 1
    # Windows see less left context than one full pass, so NLLs agree closely, not exactly.
    assert strided[0].nll == pytest.approx(whole.nll, rel=0.01)
    assert strided[1].n_tokens == 2


@pytest.fixture(scope="module")
def adapter_setup(tiny, tmp_path_factory):
    """A LoRA adapter on the tiny model's language tower, saved the way training saves it."""
    torch.manual_seed(0)
    lora = peft.LoraConfig(
        r=4, target_modules=r".*language_model.*\.(q_proj|v_proj)", init_lora_weights=False
    )
    adapter_dir = tmp_path_factory.mktemp("adapter")
    peft.get_peft_model(_load_full_model(REPO, SHA, torch.float32), lora).save_pretrained(
        adapter_dir
    )
    models = ModelsConfig.model_validate(
        {
            "reference_models": [
                {"id": "tiny-base", "role": "base", "checkpoint": f"hf://{REPO}@{SHA}"}
            ],
            "target_models": [
                {
                    "id": "tiny-lora",
                    "role": "target",
                    "checkpoint": str(adapter_dir),
                    "adapter_of": "tiny-base",
                }
            ],
        }
    )
    return models, Path(adapter_dir)


def _load(models: ModelsConfig, model_id: str) -> HFBackend:
    spec = models.get(model_id)
    identity = checkpoint_identity(spec, models, resolver=FakeResolver())
    return HFBackend.load(spec, models, identity, chat_template=False, device=DEVICE)


def test_adapter_is_loaded_on_its_parent_and_changes_the_scores(adapter_setup) -> None:
    models, _ = adapter_setup
    base, adapted = _load(models, "tiny-base"), _load(models, "tiny-lora")
    assert type(base.model).__name__ == "Gemma4ForConditionalGeneration"
    assert type(adapted.model).__name__.startswith("Peft")
    assert base.loglikelihood(REQUESTS[:1]) != adapted.loglikelihood(REQUESTS[:1])


def test_adapter_that_does_not_match_the_parent_is_rejected(adapter_setup, tmp_path) -> None:
    from safetensors.torch import load_file, save_file

    models, adapter_dir = adapter_setup
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "adapter_config.json").write_text((adapter_dir / "adapter_config.json").read_text())
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    renamed = {k.replace("language_model", "language_tower"): v for k, v in tensors.items()}
    save_file(renamed, str(broken / "adapter_model.safetensors"))
    target = models.get("tiny-lora").model_copy(update={"checkpoint": str(broken)})
    patched = models.model_copy(update={"target_models": [target]})
    identity = checkpoint_identity(target, patched, resolver=FakeResolver())
    with pytest.raises(BackendError, match="did not load cleanly"):
        HFBackend.load(target, patched, identity, chat_template=False, device=DEVICE)


def test_chat_template_required_when_requested(tiny) -> None:
    import copy

    tokenizer = copy.copy(tiny[1])
    tokenizer.chat_template = None
    with pytest.raises(BackendError, match="no chat template"):
        HFBackend(tiny[0], tokenizer, chat_template=True)


def test_run_unit_end_to_end_on_a_real_model(tiny) -> None:
    backend = HFBackend(*tiny)
    bench = rows.one_language("afrixnli")
    loader = rows.loader_for(
        bench,
        {
            "test": [rows.xnli(i, label=i % 3) for i in range(6)],
            "validation": [rows.xnli(100 + i, label=i % 3) for i in range(8)],
        },
    )
    prompting = Prompting(shots=[0, 5])
    for shots in (0, 5):
        preds = run_unit(bench, backend=backend, loader=loader, prompting=prompting, shots=shots)
        assert len(preds["zul"]) == 6
        assert all(len(p["scores"]) == 3 and p["pred"] in (0, 1, 2) for p in preds["zul"])
        assert all(all(s < 0 for s in p["scores"]) for p in preds["zul"])
