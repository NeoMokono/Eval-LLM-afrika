"""Hugging Face Transformers backend: full checkpoints and PEFT adapters on a parent model.

Memory note: Gemma's vocabulary has 262k entries, so materialising logits for every position
of a batch costs gigabytes. For log-likelihood scoring a pre-hook on the output head keeps only
the hidden states at the positions being scored, so logits are computed for those alone.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from eval.backends.base import LoglikelihoodRequest, ModelBackend, Prompt, TokenNLL
from eval.fingerprint import CheckpointIdentity
from eval.registry import Decoding, ModelsConfig, ModelSpec


class BackendError(Exception):
    pass


def _local_dir(spec: ModelSpec, identity: CheckpointIdentity) -> tuple[str, str | None]:
    """(path or repo id, revision) to hand to from_pretrained, pinned to the planned commit."""
    ref = spec.checkpoint_ref
    if ref.scheme == "hf":
        return ref.location, identity.resolved_revision
    if ref.scheme == "local":
        return ref.location, None
    raise BackendError(f"model {spec.id!r}: {ref.scheme}:// checkpoints cannot be loaded yet")


def _load_full_model(source: str, revision: str | None, dtype: torch.dtype) -> Any:
    import transformers

    config = transformers.AutoConfig.from_pretrained(source, revision=revision)
    arch = (getattr(config, "architectures", None) or [None])[0]
    cls = getattr(transformers, arch, None) if arch else None
    if cls is None:
        cls = transformers.AutoModelForCausalLM
    return cls.from_pretrained(source, revision=revision, dtype=dtype)


def verify_adapter_loaded(model: Any, adapter_dir: Path) -> int:
    """Check every tensor in the adapter file landed in the model unchanged.

    PEFT only warns when adapter keys do not match the parent's module names; a silently
    unloaded adapter would be evaluated as if it were the base model.
    """
    from safetensors.torch import load_file

    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    loaded = {
        name.replace(".default.", "."): param
        for name, param in model.named_parameters()
        if "lora_" in name
    }
    missing = sorted(set(saved) - set(loaded))
    extra = sorted(set(loaded) - set(saved))
    if missing or extra:
        raise BackendError(
            f"adapter {adapter_dir} did not load cleanly: {len(missing)} tensors in the file "
            f"have no module in the model (e.g. {missing[:2]}), {len(extra)} LoRA modules got "
            f"no weights (e.g. {extra[:2]})"
        )
    for name, tensor in saved.items():
        if not torch.equal(loaded[name].detach().cpu().to(tensor.dtype), tensor):
            raise BackendError(f"adapter tensor {name} differs from the file after loading")
    return len(saved)


class HFBackend(ModelBackend):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        chat_template: bool = True,
        max_batch_size: int = 64,
        max_batch_tokens: int = 48_000,
        ppl_window: int = 1024,
        ppl_stride: int = 512,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.ppl_window = ppl_window
        self.ppl_stride = ppl_stride
        self.device = next(model.parameters()).device
        if chat_template and not getattr(tokenizer, "chat_template", None):
            raise BackendError("chat_template=true but this tokenizer has no chat template")
        self._pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        models: ModelsConfig,
        identity: CheckpointIdentity,
        *,
        chat_template: bool = True,
        device: str | None = None,
        **options: Any,
    ) -> HFBackend:
        import transformers

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        source, revision = _local_dir(spec, identity)

        if spec.adapter_of is None:
            model = _load_full_model(source, revision, dtype)
            tokenizer_source, tokenizer_revision = source, revision
        else:
            from huggingface_hub import snapshot_download
            from peft import PeftModel

            if identity.parent is None:
                raise BackendError(f"model {spec.id!r}: adapter identity has no parent")
            parent_source, parent_revision = _local_dir(
                models.get(spec.adapter_of), identity.parent
            )
            base = _load_full_model(parent_source, parent_revision, dtype)
            adapter_dir = Path(
                source if revision is None else snapshot_download(source, revision=revision)
            )
            model = PeftModel.from_pretrained(base, str(adapter_dir))
            verify_adapter_loaded(model, adapter_dir)
            if (adapter_dir / "tokenizer.json").is_file():
                tokenizer_source, tokenizer_revision = str(adapter_dir), None
            else:
                tokenizer_source, tokenizer_revision = parent_source, parent_revision

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_source, revision=tokenizer_revision
        )
        return cls(model.to(device), tokenizer, chat_template=chat_template, **options)

    def close(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- rendering

    def render(self, prompt: Prompt) -> tuple[str, list[int]]:
        """The prompt as text and token ids, ending where the model's answer begins."""
        if not self.chat_template:
            text = prompt.raw()
            return text, self.tokenizer.encode(text, add_special_tokens=True)
        extra = (
            {"enable_thinking": False} if "enable_thinking" in self.tokenizer.chat_template else {}
        )
        text = self.tokenizer.apply_chat_template(
            prompt.messages(), tokenize=False, add_generation_prompt=True, **extra
        )
        return text, self.tokenizer.encode(text, add_special_tokens=False)

    def _continuation_ids(self, text: str, context_ids: list[int], continuation: str) -> list[int]:
        # Raw prompts take the answer after one space (lm-eval's target delimiter); chat prompts
        # start the model turn directly with it.
        piece = continuation if self.chat_template else " " + continuation
        whole = self.tokenizer.encode(text + piece, add_special_tokens=not self.chat_template)
        if whole[: len(context_ids)] == context_ids and len(whole) > len(context_ids):
            return whole[len(context_ids) :]
        ids = self.tokenizer.encode(piece, add_special_tokens=False)
        if not ids:
            raise BackendError(f"continuation {continuation!r} encodes to no tokens")
        return ids

    # ----------------------------------------------------------------- batching

    def _batches(self, lengths: Sequence[int]) -> Iterator[list[int]]:
        """Indices grouped longest-first under both the size and the padded-token budget."""
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        batch: list[int] = []
        for i in order:
            width = lengths[batch[0]] if batch else lengths[i]
            if batch and (
                len(batch) >= self.max_batch_size
                or (len(batch) + 1) * width > self.max_batch_tokens
            ):
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def _pad(
        self, seqs: Sequence[Sequence[int]], *, left: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self._pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for row, seq in enumerate(seqs):
            span = slice(width - len(seq), width) if left else slice(0, len(seq))
            ids[row, span] = torch.tensor(seq, dtype=torch.long)
            mask[row, span] = 1
        return ids.to(self.device), mask.to(self.device)

    @contextlib.contextmanager
    def _only_positions(self, positions: torch.Tensor) -> Iterator[None]:
        """While active, the output head only sees hidden states at `positions` (batch, k)."""
        head = self.model.get_output_embeddings()

        def select(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            hidden = args[0]
            index = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            return (hidden.gather(1, index), *args[1:])

        handle = head.register_forward_pre_hook(select)
        try:
            yield
        finally:
            handle.remove()

    def _score_rows(
        self, seqs: list[list[int]], targets: list[list[tuple[int, int]]], rows: list[int]
    ) -> list[list[float]]:
        return self._score([seqs[i] for i in rows], [targets[i] for i in rows])

    @torch.inference_mode()
    def _score(
        self, seqs: list[list[int]], targets: list[list[tuple[int, int]]]
    ) -> list[list[float]]:
        """For each sequence, log P(token | prefix) for each (position, token) target, where
        `position` indexes the input token whose next-token distribution is read."""
        out: list[list[float]] = [[] for _ in seqs]
        for batch in self._batches([len(s) for s in seqs]):
            ids, mask = self._pad([seqs[i] for i in batch], left=False)
            k = max(len(targets[i]) for i in batch)
            positions = torch.zeros((len(batch), k), dtype=torch.long, device=self.device)
            tokens = torch.zeros((len(batch), k), dtype=torch.long, device=self.device)
            for row, i in enumerate(batch):
                for col, (pos, tok) in enumerate(targets[i]):
                    positions[row, col], tokens[row, col] = pos, tok
            try:
                with self._only_positions(positions):
                    logits = self.model(input_ids=ids, attention_mask=mask).logits
            except torch.cuda.OutOfMemoryError:
                # Halve the batch and retry; a single sequence that does not fit is fatal.
                if len(batch) == 1:
                    raise
                del ids, mask, positions, tokens
                torch.cuda.empty_cache()
                half = len(batch) // 2
                for part in (batch[:half], batch[half:]):
                    scored = self._score_rows(seqs, targets, part)
                    for i, values in zip(part, scored, strict=True):
                        out[i] = values
                continue
            logprobs = torch.log_softmax(logits.float(), dim=-1).gather(-1, tokens.unsqueeze(-1))
            values = logprobs.squeeze(-1).cpu().tolist()
            for row, i in enumerate(batch):
                out[i] = values[row][: len(targets[i])]
        return out

    # -------------------------------------------------------------- the interface

    def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[list[float]]:
        self.calls["loglikelihood"] += len(requests)
        results: list[list[float]] = [[0.0] * len(r.continuations) for r in requests]
        seqs: list[list[int]] = []
        targets: list[list[tuple[int, int]]] = []
        owners: list[tuple[int, int | None]] = []  # (request, option) or (request, None) = all

        for r_idx, request in enumerate(requests):
            text, ctx = self.render(request.prompt)
            conts = [self._continuation_ids(text, ctx, c) for c in request.continuations]
            if all(len(c) == 1 for c in conts):
                # Every option is one token: a single forward pass scores them all.
                seqs.append(ctx)
                targets.append([(len(ctx) - 1, c[0]) for c in conts])
                owners.append((r_idx, None))
                continue
            for o_idx, cont in enumerate(conts):
                seqs.append((ctx + cont)[:-1])
                targets.append([(len(ctx) - 1 + j, tok) for j, tok in enumerate(cont)])
                owners.append((r_idx, o_idx))

        for (r_idx, o_idx), scores in zip(owners, self._score(seqs, targets), strict=True):
            if o_idx is None:
                results[r_idx] = scores
            else:
                results[r_idx][o_idx] = float(sum(scores))
        return results

    @torch.inference_mode()
    def generate(self, prompts: Sequence[Prompt], decoding: Decoding) -> list[str]:
        self.calls["generate"] += len(prompts)
        encoded = [self.render(p)[1] for p in prompts]
        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": decoding.temperature, "top_p": decoding.top_p}
            if decoding.strategy == "sample"
            else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        )
        outputs = [""] * len(prompts)
        for batch in self._batches([len(e) + decoding.max_new_tokens for e in encoded]):
            ids, mask = self._pad([encoded[i] for i in batch], left=True)
            generated = self.model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=decoding.max_new_tokens,
                pad_token_id=self._pad_id,
                **sampling,
            )
            texts = self.tokenizer.batch_decode(
                generated[:, ids.shape[1] :], skip_special_tokens=True
            )
            for i, text in zip(batch, texts, strict=True):
                outputs[i] = text.strip()
        return outputs

    @torch.inference_mode()
    def perplexity(self, texts: Sequence[str]) -> list[TokenNLL]:
        """Strided sliding window: each token is scored once, with up to
        `ppl_window - ppl_stride` tokens of left context carried over between windows."""
        self.calls["perplexity"] += len(texts)
        results = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            nll, scored, done = 0.0, 0, 1  # token 0 (BOS) is context only
            while done < len(ids):
                end = min(done + self.ppl_stride, len(ids))
                begin = max(0, end - self.ppl_window)
                window = torch.tensor([ids[begin:end]], device=self.device)
                logits = self.model(input_ids=window).logits[0, :-1].float()
                targets = window[0, 1:]
                fresh = end - done  # only the last `fresh` targets are new
                nll += torch.nn.functional.cross_entropy(
                    logits[-fresh:], targets[-fresh:], reduction="sum"
                ).item()
                scored += fresh
                done = end
            results.append(TokenNLL(nll, scored))
        return results
