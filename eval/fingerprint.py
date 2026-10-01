"""Checkpoint identity and the per-(model, benchmark, shots) fingerprint used as the cache key.

The fingerprint is a SHA-256 over canonical JSON of everything that can change a result:
checkpoint identity, benchmark id + resolved revision (+ the subsets/split actually read),
prompt template id + text hash, shot count, decoding settings and the harness version.
Components that cannot affect a given task type (decoding for log-likelihood scoring, the
chat template for perplexity) are left out so toggling them does not cause pointless reruns.
The training-data version is deliberately not part of it (see decontamination).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from eval import harness_version
from eval.atomic import atomic_write_text
from eval.prompts import PromptTemplate
from eval.registry import BenchmarkSpec, ModelsConfig, ModelSpec, Prompting, TaskType

WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt"})
_SHA = re.compile(r"[0-9a-f]{40}")


class FingerprintError(Exception):
    """A checkpoint or dataset identity could not be established."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------- HF revisions


class RevisionResolver(Protocol):
    def resolve(self, kind: Literal["model", "dataset"], repo: str, revision: str) -> str:
        """Return the commit SHA that `revision` (branch, tag or SHA) points at."""
        ...


class HfHubResolver:
    """Resolves refs to commit SHAs with the HF Hub API (HF_TOKEN is read from the env)."""

    def __init__(self) -> None:
        self._memo: dict[tuple[str, str, str], str] = {}

    def resolve(self, kind: Literal["model", "dataset"], repo: str, revision: str) -> str:
        if _SHA.fullmatch(revision):
            return revision
        key = (kind, repo, revision)
        if key not in self._memo:
            from huggingface_hub import HfApi

            api = HfApi()
            info_fn = api.model_info if kind == "model" else api.dataset_info
            try:
                sha = info_fn(repo, revision=revision).sha
            except Exception as exc:
                raise FingerprintError(
                    f"could not resolve {kind} {repo}@{revision} to a commit SHA: {exc}"
                ) from exc
            if not sha:
                raise FingerprintError(f"HF Hub returned no commit SHA for {repo}@{revision}")
            self._memo[key] = sha
        return self._memo[key]


# ------------------------------------------------------------------ local identity


class IdentityCache:
    """Per-path cache of local identity hashes, keyed by a cheap stat() signature, so the small
    files that are content-hashed are only read again when something on disk changed."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, dict[str, str]] = {}
        if path is not None and path.is_file():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}  # a cache: safe to rebuild

    def get(self, key: str, signature: str) -> str | None:
        hit = self._data.get(key)
        return hit["identity"] if hit and hit.get("signature") == signature else None

    def put(self, key: str, signature: str, identity: str) -> None:
        self._data[key] = {"signature": signature, "identity": identity}
        if self.path is not None:
            atomic_write_text(self.path, json.dumps(self._data, indent=1, sort_keys=True))


def local_identity(path: Path, cache: IdentityCache | None = None) -> str:
    """Identity of a local checkpoint directory (or single file).

    Weight files contribute name, size and mtime (never their multi-GB contents); every other
    top-level file (configs, adapter_config.json, tokenizer, chat template) is content-hashed.
    Subdirectories are ignored.
    """
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith("."))
    else:
        raise FingerprintError(f"checkpoint path does not exist: {path}")
    if path.is_dir() and not any(f.suffix in WEIGHT_SUFFIXES for f in files):
        raise FingerprintError(f"no weight files ({', '.join(sorted(WEIGHT_SUFFIXES))}) in {path}")

    stats = {f: f.stat() for f in files}
    signature = sha256_hex(
        canonical_json([[f.name, s.st_size, s.st_mtime_ns] for f, s in stats.items()])
    )
    key = str(path.resolve())
    if cache is not None and (hit := cache.get(key, signature)) is not None:
        return hit

    parts = []
    for f, s in stats.items():
        if f.suffix in WEIGHT_SUFFIXES:
            parts.append([f.name, "meta", s.st_size, s.st_mtime_ns])
        else:
            parts.append([f.name, "sha256", _sha256_file(f)])
    identity = sha256_hex(canonical_json(parts))
    if cache is not None:
        cache.put(key, signature, identity)
    return identity


@dataclass(frozen=True)
class CheckpointIdentity:
    hash: str
    resolved_revision: str | None = None  # HF commit SHA to load, for hf:// checkpoints
    parent: CheckpointIdentity | None = None


def checkpoint_identity(
    model: ModelSpec,
    models: ModelsConfig,
    *,
    resolver: RevisionResolver,
    cache: IdentityCache | None = None,
) -> CheckpointIdentity:
    """Identity of a model checkpoint; adapters also fold in their parent's identity, so an
    upstream base update invalidates every adapter evaluated on top of it."""
    ref = model.checkpoint_ref
    own: dict[str, Any]
    resolved = None
    if ref.scheme == "hf":
        resolved = resolver.resolve("model", ref.location, ref.revision or "main")
        own = {"hf": ref.location, "commit": resolved}
    elif ref.scheme == "local":
        own = {"local": local_identity(Path(ref.location), cache)}
    else:
        raise FingerprintError(
            f"model {model.id!r}: identity for {ref.scheme}:// checkpoints is not implemented yet "
            "(needs object ETags); copy it locally or add s3 support"
        )
    parent = None
    if model.adapter_of is not None:
        parent = checkpoint_identity(
            models.get(model.adapter_of), models, resolver=resolver, cache=cache
        )
        own["adapter_of"] = parent.hash
    return CheckpointIdentity(sha256_hex(canonical_json(own)), resolved, parent)


def benchmark_revision(
    benchmark: BenchmarkSpec,
    *,
    resolver: RevisionResolver,
    cache: IdentityCache | None = None,
) -> str:
    """The pinned data version: an HF commit SHA, or a content identity for local files."""
    if benchmark.dataset.startswith("local:"):
        return "local:" + local_identity(Path(benchmark.dataset.removeprefix("local:")), cache)
    assert benchmark.revision is not None  # enforced by the schema
    return resolver.resolve("dataset", benchmark.dataset, benchmark.revision)


# ------------------------------------------------------------------------- fingerprint


@dataclass(frozen=True)
class Fingerprint:
    value: str  # the full cache key
    checkpoint_hash: str
    benchmark_version: str  # dataset@resolved-revision
    config_fingerprint: str  # everything except checkpoint and data version
    components: dict[str, Any] = field(hash=False)


def effective_shots(benchmark: BenchmarkSpec, prompting: Prompting) -> list[int]:
    """Shot counts actually run: perplexity has no prompt, so it runs once."""
    return [0] if benchmark.type is TaskType.PERPLEXITY else list(prompting.shots)


def compute_fingerprint(
    *,
    checkpoint_hash: str,
    benchmark: BenchmarkSpec,
    revision: str,
    template: PromptTemplate,
    prompting: Prompting,
    shots: int,
    harness: str | None = None,
) -> Fingerprint:
    is_mc = benchmark.type is TaskType.CLASSIFICATION
    uses_prompt = benchmark.type is not TaskType.PERPLEXITY
    generates = benchmark.type is TaskType.GENERATION or (
        is_mc and prompting.mc_scoring == "generate"
    )
    decoding = None
    if generates:
        d = prompting.decoding
        decoding = {
            "strategy": d.strategy,
            "temperature": d.temperature,
            "top_p": d.top_p,
            "max_new_tokens": benchmark.max_new_tokens or d.max_new_tokens,
        }
    config = {
        "benchmark": {
            "id": benchmark.id,
            "config": benchmark.config,
            "language_column": benchmark.language_column,
            "subsets": {lang: benchmark.subset_for(lang) for lang in benchmark.languages},
            "split": benchmark.split,
            "fewshot_split": benchmark.fewshot_split,
            "metric": benchmark.metric,
            "extra_metrics": list(benchmark.extra_metrics),
        },
        "prompt": {
            "template_id": template.id if uses_prompt else None,
            "template_sha256": template.sha256 if uses_prompt else None,
            "chat_template": prompting.chat_template if uses_prompt else None,
            "mc_scoring": prompting.mc_scoring if is_mc else None,
        },
        "shots": shots,
        "decoding": decoding,
        "harness_version": harness if harness is not None else harness_version(),
    }
    version = f"{benchmark.dataset}@{revision}"
    components = {"checkpoint": checkpoint_hash, "benchmark_version": version, **config}
    return Fingerprint(
        value=sha256_hex(canonical_json(components)),
        checkpoint_hash=checkpoint_hash,
        benchmark_version=version,
        config_fingerprint=sha256_hex(canonical_json(config)),
        components=components,
    )


def changed_fields(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Dotted names of the fingerprint components that differ, e.g. ['decoding.max_new_tokens']."""

    def flatten(obj: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
        # Two levels: "decoding.max_new_tokens", but "benchmark.subsets" as a whole.
        if isinstance(obj, dict) and depth < 2:
            out: dict[str, Any] = {}
            for k, v in obj.items():
                out.update(flatten(v, f"{prefix}{k}.", depth + 1))
            return out
        return {prefix.rstrip("."): obj}

    a, b = flatten(old), flatten(new)
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def identity_cache_for(results_dir: Path) -> IdentityCache:
    return IdentityCache(results_dir / ".identity_cache.json")
