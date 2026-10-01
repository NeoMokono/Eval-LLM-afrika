"""Load and validate configs/models.yaml, configs/benchmarks.yaml and configs/run.yaml.

Everything a run depends on is checked here, before any model is loaded, and errors name the
file and the offending field. Role rules (invariant 1):

* exactly one `role: base` -- the checkpoint every target is compared against, and the one any
  adapter target was trained from;
* zero or one `role: plain_baseline` -- never inferred, never substituted by base;
* any number of extra `role: reference` checkpoints (e.g. the pretrained Gemma), compared
  alongside but never treated as base or plain baseline.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from eval.languages import UnknownLanguageError, normalize


class ConfigError(Exception):
    """A config file is missing, malformed or inconsistent. The message is meant for humans."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_slug(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
        raise ValueError(f"{value!r} must be lowercase letters, digits, '.', '_' or '-'")
    return value


def _check_language(value: str) -> str:
    try:
        return normalize(value)
    except UnknownLanguageError as exc:
        raise ValueError(str(exc)) from exc


Slug = Annotated[str, AfterValidator(_check_slug)]
Language = Annotated[str, AfterValidator(_check_language)]
NonEmpty = Annotated[str, Field(min_length=1)]


def _no_duplicate_languages(raw: list[str], normalized: list[str]) -> None:
    seen: dict[str, str] = {}
    for original, code in zip(raw, normalized, strict=True):
        if code in seen:
            raise ValueError(
                f"languages {seen[code]!r} and {original!r} both normalise to {code!r}; "
                "list each language once"
            )
        seen[code] = original


# --------------------------------------------------------------------------- models.yaml


class Role(StrEnum):
    BASE = "base"
    PLAIN_BASELINE = "plain_baseline"
    REFERENCE = "reference"
    TARGET = "target"
    EXTERNAL = "external"


_SECTION_ROLES: dict[str, set[Role]] = {
    "reference_models": {Role.BASE, Role.PLAIN_BASELINE, Role.REFERENCE},
    "target_models": {Role.TARGET},
    "external_models": {Role.EXTERNAL},
}

_HF_REPO = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")
_TEXT_FIELD = re.compile(r"\w+(\[\])?(\.\w+(\[\])?)*")


class CheckpointRef(_Strict):
    """A parsed checkpoint location: `hf://org/name[@revision]`, `s3://bucket/key` or an
    absolute local path."""

    scheme: Literal["hf", "s3", "local"]
    location: str
    revision: str | None = None

    @classmethod
    def parse(cls, uri: str) -> CheckpointRef:
        if uri.startswith("hf://"):
            repo, _, revision = uri.removeprefix("hf://").partition("@")
            if not _HF_REPO.fullmatch(repo):
                raise ValueError(f"{uri!r}: expected hf://<org>/<name>[@revision]")
            return cls(scheme="hf", location=repo, revision=revision or None)
        if uri.startswith("s3://"):
            if len(uri) <= len("s3://") or "/" not in uri.removeprefix("s3://"):
                raise ValueError(f"{uri!r}: expected s3://<bucket>/<key>")
            return cls(scheme="s3", location=uri)
        path = uri.removeprefix("file://")
        if not Path(path).is_absolute():
            raise ValueError(
                f"{uri!r}: local checkpoints must be absolute paths (or hf://, s3:// URIs)"
            )
        return cls(scheme="local", location=str(Path(path)))


class SFTData(_Strict):
    """The supervised training data behind a fine-tuned model; drives decontamination status."""

    version: NonEmpty
    path: NonEmpty
    text_fields: list[str] = Field(default_factory=lambda: ["messages[].content"], min_length=1)
    languages: list[str] = Field(default_factory=list)

    @field_validator("text_fields")
    @classmethod
    def _text_field_syntax(cls, fields: list[str]) -> list[str]:
        for f in fields:
            if not _TEXT_FIELD.fullmatch(f):
                raise ValueError(f"text field {f!r} must look like 'text' or 'messages[].content'")
        return fields

    @field_validator("languages")
    @classmethod
    def _languages(cls, raw: list[str]) -> list[str]:
        normalized = [_check_language(code) for code in raw]
        _no_duplicate_languages(raw, normalized)
        return normalized


class ModelSpec(_Strict):
    id: Slug
    role: Role
    checkpoint: NonEmpty
    adapter_of: str | None = None
    description: str | None = None
    sft_data: SFTData | None = None

    @field_validator("checkpoint")
    @classmethod
    def _parse_checkpoint(cls, value: str) -> str:
        CheckpointRef.parse(value)
        return value

    @model_validator(mode="after")
    def _upstream_models_have_no_sft_data(self) -> Self:
        if self.sft_data is not None and self.role in {Role.BASE, Role.REFERENCE}:
            raise ValueError(
                f"model {self.id!r} has role {self.role.value!r}, which is an upstream checkpoint; "
                "sft_data only applies to models we fine-tuned (decontamination is "
                "not_applicable for upstream checkpoints)"
            )
        return self

    @property
    def checkpoint_ref(self) -> CheckpointRef:
        return CheckpointRef.parse(self.checkpoint)


class ModelsConfig(_Strict):
    reference_models: list[ModelSpec]
    target_models: list[ModelSpec] = Field(default_factory=list)
    external_models: list[ModelSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _role_rules(self) -> Self:
        for section, allowed in _SECTION_ROLES.items():
            for m in getattr(self, section):
                if m.role not in allowed:
                    names = ", ".join(sorted(r.value for r in allowed))
                    raise ValueError(
                        f"model {m.id!r} has role {m.role.value!r} but sits under {section}, "
                        f"which only takes: {names}"
                    )

        dupes = [i for i, n in Counter(m.id for m in self.all_models).items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate model ids: {', '.join(sorted(dupes))}")

        bases = [m.id for m in self.all_models if m.role is Role.BASE]
        if len(bases) != 1:
            found = ", ".join(bases) if bases else "none"
            raise ValueError(
                f"models.yaml needs exactly one model with role: base (found {found}); "
                "base is the checkpoint every target is compared against"
            )
        plains = [m.id for m in self.all_models if m.role is Role.PLAIN_BASELINE]
        if len(plains) > 1:
            raise ValueError(
                f"at most one model may have role: plain_baseline (found {', '.join(plains)})"
            )

        base = self.base
        plain = self.plain_baseline
        if plain is not None and plain.checkpoint_ref == base.checkpoint_ref:
            raise ValueError(
                f"plain_baseline {plain.id!r} points at the same checkpoint as base {base.id!r}; "
                "delete the plain_baseline entry instead of aliasing base"
            )

        ids = {m.id for m in self.all_models}
        for m in self.all_models:
            if m.adapter_of is None:
                continue
            if m.adapter_of not in ids:
                raise ValueError(f"model {m.id!r}: adapter_of {m.adapter_of!r} is not a model id")
            if m.role in {Role.TARGET, Role.PLAIN_BASELINE} and m.adapter_of != base.id:
                raise ValueError(
                    f"model {m.id!r} is an adapter of {m.adapter_of!r}, but base is {base.id!r}; "
                    "base must be the checkpoint our fine-tuned adapters were trained from"
                )
        return self

    @property
    def all_models(self) -> list[ModelSpec]:
        return [*self.reference_models, *self.target_models, *self.external_models]

    @property
    def base(self) -> ModelSpec:
        return next(m for m in self.all_models if m.role is Role.BASE)

    @property
    def plain_baseline(self) -> ModelSpec | None:
        """The plain SFT baseline, or None. Never falls back to base."""
        return next((m for m in self.all_models if m.role is Role.PLAIN_BASELINE), None)

    @property
    def has_plain_baseline(self) -> bool:
        return self.plain_baseline is not None

    @property
    def extra_references(self) -> list[ModelSpec]:
        return [m for m in self.all_models if m.role is Role.REFERENCE]

    def get(self, model_id: str) -> ModelSpec:
        for m in self.all_models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)


# ----------------------------------------------------------------------- benchmarks.yaml


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    PERPLEXITY = "perplexity"


class Bucket(StrEnum):
    SEEN_TASK = "seen_task"
    HELD_OUT = "held_out"
    RETENTION = "retention"


Metric = Literal["accuracy", "exact_match", "chrf++", "perplexity"]
ExtraMetric = Literal["macro_f1", "slot_f1"]

_METRICS_BY_TYPE: dict[TaskType, set[str]] = {
    TaskType.CLASSIFICATION: {"accuracy"},
    TaskType.GENERATION: {"exact_match", "chrf++"},
    TaskType.PERPLEXITY: {"perplexity"},
}
# Metrics with a meaningful chance level (invariant 6); the others must not carry one.
_CHANCE_METRICS = {"accuracy", "exact_match"}


class BenchmarkSpec(_Strict):
    id: Slug
    dataset: NonEmpty  # HF repo id, or "local:<absolute path>"
    revision: str | None = None  # commit SHA preferred; a branch/tag is resolved at plan time
    type: TaskType
    capability: Slug
    metric: Metric
    extra_metrics: list[ExtraMetric] = Field(default_factory=list)
    chance_level: float | None = None
    bucket: Bucket
    languages: list[str] = Field(min_length=1)
    config: str | None = None  # one HF config shared by all languages
    language_column: str | None = None  # filter rows of `config` on this column
    subsets: dict[str, str] = Field(default_factory=dict)  # language -> HF config / column value
    split: NonEmpty
    fewshot_split: str | None = None
    max_new_tokens: int | None = Field(default=None, gt=0)
    input_modalities: list[Literal["text", "image", "audio"]] = Field(default=["text"])
    license: str | None = None
    enabled: bool = True
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_version_shorthand(cls, data: Any) -> Any:
        """Accept the spec's `version: org/name@revision` as well as dataset + revision."""
        if isinstance(data, dict) and "version" in data:
            data = dict(data)
            if "dataset" in data or "revision" in data:
                raise ValueError("give either `version` or `dataset`/`revision`, not both")
            dataset, _, revision = str(data.pop("version")).partition("@")
            data["dataset"] = dataset
            data["revision"] = revision or None
        return data

    @field_validator("languages")
    @classmethod
    def _languages(cls, raw: list[str]) -> list[str]:
        normalized = [_check_language(code) for code in raw]
        _no_duplicate_languages(raw, normalized)
        return normalized

    @field_validator("subsets")
    @classmethod
    def _subset_keys(cls, raw: dict[str, str]) -> dict[str, str]:
        keys = [_check_language(k) for k in raw]
        _no_duplicate_languages(list(raw), keys)
        return dict(zip(keys, raw.values(), strict=True))

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.metric not in _METRICS_BY_TYPE[self.type]:
            allowed = ", ".join(sorted(_METRICS_BY_TYPE[self.type]))
            raise ValueError(
                f"{self.type.value} benchmarks use metric {allowed}, not {self.metric}"
            )
        if self.metric in _CHANCE_METRICS:
            if self.chance_level is None:
                raise ValueError(f"metric {self.metric} requires chance_level (invariant 6)")
            if not 0.0 <= self.chance_level < 1.0:
                raise ValueError("chance_level must be in [0, 1)")
        elif self.chance_level is not None:
            raise ValueError(f"metric {self.metric} has no chance level; remove chance_level")
        if self.extra_metrics and self.type is not TaskType.CLASSIFICATION:
            raise ValueError("extra_metrics (macro_f1, slot_f1) only apply to classification")

        if self.dataset.startswith("local:"):
            if not Path(self.dataset.removeprefix("local:")).is_absolute():
                raise ValueError("local datasets must be given as local:<absolute path>")
        else:
            if not _HF_REPO.fullmatch(self.dataset):
                raise ValueError(f"dataset {self.dataset!r} is not an HF repo id or local:<path>")
            if not self.revision:
                raise ValueError("HF datasets need a revision (commit SHA, or a ref to resolve)")

        missing = set(self.subsets) - set(self.languages)
        if missing:
            raise ValueError(f"subsets name languages not in `languages`: {sorted(missing)}")
        for code, subset in self.subsets.items():
            try:
                tagged = normalize(subset)
            except UnknownLanguageError:
                continue  # not a language tag (e.g. "eng-zul", "Zulu_test")
            if tagged != code:
                raise ValueError(f"subset {subset!r} is {tagged!r}, but is listed under {code!r}")
        values = Counter(self.subsets.values())
        shared = sorted(v for v, n in values.items() if n > 1)
        if shared:
            raise ValueError(f"several languages map to the same subset {shared}; double count")
        if self.language_column and not self.config:
            raise ValueError("language_column filters rows within `config`; set config too")
        if self.fewshot_split is not None and self.fewshot_split == self.split:
            raise ValueError("fewshot_split must differ from the scored split (no test leakage)")
        return self

    @property
    def version(self) -> str:
        return f"{self.dataset}@{self.revision}" if self.revision else self.dataset

    def subset_for(self, language: str) -> str:
        """The HF config name (or `language_column` value) holding `language`."""
        code = normalize(language)
        if code not in self.languages:
            raise KeyError(f"{self.id} has no language {code!r}")
        return self.subsets.get(code, code)


class BenchmarksConfig(_Strict):
    benchmarks: list[BenchmarkSpec]

    @model_validator(mode="after")
    def _unique_and_comparable(self) -> Self:
        dupes = [i for i, n in Counter(b.id for b in self.benchmarks).items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate benchmark ids: {', '.join(sorted(dupes))}")
        metric_of: dict[str, BenchmarkSpec] = {}
        for b in self.benchmarks:
            first = metric_of.setdefault(b.capability, b)
            if first.metric != b.metric:
                raise ValueError(
                    f"capability {b.capability!r} mixes metrics ({first.id}: {first.metric}, "
                    f"{b.id}: {b.metric}); rollups only average benchmarks on the same scale"
                )
        return self

    def get(self, benchmark_id: str) -> BenchmarkSpec:
        for b in self.benchmarks:
            if b.id == benchmark_id:
                return b
        raise KeyError(benchmark_id)


# ------------------------------------------------------------------------------ run.yaml


class Decoding(_Strict):
    strategy: Literal["greedy", "sample"] = "greedy"
    temperature: float | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_new_tokens: int = Field(default=256, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        return {"strategy": data} if isinstance(data, str) else data

    @model_validator(mode="after")
    def _params_match_strategy(self) -> Self:
        if self.strategy == "greedy" and (self.temperature is not None or self.top_p is not None):
            raise ValueError("greedy decoding takes no temperature/top_p")
        if self.strategy == "sample" and self.temperature is None:
            raise ValueError("sampled decoding needs a temperature")
        return self


class Prompting(_Strict):
    shots: list[Annotated[int, Field(ge=0)]] = Field(default=[0], min_length=1)
    headline_shots: int = 0
    chat_template: bool = True
    mc_scoring: Literal["loglikelihood", "generate"] = "loglikelihood"
    decoding: Decoding = Field(default_factory=Decoding)

    @model_validator(mode="after")
    def _headline_in_shots(self) -> Self:
        if len(set(self.shots)) != len(self.shots):
            raise ValueError("shots contains duplicates")
        if self.headline_shots not in self.shots:
            raise ValueError(f"headline_shots={self.headline_shots} is not one of shots")
        return self


class Thresholds(_Strict):
    """All score thresholds are in points on a 0-100 scale (accuracy 0.341 -> 34.1)."""

    noise_threshold: float = Field(default=1.0, ge=0)
    retention_tolerance: float | None = Field(default=None, ge=0)
    perplexity_tolerance_pct: float | None = Field(default=None, ge=0)
    near_chance_margin: float = Field(default=2.0, ge=0)


class ConfusionTriggers(_Strict):
    """Fractions of predictions (0-1), plus a z-score for outlier languages."""

    class_collapse_pct: float = Field(default=0.6, gt=0, le=1)
    position_bias_pct: float = Field(default=0.5, gt=0, le=1)
    outlier_z: float = Field(default=2.0, gt=0)


class Bootstrap(_Strict):
    n_resamples: int = Field(default=1000, gt=0)
    seed: int = 0
    confidence: float = Field(default=0.95, gt=0, lt=1)


class ForceRerun(_Strict):
    model: NonEmpty
    benchmark: NonEmpty
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("force_rerun needs a non-empty reason; it is logged in the manifest")
        return value.strip()


class RunConfig(_Strict):
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    models_file: str = "models.yaml"
    benchmarks_file: str = "benchmarks.yaml"
    models: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(min_length=1)
    prompting: Prompting = Field(default_factory=Prompting)
    force_rerun_all: bool = False
    force_rerun: list[ForceRerun] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    confusion: ConfusionTriggers = Field(default_factory=ConfusionTriggers)
    bootstrap: Bootstrap = Field(default_factory=Bootstrap)
    results_dir: str = "../results"  # relative to run.yaml
    reports_dir: str = "../reports"

    @model_validator(mode="before")
    @classmethod
    def _harness_version_is_not_configurable(cls, data: Any) -> Any:
        if isinstance(data, dict) and "harness_version" in data:
            raise ValueError(
                "remove harness_version from run.yaml: it is read from the installed "
                "afrigemma-eval package version so it cannot drift from the code"
            )
        return data

    @model_validator(mode="after")
    def _no_duplicates(self) -> Self:
        for name in ("models", "benchmarks"):
            dupes = [i for i, n in Counter(getattr(self, name)).items() if n > 1]
            if dupes:
                raise ValueError(f"{name} lists {', '.join(dupes)} more than once")
        pairs = Counter((f.model, f.benchmark) for f in self.force_rerun)
        dupes = [f"{m}:{b}" for (m, b), n in pairs.items() if n > 1]
        if dupes:
            raise ValueError(f"force_rerun lists {', '.join(dupes)} more than once")
        return self


# ------------------------------------------------------------------- cross-file config


class EvalConfig(BaseModel):
    """The three files, validated against each other. Build it with `load_config`."""

    model_config = ConfigDict(frozen=True)

    run: RunConfig
    models: ModelsConfig
    benchmarks: BenchmarksConfig
    config_dir: Path

    @model_validator(mode="after")
    def _cross_references(self) -> Self:
        known_models = {m.id for m in self.models.all_models}
        unknown = [m for m in self.run.models if m not in known_models]
        if unknown:
            raise ValueError(
                f"run.yaml models {unknown} are not in models.yaml "
                f"(known: {', '.join(sorted(known_models))})"
            )
        known_bench = {b.id for b in self.benchmarks.benchmarks}
        unknown = [b for b in self.run.benchmarks if b not in known_bench]
        if unknown:
            raise ValueError(
                f"run.yaml benchmarks {unknown} are not in benchmarks.yaml "
                f"(known: {', '.join(sorted(known_bench))})"
            )
        disabled = [b for b in self.run.benchmarks if not self.benchmarks.get(b).enabled]
        if disabled:
            raise ValueError(f"run.yaml selects disabled benchmarks {disabled}")
        problems = self.force_rerun_problems(self.run.force_rerun)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def force_rerun_problems(self, entries: Sequence[ForceRerun]) -> list[str]:
        """Why each forced (model, benchmark) pair is not part of this run, if it isn't."""
        selected = {m.id for m in self.selected_models}
        problems = []
        for f in entries:
            if f.model not in selected:
                problems.append(f"force_rerun model {f.model!r} is not part of this run")
            if f.benchmark not in self.run.benchmarks:
                problems.append(f"force_rerun benchmark {f.benchmark!r} is not part of this run")
        return problems

    @property
    def results_path(self) -> Path:
        """results_dir, resolved relative to run.yaml's directory."""
        return (self.config_dir / self.run.results_dir).resolve()

    @property
    def reports_path(self) -> Path:
        return (self.config_dir / self.run.reports_dir).resolve()

    @property
    def selected_models(self) -> list[ModelSpec]:
        """Every reference model (always compared; served from cache once run), then the
        targets/externals listed in run.yaml, in that order."""
        chosen = [m.id for m in self.models.reference_models]
        chosen += [m for m in self.run.models if m not in chosen]
        return [self.models.get(m) for m in chosen]

    @property
    def selected_benchmarks(self) -> list[BenchmarkSpec]:
        return [self.benchmarks.get(b) for b in self.run.benchmarks]


# ----------------------------------------------------------------------------- loading


_MERGE_TAG = "tag:yaml.org,2002:merge"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    # Only the mapping's own keys; `<<:` merge keys may legitimately be overridden.
    keys = [loader.construct_object(k) for k, _ in node.value if k.tag != _MERGE_TAG]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    if dupes:
        raise ConfigError(f"line {node.start_mark.line + 1}: duplicate keys {dupes}")
    return loader.construct_mapping(node)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(f"{path}: file not found")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _format_validation_error(path: Path | str, exc: ValidationError) -> str:
    lines = [f"{path}: {exc.error_count()} problem(s)"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<top level>"
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)


_M = TypeVar("_M", bound=BaseModel)


def _validate(model: type[_M], data: Any, path: Path | str) -> _M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from None


def check_adapter_parents(models: ModelsConfig) -> None:
    """For local PEFT adapters, check the parent recorded in adapter_config.json against
    `adapter_of`, so the declared base really is what the adapter was trained from."""
    for m in models.all_models:
        ref = m.checkpoint_ref
        if ref.scheme != "local":
            continue
        adapter_config = Path(ref.location) / "adapter_config.json"
        if not adapter_config.is_file():
            continue
        if m.adapter_of is None:
            raise ConfigError(
                f"model {m.id!r} is a PEFT adapter ({adapter_config}); "
                "set adapter_of to the model id of its parent checkpoint"
            )
        recorded = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
        parent = models.get(m.adapter_of).checkpoint_ref
        expected = parent.location
        if not recorded or recorded.rstrip("/").lower() != expected.rstrip("/").lower():
            raise ConfigError(
                f"model {m.id!r}: adapter_config.json says it was trained from {recorded!r}, "
                f"but adapter_of {m.adapter_of!r} points at {expected!r}"
            )


def load_config(run_path: str | Path, *, check_adapters: bool = True) -> EvalConfig:
    """Load run.yaml plus the models/benchmarks files it names (relative to run.yaml)."""
    run_path = Path(run_path)
    run = _validate(RunConfig, _read_yaml(run_path), run_path)
    config_dir = run_path.parent
    models_path = config_dir / run.models_file
    benchmarks_path = config_dir / run.benchmarks_file
    models = _validate(ModelsConfig, _read_yaml(models_path), models_path)
    benchmarks = _validate(BenchmarksConfig, _read_yaml(benchmarks_path), benchmarks_path)
    config = _validate(
        EvalConfig,
        {"run": run, "models": models, "benchmarks": benchmarks, "config_dir": config_dir},
        f"{run_path} (cross-file checks)",
    )
    if check_adapters:
        check_adapter_parents(models)
    return config
