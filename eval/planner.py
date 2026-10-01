"""Decide, for every (model, benchmark, shots) unit, whether it is served from cache or run.

A unit runs only if (a) it is forced (CLI --force-rerun, run.yaml force_rerun, or
force_rerun_all) with a reason, (b) no complete manifest entry exists for it, or (c) a complete
entry exists but under a different fingerprint. Everything else is CACHED.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from eval import harness_version
from eval.fingerprint import (
    CheckpointIdentity,
    Fingerprint,
    IdentityCache,
    RevisionResolver,
    benchmark_revision,
    changed_fields,
    checkpoint_identity,
    compute_fingerprint,
    effective_shots,
)
from eval.manifest import Manifest, ManifestEntry, ManifestKey
from eval.prompts import PromptTemplate, template_for
from eval.registry import BenchmarkSpec, EvalConfig, ForceRerun, ModelSpec


class DecisionKind(StrEnum):
    CACHED = "cached"
    NEW = "new"
    CHANGED = "changed"
    FORCED = "forced"


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    detail: str | None = None

    @property
    def runs(self) -> bool:
        return self.kind is not DecisionKind.CACHED

    @property
    def label(self) -> str:
        match self.kind:
            case DecisionKind.CACHED:
                return "CACHED"
            case DecisionKind.NEW:
                return f"RUN (new; {self.detail})" if self.detail else "RUN (new)"
            case DecisionKind.CHANGED:
                return f"RUN (fingerprint changed: {self.detail})"
            case DecisionKind.FORCED:
                return f"RUN (forced: {self.detail})"


@dataclass(frozen=True)
class PlanItem:
    model: ModelSpec
    benchmark: BenchmarkSpec
    shots: int
    identity: CheckpointIdentity
    fingerprint: Fingerprint
    decision: Decision
    cached: ManifestEntry | None

    @property
    def key(self) -> ManifestKey:
        return ManifestKey.of(self.model.id, self.benchmark.id, self.fingerprint)

    @property
    def rerun_reason(self) -> str | None:
        return self.decision.detail if self.decision.kind is DecisionKind.FORCED else None


@dataclass(frozen=True)
class Plan:
    run_id: str
    harness_version: str
    items: list[PlanItem]
    has_plain_baseline: bool

    @property
    def to_run(self) -> list[PlanItem]:
        return [i for i in self.items if i.decision.runs]

    @property
    def cached(self) -> list[PlanItem]:
        return [i for i in self.items if not i.decision.runs]


def _forced_reasons(
    config: EvalConfig, cli_force: Sequence[ForceRerun]
) -> dict[tuple[str, str], str]:
    reasons: dict[tuple[str, str], str] = {}
    if config.run.force_rerun_all:
        for m in config.selected_models:
            for b in config.selected_benchmarks:
                reasons[(m.id, b.id)] = "force_rerun_all"
    for f in config.run.force_rerun:
        reasons[(f.model, f.benchmark)] = f.reason
    for f in cli_force:  # the command line wins over run.yaml
        reasons[(f.model, f.benchmark)] = f.reason
    return reasons


def _decide(
    manifest: Manifest, item_key: ManifestKey, shots: int, fp: Fingerprint, forced: str | None
) -> tuple[Decision, ManifestEntry | None]:
    if forced is not None:
        return Decision(DecisionKind.FORCED, forced), None
    hit = manifest.complete(item_key)
    if hit is not None:
        return Decision(DecisionKind.CACHED), hit
    previous = manifest.latest_complete(item_key.model_id, item_key.benchmark_id, shots)
    if previous is not None:
        fields = changed_fields(previous.components, fp.components)
        return Decision(DecisionKind.CHANGED, ", ".join(fields) or "fingerprint"), None
    attempt = manifest.find(item_key)
    note = f"previous attempt {attempt.status}" if attempt is not None else None
    return Decision(DecisionKind.NEW, note), None


def build_plan(
    config: EvalConfig,
    manifest: Manifest,
    *,
    resolver: RevisionResolver,
    identity_cache: IdentityCache | None = None,
    cli_force: Sequence[ForceRerun] = (),
    templates: Callable[[BenchmarkSpec], PromptTemplate] = template_for,
) -> Plan:
    harness = harness_version()
    forced = _forced_reasons(config, cli_force)
    revisions = {
        b.id: benchmark_revision(b, resolver=resolver, cache=identity_cache)
        for b in config.selected_benchmarks
    }
    items: list[PlanItem] = []
    for model in config.selected_models:
        identity = checkpoint_identity(
            model, config.models, resolver=resolver, cache=identity_cache
        )
        for bench in config.selected_benchmarks:
            for shots in effective_shots(bench, config.run.prompting):
                fp = compute_fingerprint(
                    checkpoint_hash=identity.hash,
                    benchmark=bench,
                    revision=revisions[bench.id],
                    template=templates(bench),
                    prompting=config.run.prompting,
                    shots=shots,
                    limit=config.run.limit,
                    harness=harness,
                )
                key = ManifestKey.of(model.id, bench.id, fp)
                decision, hit = _decide(manifest, key, shots, fp, forced.get((model.id, bench.id)))
                items.append(PlanItem(model, bench, shots, identity, fp, decision, hit))
    return Plan(config.run.run_id, harness, items, config.models.has_plain_baseline)


def format_plan(plan: Plan) -> str:
    rows = [
        (i.model.id, i.model.role.value, i.benchmark.id, str(i.shots), i.decision.label)
        for i in plan.items
    ]
    header = ("MODEL", "ROLE", "BENCHMARK", "SHOTS", "DECISION")
    widths = [max(len(r[c]) for r in [header, *rows]) for c in range(4)]

    def line(r: tuple[str, ...]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(r[:4], widths, strict=True)) + "  " + r[4]

    plain = (
        "present"
        if plan.has_plain_baseline
        else "none -> comparisons against base only; recipe claims marked not yet verified"
    )
    out = [
        f"Plan for run {plan.run_id} (harness {plan.harness_version})",
        f"plain_baseline: {plain}",
        "",
        line(header),
        *(line(r) for r in rows),
        "",
    ]
    counts = {k: sum(i.decision.kind is k for i in plan.items) for k in DecisionKind}
    out.append(
        f"{len(plan.to_run)} to run ({counts[DecisionKind.NEW]} new, "
        f"{counts[DecisionKind.CHANGED]} changed, {counts[DecisionKind.FORCED]} forced), "
        f"{len(plan.cached)} cached. Nothing was run."
    )
    return "\n".join(out)
