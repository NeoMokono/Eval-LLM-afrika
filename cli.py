"""afrigemma-eval command line.

    afrigemma-eval plan --config configs/run.yaml [--force-rerun MODEL:BENCHMARK ... --reason "..."]

    afrigemma-eval run  --config configs/run.yaml [--force-rerun MODEL:BENCHMARK ... --reason "..."]

`plan` prints, for every (model, benchmark, shots) unit, whether it would be CACHED or RUN and
why, and runs nothing. `run` executes exactly the units `plan` marks as RUN, scores them per
language and stores records and predictions; cached units are not re-inferred.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.backends.base import ModelBackend
from eval.data import DataError, DataLoader, HFDataLoader
from eval.env import load_dotenv
from eval.fingerprint import FingerprintError, HfHubResolver, RevisionResolver, identity_cache_for
from eval.manifest import Manifest, ManifestError
from eval.pipeline import BackendFactory, execute
from eval.planner import Plan, PlanItem, build_plan, format_plan
from eval.registry import ConfigError, EvalConfig, ForceRerun, load_config


def _resolver() -> RevisionResolver:
    """Factory seam so tests can substitute an offline resolver."""
    return HfHubResolver()


def _parse_force(pairs: Sequence[str], reason: str | None) -> list[ForceRerun]:
    if pairs and not (reason and reason.strip()):
        raise ConfigError('--force-rerun needs --reason "..." (it is logged in the manifest)')
    if reason and not pairs:
        raise ConfigError("--reason only applies together with --force-rerun")
    forced = []
    for pair in pairs:
        model, sep, benchmark = pair.partition(":")
        if not sep or not model or not benchmark:
            raise ConfigError(f"--force-rerun expects MODEL:BENCHMARK, got {pair!r}")
        forced.append(ForceRerun(model=model, benchmark=benchmark, reason=reason or ""))
    return forced


def _load(config_path: str) -> EvalConfig:
    path = Path(config_path)
    for env_file in (Path.cwd() / ".env", path.resolve().parent.parent / ".env"):
        load_dotenv(env_file)
    return load_config(path)


def _backend_factory(config: EvalConfig) -> BackendFactory:
    """Factory seam: the real Transformers backend (imported lazily; needs the hf extra)."""

    def load(item: PlanItem) -> ModelBackend:
        from eval.backends.hf import HFBackend

        return HFBackend.load(
            item.model,
            config.models,
            item.identity,
            chat_template=config.run.prompting.chat_template,
        )

    return load


def _data_loader(plan: Plan) -> DataLoader:
    """Load every dataset at the revision the plan fingerprinted."""
    revisions = {
        i.benchmark.id: i.fingerprint.benchmark_version.rpartition("@")[2] for i in plan.items
    }
    return HFDataLoader(revisions)


def _plan(args: argparse.Namespace) -> tuple[EvalConfig, Manifest, Plan]:
    config = _load(args.config)
    forced = _parse_force(args.force_rerun, args.reason)
    problems = config.force_rerun_problems(forced)
    if problems:
        raise ConfigError("; ".join(problems))
    manifest = Manifest.load(config.results_path / "manifest.json")
    plan = build_plan(
        config,
        manifest,
        resolver=_resolver(),
        identity_cache=identity_cache_for(config.results_path),
        cli_force=forced,
    )
    return config, manifest, plan


def cmd_plan(args: argparse.Namespace) -> int:
    _, _, plan = _plan(args)
    print(format_plan(plan))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config, manifest, plan = _plan(args)
    print(format_plan(plan).replace(" Nothing was run.", ""), flush=True)
    if not plan.to_run:
        print("Everything is cached; no inference needed.")
        return 0
    summary = execute(
        config,
        plan,
        manifest,
        backend_factory=_backend_factory(config),
        loader=_data_loader(plan),
        log=lambda message: print(message, flush=True),
    )
    print(
        f"Run {plan.run_id}: {len(summary.completed)} unit(s) completed, "
        f"{len(summary.failed)} failed, {len(plan.cached)} served from cache."
    )
    for item, trace in summary.failed:
        print(
            f"--- {item.model.id} / {item.benchmark.id} / {item.shots}-shot\n{trace}",
            file=sys.stderr,
        )
    return 1 if summary.failed else 0


def _not_yet(args: argparse.Namespace) -> int:
    print(
        f"`{args.command}` is not implemented yet (Phase 6). Use `plan` or `run`.", file=sys.stderr
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afrigemma-eval", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    commands = {
        "plan": ("show what would run or be served from cache", cmd_plan),
        "run": ("run whatever `plan` marks as RUN, then score and store it", cmd_run),
    }
    for name, (help_text, func) in commands.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="path to run.yaml")
        p.add_argument(
            "--force-rerun",
            action="append",
            default=[],
            metavar="MODEL:BENCHMARK",
            help="rerun this pair regardless of cache (repeatable; needs --reason)",
        )
        p.add_argument("--reason", help="why the forced pairs must rerun")
        p.set_defaults(func=func)

    report = sub.add_parser("report", help="(Phase 6)")
    report.add_argument("--config", required=True)
    report.set_defaults(func=_not_yet)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ManifestError, FingerprintError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
