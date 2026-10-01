"""afrigemma-eval command line.

    afrigemma-eval plan --config configs/run.yaml [--force-rerun MODEL:BENCHMARK ... --reason "..."]

`plan` prints, for every (model, benchmark, shots) unit, whether it would be CACHED or RUN and
why, and runs nothing. `run` and `report` arrive in Phase 7.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.env import load_dotenv
from eval.fingerprint import FingerprintError, HfHubResolver, RevisionResolver, identity_cache_for
from eval.manifest import Manifest, ManifestError
from eval.planner import build_plan, format_plan
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


def cmd_plan(args: argparse.Namespace) -> int:
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
    print(format_plan(plan))
    return 0


def _not_yet(args: argparse.Namespace) -> int:
    print(f"`{args.command}` is not implemented yet (Phase 7). Use `plan`.", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afrigemma-eval", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="show what would run or be served from cache")
    plan.add_argument("--config", required=True, help="path to run.yaml")
    plan.add_argument(
        "--force-rerun",
        action="append",
        default=[],
        metavar="MODEL:BENCHMARK",
        help="rerun this pair regardless of cache (repeatable; needs --reason)",
    )
    plan.add_argument("--reason", help="why the forced pairs must rerun")
    plan.set_defaults(func=cmd_plan)

    for name in ("run", "report"):
        p = sub.add_parser(name, help="(Phase 7)")
        p.add_argument("--config", required=True)
        p.set_defaults(func=_not_yet)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ManifestError, FingerprintError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
