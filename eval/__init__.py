"""afrigemma-eval: evaluation core (config, caching, runners, scoring, storage)."""

from importlib.metadata import PackageNotFoundError, version


def harness_version() -> str:
    """The installed package version; the only source of `harness_version` (never typed by hand)."""
    try:
        return version("afrigemma-eval")
    except PackageNotFoundError as exc:  # pragma: no cover - only hit when not installed
        raise RuntimeError(
            "afrigemma-eval is not installed; run `uv pip install -e .` so the harness "
            "version can be read from package metadata"
        ) from exc
