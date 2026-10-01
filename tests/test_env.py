import os
from pathlib import Path

from eval.env import load_dotenv


def test_only_hf_variables_are_loaded(tmp_path: Path, monkeypatch) -> None:
    for name in ("HF_TOKEN", "HF_HOME", "TRACKER_API_KEY", "OTHER"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nTRACKER_API_KEY=secret\nexport HF_TOKEN='hf_abc'\nHF_HOME=\"/cache\"\nOTHER=1\n"
    )
    assert load_dotenv(env) == ["HF_TOKEN", "HF_HOME"]
    assert os.environ["HF_TOKEN"] == "hf_abc"
    assert os.environ["HF_HOME"] == "/cache"
    assert "TRACKER_API_KEY" not in os.environ
    assert "OTHER" not in os.environ
    for name in ("HF_TOKEN", "HF_HOME"):
        monkeypatch.delenv(name)


def test_existing_environment_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=from-file\n")
    assert load_dotenv(env) == []
    assert os.environ["HF_TOKEN"] == "from-shell"


def test_missing_file_is_fine(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / ".env") == []
