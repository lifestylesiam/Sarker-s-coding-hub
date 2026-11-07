from pathlib import Path

import pytest

from quantlab.settings import load_config, load_settings

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Tests must never pick up real keys or env overrides from the developer's shell."""
    import os

    for name in list(os.environ):
        if name.startswith("QUANTLAB") or name.startswith("ALPACA") or name.startswith("APCA"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings():
    return load_settings(CONFIG_DIR, overrides={"data.provider": "synthetic", "credentials.dotenv_path": None})


@pytest.fixture
def config_dir():
    return CONFIG_DIR


@pytest.fixture
def impact_cfg():
    return load_config("impact", CONFIG_DIR)
