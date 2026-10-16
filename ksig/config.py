from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "site"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.toml", "rb") as fh:
        return tomllib.load(fh)
