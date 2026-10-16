"""Bygger den statiske nettsiden til site/."""
from __future__ import annotations

import json

from .config import SITE_DIR, TEMPLATE_DIR
from .signals import read_log


def build_site(research: dict, signals: dict) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"research": research, "signals": signals, "log": read_log(100)}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (TEMPLATE_DIR / "index.html").read_text().replace("__DATA_JSON__", data)
    (SITE_DIR / "index.html").write_text(html)
    (SITE_DIR / "research.json").write_text(json.dumps(research, ensure_ascii=False))
    (SITE_DIR / "signals.json").write_text(json.dumps(signals, ensure_ascii=False))
    (SITE_DIR / ".nojekyll").write_text("")
