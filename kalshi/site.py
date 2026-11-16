"""Bygger Kalshi-fanen som en egen statisk side (kalshi/public/index.html)."""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import store

TEMPLATE = Path(__file__).resolve().parent / "templates" / "index.html"


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def evaluation(df: pd.DataFrame) -> dict:
    """Hvor godt har ukemarkedet truffet? Bare øyeblikksbilder der fredagens fasit er kjent."""
    if df.empty or "wk_outcome" not in df:
        return {"n": 0}
    e = df[pd.to_numeric(df["wk_outcome"], errors="coerce").notna() & df["wk_pup"].notna()].copy()
    if e.empty:
        return {"n": 0}
    e["wk_outcome"] = e["wk_outcome"].astype(float)
    up = (e["wk_outcome"] > e["spot"]).astype(float)
    brier = float(((e["wk_pup"] - up) ** 2).mean())
    hi, lo = e["wk_pup"] >= 0.5, e["wk_pup"] < 0.5
    return {
        "n": int(len(e)),
        "events": int(e["wk_event"].nunique()),
        "share_up": float(up.mean()),
        "avg_pup": float(e["wk_pup"].mean()),
        "brier": brier,
        "brier_flat": 0.25,
        "skill": 1 - brier / 0.25,
        "coverage80": float(((e["wk_outcome"] >= e["wk_q10"]) & (e["wk_outcome"] <= e["wk_q90"])).mean()),
        "n_hi": int(hi.sum()), "up_when_hi": float(up[hi].mean()) if hi.any() else None,
        "n_lo": int(lo.sum()), "up_when_lo": float(up[lo].mean()) if lo.any() else None,
    }


def build_site(out: str | Path | None = None) -> Path:
    out = Path(out) if out else store.PUBLIC
    out.mkdir(parents=True, exist_ok=True)
    df = store.load()
    rows = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")] if not df.empty else []
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "latest": store.load_latest(),
        "history": rows,
        "evaluation": evaluation(df),
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (out / "index.html").write_text(TEMPLATE.read_text().replace("__DATA_JSON__", data))
    if store.SNAPSHOTS.exists():
        shutil.copy(store.SNAPSHOTS, out / "snapshots.csv")
    if store.LATEST.exists():
        shutil.copy(store.LATEST, out / "latest.json")
    return out
