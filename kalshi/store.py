"""Lagring av Kalshi-øyeblikksbilder (egen mappe, deles ikke med ksig)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAPSHOTS = DATA / "snapshots.csv"
LATEST = DATA / "latest.json"
PUBLIC = ROOT / "public"


def present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return str(v) != ""


def load() -> pd.DataFrame:
    if not SNAPSHOTS.exists():
        return pd.DataFrame()
    return pd.read_csv(SNAPSHOTS)


def save(df: pd.DataFrame) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    df.sort_values("date").to_csv(SNAPSHOTS, index=False)


def upsert(row: dict) -> None:
    """Én rad per UTC-dato. Nyeste øyeblikksbilde for en dato vinner, men en kjent fasit beholdes."""
    df = load()
    if not df.empty and row["date"] in set(df["date"]):
        old = df[df["date"] == row["date"]].iloc[0].to_dict()
        if old.get("wk_event") == row.get("wk_event") and present(old.get("wk_outcome")):
            row = {**row, "wk_outcome": old["wk_outcome"]}
        df = df[df["date"] != row["date"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save(df)


def save_latest(detail: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(detail, ensure_ascii=False, indent=1))


def load_latest() -> dict | None:
    return json.loads(LATEST.read_text()) if LATEST.exists() else None
