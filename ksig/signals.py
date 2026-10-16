"""Dagens signaler og endringsdeteksjon.

Signalet beregnes fra siste LUKKEDE døgn med strategien som er satt for hver coin
i config.toml. Endringer måles mot den siste tilstanden som ble varslet
(state/positions.json), ikke bare mot gårsdagen. Slik blir en endring varslet
selv om en daglig kjøring skulle falle ut.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone

import pandas as pd

from . import strategies as S
from .config import STATE_DIR
from .data import load_prices, utc_today

STATE_FILE = STATE_DIR / "positions.json"
LOG_FILE = STATE_DIR / "signal_log.csv"
HISTORY_DAYS = 400


def pct(x: float) -> str:
    return f"{x:.0%}".replace("%", " %")


def describe_action(prev: float | None, new: float) -> str:
    if prev is None:
        return f"Startposisjon: {pct(new)}"
    if abs(new - prev) < 1e-9:
        return "Ingen endring"
    if new == 0:
        return f"SELG ALT ({pct(prev)} → 0 %)"
    if prev == 0:
        return f"KJØP til {pct(new)}"
    if new < prev:
        return f"SELG ned til {pct(new)} (fra {pct(prev)})"
    return f"KJØP opp til {pct(new)} (fra {pct(prev)})"


def load_state() -> dict | None:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if math.isnan(f) else f


def compute_signals(cfg: dict) -> dict:
    alerts = cfg.get("alerts", {})
    state = load_state()
    notified = (state or {}).get("targets", {})
    old_specs = (state or {}).get("strategies", {})
    yesterday = utc_today() - pd.Timedelta(days=1)
    rows = []
    for asset in cfg["assets"]:
        spec = asset["strategy"]
        df = load_prices(asset["symbol"])
        sig = S.run(df["Close"], spec)
        last = sig.iloc[-1]
        tgt = sig["target"]
        changes = tgt[tgt.diff().fillna(0).abs() > 1e-9]
        prev_notified = notified.get(asset["short"])
        target = float(last["target"])
        stale = int((yesterday - sig.index[-1]).days)
        notes = S.notes(last, spec, alerts.get("near_pct", 0.03))
        if stale >= alerts.get("stale_after_days", 2):
            notes.append(f"OBS: siste data er fra {sig.index[-1].date()} ({stale} døgn gammelt).")
        strategy_changed = bool(state) and old_specs.get(asset["short"]) not in (None, spec)
        if strategy_changed:
            notes.insert(0, f"Ny strategi for {asset['short']}: {S.label(spec)}.")
        lines = S.chart_lines(spec)
        h = sig.iloc[-HISTORY_DAYS:]
        rows.append({
            "symbol": asset["symbol"], "short": asset["short"], "name": asset["name"],
            "spec": spec, "label": S.label(spec), "rules": S.rules(spec),
            "date": str(sig.index[-1].date()),
            "close": float(last["close"]),
            "display": [{"name": n, "kind": k, "value": (bool(v) if k == "check" else _num(v))}
                        for n, k, v in S.display(last, spec)],
            "target": target,
            "prev_day_target": float(tgt.iloc[-2]) if len(tgt) > 1 else None,
            "notified_target": prev_notified,
            "changed": prev_notified is None or abs(prev_notified - target) > 1e-9,
            "strategy_changed": strategy_changed,
            "action": describe_action(prev_notified, target),
            "last_change": str(changes.index[-1].date()) if len(changes) else None,
            "stale_days": stale,
            "notes": notes,
            "history": {
                "dates": [str(d.date()) for d in h.index],
                "close": [round(float(v), 6) for v in h["close"]],
                "lines": [{"name": nm, "values": [None if pd.isna(v) else round(float(v), 6) for v in h[col]]}
                          for col, nm in lines],
                "target": [float(v) for v in h["target"]],
            },
        })
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "first_run": state is None,
        "params_changed": any(r["strategy_changed"] for r in rows),
        "assets": rows,
        "any_change": any(r["changed"] for r in rows),
        "max_stale_days": max(r["stale_days"] for r in rows),
    }


def save_state(sig: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "updated": sig["generated"],
        "data_date": max(r["date"] for r in sig["assets"]),
        "strategies": {r["short"]: r["spec"] for r in sig["assets"]},
        "targets": {r["short"]: r["target"] for r in sig["assets"]},
    }, indent=2, ensure_ascii=False))
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["data_date", "coin", "fra", "til", "pris", "handling"])
        for r in sig["assets"]:
            if r["changed"]:
                w.writerow([r["date"], r["short"],
                            "" if r["notified_target"] is None else r["notified_target"],
                            r["target"], f"{r['close']:.6g}", r["action"]])


def read_log(limit: int = 50) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as fh:
        rows = list(csv.DictReader(fh))
    return rows[-limit:][::-1]
