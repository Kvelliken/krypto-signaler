"""Dagens signaler og endringsdeteksjon.

Signalet beregnes fra siste LUKKEDE døgn. Endringer måles mot den siste
tilstanden som ble varslet (state/positions.json), ikke bare mot gårsdagen.
Slik blir en endring varslet selv om en daglig kjøring skulle falle ut.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pandas as pd

from .config import STATE_DIR
from .data import load_prices, utc_today
from .strategy import Params, compute

STATE_FILE = STATE_DIR / "positions.json"
LOG_FILE = STATE_DIR / "signal_log.csv"


def pct(x: float) -> str:
    return f"{x:.0%}".replace("%", " %")


def describe_action(prev: float | None, new: float) -> str:
    if prev is None:
        return f"Startposisjon: {pct(new)}"
    if new == prev:
        return "Ingen endring"
    if new == 0:
        return f"SELG ALT ({pct(prev)} → 0 %): prisen har falt under SMA"
    if prev == 0:
        return f"KJØP til {pct(new)}: prisen har krysset over SMA"
    if new < prev:
        return f"SELG ned til {pct(new)} (gevinstsikring, RSI høy)"
    return f"KJØP tilbake til {pct(new)} (RSI har kjølnet)"


def load_state() -> dict | None:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def compute_signals(cfg: dict) -> dict:
    p = Params.from_cfg(cfg["strategy"])
    alerts = cfg.get("alerts", {})
    state = load_state()
    notified = (state or {}).get("targets", {})
    params_changed = bool(state) and state.get("params") != p.to_dict()
    yesterday = utc_today() - pd.Timedelta(days=1)
    rows = []
    for asset in cfg["assets"]:
        df = load_prices(asset["symbol"])
        sig = compute(df["Close"], p)
        last = sig.iloc[-1]
        tgt = sig["target"]
        changes = tgt[tgt.diff().fillna(0) != 0]
        last_change = str(changes.index[-1].date()) if len(changes) else None
        prev_notified = notified.get(asset["short"])
        target = float(last["target"])
        dist = float(last["close"] / last["sma"] - 1) if pd.notna(last["sma"]) else None
        notes = []
        if dist is not None and abs(dist) <= alerts.get("near_sma_pct", 0.03):
            notes.append(f"Prisen er {dist:+.1%} fra SMA{p.sma} – et brudd kan gi signal snart.")
        if p.uses_rsi and pd.notna(last["rsi"]) and target == 1.0 and last["rsi"] >= alerts.get("near_rsi", 80):
            notes.append(f"RSI er {last['rsi']:.0f}, nær gevinstsikring ved {p.rsi_exit:g}.")
        if bool(last["trimmed"]):
            notes.append(f"Gevinstsikring aktiv: kjøper tilbake når RSI ≤ {p.rsi_reentry:g}.")
        stale = int((yesterday - sig.index[-1]).days)
        if stale >= alerts.get("stale_after_days", 2):
            notes.append(f"OBS: siste data er fra {sig.index[-1].date()} ({stale} døgn gammelt).")
        rows.append({
            "symbol": asset["symbol"], "short": asset["short"], "name": asset["name"],
            "date": str(sig.index[-1].date()),
            "close": float(last["close"]),
            "sma": float(last["sma"]) if pd.notna(last["sma"]) else None,
            "dist_sma": dist,
            "rsi": float(last["rsi"]) if pd.notna(last["rsi"]) else None,
            "target": target,
            "prev_day_target": float(tgt.iloc[-2]) if len(tgt) > 1 else None,
            "notified_target": prev_notified,
            "changed": prev_notified is None or prev_notified != target,
            "action": describe_action(prev_notified, target),
            "trimmed": bool(last["trimmed"]),
            "last_change": last_change,
            "stale_days": stale,
            "notes": notes,
            "history": {
                "dates": [str(d.date()) for d in sig.index[-400:]],
                "close": [round(float(v), 6) for v in sig["close"].iloc[-400:]],
                "sma": [None if pd.isna(v) else round(float(v), 6) for v in sig["sma"].iloc[-400:]],
                "rsi": [None if pd.isna(v) else round(float(v), 2) for v in sig["rsi"].iloc[-400:]],
                "target": [float(v) for v in sig["target"].iloc[-400:]],
            },
        })
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "params": p.to_dict(),
        "params_label": p.label(),
        "params_changed": params_changed,
        "first_run": state is None,
        "assets": rows,
        "any_change": any(r["changed"] for r in rows),
        "max_stale_days": max(r["stale_days"] for r in rows),
    }


def save_state(sig: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "updated": sig["generated"],
        "data_date": max(r["date"] for r in sig["assets"]),
        "params": sig["params"],
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
