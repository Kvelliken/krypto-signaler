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

from . import portfolio as PF
from . import strategies as S
from .config import STATE_DIR
from .data import load_prices, utc_today

STATE_FILE = STATE_DIR / "positions.json"
LOG_FILE = STATE_DIR / "portfolio_log.csv"
HISTORY_DAYS = 400


def pct(x: float) -> str:
    v = round(x * 100, 1)
    return (f"{v:.0f}" if v == int(v) else f"{v:.1f}".replace(".", ",")) + " %"


def describe_action(prev: float | None, new: float) -> str:
    if prev is None:
        return f"Startfordeling: {pct(new)} av porteføljen"
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
    notified_shares = (state or {}).get("shares", {})
    old_specs = (state or {}).get("strategies", {})
    yesterday = utc_today() - pd.Timedelta(days=1)

    # Fordeling av hele porteføljen
    method, pparams, band = PF.from_cfg(cfg)
    C, T = PF.coin_frames(cfg)
    shares = PF.allocate(T, C, method, pparams)
    mom, rank = PF.momentum_info(C, int(pparams.get("mom_window", 90)))
    base_w = PF.base_weights(C, method, pparams)

    rows = []
    for asset in cfg["assets"]:
        short = asset["short"]
        spec = asset["strategy"]
        df = load_prices(asset["symbol"])
        sig = S.run(df["Close"], spec)
        last = sig.iloc[-1]
        tgt = sig["target"]
        target = float(last["target"])
        sh = shares[short]
        share = float(sh.iloc[-1])
        prev_share = notified_shares.get(short)
        if prev_share is None:
            changed = True
        else:
            diff = abs(share - prev_share)
            changed = diff >= band - 1e-9 or (diff > 1e-9 and (share == 0 or prev_share == 0))
        share_changes = sh[sh.diff().fillna(0).abs() > 1e-9]
        stale = int((yesterday - sig.index[-1]).days)
        notes = S.notes(last, spec, alerts.get("near_pct", 0.03))
        if stale >= alerts.get("stale_after_days", 2):
            notes.append(f"OBS: siste data er fra {sig.index[-1].date()} ({stale} døgn gammelt).")
        strategy_changed = bool(state) and old_specs.get(short) not in (None, spec)
        if strategy_changed:
            notes.insert(0, f"Ny strategi for {short}: {S.label(spec)}.")
        if prev_share is not None and not changed and abs(share - prev_share) > 1e-9:
            notes.append(f"Beregnet andel er {pct(share)}, men endringen fra {pct(prev_share)} er under "
                         f"{band * 100:g} prosentpoeng. Behold {pct(prev_share)}.")
        lines = S.chart_lines(spec)
        h = sig.iloc[-HISTORY_DAYS:]
        hs = sh.reindex(h.index).ffill().fillna(0.0)
        rows.append({
            "symbol": asset["symbol"], "short": short, "name": asset["name"],
            "spec": spec, "label": S.label(spec), "rules": S.rules(spec),
            "date": str(sig.index[-1].date()),
            "close": float(last["close"]),
            "display": [{"name": n, "kind": k, "value": (bool(v) if k == "check" else _num(v))}
                        for n, k, v in S.display(last, spec)],
            "target": target,
            "prev_day_target": float(tgt.iloc[-2]) if len(tgt) > 1 else None,
            "weight": float(base_w[short].iloc[-1]),
            "mom_ret": _num(mom[short].iloc[-1]),
            "mom_rank": int(rank[short].iloc[-1]) if pd.notna(rank[short].iloc[-1]) else None,
            "share": share,
            "prev_day_share": float(sh.iloc[-2]) if len(sh) > 1 else None,
            "notified_share": prev_share,
            "hold_share": share if changed else prev_share,
            "changed": changed,
            "strategy_changed": strategy_changed,
            "action": describe_action(prev_share, share) if changed else "Ingen endring",
            "last_change": str(share_changes.index[-1].date()) if len(share_changes) else None,
            "stale_days": stale,
            "notes": notes,
            "history": {
                "dates": [str(d.date()) for d in h.index],
                "close": [round(float(v), 6) for v in h["close"]],
                "lines": [{"name": nm, "values": [None if pd.isna(v) else round(float(v), 6) for v in h[col]]}
                          for col, nm in lines],
                "target": [float(v) for v in h["target"]],
                "share": [float(v) for v in hs],
            },
        })
    hold_total = sum(r["hold_share"] for r in rows)
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "first_run": state is None,
        "first_portfolio": not notified_shares,
        "params_changed": any(r["strategy_changed"] for r in rows),
        "portfolio": {"method": method, "params": pparams, "band": band, "rules": PF.describe(cfg),
                      "cash": round(1 - hold_total, 6)},
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
        "shares": {r["short"]: r["hold_share"] for r in sig["assets"]},
    }, indent=2, ensure_ascii=False))
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["data_date", "coin", "fra", "til", "pris", "handling"])
        for r in sig["assets"]:
            if r["changed"]:
                w.writerow([r["date"], r["short"],
                            "" if r["notified_share"] is None else r["notified_share"],
                            r["share"], f"{r['close']:.6g}", r["action"]])


def read_log(limit: int = 50) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as fh:
        rows = list(csv.DictReader(fh))
    return rows[-limit:][::-1]
