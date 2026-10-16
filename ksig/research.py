"""Analyse av gjeldende strategi per coin mot kjøp og hold.

Alt beregnes etter Firi-kostnader og lagres i results/research.json.
Delperioder:
  full      – hele evalueringsperioden
  dev       – før hold-out (perioden reglene først ble valgt på)
  holdout   – fra og med holdout_start
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import strategies as S
from .backtest import Costs, metrics, simulate, yearly_returns
from .config import RESULTS_DIR
from .data import load_prices

BH = "buy_hold"

# Nabolagskart per familie: (dim1, verdier), (dim2, verdier)
NEIGHBOR_DIMS = {
    "ens_regime": (("set", ["kort", "middels", "lang"]), ("regime_set", ["kort", "middels", "lang"])),
    "sma_band": (("n", [100, 125, 150, 175, 200, 250]), ("band", [0.0, 0.01, 0.02, 0.03, 0.05])),
    "ens_sma": (("set", ["kort", "middels", "lang"]), ("floor", [0.0, 0.25, 0.5])),
    "ens_mom": (("set", ["kort", "middels", "lang"]), ("floor", [0.0, 0.25, 0.5])),
    "ema_cross": (("fast", [10, 20, 30, 50]), ("slow", [50, 100, 150, 200])),
    "donchian": (("entry", [20, 30, 50, 75, 100]), ("exit", [10, 20, 30, 50, 75])),
    "tsmom": (("lookback", [60, 90, 120, 180, 270, 365]), ("band", [0.0, 0.05, 0.1, 0.2])),
    "trail": (("stop", [0.2, 0.25, 0.3, 0.4, 0.5]), ("reentry", [50, 100, 150, 200])),
}


def _clean(obj):
    if isinstance(obj, (float, np.floating)):
        obj = float(obj)
        return None if math.isnan(obj) or math.isinf(obj) else round(obj, 6)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _downsample(res_map: dict[str, pd.DataFrame], step: int = 3) -> dict:
    any_res = next(iter(res_map.values()))
    idx = list(range(0, len(any_res), step))
    if idx[-1] != len(any_res) - 1:
        idx.append(len(any_res) - 1)
    out = {"dates": [str(any_res.index[i].date()) for i in idx]}
    for key, res in res_map.items():
        eq = res["equity"].to_numpy()
        out[key] = [round(float(eq[i]) * 100, 4) for i in idx]
    return out


def _drawdown(res: pd.DataFrame, step: int = 3) -> list[float]:
    eq = res["equity"]
    dd = (eq / eq.cummax() - 1).to_numpy()
    idx = list(range(0, len(dd), step))
    if idx[-1] != len(dd) - 1:
        idx.append(len(dd) - 1)
    return [round(float(dd[i]) * 100, 2) for i in idx]


def bootstrap_prob(r_s: np.ndarray, r_b: np.ndarray, block: int, n: int, seed: int = 0) -> float:
    """Andel av blokk-bootstrap-utvalg der strategien ender over kjøp og hold."""
    rng = np.random.default_rng(seed)
    d = np.log1p(r_s) - np.log1p(r_b)
    L = len(d)
    nb = L // block + 1
    wins = 0
    for _ in range(n):
        st = rng.integers(0, L - block, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:L]
        wins += d[idx].sum() > 0
    return wins / n


def analyze_asset(asset: dict, cfg: dict) -> dict:
    rc = cfg["research"]
    costs = Costs.from_cfg(cfg["costs"])
    spec = asset["strategy"]
    close = load_prices(asset["symbol"])["Close"]
    start_idx = min(max(rc["eval_warmup_days"], S.warmup(spec)), len(close) - 2)
    eval_start = close.index[start_idx]
    holdout = pd.Timestamp(rc["holdout_start"])
    c = close[close.index >= eval_start]
    target = S.run(close, spec)["target"]
    one = pd.Series(1.0, index=close.index)

    def sim(tgt, a=None, b=None, cst=costs):
        cc = c if a is None else c[(c.index >= a) & (c.index <= b)]
        return simulate(cc, tgt[cc.index], cst)

    main = {"strategy": sim(target), BH: sim(one)}
    periods = {
        "full": (c.index[0], c.index[-1]),
        "dev": (c.index[0], holdout - pd.Timedelta(days=1)),
        "holdout": (holdout, c.index[-1]),
    }
    per = {}
    for name, (a, b) in periods.items():
        if (c.index >= a).sum() < 30 or a >= b:
            continue
        per[name] = {"start": str(a.date()), "end": str(b.date()),
                     "strategy": metrics(sim(target, a, b)), BH: metrics(sim(one, a, b))}

    cost_sens = []
    for lvl in rc["cost_levels"]:
        m = metrics(sim(target, cst=Costs(fee=lvl, slippage=0.0, lag_days=costs.lag_days)))
        cost_sens.append({"cost": lvl, "cagr": m["cagr"], "sharpe": m["sharpe"],
                          "max_dd": m["max_dd"], "final_multiple": m["final_multiple"]})
    lag_sens = []
    for lag in rc["lag_days"]:
        m = metrics(sim(target, cst=Costs(fee=costs.fee, slippage=costs.slippage, lag_days=lag)))
        lag_sens.append({"lag_days": lag, "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"]})

    # Nabolagskart: samme familie, nærliggende parametere
    fam = spec["type"]
    grid = None
    if fam in NEIGHBOR_DIMS:
        (k1, v1), (k2, v2) = NEIGHBOR_DIMS[fam]
        v1 = list(v1) if spec[k1] in v1 else sorted([*v1, spec[k1]])
        v2 = list(v2) if spec[k2] in v2 else sorted([*v2, spec[k2]])
        cagr = []
        for a in v1:
            row = []
            for b in v2:
                s2 = {**spec, k1: a, k2: b}
                if fam == "ema_cross" and s2["fast"] >= s2["slow"]:
                    row.append(None)
                    continue
                row.append(metrics(sim(S.run(close, s2)["target"]))["cagr"])
            cagr.append(row)
        grid = {"dim1": k1, "values1": v1, "dim2": k2, "values2": v2, "cagr": cagr,
                "current": [spec[k1], spec[k2]], "buy_hold_cagr": metrics(main[BH])["cagr"]}

    yr_s, yr_b = yearly_returns(main["strategy"]), yearly_returns(main[BH])
    yearly = [{"year": int(y), "strategy": float(yr_s[y]), "buy_hold": float(yr_b[y]),
               "partial": bool(y == yr_s.index[0] or y == yr_s.index[-1])} for y in yr_s.index]
    full_years = [y for y in yearly if not y["partial"]]

    prob = bootstrap_prob(main["strategy"]["ret"].to_numpy(), main[BH]["ret"].to_numpy(),
                          rc["bootstrap_block_days"], rc["bootstrap_samples"])

    return {
        "symbol": asset["symbol"], "short": asset["short"], "name": asset["name"],
        "spec": spec, "label": S.label(spec), "rules": S.rules(spec),
        "data_start": str(close.index[0].date()), "data_end": str(close.index[-1].date()),
        "eval_start": str(eval_start.date()),
        "metrics": {k: metrics(v) for k, v in main.items()},
        "periods": per,
        "equity": _downsample(main),
        "drawdown": {k: _drawdown(v) for k, v in main.items()},
        "cost_sensitivity": cost_sens,
        "lag_sensitivity": lag_sens,
        "neighbors": grid,
        "yearly": yearly,
        "years_beaten": sum(1 for y in full_years if y["strategy"] > y["buy_hold"]),
        "years_total": len(full_years),
        "prob_beats": prob,
        "_frames": {"close": close, "target": target, "eval_start": eval_start},
    }


def portfolio(assets: list[dict], cfg: dict) -> dict:
    """Lik vekt (1/3) per coin fra felles startdato, uten rebalansering mellom coins."""
    costs = Costs.from_cfg(cfg["costs"])
    start = max(a["_frames"]["eval_start"] for a in assets)
    sleeves = {"strategy": [], BH: []}
    for a in assets:
        f = a["_frames"]
        c = f["close"][f["close"].index >= start]
        sleeves["strategy"].append(simulate(c, f["target"][c.index], costs))
        sleeves[BH].append(simulate(c, pd.Series(1.0, index=c.index), costs))
    combined = {}
    for key, parts in sleeves.items():
        idx = parts[0].index
        for p in parts[1:]:
            idx = idx.intersection(p.index)
        eq = sum(p["equity"].reindex(idx) for p in parts) / len(parts)
        ret = eq.pct_change()
        ret.iloc[0] = eq.iloc[0] - 1
        combined[key] = pd.DataFrame({
            "equity": eq, "ret": ret,
            "cost": sum(p["cost"].reindex(idx) for p in parts) / len(parts),
            "traded": sum(p["traded"].reindex(idx) for p in parts),
            "exposure": sum(p["exposure"].reindex(idx) for p in parts) / len(parts)})
    yr_s, yr_b = yearly_returns(combined["strategy"]), yearly_returns(combined[BH])
    return {
        "start": str(start.date()),
        "metrics": {k: metrics(v) for k, v in combined.items()},
        "equity": _downsample(combined),
        "drawdown": {k: _drawdown(v) for k, v in combined.items()},
        "yearly": [{"year": int(y), "strategy": float(yr_s[y]), "buy_hold": float(yr_b[y]),
                    "partial": bool(y == yr_s.index[0] or y == yr_s.index[-1])} for y in yr_s.index],
    }


def _search_summary() -> dict | None:
    """Kort sammendrag av søket (fra research/search.py), hvis det finnes."""
    path = RESULTS_DIR / "search.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    o1 = data.get("methods", {}).get("O1", {})
    return {"n_candidates": data.get("n_candidates"),
            "walk_forward": {k: {"start": v["start"], "strategy_cagr": v["strategy"]["cagr"],
                                 "buy_hold_cagr": v["buy_hold"]["cagr"],
                                 "strategy_dd": v["strategy"]["max_dd"], "buy_hold_dd": v["buy_hold"]["max_dd"]}
                             for k, v in o1.items()}}


def run_research(cfg: dict, save: bool = True) -> dict:
    assets = [analyze_asset(a, cfg) for a in cfg["assets"]]
    result = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "costs": cfg["costs"],
        "holdout_start": cfg["research"]["holdout_start"],
        "portfolio": portfolio(assets, cfg),
        "assets": [{k: v for k, v in a.items() if k != "_frames"} for a in assets],
        "search": _search_summary(),
    }
    result = _clean(result)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "research.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result
