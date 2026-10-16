"""Analyse: baseline, robusthet, walk-forward og portefølje.

Resultatet lagres i results/research.json og vises på nettsiden.

Rettferdig sammenligning: alle varianter for en coin evalueres fra samme
startdato (første dag der den lengste SMA-en i rutenettet er tilgjengelig).
"""
from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import Costs, metrics, sharpe_of, simulate, yearly_returns
from .config import RESULTS_DIR
from .data import load_prices
from .strategy import Params, compute, warmup_days

BH = "buy_hold"


def _grid(cfg: dict, base: Params) -> list[Params]:
    wf = cfg["walkforward"]
    seen, out = set(), []
    for s, e, r in itertools.product(wf["sma"], wf["rsi_exit"], wf["rsi_reentry"]):
        p = Params(sma=s, rsi_period=base.rsi_period, rsi_exit=e,
                   rsi_reentry=r if e < 100 else 0, trim_to=base.trim_to)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _downsample(res_map: dict[str, pd.DataFrame], step: int = 3) -> dict:
    """Egenkapitalkurver for nettsiden (hver n-te dag + siste dag)."""
    any_res = next(iter(res_map.values()))
    idx = list(range(0, len(any_res), step))
    if idx[-1] != len(any_res) - 1:
        idx.append(len(any_res) - 1)
    out = {"dates": [str(any_res.index[i].date()) for i in idx]}
    for key, res in res_map.items():
        eq = res["equity"].to_numpy()
        out[key] = [round(float(eq[i]) * 100, 4) for i in idx]
    return out


def _drawdown_series(res: pd.DataFrame, step: int = 3) -> list[float]:
    eq = res["equity"]
    dd = (eq / eq.cummax() - 1).to_numpy()
    idx = list(range(0, len(dd), step))
    if idx[-1] != len(dd) - 1:
        idx.append(len(dd) - 1)
    return [round(float(dd[i]) * 100, 2) for i in idx]


def _clean(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else round(obj, 6)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return _clean(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def walk_forward(close: pd.Series, eval_start: pd.Timestamp, grid: list[Params],
                 targets: dict[Params, pd.Series], costs: Costs, cfg: dict) -> dict | None:
    """Ekspanderende walk-forward.

    For hver testperiode velges parameterne med høyest Sharpe (etter
    kostnader) på ALL data før perioden. De valgte parameterne brukes så
    på perioden etterpå, som modellen ikke har sett. Testperiodene skjøtes
    sammen til én sammenhengende out-of-sample-kurve.
    """
    wf = cfg["walkforward"]
    c = close[close.index >= eval_start]
    full_ret = {p: simulate(c, targets[p], costs)["ret"] for p in grid}

    first_test = eval_start + pd.Timedelta(days=wf["min_train_days"])
    if first_test >= c.index[-1] - pd.Timedelta(days=60):
        return None
    starts = list(pd.date_range(first_test, c.index[-1], freq=f"{wf['test_days']}D"))
    segments, stitched = [], pd.Series(0.0, index=c.index)
    for k, seg_start in enumerate(starts):
        seg_end = starts[k + 1] - pd.Timedelta(days=1) if k + 1 < len(starts) else c.index[-1]
        train_mask = c.index < seg_start
        scores = {p: sharpe_of(full_ret[p][train_mask]) for p in grid}
        best = max(grid, key=lambda p: scores[p])
        seg_mask = (c.index >= seg_start) & (c.index <= seg_end)
        stitched[seg_mask] = targets[best][c.index[seg_mask]]
        segments.append({"start": str(seg_start.date()), "end": str(seg_end.date()),
                         "params": best.to_dict(), "label": best.label(),
                         "train_sharpe": scores[best]})

    oos = c[c.index >= first_test]
    base = Params.from_cfg(cfg["strategy"])
    sma_only = Params(sma=base.sma, rsi_exit=100, rsi_reentry=0)
    res = {
        "walk_forward": simulate(oos, stitched[oos.index], costs),
        "baseline": simulate(oos, targets[base][oos.index], costs),
        "sma_only": simulate(oos, targets[sma_only][oos.index], costs),
        BH: simulate(oos, pd.Series(1.0, index=oos.index), costs),
    }
    # Optimismegap: beste parametere på HELE historikken (inkl. fremtiden)
    is_scores = {p: sharpe_of(full_ret[p]) for p in grid}
    is_best = max(grid, key=lambda p: is_scores[p])
    is_best_oos = simulate(oos, targets[is_best][oos.index], costs)
    return {
        "oos_start": str(first_test.date()),
        "segments": segments,
        "metrics": {k: metrics(v) for k, v in res.items()},
        "insample_best": {
            "label": is_best.label(), "params": is_best.to_dict(),
            "full_period_sharpe": is_scores[is_best],
            "oos_period_sharpe": metrics(is_best_oos)["sharpe"],
        },
        "equity": _downsample(res),
        "n_candidates": len(grid),
    }


def analyze_asset(asset: dict, cfg: dict) -> dict:
    df = load_prices(asset["symbol"])
    close = df["Close"]
    base = Params.from_cfg(cfg["strategy"])
    costs = Costs.from_cfg(cfg["costs"])
    grid = _grid(cfg, base)
    sma_only = Params(sma=base.sma, rsi_exit=100, rsi_reentry=0)
    all_params = list(dict.fromkeys([base, sma_only, *grid]))

    warm = max(warmup_days(p) for p in all_params)
    eval_start = close.index[min(warm, len(close) - 1)]
    targets = {p: compute(close, p)["target"] for p in all_params}
    c = close[close.index >= eval_start]

    def run(p: Params | None, cst: Costs = costs) -> pd.DataFrame:
        tgt = pd.Series(1.0, index=c.index) if p is None else targets[p][c.index]
        return simulate(c, tgt, cst)

    main = {"strategy": run(base), "sma_only": run(sma_only), BH: run(None)}

    # Kostnadsfølsomhet (fordelt som kurtasje, ingen ekstra slippage)
    cost_sens = []
    for lvl in cfg["robustness"]["cost_levels"]:
        cst = Costs(fee=lvl, slippage=0.0, lag_days=costs.lag_days)
        m = metrics(run(base, cst))
        cost_sens.append({"cost": lvl, "cagr": m["cagr"], "sharpe": m["sharpe"],
                          "final_multiple": m["final_multiple"], "max_dd": m["max_dd"]})
    bh_m = metrics(main[BH])

    # Forsinkelse i utførelse
    lag_sens = []
    for lag in cfg["robustness"]["lag_days"]:
        cst = Costs(fee=costs.fee, slippage=costs.slippage, lag_days=lag)
        m = metrics(run(base, cst))
        lag_sens.append({"lag_days": lag, "cagr": m["cagr"], "sharpe": m["sharpe"],
                         "max_dd": m["max_dd"], "final_multiple": m["final_multiple"]})

    # Parameterrutenett (reentry = baseline-verdien)
    wf = cfg["walkforward"]
    grid_tbl = {"sma": wf["sma"], "rsi_exit": wf["rsi_exit"], "sharpe": [], "cagr": [], "max_dd": []}
    for s in wf["sma"]:
        row_s, row_c, row_d = [], [], []
        for e in wf["rsi_exit"]:
            p = Params(sma=s, rsi_period=base.rsi_period, rsi_exit=e,
                       rsi_reentry=base.rsi_reentry if e < 100 else 0, trim_to=base.trim_to)
            if p not in targets:
                targets[p] = compute(close, p)["target"]
            m = metrics(run(p))
            row_s.append(m["sharpe"]); row_c.append(m["cagr"]); row_d.append(m["max_dd"])
        grid_tbl["sharpe"].append(row_s); grid_tbl["cagr"].append(row_c); grid_tbl["max_dd"].append(row_d)

    yr_s, yr_b = yearly_returns(main["strategy"]), yearly_returns(main[BH])
    yearly = [{"year": int(y), "strategy": float(yr_s[y]), "buy_hold": float(yr_b[y]),
               "partial": bool(y == yr_s.index[0] or y == yr_s.index[-1])} for y in yr_s.index]

    return {
        "symbol": asset["symbol"], "short": asset["short"], "name": asset["name"],
        "data_start": str(close.index[0].date()), "data_end": str(close.index[-1].date()),
        "eval_start": str(eval_start.date()),
        "metrics": {k: metrics(v) for k, v in main.items()},
        "equity": _downsample(main),
        "drawdown": {k: _drawdown_series(v) for k, v in main.items()},
        "cost_sensitivity": cost_sens,
        "buy_hold_reference": {"cagr": bh_m["cagr"], "sharpe": bh_m["sharpe"], "max_dd": bh_m["max_dd"]},
        "lag_sensitivity": lag_sens,
        "grid": grid_tbl,
        "walkforward": walk_forward(close, eval_start, grid, targets, costs, cfg),
        "yearly": yearly,
        "_frames": {"close": close, "targets": targets, "base": base, "sma_only": sma_only},
    }


def portfolio(assets: list[dict], cfg: dict) -> dict:
    """Lik vekt (1/3) per coin fra felles startdato, uten rebalansering mellom coins."""
    costs = Costs.from_cfg(cfg["costs"])
    start = max(pd.Timestamp(a["eval_start"]) for a in assets)
    sleeves = {"strategy": [], "sma_only": [], BH: []}
    for a in assets:
        f = a["_frames"]
        c = f["close"][f["close"].index >= start]
        sleeves["strategy"].append(simulate(c, f["targets"][f["base"]][c.index], costs))
        sleeves["sma_only"].append(simulate(c, f["targets"][f["sma_only"]][c.index], costs))
        sleeves[BH].append(simulate(c, pd.Series(1.0, index=c.index), costs))
    combined = {}
    for key, parts in sleeves.items():
        idx = parts[0].index
        for p in parts[1:]:
            idx = idx.intersection(p.index)
        eq = sum(p["equity"].reindex(idx) for p in parts) / len(parts)
        cost = sum(p["cost"].reindex(idx) for p in parts) / len(parts)
        traded = sum(p["traded"].reindex(idx) for p in parts)
        expo = sum(p["exposure"].reindex(idx) for p in parts) / len(parts)
        ret = eq.pct_change()
        ret.iloc[0] = eq.iloc[0] - 1
        combined[key] = pd.DataFrame({"equity": eq, "ret": ret, "cost": cost,
                                      "traded": traded, "exposure": expo})
    yr_s, yr_b = yearly_returns(combined["strategy"]), yearly_returns(combined[BH])
    return {
        "start": str(start.date()),
        "metrics": {k: metrics(v) for k, v in combined.items()},
        "equity": _downsample(combined),
        "drawdown": {k: _drawdown_series(v) for k, v in combined.items()},
        "yearly": [{"year": int(y), "strategy": float(yr_s[y]), "buy_hold": float(yr_b[y]),
                    "partial": bool(y == yr_s.index[0] or y == yr_s.index[-1])} for y in yr_s.index],
    }


def run_research(cfg: dict, save: bool = True) -> dict:
    assets = [analyze_asset(a, cfg) for a in cfg["assets"]]
    result = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "strategy": Params.from_cfg(cfg["strategy"]).to_dict(),
        "strategy_label": Params.from_cfg(cfg["strategy"]).label(),
        "costs": cfg["costs"],
        "walkforward_cfg": {k: cfg["walkforward"][k] for k in ("min_train_days", "test_days")},
        "portfolio": portfolio(assets, cfg),
        "assets": [{k: v for k, v in a.items() if k != "_frames"} for a in assets],
    }
    result = _clean(result)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "research.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result
