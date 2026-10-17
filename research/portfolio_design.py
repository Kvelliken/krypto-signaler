"""Fordeling mellom BTC, ETH og SOL: hvilken metode gir høyest avkastning uten større fall?

Fastsatt før kjøring:
  * Coin-reglene er de som er i drift (config.toml).
  * Felles vindu: fra 11. april 2021 (SOL-regelen har oppvarming) til siste døgn.
  * Kostnad 1,0 % per handel, handelsbånd 5 prosentpoeng, handel døgnet etter signalet.
  * Metoder: lik vekt (dagens), lik vekt med omfordeling, invers volatilitet (± omfordeling),
    momentum-vekting (± omfordeling), og optimerte faste vekter valgt walk-forward.
  * «I ettertid»: beste faste vekter over hele perioden – bare som øvre grense.
  * Valgregel: høyest avkastning per år i hele vinduet, med maks fall høyst 5 prosentpoeng
    dypere enn lik vekt, OG bedre enn lik vekt i begge delperioder (før/etter april 2023)
    OG i kontrollen med bare BTC og ETH fra november 2018. Består ingen: behold lik vekt.

Kjør:  python research/portfolio_design.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ksig.backtest import Costs, metrics  # noqa: E402
from ksig.config import RESULTS_DIR, load_config  # noqa: E402
from ksig.portfolio import allocate, coin_frames, simulate_portfolio  # noqa: E402

COSTS = Costs(0.007, 0.003, 1)
BAND = 0.05
START3 = pd.Timestamp("2021-04-11")
SPLIT = pd.Timestamp("2023-04-11")
START2 = pd.Timestamp("2018-11-10")

METHODS = {
    "Lik vekt (dagens)": ("equal", {}),
    "Lik vekt, omfordelt (maks 2×)": ("equal_redist", {"max_scale": 2.0}),
    "Lik vekt, omfordelt (maks 3×)": ("equal_redist", {"max_scale": 3.0}),
    "Invers volatilitet": ("inv_vol", {"vol_window": 90}),
    "Invers volatilitet, omfordelt (maks 2×)": ("inv_vol_redist", {"vol_window": 90, "max_scale": 2.0}),
    "Momentum 50/30/20": ("momentum", {"mom_window": 90}),
    "Momentum 50/30/20, omfordelt (maks 2×)": ("momentum_redist", {"mom_window": 90, "max_scale": 2.0}),
}


def run(C, T, method, params, start, end=None):
    sh = allocate(T, C, method, params)
    res = simulate_portfolio(C, sh, COSTS, BAND, start)
    if end is not None:
        res = res[res.index <= end]
    return res


def window_metrics(res, a, b):
    r = res[(res.index >= a) & (res.index <= b)].copy()
    r["equity"] = r["equity"] / r["equity"].iloc[0] * (1 + r["ret"].iloc[0])
    return metrics(r)


def fixed_grid(cols, step=0.1):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    out = []
    for combo in itertools.product(vals, repeat=len(cols) - 1):
        last = round(1 - sum(combo), 2)
        if last >= -1e-9:
            out.append(dict(zip(cols, [*combo, last])))
    return out


def main():
    cfg = load_config()
    C, T = coin_frames(cfg)
    end = C.index[-1]
    cfg2 = {**cfg, "assets": [a for a in cfg["assets"] if a["short"] in ("BTC", "ETH")]}
    C2, T2 = coin_frames(cfg2)
    out = {"methods": {}}

    def report(name, method, params):
        r3 = run(C, T, method, params, START3)
        r2 = run(C2, T2, method, params, START2)
        m = {"full": metrics(r3), "pre": window_metrics(r3, START3, SPLIT - pd.Timedelta(days=1)),
             "post": window_metrics(r3, SPLIT, end), "two": metrics(r2),
             "avg_cash": float(1 - r3["exposure"].mean())}
        out["methods"][name] = {"method": method, "params": params, **m}
        return m

    for name, (method, params) in METHODS.items():
        report(name, method, params)

    # Kjøp og hold, lik vekt, uten signaler (referanse)
    ones = pd.DataFrame(1.0, index=C.index, columns=C.columns)
    r = simulate_portfolio(C, allocate(ones, C, "equal"), COSTS, BAND, START3)
    ones2 = pd.DataFrame(1.0, index=C2.index, columns=C2.columns)
    r2 = simulate_portfolio(C2, allocate(ones2, C2, "equal"), COSTS, BAND, START2)
    out["buy_hold"] = {"full": metrics(r), "pre": window_metrics(r, START3, SPLIT - pd.Timedelta(days=1)),
                       "post": window_metrics(r, SPLIT, end), "two": metrics(r2)}

    # Faste vekter: i ettertid og walk-forward
    grid = fixed_grid(list(C.columns))
    sims = {i: run(C, T, "fixed", {"weights": w}, START3) for i, w in enumerate(grid)}
    full_cagr = {i: metrics(s)["cagr"] for i, s in sims.items()}
    best_i = max(full_cagr, key=full_cagr.get)
    eq_dd = out["methods"]["Lik vekt (dagens)"]["full"]["max_dd"]
    ok = {i: s for i, s in sims.items() if metrics(s)["max_dd"] >= eq_dd - 0.05}
    best_ok = max(ok, key=lambda i: full_cagr[i])
    out["hindsight"] = {"best": {"weights": grid[best_i], **{"full": metrics(sims[best_i])}},
                        "best_dd_limited": {"weights": grid[best_ok], **{"full": metrics(sims[best_ok])}}}

    # Walk-forward: hvert halvår velges faste vekter med høyest avkastning hittil
    # (med samme fallgrense), og brukes til neste halvår.
    first = START3 + pd.Timedelta(days=365)
    starts = list(pd.date_range(first, end, freq="182D"))
    stitched = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    picks = []
    for k, st in enumerate(starts):
        en = starts[k + 1] - pd.Timedelta(days=1) if k + 1 < len(starts) else end
        def score(i):
            s = sims[i][sims[i].index < st]
            eqv = s["equity"]
            dd = (eqv / eqv.cummax() - 1).min()
            return (eqv.iloc[-1] ** (365.25 / len(eqv)) - 1) if dd >= -0.70 else -9
        j = max(sims, key=score)
        sh = allocate(T, C, "fixed", {"weights": grid[j]})
        m = (C.index >= st) & (C.index <= en)
        stitched[m] = sh[m]
        picks.append({"start": str(st.date()), "weights": grid[j]})
    eqw = allocate(T, C, "equal")
    stitched[C.index < first] = eqw[C.index < first]
    rw = simulate_portfolio(C, stitched, COSTS, BAND, START3)
    out["methods"]["Optimerte faste vekter (walk-forward)"] = {
        "method": "wf_fixed", "params": {}, "full": metrics(rw),
        "pre": window_metrics(rw, START3, SPLIT - pd.Timedelta(days=1)), "post": window_metrics(rw, SPLIT, end),
        "two": None, "avg_cash": float(1 - rw["exposure"].mean()), "picks": picks}

    # Valgregel
    base = out["methods"]["Lik vekt (dagens)"]
    passing = []
    for name, m in out["methods"].items():
        if name == "Lik vekt (dagens)" or m["two"] is None:
            continue
        ok_ = (m["full"]["max_dd"] >= base["full"]["max_dd"] - 0.05
               and m["pre"]["cagr"] > base["pre"]["cagr"] and m["post"]["cagr"] > base["post"]["cagr"]
               and m["two"]["cagr"] > base["two"]["cagr"])
        m["passes"] = bool(ok_)
        if ok_:
            passing.append(name)
    chosen = max(passing, key=lambda n: out["methods"][n]["full"]["cagr"]) if passing else "Lik vekt (dagens)"
    out["chosen"] = chosen

    f = lambda m: f"{m['cagr']:6.1%} DD {m['max_dd']:6.1%} Sh {m['sharpe']:.2f}"
    print(f"Vindu {START3.date()} – {end.date()}, delt {SPLIT.date()}; kontroll BTC+ETH fra {START2.date()}")
    bh = out["buy_hold"]
    print(f"{'Kjøp og hold, lik vekt':42s} hele {f(bh['full'])} | før {bh['pre']['cagr']:6.1%} | etter {bh['post']['cagr']:6.1%} | BTC+ETH {f(bh['two'])}")
    for name, m in out["methods"].items():
        two = f(m["two"]) if m["two"] else "–"
        flag = "✓" if m.get("passes") else " "
        print(f"{flag} {name:40s} hele {f(m['full'])} | før {m['pre']['cagr']:6.1%} | etter {m['post']['cagr']:6.1%} | BTC+ETH {two} | snitt kontanter {m['avg_cash']:.0%} | handler/år {m['full']['trades_per_year']:.1f}")
    h = out["hindsight"]
    print(f"\nI ettertid, beste faste vekter: {h['best']['weights']} {f(h['best']['full'])}")
    print(f"I ettertid, med fallgrense:     {h['best_dd_limited']['weights']} {f(h['best_dd_limited']['full'])}")
    print("Walk-forward-valg:", [(p['start'], p['weights']) for p in picks])
    print("\nVALGT:", chosen)

    def clean(o):
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, dict):
            return {kk: clean(v) for kk, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "portfolio_design.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
