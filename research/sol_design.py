"""SOL-strategidesign med rettferdig sammenligning mot dagens regel.

Dagens SOL-regel (SMA200 ±2 %) ble valgt blant 320 varianter etter å ha sett
hele SOL-historikken. For å fjerne den fordelen sammenlignes UTVELGELSESPROSESSEN
for hver regeltype, aldri den faste regelen:

  Test 1  Walk-forward på SOL: ny utvelgelse hvert halvår, bare med SOL-data før perioden.
  Test 2  Overføring: utvelgelse på BTC/ETH, brukt uendret på SOL.
  Test 3  Deflatert Sharpe (Bailey & López de Prado): justert for antall varianter (320 vs 24).

Regeltyper (familier), hver med fast rutenett:
  INC  SMA med bånd (dagens type)        108 varianter (samme som det opprinnelige søket)
  H1   egen trend × BTC-trend               6
  H2   egen trend × ETH-trend               6
  H3   egen trend × volatilitetsmål         6
  H4   over både langt og kort snitt        6

Utvelgelse overalt: høyest median-Sharpe blant varianten og naboene
(ett steg i hver ordnet dimensjon, samme sjekkfrekvens).

Kjør:  python research/sol_design.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ksig import strategies as S  # noqa: E402
from ksig.backtest import Costs, metrics, simulate  # noqa: E402
from ksig.config import RESULTS_DIR  # noqa: E402
from ksig.data import load_prices  # noqa: E402

COSTS = Costs(0.007, 0.003, 1)
SETS = {"kort": [20, 50, 100, 200], "middels": [50, 100, 150, 200], "lang": [100, 150, 200, 250, 300]}
SOL_START = pd.Timestamp("2021-04-11")
WF_FIRST = pd.Timestamp("2023-04-11")
STEP = 182
TRAIN = {"BTC": pd.Timestamp("2015-09-18"), "ETH": pd.Timestamp("2018-11-10")}
N_DIST = NormalDist()

# ---------------------------------------------------------------------------
# Familier: dimensjoner (ordnede lister) og målfunksjon
# ---------------------------------------------------------------------------
GRIDS = {
    "INC": {"n": [50, 75, 100, 125, 150, 175, 200, 250, 300], "band": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08],
            "check": ["daily", "weekly"]},
    "H1": {"set": ["kort", "middels", "lang"], "check": ["daily", "weekly"]},
    "H2": {"set": ["kort", "middels", "lang"], "check": ["daily", "weekly"]},
    "H3": {"vt": [0.6, 0.8, 1.0], "check": ["daily", "weekly"]},
    "H4": {"long": [100, 150, 200], "short": [20, 50]},
}
NAMES = {"INC": "Dagens regeltype (SMA med bånd)", "H1": "H1 BTC som markedsregime",
         "H2": "H2 ETH som sektorregime", "H3": "H3 Volatilitetsmål", "H4": "H4 Rask utgang, treg inngang"}


def votes(series: pd.Series, wins) -> pd.Series:
    v = []
    for n in wins:
        m = series.rolling(n, min_periods=n).mean()
        v.append((series > m).astype(float).where(m.notna()))
    return pd.concat(v, axis=1).mean(axis=1, skipna=False)


def weekly(t: pd.Series) -> pd.Series:
    return t.where(t.index.dayofweek == 6).ffill().fillna(0.0)


def vol_scale(close: pd.Series, target: float, window: int = 30) -> pd.Series:
    vol = np.log(close).diff().rolling(window).std() * np.sqrt(365)
    v = (target / vol).clip(upper=1.0)
    return (np.round(v * 4) / 4).clip(lower=0.25)


def target(v: dict, own: pd.Series, btc: pd.Series, eth: pd.Series) -> pd.Series:
    fam = v["fam"]
    if fam == "INC":
        return S.run(own, {"type": "sma_band", "n": v["n"], "band": v["band"], "check": v["check"]})["target"]
    if fam in ("H1", "H2"):
        reg = btc if fam == "H1" else eth
        x = votes(own, SETS[v["set"]]) * votes(reg, SETS[v["set"]])
    elif fam == "H3":
        x = votes(own, SETS["middels"]) * vol_scale(own, v["vt"])
    elif fam == "H4":
        ml = own.rolling(v["long"], min_periods=v["long"]).mean()
        ms = own.rolling(v["short"], min_periods=v["short"]).mean()
        x = ((own > ml) & (own > ms)).astype(float).where(ml.notna())
    x = (np.round(x * 8) / 8).fillna(0.0)
    return weekly(x) if v.get("check") == "weekly" else x


def variants() -> list[dict]:
    out = []
    for fam, grid in GRIDS.items():
        keys = list(grid)
        for vals in itertools.product(*(grid[k] for k in keys)):
            out.append({"fam": fam, **dict(zip(keys, vals))})
    return out


def label(v: dict) -> str:
    chk = {"daily": "daglig", "weekly": "ukentlig"}.get(v.get("check"), "daglig")
    if v["fam"] == "INC":
        b = f" ±{v['band'] * 100:g} %".replace(".", ",") if v["band"] else ""
        return f"SMA{v['n']}{b}, {chk}"
    if v["fam"] in ("H1", "H2"):
        reg = "BTC" if v["fam"] == "H1" else "ETH"
        return f"Trend × {reg}-trend, {v['set']}, {chk}"
    if v["fam"] == "H3":
        return f"Trend × vol.mål {v['vt']:.0%}, {chk}".replace("%", " %")
    return f"Over SMA{v['long']} og SMA{v['short']}"


def neighbor_lists(vs: list[dict]) -> list[np.ndarray]:
    out = []
    for a in vs:
        grid, idx = GRIDS[a["fam"]], []
        for j, b in enumerate(vs):
            if b["fam"] != a["fam"]:
                continue
            ok = True
            for k, vals in grid.items():
                if k == "check":
                    ok = a[k] == b[k]
                elif abs(vals.index(a[k]) - vals.index(b[k])) > 1:
                    ok = False
                if not ok:
                    break
            if ok:
                idx.append(j)
        out.append(np.array(idx))
    return out


def sharpe(ret: np.ndarray) -> float:
    s = ret.std()
    return float(ret.mean() / s * math.sqrt(365)) if s > 0 else -np.inf


def choose(scores: np.ndarray, nbrs, members: list[int]) -> int:
    best, best_sc = None, -np.inf
    for i in members:
        sc = float(np.median(scores[nbrs[i]]))
        if sc > best_sc:
            best, best_sc = i, sc
    return best


# ---------------------------------------------------------------------------
# Deflatert Sharpe
# ---------------------------------------------------------------------------
def deflated_sharpe(ret: np.ndarray, trial_srs: np.ndarray, n_trials: int) -> dict:
    """Sharpe per døgn. trial_srs = Sharpe per døgn for variantene som ble prøvd."""
    sr = ret.mean() / ret.std()
    T = len(ret)
    z = (ret - ret.mean()) / ret.std()
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    var_sr = float(np.var(trial_srs, ddof=1))
    g = 0.5772156649
    sr0 = math.sqrt(var_sr) * ((1 - g) * N_DIST.inv_cdf(1 - 1 / n_trials)
                               + g * N_DIST.inv_cdf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    dsr = N_DIST.cdf((sr - sr0) * math.sqrt(T - 1) / denom)
    psr = N_DIST.cdf(sr * math.sqrt(T - 1) / denom)
    return {"sharpe_ann": sr * math.sqrt(365), "sr0_ann": sr0 * math.sqrt(365), "psr": psr, "dsr": dsr,
            "skew": skew, "kurt": kurt, "n_trials": n_trials}


def main():
    sol = load_prices("SOL-USD")["Close"]
    btc_all = load_prices("BTC-USD")["Close"]
    eth_all = load_prices("ETH-USD")["Close"]
    vs = variants()
    nbrs = neighbor_lists(vs)
    fam_members = {f: [i for i, v in enumerate(vs) if v["fam"] == f] for f in GRIDS}
    new_members = [i for i, v in enumerate(vs) if v["fam"] != "INC"]

    # ---------------- SOL: alle varianter fra felles start ----------------
    btc_s, eth_s = btc_all.reindex(sol.index).ffill(), eth_all.reindex(sol.index).ffill()
    c = sol[sol.index >= SOL_START]
    tg = [target(v, sol, btc_s, eth_s) for v in vs]
    sims = [simulate(c, t[c.index], COSTS) for t in tg]
    R = np.column_stack([s["ret"].to_numpy() for s in sims])
    one = pd.Series(1.0, index=sol.index)
    bh_full = simulate(c, one[c.index], COSTS)

    # ---------------- Test 1: walk-forward ----------------
    end = c.index[-1]
    starts = list(pd.date_range(WF_FIRST, end, freq=f"{STEP}D"))
    wf = {}
    groups = {**{f: m for f, m in fam_members.items()}, "NEW": new_members}
    for g, members in groups.items():
        stitched = pd.Series(0.0, index=c.index)
        picks = []
        for k, st in enumerate(starts):
            en = starts[k + 1] - pd.Timedelta(days=1) if k + 1 < len(starts) else end
            mask = np.asarray(c.index < st)
            scores = np.array([sharpe(R[mask, j]) for j in range(len(vs))])
            j = choose(scores, nbrs, members)
            seg = (c.index >= st) & (c.index <= en)
            stitched[seg] = tg[j][c.index[seg]]
            picks.append({"start": str(st.date()), "label": label(vs[j])})
        oos = c[c.index >= WF_FIRST]
        res = simulate(oos, stitched[oos.index], COSTS)
        wf[g] = {"metrics": metrics(res), "picks": picks, "equity": res["equity"]}
    oos = c[c.index >= WF_FIRST]
    wf_bh = metrics(simulate(oos, one[oos.index], COSTS))
    inc_fixed = {"type": "sma_band", "n": 200, "band": 0.02, "check": "daily"}
    inc_fixed_t = S.run(sol, inc_fixed)["target"]
    wf_fixed = metrics(simulate(oos, inc_fixed_t[oos.index], COSTS))

    # ---------------- Test 2: overføring fra BTC/ETH ----------------
    train_scores = {}
    for coin, own_all, start in (("BTC", btc_all, TRAIN["BTC"]), ("ETH", eth_all, TRAIN["ETH"])):
        b = btc_all.reindex(own_all.index).ffill()
        e = eth_all.reindex(own_all.index)  # NaN før ETH finnes
        cc = own_all[own_all.index >= start]
        sc = np.full(len(vs), np.nan)
        for j, v in enumerate(vs):
            if (v["fam"] == "H1" and coin != "ETH") or (v["fam"] == "H2" and coin != "BTC"):
                continue
            t = target(v, own_all, b, e)
            sc[j] = sharpe(simulate(cc, t[cc.index], COSTS)["ret"].to_numpy())
        train_scores[coin] = sc
    avg = np.nanmean(np.vstack([train_scores["BTC"], train_scores["ETH"]]), axis=0)
    transfer = {}
    for f, members in fam_members.items():
        j = choose(avg, nbrs, members)
        full = metrics(sims[j])
        tw = metrics(simulate(oos, tg[j][oos.index], COSTS))
        c2 = metrics(simulate(c, tg[j][c.index], Costs(0.02, 0, 1)))
        l2 = metrics(simulate(c, tg[j][c.index], Costs(0.007, 0.003, 2)))
        transfer[f] = {"idx": j, "label": label(vs[j]), "variant": vs[j], "full": full, "wf_window": tw,
                       "cost2_cagr": c2["cagr"], "lag2_cagr": l2["cagr"],
                       "train_score": float(np.median(avg[nbrs[j]]))}

    # ---------------- Test 3: deflatert Sharpe ----------------
    sr_day = R.mean(axis=0) / R.std(axis=0)
    dsr = {}
    # Dagens regel: 320 varianter fra det opprinnelige søket
    from search import candidates as search_candidates  # research/search.py
    old = search_candidates()
    old_sr = []
    for spec in old:
        t = S.run(sol, spec)["target"]
        r = simulate(c, t[c.index], COSTS)["ret"].to_numpy()
        old_sr.append(r.mean() / r.std() if r.std() > 0 else 0.0)
    inc_ret = simulate(c, inc_fixed_t[c.index], COSTS)["ret"].to_numpy()
    dsr["INC"] = {**deflated_sharpe(inc_ret, np.array(old_sr), len(old)), "label": "SMA200 ±2 %, daglig (fast)"}
    new_srs = sr_day[new_members]
    for f in ("H1", "H2", "H3", "H4"):
        j = max(fam_members[f], key=lambda i: sr_day[i])
        dsr[f] = {**deflated_sharpe(R[:, j], new_srs, len(new_members)), "label": label(vs[j])}

    # ---------------- Beslutning ----------------
    bh_m = metrics(bh_full)
    decision = {}
    for f in ("H1", "H2", "H3", "H4"):
        tr = transfer[f]
        crit = {
            "1_wf": wf[f]["metrics"]["sharpe"] > wf["INC"]["metrics"]["sharpe"],
            "2_transfer": tr["full"]["sharpe"] > transfer["INC"]["full"]["sharpe"],
            "3_dsr": dsr[f]["dsr"] > dsr["INC"]["dsr"],
            "4a_cagr": tr["full"]["cagr"] > bh_m["cagr"],
            "4b_dd": tr["full"]["max_dd"] >= bh_m["max_dd"] + 0.10,
            "4c_cost2": tr["cost2_cagr"] > bh_m["cagr"],
        }
        decision[f] = {"criteria": crit, "passes": all(crit.values())}

    # ---------------- Utskrift ----------------
    def m(x):
        return f"CAGR {x['cagr']:6.1%} Sh {x['sharpe']:5.2f} DD {x['max_dd']:6.1%} tr/år {x['trades_per_year']:5.1f}"
    print(f"SOL {c.index[0].date()} – {end.date()}, {len(vs)} varianter")
    print(f"\nTEST 1 walk-forward fra {WF_FIRST.date()}   K&H {m(wf_bh)}   fast SMA200±2 % {m(wf_fixed)}")
    for g in groups:
        print(f"  {g:4s} {m(wf[g]['metrics'])}  valg: {' → '.join(dict.fromkeys(p['label'] for p in wf[g]['picks']))}")
    print(f"\nTEST 2 overføring   K&H SOL hele {m(bh_m)}")
    for f, tr in transfer.items():
        print(f"  {f:4s} {tr['label']:34s} SOL hele {m(tr['full'])} | WF-vindu {m(tr['wf_window'])} | k2% {tr['cost2_cagr']:6.1%} lag2 {tr['lag2_cagr']:6.1%}")
    print("\nTEST 3 deflatert Sharpe")
    for f, d in dsr.items():
        print(f"  {f:4s} {d['label']:34s} Sharpe {d['sharpe_ann']:.2f}  terskel {d['sr0_ann']:.2f}  PSR {d['psr']:.2f}  DSR {d['dsr']:.2f}  N={d['n_trials']}  skew {d['skew']:.2f} kurt {d['kurt']:.1f}")
    print("\nBESLUTNING")
    for f, d in decision.items():
        print(f"  {f}: {'BESTÅR' if d['passes'] else 'består ikke'}  {d['criteria']}")

    out = {"window": [str(c.index[0].date()), str(end.date())], "wf_first": str(WF_FIRST.date()),
           "buy_hold_full": bh_m, "buy_hold_wf": wf_bh, "fixed_wf": wf_fixed,
           "fixed_full": metrics(simulate(c, inc_fixed_t[c.index], COSTS)),
           "wf": {g: {"metrics": w["metrics"], "picks": w["picks"]} for g, w in wf.items()},
           "transfer": {f: {k: v for k, v in tr.items() if k != "idx"} for f, tr in transfer.items()},
           "dsr": dsr, "decision": decision,
           "variants": [{"label": label(v), **v, "sol_sharpe": float(sr_day[i] * math.sqrt(365))}
                        for i, v in enumerate(vs)]}

    def clean(o):
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "sol_design.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1))
    # Kurver for diagram
    eq = pd.DataFrame({g: w["equity"] for g, w in wf.items()})
    eq["BH"] = simulate(oos, one[oos.index], COSTS)["equity"]
    eq.to_csv(RESULTS_DIR / "sol_design_wf_equity.csv", float_format="%.5f")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
