"""Kan noe slå kjøp og hold for SOL etter april 2023?

Opplegg (fastsatt før kjøring):
  * Testperiode: 11. april 2023 – siste døgn. Ingen valg tar hensyn til den.
  * Utvelgelse på data før 11. april 2023, på to måter:
      SOL   SOL fra 11. april 2021 (etter oppvarming) til 10. april 2023
      BTC/ETH  BTC fra sept. 2015 (fra mars 2018 for fryktindeks-regler) og ETH fra nov. 2018,
               begge til 10. april 2023. SOL brukes ikke.
  * Kriterium: median (blant varianten og naboene) av årlig meravkastning mot kjøp og hold.
  * Kostnad 1,0 % per handel, handel døgnet etter signalet.
  * «I ettertid»: beste variant målt på selve testperioden – bare for å vise hva som var mulig.

Regeltyper (alle long/flat, 0–100 % SOL):
  TS     Trailing stop, inn igjen etter rebound: ut når kursen faller X % fra toppen,
         inn igjen når den har steget Y % fra bunnen.
  TSMA   Trailing stop, inn igjen over SMA.
  SL     Stop loss fra kjøpskurs: inn over SMA, ut ved X % under kjøpskurs eller under SMA.
  VT     Volatilitetsmål uten trendfilter: andel = min(1, mål / volatilitet).
  MANIA  Selg i mani: 100 %, men ned til T når kursen er k × SMA200, tilbake under m × SMA200.
  DIP    Kjerne + dip-kjøp: kjerne c, 100 % når kursen er D under 90-døgns topp.
  FG1    Fryktindeks – selg grådighet: 100 %, ned til T når indeksen ≥ G, tilbake når ≤ R.
  FG2    Fryktindeks – kjøp frykt: kjerne c, 100 % når indeksen ≤ F, tilbake til kjerne når ≥ 60.
  FG3    Trend + kjøp frykt: SMA200 ±2 %, men 100 % når indeksen ≤ F.
  Referanse: INC (SMA med bånd, 108 varianter) og H1 (SOL × BTC-trend, lang, ukentlig).

Kjør:  python research/sol_extra.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ksig import strategies as S  # noqa: E402
from ksig.backtest import Costs, metrics, simulate  # noqa: E402
from ksig.config import RESULTS_DIR  # noqa: E402
from ksig.data import load_fng, load_prices  # noqa: E402

COSTS = Costs(0.007, 0.003, 1)
TEST = pd.Timestamp("2023-04-11")
SOL_TRAIN = pd.Timestamp("2021-04-11")
TRAIN_END = TEST - pd.Timedelta(days=1)
TRAIN_START = {"BTC": pd.Timestamp("2015-09-18"), "ETH": pd.Timestamp("2018-11-10")}
FNG_START = pd.Timestamp("2018-03-01")
SETS_LANG = [100, 150, 200, 250, 300]

GRIDS = {
    "TS": {"x": [0.2, 0.25, 0.3, 0.4, 0.5], "y": [0.2, 0.3, 0.4, 0.6]},
    "TSMA": {"x": [0.2, 0.25, 0.3, 0.4, 0.5], "sma": [20, 50, 100, 200]},
    "SL": {"sma": [50, 100, 200], "x": [0.1, 0.15, 0.2, 0.3]},
    "VT": {"vt": [0.5, 0.7, 0.9, 1.1], "win": [14, 30, 60], "check": ["daily", "weekly"]},
    "MANIA": {"k": [1.5, 2.0, 2.5, 3.0], "trim": [0.0, 0.5], "back": [1.0, 1.3]},
    "DIP": {"d": [0.2, 0.3, 0.4, 0.5], "core": [0.25, 0.5, 0.75]},
    "FG1": {"g": [75, 80, 85, 90], "trim": [0.0, 0.5], "r": [50, 60]},
    "FG2": {"f": [10, 20, 25, 30], "core": [0.25, 0.5, 0.75]},
    "FG3": {"f": [10, 15, 20, 25]},
    "INC": {"n": [50, 75, 100, 125, 150, 175, 200, 250, 300], "band": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08],
            "check": ["daily", "weekly"]},
}
NAMES = {
    "TS": "Trailing stop, inn igjen etter rebound", "TSMA": "Trailing stop, inn igjen over SMA",
    "SL": "Stop loss fra kjøpskurs", "VT": "Volatilitetsmål uten trendfilter", "MANIA": "Selg i mani",
    "DIP": "Kjerne + kjøp dipper", "FG1": "Fryktindeks: selg grådighet", "FG2": "Fryktindeks: kjøp frykt",
    "FG3": "Trend + kjøp frykt", "INC": "Dagens regeltype (SMA med bånd)",
}
USES_FNG = {"FG1", "FG2", "FG3"}


def weekly(t: pd.Series) -> pd.Series:
    return t.where(t.index.dayofweek == 6).ffill().fillna(0.0)


def target(v: dict, close: pd.Series, fng: pd.Series) -> pd.Series:
    f, c = v["fam"], close.to_numpy()
    n = len(c)
    out = np.zeros(n)
    if f == "INC":
        return S.run(close, {"type": "sma_band", "n": v["n"], "band": v["band"], "check": v["check"]})["target"]
    if f == "TS":
        inside, peak, trough = True, c[0], c[0]
        for i in range(n):
            if inside:
                peak = max(peak, c[i])
                if c[i] < peak * (1 - v["x"]):
                    inside, trough = False, c[i]
            else:
                trough = min(trough, c[i])
                if c[i] > trough * (1 + v["y"]):
                    inside, peak = True, c[i]
            out[i] = 1.0 if inside else 0.0
        return pd.Series(out, index=close.index)
    if f == "TSMA":
        return S.run(close, {"type": "trail", "stop": v["x"], "reentry": v["sma"]})["target"]
    if f == "SL":
        m = close.rolling(v["sma"], min_periods=v["sma"]).mean().to_numpy()
        inside, entry = False, np.nan
        for i in range(n):
            if np.isnan(m[i]):
                continue
            if inside and (c[i] < entry * (1 - v["x"]) or c[i] < m[i]):
                inside = False
            elif not inside and c[i] > m[i] and (i == 0 or c[i - 1] <= m[i - 1] or np.isnan(m[i - 1])):
                inside, entry = True, c[i]   # bare på et nytt kryss opp, ikke rett etter stopp
            out[i] = 1.0 if inside else 0.0
        return pd.Series(out, index=close.index)
    if f == "VT":
        vol = np.log(close).diff().rolling(v["win"]).std() * math.sqrt(365)
        x = (np.round((v["vt"] / vol).clip(upper=1.0) * 8) / 8).fillna(0.0)
        return weekly(x) if v["check"] == "weekly" else x
    if f == "MANIA":
        m = close.rolling(200, min_periods=200).mean().to_numpy()
        trimmed = False
        for i in range(n):
            if np.isnan(m[i]):
                out[i] = 1.0
                continue
            if not trimmed and c[i] > v["k"] * m[i]:
                trimmed = True
            elif trimmed and c[i] < v["back"] * m[i]:
                trimmed = False
            out[i] = v["trim"] if trimmed else 1.0
        return pd.Series(out, index=close.index)
    if f == "DIP":
        hi = close.rolling(90, min_periods=1).max()
        dip = close < (1 - v["d"]) * hi
        return pd.Series(np.where(dip, 1.0, v["core"]), index=close.index)
    fg = fng.reindex(close.index).ffill().to_numpy()
    if f == "FG1":
        trimmed = False
        for i in range(n):
            if np.isnan(fg[i]):
                out[i] = 1.0
                continue
            if not trimmed and fg[i] >= v["g"]:
                trimmed = True
            elif trimmed and fg[i] <= v["r"]:
                trimmed = False
            out[i] = v["trim"] if trimmed else 1.0
        return pd.Series(out, index=close.index)
    if f == "FG2":
        full = False
        for i in range(n):
            if np.isnan(fg[i]):
                out[i] = v["core"]
                continue
            if not full and fg[i] <= v["f"]:
                full = True
            elif full and fg[i] >= 60:
                full = False
            out[i] = 1.0 if full else v["core"]
        return pd.Series(out, index=close.index)
    if f == "FG3":
        base = S.run(close, {"type": "sma_band", "n": 200, "band": 0.02, "check": "daily"})["target"].to_numpy()
        fear = np.nan_to_num(fg, nan=100) <= v["f"]
        return pd.Series(np.where(fear, 1.0, base), index=close.index)
    raise ValueError(f)


def variants():
    out = []
    for fam, grid in GRIDS.items():
        keys = list(grid)
        for vals in itertools.product(*(grid[k] for k in keys)):
            out.append({"fam": fam, **dict(zip(keys, vals))})
    return out


def label(v):
    f = v["fam"]
    p = lambda x: f"{x * 100:g} %".replace(".", ",")
    chk = " ukentlig" if v.get("check") == "weekly" else ""
    return {
        "TS": lambda: f"Stopp {p(v['x'])} fra topp, inn ved +{p(v['y'])} fra bunn",
        "TSMA": lambda: f"Stopp {p(v['x'])} fra topp, inn over SMA{v['sma']}",
        "SL": lambda: f"Inn over SMA{v['sma']}, stopp {p(v['x'])} under kjøpskurs",
        "VT": lambda: f"Vol.mål {p(v['vt'])}, {v['win']} døgn{chk}",
        "MANIA": lambda: f"Ned til {p(v['trim'])} over {v['k']:g}×SMA200, tilbake under {v['back']:g}×",
        "DIP": lambda: f"Kjerne {p(v['core'])}, 100 % ved {p(v['d'])} under 90d-topp",
        "FG1": lambda: f"Ned til {p(v['trim'])} ved indeks ≥ {v['g']}, tilbake ≤ {v['r']}",
        "FG2": lambda: f"Kjerne {p(v['core'])}, 100 % ved indeks ≤ {v['f']}",
        "FG3": lambda: f"SMA200 ±2 %, 100 % ved indeks ≤ {v['f']}",
        "INC": lambda: f"SMA{v['n']}" + (f" ±{p(v['band'])}" if v["band"] else "") + chk,
    }[f]()


def neighbor_lists(vs):
    out = []
    for a in vs:
        idx = []
        for j, b in enumerate(vs):
            if b["fam"] != a["fam"]:
                continue
            ok = all((a[k] == b[k]) if k == "check" else abs(vals.index(a[k]) - vals.index(b[k])) <= 1
                     for k, vals in GRIDS[a["fam"]].items())
            if ok:
                idx.append(j)
        out.append(np.array(idx))
    return out


def cagr_of(ret: np.ndarray) -> float:
    eq = np.prod(1 + ret)
    return eq ** (365.25 / len(ret)) - 1 if eq > 0 else -1.0


def main():
    fng = load_fng()
    vs = variants()
    nbrs = neighbor_lists(vs)
    fams = list(GRIDS)
    members = {f: [i for i, v in enumerate(vs) if v["fam"] == f] for f in fams}

    sol = load_prices("SOL-USD")["Close"]
    btc_all = load_prices("BTC-USD")["Close"]
    eth_all = load_prices("ETH-USD")["Close"]
    tg = [target(v, sol, fng) for v in vs]
    one = pd.Series(1.0, index=sol.index)

    def window(series, tgt, a, b, costs=COSTS):
        cc = series[(series.index >= a) & (series.index <= b)]
        return simulate(cc, tgt[cc.index], costs)

    end = sol.index[-1]
    bh_test = metrics(window(sol, one, TEST, end))
    bh_train_ret = window(sol, one, SOL_TRAIN, TRAIN_END)["ret"].to_numpy()

    # Testperiode for alle varianter
    test_res = [window(sol, t, TEST, end) for t in tg]
    test_m = [metrics(r) for r in test_res]
    test_m0 = [metrics(window(sol, t, TEST, end, Costs(0, 0, 1))) for t in tg]

    # Utvelgelse 1: SOL før testperioden
    sol_score = np.array([cagr_of(window(sol, t, SOL_TRAIN, TRAIN_END)["ret"].to_numpy()) for t in tg])
    sol_score -= cagr_of(bh_train_ret)

    # Utvelgelse 2: BTC og ETH før testperioden
    tr_scores = []
    for coin, series in (("BTC", btc_all), ("ETH", eth_all)):
        sc = np.zeros(len(vs))
        for j, v in enumerate(vs):
            start = max(TRAIN_START[coin], FNG_START) if v["fam"] in USES_FNG else TRAIN_START[coin]
            t = target(v, series, fng)
            r = window(series, t, start, TRAIN_END)["ret"].to_numpy()
            rb = window(series, pd.Series(1.0, index=series.index), start, TRAIN_END)["ret"].to_numpy()
            sc[j] = cagr_of(r) - cagr_of(rb)
        tr_scores.append(sc)
    be_score = np.mean(tr_scores, axis=0)

    def pick(scores, fam):
        return max(members[fam], key=lambda i: float(np.median(scores[nbrs[i]])))

    rows = {}
    for f in fams:
        js, jb = pick(sol_score, f), pick(be_score, f)
        jh = max(members[f], key=lambda i: test_m[i]["cagr"])
        beat = sum(test_m[i]["cagr"] > bh_test["cagr"] for i in members[f])
        beat0 = sum(test_m0[i]["cagr"] > bh_test["cagr"] for i in members[f])
        rows[f] = {"name": NAMES[f], "n": len(members[f]),
                   "sol_pick": {"label": label(vs[js]), "test": test_m[js]},
                   "be_pick": {"label": label(vs[jb]), "test": test_m[jb]},
                   "hindsight": {"label": label(vs[jh]), "test": test_m[jh]},
                   "median_test_cagr": float(np.median([test_m[i]["cagr"] for i in members[f]])),
                   "n_beat": int(beat), "n_beat_nocost": int(beat0)}

    # Referanse H1
    btc_s = btc_all.reindex(sol.index).ffill()
    h1 = S.run(sol, {"type": "ens_regime", "set": "lang", "regime": "BTC-USD", "regime_set": "lang",
                     "check": "weekly", "regime_close": btc_s})["target"]
    h1_test = metrics(window(sol, h1, TEST, end))

    # Hvor konsentrert er avkastningen?
    r = sol[(sol.index >= TEST)].pct_change().dropna()
    best = r.sort_values(ascending=False)
    miss = {k: float((1 + r.drop(best.index[:k])).prod() ** (365.25 / len(r)) - 1) for k in (5, 10, 20)}

    f = lambda m: f"{m['cagr']:7.1%} DD {m['max_dd']:6.1%} tr/år {m['trades_per_year']:5.1f}"
    print(f"Testperiode {TEST.date()} – {end.date()}   Kjøp og hold {f(bh_test)}   H1 {f(h1_test)}")
    print(f"Kjøp og hold uten de 5/10/20 beste døgnene: {miss[5]:.1%} / {miss[10]:.1%} / {miss[20]:.1%}")
    for k, rr in rows.items():
        print(f"\n{k:6s} {rr['name']} ({rr['n']} var.)  slår K&H: {rr['n_beat']} (uten kostnad {rr['n_beat_nocost']})  median {rr['median_test_cagr']:.1%}")
        print(f"   valgt på SOL før 2023-04: {rr['sol_pick']['label']:48s} {f(rr['sol_pick']['test'])}")
        print(f"   valgt på BTC/ETH:         {rr['be_pick']['label']:48s} {f(rr['be_pick']['test'])}")
        print(f"   beste i ettertid:         {rr['hindsight']['label']:48s} {f(rr['hindsight']['test'])}")

    def clean(o):
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "sol_extra.json").write_text(json.dumps(clean(
        {"test_start": str(TEST.date()), "test_end": str(end.date()), "buy_hold": bh_test, "h1": h1_test,
         "miss_best_days": miss, "families": rows}), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
