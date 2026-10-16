"""Søk etter strategi per coin som slår kjøp og hold.

Metode (bestemt FØR søket, for å begrense datasnoking):
  * Kostnad 1,0 % per handel (Firi 0,7 % + ca. 0,3 % spread/påslag), handel 1 døgn etter signal.
  * Kandidater fra 7 strategifamilier (se GRIDS).
  * Krav: høyere årlig avkastning enn kjøp og hold OG ikke dypere maks fall.
  * Robust poengsum: median over kandidaten og nabo-parameterne (ett steg i hver dimensjon).
  * Finnes ingen kandidat med positiv robust poengsum, velges KJØP OG HOLD.
  * Tre utvelgelsesmetoder testes med walk-forward (hvert halvår, bare data før perioden):
      O1  meravkastning (CAGR) over hele treningsperioden
      O2  median av årlig meravkastning (mindre styrt av ett enkelt år)
      O3  som O1, men rangert sammen med bevis fra de andre coinene
  * Hold-out: de siste to årene vises separat. De ble ikke brukt til å velge familier
    i runde 1; etter det er de sett og regnes ikke lenger som helt rene.

Kjør:  python research/search.py   (skriver results/search.json)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ksig import strategies as S  # noqa: E402
from ksig.backtest import Costs, metrics, simulate  # noqa: E402
from ksig.config import RESULTS_DIR, load_config  # noqa: E402
from ksig.data import load_prices  # noqa: E402

HOLDOUT_START = pd.Timestamp("2024-10-16")
COSTS = Costs(fee=0.007, slippage=0.003, lag_days=1)
MIN_TRAIN_DAYS = 730
STEP_DAYS = 182
HOLD = {"type": "hold"}

GRIDS = {
    "sma_band": {"n": [50, 75, 100, 125, 150, 175, 200, 250, 300],
                 "band": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08], "check": ["daily", "weekly"]},
    "ema_cross": {"fast": [10, 20, 30, 50], "slow": [50, 100, 150, 200], "check": ["daily", "weekly"]},
    "donchian": {"entry": [20, 30, 50, 75, 100], "exit": [10, 20, 30, 50, 75], "check": ["daily", "weekly"]},
    "tsmom": {"lookback": [30, 60, 90, 120, 180, 270, 365], "band": [0.0, 0.05, 0.1, 0.2],
              "check": ["daily", "weekly"]},
    "ens_sma": {"set": ["kort", "middels", "lang"], "floor": [0.0, 0.25, 0.5], "check": ["daily", "weekly"]},
    "ens_mom": {"set": ["kort", "middels", "lang"], "floor": [0.0, 0.25, 0.5], "check": ["daily", "weekly"]},
    "trail": {"stop": [0.2, 0.25, 0.3, 0.4, 0.5], "reentry": [50, 100, 150, 200], "check": ["daily", "weekly"]},
}


def candidates() -> list[dict]:
    out = []
    for fam, grid in GRIDS.items():
        keys = list(grid)
        for vals in itertools.product(*(grid[k] for k in keys)):
            p = dict(zip(keys, vals))
            if fam == "ema_cross" and p["fast"] >= p["slow"]:
                continue
            out.append({"type": fam, **p})
    return out


def neighbor_lists(specs: list[dict]) -> list[np.ndarray]:
    out = []
    for spec in specs:
        grid = GRIDS[spec["type"]]
        idx = []
        for j, other in enumerate(specs):
            if other["type"] != spec["type"]:
                continue
            ok = True
            for k, vals in grid.items():
                a, b = vals.index(spec[k]), vals.index(other[k])
                if (isinstance(vals[0], str) and a != b) or abs(a - b) > 1:
                    ok = False
                    break
            if ok:
                idx.append(j)
        out.append(np.array(idx))
    return out


def label(spec: dict) -> str:
    return "Kjøp og hold" if spec["type"] == "hold" else S.label(spec)


class Coin:
    """Forhåndsberegnede egenkapitalkurver for alle kandidater fra felles start."""

    def __init__(self, symbol: str, specs: list[dict]):
        self.symbol = symbol
        self.close = load_prices(symbol)["Close"]
        warm = max(S.warmup(s) for s in specs)
        self.eval_start = self.close.index[warm]
        self.c = self.close[self.close.index >= self.eval_start]
        self.targets = [S.run(self.close, s)["target"] for s in specs]
        sims = [simulate(self.c, t[self.c.index], COSTS) for t in self.targets]
        self.E = np.column_stack([s["equity"].to_numpy() for s in sims])
        self.bh_target = pd.Series(1.0, index=self.close.index)
        self.bh = simulate(self.c, self.bh_target[self.c.index], COSTS)["equity"].to_numpy()
        self.years = self.c.index.year.to_numpy()

    def train_mask(self, t: pd.Timestamp) -> np.ndarray:
        return np.asarray(self.c.index < t)


def _cagr(E_last, n_days):
    return np.power(np.maximum(E_last, 1e-12), 365.25 / n_days) - 1


def _maxdd(E):
    return (E / np.maximum.accumulate(E, axis=0) - 1).min(axis=0)


def _robust(raw: np.ndarray, ok: np.ndarray, nbrs) -> np.ndarray:
    sc = np.array([np.median(raw[n]) for n in nbrs])
    sc[~ok] = -np.inf
    return sc


def score_o1(coin: Coin, mask, nbrs):
    E, b, n = coin.E[mask], coin.bh[mask], mask.sum()
    raw = _cagr(E[-1], n) - _cagr(np.array([b[-1]]), n)[0]
    ok = _maxdd(E) >= _maxdd(b[:, None])[0]
    return _robust(raw, ok, nbrs)


def score_o2(coin: Coin, mask, nbrs):
    E, b = coin.E[mask], coin.bh[mask]
    yrs = coin.years[mask]
    ends = np.r_[np.nonzero(np.diff(yrs))[0], len(yrs) - 1]
    Ey = np.vstack([np.ones((1, E.shape[1])), E[ends]])
    by = np.r_[1.0, b[ends]]
    ry = Ey[1:] / Ey[:-1] - 1
    rb = by[1:] / by[:-1] - 1
    raw = np.median(ry - rb[:, None], axis=0)
    ok = _maxdd(E) >= _maxdd(b[:, None])[0]
    return _robust(raw, ok, nbrs)


def pct_rank(x: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(x), x, -1e9)
    return pd.Series(finite).rank(pct=True).to_numpy()


def choose(scores: np.ndarray, pooled: np.ndarray | None = None) -> int | None:
    """Indeks for valgt kandidat, eller None = kjøp og hold."""
    if not np.isfinite(scores).any() or np.nanmax(scores[np.isfinite(scores)]) <= 0:
        return None
    if pooled is None:
        return int(np.argmax(scores))
    eligible = np.where(scores > 0, pooled, -np.inf)
    return int(np.argmax(eligible))


def selection(coins: dict[str, Coin], name: str, t: pd.Timestamp, method: str, nbrs) -> int | None:
    coin = coins[name]
    mask = coin.train_mask(t)
    if method == "O1":
        return choose(score_o1(coin, mask, nbrs))
    if method == "O2":
        return choose(score_o2(coin, mask, nbrs))
    own = score_o1(coin, mask, nbrs)
    ranks = [pct_rank(own)]
    for other_name, other in coins.items():
        if other_name == name:
            continue
        om = other.train_mask(t)
        if om.sum() >= 365:
            ranks.append(pct_rank(score_o1(other, om, nbrs)))
    pooled = 0.5 * ranks[0] + 0.5 * np.mean(ranks[1:], axis=0) if len(ranks) > 1 else ranks[0]
    return choose(own, pooled)


def walk_forward(coins, name, method, specs, nbrs):
    coin = coins[name]
    first = coin.eval_start + pd.Timedelta(days=MIN_TRAIN_DAYS)
    end = coin.c.index[-1]
    starts = list(pd.date_range(first, end, freq=f"{STEP_DAYS}D"))
    stitched = pd.Series(1.0, index=coin.c.index)
    segs = []
    for k, st in enumerate(starts):
        en = starts[k + 1] - pd.Timedelta(days=1) if k + 1 < len(starts) else end
        j = selection(coins, name, st, method, nbrs)
        tgt = coin.bh_target if j is None else coin.targets[j]
        m = (coin.c.index >= st) & (coin.c.index <= en)
        stitched[m] = tgt[coin.c.index[m]]
        segs.append({"start": str(st.date()), "end": str(en.date()),
                     "label": label(HOLD if j is None else specs[j])})
    oos = coin.c[coin.c.index >= first]
    res = simulate(oos, stitched[oos.index], COSTS)
    hold = oos[oos.index >= HOLDOUT_START]
    return {"start": str(first.date()), "segments": segs,
            "strategy": metrics(res),
            "buy_hold": metrics(simulate(oos, coin.bh_target[oos.index], COSTS)),
            "holdout_strategy": metrics(simulate(hold, stitched[hold.index], COSTS)),
            "holdout_buy_hold": metrics(simulate(hold, coin.bh_target[hold.index], COSTS)),
            "equity": {"dates": [str(d.date()) for d in res.index[::3]],
                       "strategy": [round(float(v) * 100, 3) for v in res["equity"].iloc[::3]],
                       "buy_hold": [round(float(v) * 100, 3) for v in
                                    simulate(oos, coin.bh_target[oos.index], COSTS)["equity"].iloc[::3]]}}


def beats(r: dict) -> bool:
    return r["strategy"]["cagr"] > r["buy_hold"]["cagr"] and r["strategy"]["max_dd"] >= r["buy_hold"]["max_dd"]


def fmt(m):
    return (f"CAGR {m['cagr']:7.1%}  Sharpe {m['sharpe']:5.2f}  MaxDD {m['max_dd']:7.1%}  "
            f"handler/år {m['trades_per_year']:4.1f}")


def clean(o):
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def main():
    cfg = load_config()
    specs = candidates()
    nbrs = neighbor_lists(specs)
    coins = {a["short"]: Coin(a["symbol"], specs) for a in cfg["assets"]}
    print(f"{len(specs)} kandidater per coin, kostnad {COSTS.per_trade:.1%} per handel")
    out = {"n_candidates": len(specs), "costs": COSTS.per_trade, "methods": {}}
    for method in ("O1", "O2", "O3"):
        print(f"\n######## Metode {method} ########")
        out["methods"][method] = {}
        for name in coins:
            r = walk_forward(coins, name, method, specs, nbrs)
            now = selection(coins, name, coins[name].c.index[-1] + pd.Timedelta(days=1), method, nbrs)
            r["current_pick"] = HOLD if now is None else specs[now]
            r["current_label"] = label(r["current_pick"])
            r["beats"] = beats(r)
            out["methods"][method][name] = r
            print(f"{name} WF fra {r['start']}  {'SLÅR K&H' if r['beats'] else 'slår ikke'}")
            print(f"   strategi {fmt(r['strategy'])}")
            print(f"   K&H      {fmt(r['buy_hold'])}")
            print(f"   hold-out strategi {r['holdout_strategy']['cagr']:6.1%} DD {r['holdout_strategy']['max_dd']:6.1%}"
                  f" | K&H {r['holdout_buy_hold']['cagr']:6.1%} DD {r['holdout_buy_hold']['max_dd']:6.1%}")
            print("   valg:", " → ".join(dict.fromkeys(s["label"] for s in r["segments"])))
            print("   valg i dag:", r["current_label"])
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "search.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
