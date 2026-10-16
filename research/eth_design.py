"""ETH-strategidesign: test av forhåndsdefinerte hypoteser H1–H4.

  T  = andel av SMA50/100/150/200 som ETH er over           (H1, egen trend)
  B  = andel av SMA50/100/150/200 som BTC er over           (H2, markedsregime)
  RS = andel av SMA50/100/150/200 som ETH/BTC er over       (H3, relativ styrke)
  V  = min(1, mål-volatilitet / volatilitet siste 30 døgn), i trinn på 25 %   (H4)

Eksponering = kjerne + (1 - kjerne) * kombinasjon, avrundet til nærmeste 12,5 %.
Kostnad 1,0 % per handlet andel, handel døgnet etter signalet.

Kjør:  python research/eth_design.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ksig.backtest import Costs, metrics, simulate  # noqa: E402
from ksig.config import RESULTS_DIR  # noqa: E402
from ksig.data import load_prices  # noqa: E402
from ksig.research import bootstrap_prob  # noqa: E402

COSTS = Costs(0.007, 0.003, 1)
HOLDOUT = pd.Timestamp("2024-10-16")
SETS = {"kort": [20, 50, 100, 200], "middels": [50, 100, 150, 200], "lang": [100, 150, 200, 250, 300]}


def votes(series: pd.Series, wins) -> pd.Series:
    v = [(series > series.rolling(n, min_periods=n).mean()).astype(float).where(series.rolling(n).mean().notna())
         for n in wins]
    return pd.concat(v, axis=1).mean(axis=1, skipna=False)


def vol_scale(close: pd.Series, target: float, window: int = 30) -> pd.Series:
    vol = np.log(close).diff().rolling(window).std() * np.sqrt(365)
    v = (target / vol).clip(upper=1.0)
    return (np.round(v * 4) / 4).clip(lower=0.25)


def weekly(t: pd.Series) -> pd.Series:
    return t.where(t.index.dayofweek == 6).ffill().fillna(0.0)


def build(eth: pd.Series, btc: pd.Series, combo: str, floor: float, check: str,
          sset: str = "middels", vol_target: float = 0.8) -> pd.Series:
    wins = SETS[sset]
    T = votes(eth, wins)
    B = votes(btc, wins)
    RS = votes(eth / btc, wins)
    V = vol_scale(eth, vol_target)
    parts = {"T": T, "B": B, "RS": RS}
    base_key = combo.replace("*V", "")
    keys = base_key.split("+")
    if combo.startswith("T×B"):
        x = T * B
    else:
        x = sum(parts[k] for k in keys) / len(keys)
    if combo.endswith("*V"):
        x = x * V
    x = floor + (1 - floor) * x
    x = (np.round(x * 8) / 8).fillna(0.0)
    return weekly(x) if check == "weekly" else x


def evaluate(eth, tgt, a=None, b=None, costs=COSTS):
    c = eth if a is None else eth[(eth.index >= a) & (eth.index <= b)]
    return simulate(c, tgt[c.index], costs)


def main():
    eth_all = load_prices("ETH-USD")["Close"]
    btc_all = load_prices("BTC-USD")["Close"].reindex(eth_all.index).ffill()
    start = eth_all.index[366]
    eth = eth_all[eth_all.index >= start]
    end = eth.index[-1]
    one = pd.Series(1.0, index=eth_all.index)
    periods = {"full": (start, end), "dev": (start, HOLDOUT - pd.Timedelta(days=1)), "hold": (HOLDOUT, end)}

    def row(name, tgt):
        r = {"name": name}
        for p, (a, b) in periods.items():
            r[p] = metrics(evaluate(eth, tgt, a, b))
        r["cost2"] = metrics(evaluate(eth, tgt, costs=Costs(0.02, 0, 1)))["cagr"]
        r["lag2"] = metrics(evaluate(eth, tgt, costs=Costs(0.007, 0.003, 2)))["cagr"]
        rs = evaluate(eth, tgt)["ret"].to_numpy()
        rb = evaluate(eth, one)["ret"].to_numpy()
        r["p_beat"] = bootstrap_prob(rs, rb, 60, 1000)
        return r

    bh = row("Kjøp og hold", one)
    current = row("Dagens regel (T, kjerne 50 %, ukentlig)", build(eth_all, btc_all, "T", 0.5, "weekly"))
    combos = ["T", "T+B", "T×B", "T+RS", "T+B+RS", "T*V", "T+B*V", "T+B+RS*V"]
    rows = []
    for combo, floor, check in itertools.product(combos, [0.0, 0.25, 0.5], ["daily", "weekly"]):
        tgt = build(eth_all, btc_all, combo, floor, check)
        rows.append({**row(f"{combo} | kjerne {floor:.0%} | {check}", tgt),
                     "combo": combo, "floor": floor, "check": check})

    def passes(r):
        return (all(r[p]["cagr"] > bh[p]["cagr"] for p in periods)
                and r["full"]["max_dd"] >= bh["full"]["max_dd"] + 0.10
                and r["full"]["sharpe"] > current["full"]["sharpe"]
                and r["cost2"] > bh["full"]["cagr"] and r["lag2"] > bh["full"]["cagr"])

    def line(r):
        return (f"{r['name']:42s} full {r['full']['cagr']:6.1%} dev {r['dev']['cagr']:6.1%} hold {r['hold']['cagr']:6.1%} "
                f"DD {r['full']['max_dd']:6.1%} Sh {r['full']['sharpe']:.2f} tr/år {r['full']['trades_per_year']:5.1f} "
                f"k2% {r['cost2']:6.1%} lag2 {r['lag2']:6.1%} P {r['p_beat']:.0%}")
    print(line(bh)); print(line(current)); print("-" * 160)
    for r in sorted(rows, key=lambda r: -r["full"]["sharpe"]):
        print(("✓ " if passes(r) else "  ") + line(r))
    RESULTS_DIR.mkdir(exist_ok=True)
    clean = lambda o: (None if isinstance(o, float) and not np.isfinite(o) else
                       {k: clean(v) for k, v in o.items()} if isinstance(o, dict) else
                       [clean(v) for v in o] if isinstance(o, list) else o)
    (RESULTS_DIR / "eth_design.json").write_text(json.dumps(clean(
        {"buy_hold": bh, "current": current, "rows": rows, "passing": [r["name"] for r in rows if passes(r)]}),
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
