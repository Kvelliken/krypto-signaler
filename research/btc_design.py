"""BTC-strategidesign: test av forhåndsdefinerte hypoteser H1–H4.

  T = andel av SMA50/100/150/200 som BTC er over                  (H1, egen trend)
  M = andel av 90-, 180- og 365-døgns avkastning som er positiv  (H2, tidsseriemomentum)
  E = andel av SMA50/100/150/200 som ETH er over                  (H3, risikoappetitt)
  V = min(1, 60 % / volatilitet siste 30 døgn), i trinn på 25 %   (H4, volatilitet)

Eksponering = kjerne + (1 - kjerne) * kombinasjon, avrundet til nærmeste 12,5 %.
Kostnad 1,0 % per handlet andel, handel døgnet etter signalet.

To evalueringsvinduer:
  lang   – fra sept. 2015 (bare BTC-data; H3 utelatt)
  felles – fra nov. 2018 (alle hypoteser, samme vindu som ETH-designet)

Kjør:  python research/btc_design.py
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
from ksig.config import RESULTS_DIR  # noqa: E402
from ksig.data import load_prices  # noqa: E402
from ksig.research import bootstrap_prob  # noqa: E402

COSTS = Costs(0.007, 0.003, 1)
HOLDOUT = pd.Timestamp("2024-10-16")
SMA_WINS = [50, 100, 150, 200]
MOM_LBS = [90, 180, 365]
VOL_TARGET = 0.6


def votes_sma(series: pd.Series, wins=SMA_WINS) -> pd.Series:
    v = []
    for n in wins:
        m = series.rolling(n, min_periods=n).mean()
        v.append((series > m).astype(float).where(m.notna()))
    return pd.concat(v, axis=1).mean(axis=1, skipna=False)


def votes_mom(series: pd.Series, lbs=MOM_LBS) -> pd.Series:
    v = []
    for lb in lbs:
        r = series / series.shift(lb) - 1
        v.append((r > 0).astype(float).where(r.notna()))
    return pd.concat(v, axis=1).mean(axis=1, skipna=False)


def vol_scale(close: pd.Series, target: float = VOL_TARGET, window: int = 30) -> pd.Series:
    vol = np.log(close).diff().rolling(window).std() * np.sqrt(365)
    v = (target / vol).clip(upper=1.0)
    return (np.round(v * 4) / 4).clip(lower=0.25)


def weekly(t: pd.Series, day: int = 6) -> pd.Series:
    return t.where(t.index.dayofweek == day).ffill().fillna(0.0)


def build(btc: pd.Series, eth: pd.Series, combo: str, floor: float, check: str, day: int = 6,
          sma_wins=SMA_WINS, mom_lbs=MOM_LBS, vol_target=VOL_TARGET) -> pd.Series:
    T, M, E = votes_sma(btc, sma_wins), votes_mom(btc, mom_lbs), votes_sma(eth, sma_wins)
    base = combo.replace("*V", "")
    if base == "T×E":
        x = T * E
    else:
        parts = {"T": T, "M": M, "E": E}
        keys = base.split("+")
        x = sum(parts[k] for k in keys) / len(keys)
    if combo.endswith("*V"):
        x = x * vol_scale(btc, vol_target)
    x = floor + (1 - floor) * x
    x = (np.round(x * 8) / 8).fillna(0.0)
    return weekly(x, day) if check == "weekly" else x


def main():
    btc = load_prices("BTC-USD")["Close"]
    eth = load_prices("ETH-USD")["Close"].reindex(btc.index)  # NaN før ETH finnes
    one = pd.Series(1.0, index=btc.index)
    windows = {"lang": btc.index[366], "felles": pd.Timestamp("2018-11-10")}
    end = btc.index[-1]
    current_spec = {"type": "sma_band", "n": 150, "band": 0.02, "check": "daily"}
    current = S.run(btc, current_spec)["target"]
    current_w = S.run(btc, {**current_spec, "check": "weekly"})["target"]

    combos = ["T", "M", "T+M", "T×E", "T+E", "T+M+E", "T*V", "T+M*V"]
    out = {}
    for wname, start in windows.items():
        c = btc[btc.index >= start]
        periods = {"full": (start, end), "dev": (start, HOLDOUT - pd.Timedelta(days=1)), "hold": (HOLDOUT, end)}

        def ev(tgt, a, b, costs=COSTS):
            cc = c[(c.index >= a) & (c.index <= b)]
            return metrics(simulate(cc, tgt[cc.index], costs))

        def row(name, tgt):
            r = {"name": name}
            for p, (a, b) in periods.items():
                r[p] = ev(tgt, a, b)
            r["cost2"] = ev(tgt, start, end, Costs(0.02, 0, 1))["cagr"]
            r["lag2"] = ev(tgt, start, end, Costs(0.007, 0.003, 2))["cagr"]
            rs = simulate(c, tgt[c.index], COSTS)["ret"].to_numpy()
            rb = simulate(c, one[c.index], COSTS)["ret"].to_numpy()
            r["p_beat"] = bootstrap_prob(rs, rb, 60, 1000)
            return r

        bh = row("Kjøp og hold", one)
        cur = row("Dagens regel (SMA150 ±2 %, daglig)", current)
        curw = row("SMA150 ±2 %, ukentlig", current_w)
        rows = []
        for combo, floor, check in itertools.product(combos, [0.0, 0.25, 0.5], ["daily", "weekly"]):
            if wname == "lang" and "E" in combo:
                continue
            rows.append({**row(f"{combo} | kjerne {floor:.0%} | {check}", build(btc, eth, combo, floor, check)),
                         "combo": combo, "floor": floor, "check": check})

        def passes(r):
            return (all(r[p]["cagr"] > bh[p]["cagr"] for p in periods)
                    and r["full"]["max_dd"] >= bh["full"]["max_dd"] + 0.10
                    and r["full"]["sharpe"] > cur["full"]["sharpe"]
                    and r["cost2"] > bh["full"]["cagr"] and r["lag2"] > bh["full"]["cagr"])

        def line(r):
            return (f"{r['name']:38s} full {r['full']['cagr']:6.1%} dev {r['dev']['cagr']:6.1%} hold {r['hold']['cagr']:6.1%} "
                    f"DD {r['full']['max_dd']:6.1%} Sh {r['full']['sharpe']:.2f} tr/år {r['full']['trades_per_year']:5.1f} "
                    f"k2% {r['cost2']:6.1%} lag2 {r['lag2']:6.1%} P {r['p_beat']:.0%}")
        print(f"\n===== Vindu {wname} fra {start.date()} =====")
        for r in (bh, cur, curw):
            print("  " + line(r))
        print("  " + "-" * 150)
        for r in sorted(rows, key=lambda r: -r["full"]["sharpe"])[:20]:
            print(("✓ " if passes(r) else "  ") + line(r))
        out[wname] = {"start": str(start.date()), "buy_hold": bh, "current": cur, "current_weekly": curw,
                      "rows": rows, "passing": [r["name"] for r in rows if passes(r)]}
    RESULTS_DIR.mkdir(exist_ok=True)

    def clean(o):
        if isinstance(o, float):
            return None if not np.isfinite(o) else round(o, 6)
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    (RESULTS_DIR / "btc_design.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
