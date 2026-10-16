"""Klassiske, ikke-optimaliserte regler: slår de kjøp og hold i alle delperioder?
Inkluderer blokk-bootstrap av sannsynligheten for at strategien har høyere CAGR enn K&H."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ksig import strategies as S
from ksig.backtest import Costs, metrics, simulate
from ksig.data import load_prices

COSTS = Costs(0.007, 0.003, 1)
RULES = [
    {"type": "sma_band", "n": 200, "band": 0.0, "check": "daily"},
    {"type": "sma_band", "n": 200, "band": 0.02, "check": "weekly"},
    {"type": "sma_band", "n": 150, "band": 0.02, "check": "daily"},
    {"type": "ema_cross", "fast": 50, "slow": 200, "check": "daily"},
    {"type": "ens_sma", "set": "middels", "floor": 0.0, "check": "weekly"},
    {"type": "ens_sma", "set": "middels", "floor": 0.5, "check": "weekly"},
    {"type": "tsmom", "lookback": 365, "band": 0.0, "check": "weekly"},
]
def boot_prob(r_s, r_b, block=60, n=2000, seed=0):
    rng = np.random.default_rng(seed); L = len(r_s); nb = L // block + 1
    d = np.log1p(r_s) - np.log1p(r_b); wins = 0
    for _ in range(n):
        st = rng.integers(0, L - block, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:L]
        wins += d[idx].sum() > 0
    return wins / n
for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]:
    close = load_prices(sym)["Close"]
    start = close.index[366]
    periods = {"full": (start, close.index[-1]), "før 2024-10": (start, pd.Timestamp("2024-10-15")),
               "hold-out": (pd.Timestamp("2024-10-16"), close.index[-1])}
    print(f"\n==== {sym} (fra {start.date()}) ====")
    bhm = {}
    for p, (a, b) in periods.items():
        c = close[(close.index >= a) & (close.index <= b)]
        bhm[p] = metrics(simulate(c, pd.Series(1.0, index=c.index), COSTS))
    print(f"  {'Kjøp og hold':34s}" + "".join(f" | {p}: {bhm[p]['cagr']:6.1%} DD {bhm[p]['max_dd']:6.1%}" for p in periods))
    for spec in RULES:
        t = S.run(close, spec)["target"]
        line = f"  {S.label(spec):34s}"
        for p, (a, b) in periods.items():
            c = close[(close.index >= a) & (close.index <= b)]
            m = metrics(simulate(c, t[c.index], COSTS))
            mark = "✓" if m["cagr"] > bhm[p]["cagr"] else " "
            line += f" | {p}: {m['cagr']:6.1%}{mark} DD {m['max_dd']:6.1%}"
        c = close[close.index >= start]
        rs = simulate(c, t[c.index], COSTS)["ret"].to_numpy()
        rb = simulate(c, pd.Series(1.0, index=c.index), COSTS)["ret"].to_numpy()
        yrs = pd.DataFrame({"s": rs, "b": rb}, index=c.index).groupby(c.index.year).apply(lambda g: (1+g.s).prod() - (1+g.b).prod())
        line += f" | år slått {int((yrs>0).sum())}/{len(yrs)} | P(slår) {boot_prob(rs, rb):.0%}"
        print(line)
