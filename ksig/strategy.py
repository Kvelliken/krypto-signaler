"""Indikatorer og posisjonsregler.

Alle indikatorer bruker bare data til og med dagen de beregnes for.
Signalet for dag t er altså kjent ved døgnslutt t, og backtesten handler
tidligst etter det (se backtest.simulate).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    sma: int = 200
    rsi_period: int = 14
    rsi_exit: float = 85      # >= 100 betyr ingen gevinstsikring
    rsi_reentry: float = 70
    trim_to: float = 0.5

    @property
    def uses_rsi(self) -> bool:
        return self.rsi_exit < 100

    def label(self) -> str:
        if not self.uses_rsi:
            return f"SMA{self.sma}"
        return f"SMA{self.sma} · RSI>{self.rsi_exit:g}→{self.trim_to:.0%} · tilbake ≤{self.rsi_reentry:g}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_cfg(cls, d: dict) -> "Params":
        return cls(**{k: d[k] for k in ("sma", "rsi_period", "rsi_exit", "rsi_reentry", "trim_to") if k in d})


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & avg_gain.notna()] = 100.0
    return out


def compute(close: pd.Series, p: Params) -> pd.DataFrame:
    """Returnerer indikatorer, ønsket eksponering (target) og tilstand per dag.

    Regler, vurdert ved hver døgnslutt:
      1. Pris under SMA                     -> 0 %, og gevinstsikring nullstilles
      2. Over SMA, ikke trimmet, RSI > exit -> trim til trim_to (f.eks. 50 %)
      3. Trimmet og RSI <= reentry          -> tilbake til 100 %
      4. Ellers                             -> uendret tilstand
    Før SMA kan beregnes (oppvarming) er target 0.
    """
    s = sma(close, p.sma)
    r = rsi_wilder(close, p.rsi_period)
    c, sv, rv = close.to_numpy(), s.to_numpy(), r.to_numpy()
    target = np.zeros(len(close))
    trimmed_arr = np.zeros(len(close), dtype=bool)
    trimmed = False
    for i in range(len(close)):
        if np.isnan(sv[i]) or np.isnan(rv[i]):
            trimmed = False
            continue
        if c[i] < sv[i]:
            trimmed = False
            target[i] = 0.0
            continue
        if p.uses_rsi:
            if trimmed:
                if rv[i] <= p.rsi_reentry:
                    trimmed = False
            elif rv[i] > p.rsi_exit:
                trimmed = True
        target[i] = p.trim_to if trimmed else 1.0
        trimmed_arr[i] = trimmed
    return pd.DataFrame({"close": close, "sma": s, "rsi": r,
                         "target": target, "trimmed": trimmed_arr}, index=close.index)


def warmup_days(p: Params) -> int:
    return max(p.sma, p.rsi_period + 1)
