"""Backtest med realistisk utførelse.

Forutsetninger (alle kan justeres i config.toml):
- Signal fra døgnslutt t. Med lag_days=1 handles det rett etter døgnslutt,
  til sluttkursen for dag t pluss kostnader. Med lag_days=2 handles det
  først ved neste døgnslutt (et døgn for sent).
- Kostnad per handel = (kurtasje + slippage) x handlet andel av porteføljen.
- Beholdningen simuleres dag for dag. En halv posisjon driver med prisen og
  rebalanseres IKKE gratis hver dag – det handles bare når signalet endres.
- Kontanter gir 0 % avkastning (ingen rente på stablecoins e.l.).
- Ingen skatt er trukket fra (se fallgruver på nettsiden).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365


@dataclass(frozen=True)
class Costs:
    fee: float = 0.001
    slippage: float = 0.0005
    lag_days: int = 1

    @property
    def per_trade(self) -> float:
        return self.fee + self.slippage

    @classmethod
    def from_cfg(cls, d: dict) -> "Costs":
        return cls(fee=d.get("fee", 0.001), slippage=d.get("slippage", 0.0005),
                   lag_days=int(d.get("lag_days", 1)))


def simulate(close: pd.Series, target: pd.Series, costs: Costs) -> pd.DataFrame:
    """Simulerer en portefølje som starter med 1,0 i kontanter.

    Kolonner i resultatet:
      exposure – andel i coin gjennom dagen (etter drift)
      ret      – netto dagsavkastning etter kostnader
      cost     – kostnad trukket ved døgnslutt (andel av egenkapital)
      traded   – handlet andel ved døgnslutt
      equity   – egenkapital ved døgnslutt
    """
    target = target.reindex(close.index).fillna(0.0)
    r = close.pct_change().fillna(0.0).to_numpy()
    lag = max(1, int(costs.lag_days))
    desired = target.shift(lag - 1).fillna(0.0).to_numpy()
    n = len(r)
    equity = np.empty(n)
    exposure = np.empty(n)
    cost_arr = np.zeros(n)
    traded = np.zeros(n)
    eq, w, prev_desired = 1.0, 0.0, None
    k = costs.per_trade
    for i in range(n):
        exposure[i] = w
        gross = w * r[i]
        eq *= 1.0 + gross
        if eq > 0 and gross != -1:
            w = w * (1.0 + r[i]) / (1.0 + gross)
        d = desired[i]
        if prev_desired is None or d != prev_desired:
            delta = abs(d - w)
            if delta > 1e-12:
                cost_arr[i] = delta * k
                traded[i] = delta
                eq *= 1.0 - cost_arr[i]
            w = d
        prev_desired = d
        equity[i] = eq
    eq_s = pd.Series(equity, index=close.index)
    ret = eq_s.pct_change()
    ret.iloc[0] = equity[0] - 1.0
    return pd.DataFrame({"exposure": exposure, "ret": ret.to_numpy(), "cost": cost_arr,
                         "traded": traded, "equity": equity}, index=close.index)


def max_drawdown(equity: pd.Series) -> float:
    eq = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)])
    return float((eq / eq.cummax() - 1).min())


def metrics(res: pd.DataFrame) -> dict:
    ret, eq = res["ret"], res["equity"]
    days = len(res)
    years = days / 365.25
    final = float(eq.iloc[-1])
    std = ret.std()
    downside = ret[ret < 0].std()
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else float("nan")
    mdd = max_drawdown(eq)
    trades = int((res["traded"] > 0).sum())
    return {
        "start": str(res.index[0].date()),
        "end": str(res.index[-1].date()),
        "years": round(years, 2),
        "final_multiple": final,
        "total_return": final - 1,
        "cagr": cagr,
        "vol": float(std * np.sqrt(DAYS_PER_YEAR)),
        "sharpe": float(ret.mean() / std * np.sqrt(DAYS_PER_YEAR)) if std > 0 else float("nan"),
        "sortino": float(ret.mean() / downside * np.sqrt(DAYS_PER_YEAR)) if downside > 0 else float("nan"),
        "max_dd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "time_in_market": float((res["exposure"] > 0).mean()),
        "avg_exposure": float(res["exposure"].mean()),
        "trades": trades,
        "trades_per_year": trades / years if years > 0 else float("nan"),
        "cost_drag": float(res["cost"].sum()),
        "worst_day": float(ret.min()),
    }


def sharpe_of(ret: pd.Series) -> float:
    std = ret.std()
    return float(ret.mean() / std * np.sqrt(DAYS_PER_YEAR)) if std and std > 0 else float("-inf")


def yearly_returns(res: pd.DataFrame) -> pd.Series:
    eq = res["equity"]
    by_year = eq.groupby(eq.index.year).last()
    start = pd.Series([1.0], index=[by_year.index[0] - 1])
    full = pd.concat([start, by_year])
    return (full / full.shift(1) - 1).iloc[1:]
