"""Fordeling av HELE porteføljen mellom coinene og kontanter.

Hver coin har sin egen regel som gir en målandel t_i (0–1). Fordelingen gjør
dette om til andeler av totalporteføljen s_i, og resten står i kontanter:

  equal       s_i = t_i / k                                   (fast lik vekt)
  equal_redist  som equal, men ledig kapital fra coins som er ute fordeles på
              coinene som er inne, opp til `max_scale` × lik vekt
  inv_vol     vekter ∝ 1 / volatilitet siste `vol_window` døgn, oppdatert søndag
  momentum    vekter 50/30/20 etter avkastning siste `mom_window` døgn, oppdatert søndag
  fixed       faste vekter fra config

Handel i backtesten: en coin handles bare når målandelen endres med minst
`band` (andel av totalen), eller når den går til/fra 0. Ellers driver andelen
med prisen – slik det blir i praksis når du bare handler ved varsel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import strategies as S
from .backtest import Costs
from .data import load_prices

MOM_WEIGHTS = (0.5, 0.3, 0.2)


def coin_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sluttkurser og målandeler (per coin-pott) på felles dato-indeks."""
    closes, targets = {}, {}
    for a in cfg["assets"]:
        c = load_prices(a["symbol"])["Close"]
        closes[a["short"]] = c
        targets[a["short"]] = S.run(c, a["strategy"])["target"]
    idx = None
    for c in closes.values():
        idx = c.index if idx is None else idx.intersection(c.index)
    C = pd.DataFrame({k: v.reindex(idx) for k, v in closes.items()})
    T = pd.DataFrame({k: v.reindex(idx) for k, v in targets.items()}).fillna(0.0)
    return C, T


def _sunday_ffill(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(df.index.dayofweek == 6, index=df.index)
    return df.where(mask, axis=0).ffill()


def base_weights(C: pd.DataFrame, method: str, params: dict) -> pd.DataFrame:
    k = C.shape[1]
    if method in ("equal", "equal_redist"):
        return pd.DataFrame(1.0 / k, index=C.index, columns=C.columns)
    if method == "fixed":
        w = pd.Series(params["weights"])[C.columns].astype(float)
        w = w / w.sum()
        return pd.DataFrame(np.tile(w.to_numpy(), (len(C), 1)), index=C.index, columns=C.columns)
    if method in ("inv_vol", "inv_vol_redist"):
        vol = np.log(C).diff().rolling(params.get("vol_window", 90)).std()
        inv = 1.0 / vol
        w = inv.div(inv.sum(axis=1), axis=0)
        return _sunday_ffill(w).fillna(1.0 / k)
    if method in ("momentum", "momentum_redist"):
        mom = C / C.shift(params.get("mom_window", 90)) - 1
        ranks = mom.rank(axis=1, ascending=False, method="first")
        mw = tuple(params.get("weights", MOM_WEIGHTS))
        w = ranks.apply(lambda col: col.map({float(i + 1): mw[i] for i in range(len(mw))}))
        w = w.div(w.sum(axis=1), axis=0)
        return _sunday_ffill(w).fillna(1.0 / k)
    raise ValueError(method)


def allocate(T: pd.DataFrame, C: pd.DataFrame, method: str, params: dict | None = None) -> pd.DataFrame:
    """Andel av totalporteføljen per coin. Kontanter = 1 - sum."""
    params = params or {}
    w = base_weights(C, method, params)
    s = w * T
    if method.endswith("_redist") or params.get("redistribute"):
        cap = params.get("max_scale", 2.0)
        tot = s.sum(axis=1)
        scale = (1.0 / tot.replace(0, np.nan)).clip(upper=cap).fillna(0.0)
        s = s.mul(scale, axis=0)
    step = params.get("round_to", 0.025)
    return (np.round(s / step) * step).clip(lower=0.0)


def simulate_portfolio(C: pd.DataFrame, shares: pd.DataFrame, costs: Costs, band: float = 0.05,
                       start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Simulerer porteføljen med drift, handelsbånd og kostnader. Start: 100 % kontanter."""
    if start is not None:
        C, shares = C[C.index >= start], shares[shares.index >= start]
    lag = max(1, int(costs.lag_days))
    D = shares.shift(lag - 1).fillna(0.0).to_numpy()
    R = C.pct_change().fillna(0.0).to_numpy()
    n, k = R.shape
    v = np.zeros(k)
    cash = 1.0
    applied = np.zeros(k)
    eq = np.empty(n)
    cost_arr = np.zeros(n)
    traded = np.zeros(n)
    expo = np.zeros((n, k))
    fee = costs.per_trade
    for t in range(n):
        v = v * (1 + R[t])
        E = v.sum() + cash
        expo[t] = v / E if E > 0 else 0
        d = D[t]
        for i in range(k):
            change = abs(d[i] - applied[i])
            if change < 1e-12:
                continue
            if change >= band - 1e-12 or d[i] == 0 or applied[i] == 0:
                new = d[i] * E
                delta = new - v[i]
                c = abs(delta) * fee
                v[i] = new
                cash -= delta + c
                cost_arr[t] += c / E
                traded[t] += abs(delta) / E
                applied[i] = d[i]
        eq[t] = v.sum() + cash
    eq_s = pd.Series(eq, index=C.index)
    ret = eq_s.pct_change()
    ret.iloc[0] = eq[0] - 1
    out = pd.DataFrame({"equity": eq, "ret": ret.to_numpy(), "cost": cost_arr, "traded": traded,
                        "exposure": expo.sum(axis=1)}, index=C.index)
    for i, col in enumerate(C.columns):
        out[f"w_{col}"] = expo[:, i]
    return out


def from_cfg(cfg: dict) -> tuple[str, dict, float]:
    pc = dict(cfg.get("portfolio", {"method": "equal"}))
    method = pc.pop("method", "equal")
    band = float(pc.pop("band", 0.05))
    return method, pc, band


def momentum_info(C: pd.DataFrame, window: int) -> pd.DataFrame:
    """Avkastning siste `window` døgn og rang per coin, slik den sto ved siste søndag."""
    mom = C / C.shift(window) - 1
    rank = mom.rank(axis=1, ascending=False, method="first")
    return _sunday_ffill(mom), _sunday_ffill(rank)


def describe(cfg: dict) -> list[str]:
    method, params, band = from_cfg(cfg)
    k = len(cfg["assets"])
    if method == "momentum":
        w = params.get("weights", MOM_WEIGHTS)
        ws = "/".join(f"{x * 100:g}" for x in w)
        return [f"Hver søndag rangeres coinene etter avkastning siste {params.get('mom_window', 90)} døgn. "
                f"Beste får {ws.split('/')[0]} % av porteføljen, nest beste {ws.split('/')[1]} % og dårligste "
                f"{ws.split('/')[2]} %.",
                "Hver coin-regel bestemmer hvor mye av sin del som faktisk er investert. Resten står i kontanter.",
                f"Handle en coin bare når andelen endres minst {band * 100:g} prosentpoeng, eller går til eller fra 0."]
    return [f"Hver coin får 1/{k} av porteføljen. Coin-regelen bestemmer hvor mye av den delen som er investert."]
