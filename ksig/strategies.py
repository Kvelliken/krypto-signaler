"""Strategifamilier. Alle er long/flat: målposisjon mellom 0 og 1.

Hver familie har:
  compute(close, **p) -> DataFrame med 'target' og indikatorkolonner
  label(p)            -> kort navn
  rules(p)            -> regler i klartekst (norsk)
  display(row, p)     -> nøkkeltall for nettsiden: [(navn, type, verdi)]
  warmup(p)           -> antall dager før signalet er gyldig

Alle signaler bruker bare data til og med dagen de beregnes for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import Params, compute as compute_sma_rsi


def _fmt_band(b: float) -> str:
    return f"{b * 100:g} %".replace(".", ",")


# ---------------------------------------------------------------------------
# 1. SMA med bånd (hysterese) og valgfri ukentlig sjekk
# ---------------------------------------------------------------------------
def sma_band(close: pd.Series, n: int, band: float = 0.0, check: str = "daily") -> pd.DataFrame:
    s = close.rolling(n, min_periods=n).mean()
    upper, lower = s * (1 + band), s * (1 - band)
    c, up, lo = close.to_numpy(), upper.to_numpy(), lower.to_numpy()
    weekly = check == "weekly"
    sunday = (close.index.dayofweek == 6)
    target = np.zeros(len(c))
    inside = False
    for i in range(len(c)):
        if np.isnan(up[i]):
            continue
        if not weekly or sunday[i]:
            if c[i] > up[i]:
                inside = True
            elif c[i] < lo[i]:
                inside = False
        target[i] = 1.0 if inside else 0.0
    return pd.DataFrame({"target": target, "sma": s, "upper": upper, "lower": lower}, index=close.index)


def _sma_band_rules(p):
    n, b, chk = p["n"], p.get("band", 0), p.get("check", "daily")
    when = "ved døgnslutt" if chk == "daily" else "ved døgnslutt søndag (sjekkes én gang i uken)"
    if b:
        return [f"Kjøp til 100 % når prisen {when} er mer enn {_fmt_band(b)} over SMA{n}.",
                f"Selg alt når prisen {when} er mer enn {_fmt_band(b)} under SMA{n}.",
                "Mellom grensene beholdes posisjonen (hindrer frem-og-tilbake-handler)."]
    return [f"100 % når prisen {when} er over SMA{n}, ellers 0 %."]


def _sma_band_display(row, p):
    out = [(f"SMA{p['n']}", "money", row["sma"])]
    if p.get("band"):
        out += [("Kjøpsgrense", "money", row["upper"]), ("Salgsgrense", "money", row["lower"])]
    dist = row["close"] / row["sma"] - 1 if row["sma"] == row["sma"] else None
    out.append((f"Avstand til SMA{p['n']}", "pct_signed", dist))
    return out


# ---------------------------------------------------------------------------
# 2. EMA-kryss
# ---------------------------------------------------------------------------
def ema_cross(close: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    ef = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    es = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    target = ((ef > es) & es.notna()).astype(float)
    return pd.DataFrame({"target": target, "ema_fast": ef, "ema_slow": es}, index=close.index)


def _ema_rules(p):
    return [f"100 % når EMA{p['fast']} er over EMA{p['slow']} ved døgnslutt, ellers 0 %."]


def _ema_display(row, p):
    return [(f"EMA{p['fast']}", "money", row["ema_fast"]), (f"EMA{p['slow']}", "money", row["ema_slow"]),
            ("Avstand mellom snittene", "pct_signed", row["ema_fast"] / row["ema_slow"] - 1)]


# ---------------------------------------------------------------------------
# 3. Donchian-utbrudd
# ---------------------------------------------------------------------------
def donchian(close: pd.Series, entry: int, exit: int) -> pd.DataFrame:
    hi = close.shift(1).rolling(entry, min_periods=entry).max()
    lo = close.shift(1).rolling(exit, min_periods=exit).min()
    c, h, l = close.to_numpy(), hi.to_numpy(), lo.to_numpy()
    target = np.zeros(len(c))
    inside = False
    for i in range(len(c)):
        if np.isnan(h[i]) or np.isnan(l[i]):
            continue
        if not inside and c[i] > h[i]:
            inside = True
        elif inside and c[i] < l[i]:
            inside = False
        target[i] = 1.0 if inside else 0.0
    return pd.DataFrame({"target": target, "entry_level": hi, "exit_level": lo}, index=close.index)


def _donchian_rules(p):
    return [f"Kjøp til 100 % når døgnslutten er høyere enn alle de {p['entry']} foregående døgnene.",
            f"Selg alt når døgnslutten er lavere enn alle de {p['exit']} foregående døgnene."]


def _donchian_display(row, p):
    return [(f"Kjøpsnivå ({p['entry']}d høy)", "money", row["entry_level"]),
            (f"Salgsnivå ({p['exit']}d lav)", "money", row["exit_level"])]


# ---------------------------------------------------------------------------
# 4. Tidsseriemomentum
# ---------------------------------------------------------------------------
def tsmom(close: pd.Series, lookback: int, band: float = 0.0) -> pd.DataFrame:
    mom = close / close.shift(lookback) - 1
    m = mom.to_numpy()
    target = np.zeros(len(m))
    inside = False
    for i in range(len(m)):
        if np.isnan(m[i]):
            continue
        if m[i] > band:
            inside = True
        elif m[i] < -band:
            inside = False
        target[i] = 1.0 if inside else 0.0
    return pd.DataFrame({"target": target, "momentum": mom}, index=close.index)


def _tsmom_rules(p):
    b = p.get("band", 0)
    if b:
        return [f"Kjøp til 100 % når prisen er mer enn {_fmt_band(b)} høyere enn for {p['lookback']} døgn siden.",
                f"Selg alt når den er mer enn {_fmt_band(b)} lavere enn for {p['lookback']} døgn siden."]
    return [f"100 % når prisen er høyere enn for {p['lookback']} døgn siden, ellers 0 %."]


def _tsmom_display(row, p):
    return [(f"Endring siste {p['lookback']} døgn", "pct_signed", row["momentum"])]


# ---------------------------------------------------------------------------
# 4b. Glidende stopp: bli inne, gå ut bare ved dypt fall fra toppen
# ---------------------------------------------------------------------------
def trail(close: pd.Series, stop: float = 0.3, reentry: int = 100) -> pd.DataFrame:
    s = close.rolling(reentry, min_periods=reentry).mean()
    c, sv = close.to_numpy(), s.to_numpy()
    target = np.zeros(len(c))
    peak_arr = np.full(len(c), np.nan)
    inside, peak = False, np.nan
    for i in range(len(c)):
        if np.isnan(sv[i]):
            continue
        if inside:
            peak = max(peak, c[i])
            if c[i] < peak * (1 - stop):
                inside = False
        elif c[i] > sv[i]:
            inside, peak = True, c[i]
        target[i] = 1.0 if inside else 0.0
        peak_arr[i] = peak if inside else np.nan
    return pd.DataFrame({"target": target, "sma": s, "peak": peak_arr,
                         "stop_level": peak_arr * (1 - stop)}, index=close.index)


def _trail_rules(p):
    return [f"Inne (100 %): selg alt hvis døgnslutten faller mer enn {p['stop']:.0%} under høyeste døgnslutt siden kjøpet.",
            f"Ute (0 %): kjøp igjen når døgnslutten er over SMA{p['reentry']}."]


def _trail_display(row, p):
    out = [(f"SMA{p['reentry']} (kjøpsnivå når ute)", "money", row["sma"])]
    if row["peak"] == row["peak"]:
        out += [("Høyeste siden kjøp", "money", row["peak"]), ("Stoppnivå", "money", row["stop_level"])]
    return out


# ---------------------------------------------------------------------------
# 5. Ensembler: flere trendsignaler stemmer, eksponering i trinn
# ---------------------------------------------------------------------------
SMA_SETS = {"kort": [20, 50, 100, 200], "middels": [50, 100, 150, 200], "lang": [100, 150, 200, 250, 300]}
MOM_SETS = {"kort": [20, 40, 60, 90, 120], "middels": [30, 60, 90, 180, 365], "lang": [60, 120, 180, 270, 365]}


def ens_sma(close: pd.Series, set: str = "middels", floor: float = 0.0) -> pd.DataFrame:
    wins = SMA_SETS[set]
    cols, votes = {}, []
    for n in wins:
        s = close.rolling(n, min_periods=n).mean()
        cols[f"sma{n}"] = s
        votes.append((close > s).astype(float).where(s.notna()))
    frac = pd.concat(votes, axis=1).mean(axis=1, skipna=False)
    target = (floor + (1 - floor) * frac).fillna(0.0)
    return pd.DataFrame({"target": target, "votes": frac, **cols}, index=close.index)


def ens_mom(close: pd.Series, set: str = "middels", floor: float = 0.0) -> pd.DataFrame:
    lbs = MOM_SETS[set]
    cols, votes = {}, []
    for lb in lbs:
        m = close / close.shift(lb) - 1
        cols[f"mom{lb}"] = m
        votes.append((m > 0).astype(float).where(m.notna()))
    frac = pd.concat(votes, axis=1).mean(axis=1, skipna=False)
    target = (floor + (1 - floor) * frac).fillna(0.0)
    return pd.DataFrame({"target": target, "votes": frac, **cols}, index=close.index)


def _floor_text(p):
    f = p.get("floor", 0)
    return [f"Behold alltid minst {f:.0%} som kjerneposisjon; resten styres av signalene."] if f else []


def _ens_sma_rules(p):
    wins = SMA_SETS[p.get("set", "middels")]
    k = len(wins)
    return [f"Sjekk om prisen er over hvert av snittene SMA{', SMA'.join(map(str, wins))}.",
            f"Hvert snitt prisen er over, gir 1/{k} av posisjonen ({1 / k:.0%}). Alle over = 100 %, ingen over = 0 %.",
            *_floor_text(p)]


def _ens_mom_rules(p):
    lbs = MOM_SETS[p.get("set", "middels")]
    k = len(lbs)
    return [f"Sjekk om prisen er høyere enn for {', '.join(map(str, lbs))} døgn siden.",
            f"Hvert «ja» gir 1/{k} av posisjonen ({1 / k:.0%}). Alle ja = 100 %, ingen = 0 %.",
            *_floor_text(p)]


def _ens_sma_display(row, p):
    return [(f"Over SMA{n}", "check", row["close"] > row[f"sma{n}"]) for n in SMA_SETS[p.get("set", "middels")]]


def _ens_mom_display(row, p):
    return [(f"Endring {lb} døgn", "pct_signed", row[f"mom{lb}"]) for lb in MOM_SETS[p.get("set", "middels")]]


# ---------------------------------------------------------------------------
# 6. Original: SMA + RSI-gevinstsikring
# ---------------------------------------------------------------------------
def sma_rsi(close: pd.Series, sma: int = 200, rsi_period: int = 14, rsi_exit: float = 85,
            rsi_reentry: float = 70, trim_to: float = 0.5) -> pd.DataFrame:
    df = compute_sma_rsi(close, Params(sma, rsi_period, rsi_exit, rsi_reentry, trim_to))
    return df.drop(columns=["close"])


def _sma_rsi_rules(p):
    r = [f"Pris under SMA{p['sma']} ved døgnslutt → 0 %. Over → 100 %."]
    if p.get("rsi_exit", 100) < 100:
        r.append(f"RSI({p.get('rsi_period', 14)}) over {p['rsi_exit']:g} → selg ned til {p.get('trim_to', 0.5):.0%}; "
                 f"kjøp tilbake når RSI ≤ {p['rsi_reentry']:g}.")
    return r


def _sma_rsi_display(row, p):
    return [(f"SMA{p['sma']}", "money", row["sma"]),
            (f"Avstand til SMA{p['sma']}", "pct_signed", row["close"] / row["sma"] - 1),
            ("RSI", "num", row["rsi"])]


# ---------------------------------------------------------------------------
# 7. Egen trend × markedsregime (f.eks. ETH × BTC-trend)
#    Spesifikasjon: dokumentet "ETH-strategi: design og spesifikasjon" og research/eth_design.py
# ---------------------------------------------------------------------------
def _regime_short(symbol: str) -> str:
    return symbol.split("-")[0]


def _load_regime(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    from .data import load_prices  # sen import for å unngå sirkulær import
    return load_prices(symbol)["Close"].reindex(index).ffill()


def ens_regime(close: pd.Series, set: str = "middels", regime: str = "BTC-USD",
               regime_set: str | None = None, regime_close: pd.Series | None = None) -> pd.DataFrame:
    """Målandel = T × B / 16 avrundet til nærmeste 1/8 (halvdeler til partall).

    T = antall av snittene i `set` som egen pris er strengt over (0–4)
    B = antall av snittene i `regime_set` som regimets pris er strengt over (0–4)
    """
    wins, rwins = SMA_SETS[set], SMA_SETS[regime_set or set]
    other = regime_close if regime_close is not None else _load_regime(regime, close.index)
    cols = {}

    def votes(series: pd.Series, ns, prefix: str) -> pd.Series:
        v = []
        for n in ns:
            m = series.rolling(n, min_periods=n).mean()
            cols[f"{prefix}{n}"] = m
            v.append((series > m).astype(float).where(m.notna()))
        return pd.concat(v, axis=1).mean(axis=1, skipna=False)

    T = votes(close, wins, "sma")
    B = votes(other, rwins, "rsma")
    raw = (np.round(T * B * 8) / 8).fillna(0.0)
    return pd.DataFrame({"target": raw, "raw_target": raw, "own_votes": T * len(wins),
                         "regime_votes": B * len(rwins), "regime_close": other, **cols}, index=close.index)


def _ens_regime_rules(p):
    wins = SMA_SETS[p.get("set", "middels")]
    rwins = SMA_SETS[p.get("regime_set") or p.get("set", "middels")]
    rs = _regime_short(p.get("regime", "BTC-USD"))
    return [f"Tell hvor mange av SMA{', SMA'.join(map(str, wins))} prisen er over (T, 0–{len(wins)}), "
            f"og hvor mange av SMA{', SMA'.join(map(str, rwins))} {rs} er over (B, 0–{len(rwins)}).",
            f"Målandel = T × B / {len(wins) * len(rwins)}, avrundet til nærmeste 12,5 %: alle over for begge = 100 %, "
            f"4 og 3 = 75 %, 3 og 3 = 50 %, 2 og 2 = 25 %, null for én av dem = 0 %.",
            "Lik pris og snitt teller som ikke over."]


def _votes_text(row, prefix, ns, count):
    below = [f"SMA{n}" for n in ns if not (row["close" if prefix == "sma" else "regime_close"] > row[f"{prefix}{n}"])]
    txt = f"{int(count)} av {len(ns)}"
    return txt + (f" (under {', '.join(below)})" if below and len(below) < len(ns) else "")


def _ens_regime_display(row, p):
    wins = SMA_SETS[p.get("set", "middels")]
    rwins = SMA_SETS[p.get("regime_set") or p.get("set", "middels")]
    rs = _regime_short(p.get("regime", "BTC-USD"))
    out = [("Egen trend: snitt over", "text", _votes_text(row, "sma", wins, row["own_votes"])),
           (f"{rs}-trend: snitt over", "text", _votes_text(row, "rsma", rwins, row["regime_votes"])),
           (f"{rs} sluttkurs", "money", row["regime_close"])]
    if p.get("check") == "weekly":
        out.append(("Mål hvis søndag var i dag", "pct", row["raw_target"]))
    return out


FAMILIES = {
    "ens_regime": dict(fn=ens_regime, rules=_ens_regime_rules, display=_ens_regime_display,
                       label=lambda p: f"Trend × {_regime_short(p.get('regime', 'BTC-USD'))}-trend"
                       + {"kort": " (korte snitt)", "lang": " (lange snitt)"}.get(p.get("set", "middels"), ""),
                       warmup=lambda p: max(max(SMA_SETS[p.get("set", "middels")]),
                                            max(SMA_SETS[p.get("regime_set") or p.get("set", "middels")]))),
    "sma_band": dict(fn=sma_band, rules=_sma_band_rules, display=_sma_band_display,
                     label=lambda p: f"SMA{p['n']}" + (f" ±{_fmt_band(p['band'])}" if p.get("band") else "")
                     + (" ukentlig" if p.get("check") == "weekly" else ""),
                     warmup=lambda p: p["n"]),
    "ema_cross": dict(fn=ema_cross, rules=_ema_rules, display=_ema_display,
                      label=lambda p: f"EMA{p['fast']}/{p['slow']}", warmup=lambda p: p["slow"]),
    "donchian": dict(fn=donchian, rules=_donchian_rules, display=_donchian_display,
                     label=lambda p: f"Donchian {p['entry']}/{p['exit']}",
                     warmup=lambda p: max(p["entry"], p["exit"]) + 1),
    "tsmom": dict(fn=tsmom, rules=_tsmom_rules, display=_tsmom_display,
                  label=lambda p: f"Momentum {p['lookback']}d" + (f" ±{_fmt_band(p['band'])}" if p.get("band") else ""),
                  warmup=lambda p: p["lookback"] + 1),
    "trail": dict(fn=trail, rules=_trail_rules, display=_trail_display,
                  label=lambda p: f"Stopp {p['stop']:.0%} / inn over SMA{p['reentry']}",
                  warmup=lambda p: p["reentry"]),
    "ens_sma": dict(fn=ens_sma, rules=_ens_sma_rules, display=_ens_sma_display,
                    label=lambda p: f"SMA-ensemble {p.get('set', 'middels')}"
                    + (f" kjerne {p['floor']:.0%}" if p.get("floor") else ""),
                    warmup=lambda p: max(SMA_SETS[p.get("set", "middels")])),
    "ens_mom": dict(fn=ens_mom, rules=_ens_mom_rules, display=_ens_mom_display,
                    label=lambda p: f"Momentum-ensemble {p.get('set', 'middels')}"
                    + (f" kjerne {p['floor']:.0%}" if p.get("floor") else ""),
                    warmup=lambda p: max(MOM_SETS[p.get("set", "middels")]) + 1),
    "sma_rsi": dict(fn=sma_rsi, rules=_sma_rsi_rules, display=_sma_rsi_display,
                    label=lambda p: Params(**{k: p[k] for k in p}).label(), warmup=lambda p: p.get("sma", 200)),
}


def split_spec(spec: dict) -> tuple[str, dict]:
    spec = dict(spec)
    return spec.pop("type"), spec


def _weekly(target: pd.Series) -> pd.Series:
    """Tillat bare endringer ved døgnslutt søndag; ellers beholdes forrige nivå."""
    sunday = target.index.dayofweek == 6
    return target.where(sunday).ffill().fillna(0.0)


def run(close: pd.Series, spec: dict) -> pd.DataFrame:
    fam, p = split_spec(spec)
    weekly = fam != "sma_band" and p.pop("check", "daily") == "weekly"
    df = FAMILIES[fam]["fn"](close, **p)
    if weekly:
        df["target"] = _weekly(df["target"])
    df.insert(0, "close", close)
    return df


def _weekly_suffix(spec: dict) -> str:
    return " ukentlig" if spec.get("type") != "sma_band" and spec.get("check") == "weekly" else ""


def label(spec: dict) -> str:
    fam, p = split_spec(spec)
    p.pop("check", None) if fam != "sma_band" else None
    return FAMILIES[fam]["label"](p) + _weekly_suffix(spec)


def rules(spec: dict) -> list[str]:
    fam, p = split_spec(spec)
    weekly = fam != "sma_band" and p.pop("check", "daily") == "weekly"
    out = FAMILIES[fam]["rules"](p)
    if weekly:
        out = out + ["Signalet sjekkes bare ved døgnslutt søndag (00:00 UTC natt til mandag); "
                     "resten av uken beholdes posisjonen."]
    return out


def display(row: pd.Series, spec: dict) -> list[tuple]:
    fam, p = split_spec(spec)
    return FAMILIES[fam]["display"](row, p)


def warmup(spec: dict) -> int:
    fam, p = split_spec(spec)
    if fam != "sma_band":
        p.pop("check", None)
    return int(FAMILIES[fam]["warmup"](p))


# ---------------------------------------------------------------------------
# Visning: hvilke linjer som tegnes i minidiagrammet, og "nær signal"-merknader
# ---------------------------------------------------------------------------
def chart_lines(spec: dict) -> list[tuple[str, str]]:
    fam, p = split_spec(spec)
    if fam == "sma_band":
        return [("sma", f"SMA{p['n']}")]
    if fam in ("ens_sma", "ens_regime"):
        wins = SMA_SETS[p.get("set", "middels")]
        return [(f"sma{wins[0]}", f"SMA{wins[0]}"), (f"sma{wins[-1]}", f"SMA{wins[-1]}")]
    if fam == "ema_cross":
        return [("ema_fast", f"EMA{p['fast']}"), ("ema_slow", f"EMA{p['slow']}")]
    if fam == "donchian":
        return [("entry_level", "Kjøpsnivå"), ("exit_level", "Salgsnivå")]
    if fam == "trail":
        return [("sma", f"SMA{p['reentry']}")]
    if fam == "sma_rsi":
        return [("sma", f"SMA{p.get('sma', 200)}")]
    return []


def notes(row: pd.Series, spec: dict, near: float = 0.03) -> list[str]:
    """Merknader om at et signal kan være nær."""
    fam, p = split_spec(spec)
    out = []
    c = row["close"]
    if fam == "sma_band" and row["sma"] == row["sma"]:
        if row["target"] > 0:
            lvl, word = row["lower"], "salgsgrensen"
        else:
            lvl, word = row["upper"], "kjøpsgrensen"
        d = c / lvl - 1
        if abs(d) <= near:
            out.append(f"Prisen er {d:+.1%} fra {word} ({lvl:,.2f} $).".replace(",", " "))
    elif fam == "ens_sma":
        for n in SMA_SETS[p.get("set", "middels")]:
            s = row.get(f"sma{n}")
            if s == s and s and abs(c / s - 1) <= near:
                out.append(f"Prisen er {c / s - 1:+.1%} fra SMA{n} – posisjonen kan endres ved neste sjekk.")
    elif fam == "ens_regime":
        rs = _regime_short(p.get("regime", "BTC-USD"))
        rwins = SMA_SETS[p.get("regime_set") or p.get("set", "middels")]
        for label_, price, prefix, ns in (("Prisen", c, "sma", SMA_SETS[p.get("set", "middels")]),
                                           (rs, row["regime_close"], "rsma", rwins)):
            for n in ns:
                s = row.get(f"{prefix}{n}")
                if s == s and s and abs(price / s - 1) <= near:
                    out.append(f"{label_} er {price / s - 1:+.1%} fra sin SMA{n} – målet kan endres ved neste sjekk.")
        if p.get("check") == "weekly" and abs(row["raw_target"] - row["target"]) > 1e-9:
            pct_txt = f"{row['raw_target'] * 100:g}".replace(".", ",")
            out.append(f"Med dagens priser ville målet vært {pct_txt} % – "
                       "det gjelder først hvis det står slik ved døgnslutt søndag.")
    if p.get("check") == "weekly":
        out.append("Signalet sjekkes bare ved døgnslutt søndag.")
    return out
