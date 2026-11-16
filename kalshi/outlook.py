"""Beregner markedets «utsikt» for BTC ut fra Kalshi-priser på ett tidspunkt.

Kalshi-priser er sannsynligheter slik markedet setter dem (risikonøytrale, med
longshot-bias i halene). De sier hva markedet priser inn, ikke hva som vil skje.

Kilder (serier hos Kalshi):
  KXBTC15M     «Går BTC opp de neste 15 min?»; startprisen brukes som BTC-pris nå
  KXBTCD       «BTC over $X på tidspunkt»; det fjerneste åpne eventet (fredag kl. 17 ET)
               gir fordelingen for slutten av uken
  KXBTCY       «BTC-pris ved nyttår» i intervaller på $5 000
  KXBTCMINMON / KXBTCMAXMON   «Er BTC innom $X før månedsslutt?»
  KXBTCMINY   / KXBTCMAXY     «Er BTC innom $X før nyttår?»
"""
from __future__ import annotations

import math
from typing import Callable

from . import api

YEAR = 365.25 * 86400
QuoteFn = Callable[[str, str, list, int], dict]  # (serie, event, markeder, t) -> {ticker: midtpris}


# ---------------------------------------------------------------- matematikk
def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Lineær interpolasjon (xs stigende), flat utenfor endene."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def surv_quantile(K: list[float], P: list[float], q: float) -> float:
    """Pris der P(S >= pris) = 1 - q. P er fallende i K."""
    target = 1 - q
    for i in range(1, len(K)):
        if P[i] <= target <= P[i - 1] and P[i - 1] != P[i]:
            w = (P[i - 1] - target) / (P[i - 1] - P[i])
            return K[i - 1] + w * (K[i] - K[i - 1])
    return K[0] if target >= P[0] else K[-1]


def fit_sigma(K: list[float], P: list[float], median: float) -> float:
    """Standardavvik i log-pris for perioden, tilpasset strike-kurven (lognormal)."""
    mu = math.log(median)
    best, best_s = None, 0.05
    for i in range(1, 400):
        s = i * 0.001
        err = sum((1 - norm_cdf((math.log(k) - mu) / s) - p) ** 2 for k, p in zip(K, P))
        if best is None or err < best:
            best, best_s = err, s
    return best_s


def touch_model(level: float, spot: float, years: float, vol: float) -> float:
    """Sannsynlighet for å være innom nivået før fristen hvis prisen svinger uten retning."""
    if years <= 0 or vol <= 0:
        return float("nan")
    x = abs(math.log(level / spot)) / (vol * math.sqrt(years))
    return 2 * (1 - norm_cdf(x))


def bucket_quantile(buckets: list[tuple], q: float) -> float:
    cum = 0.0
    for lo, hi, p in buckets:
        if cum + p >= q and p > 0:
            hi_ = hi if math.isfinite(hi) else lo * 1.5
            return lo + (q - cum) / p * (hi_ - lo)
        cum += p
    return buckets[-1][0]


def bucket_p_above(buckets: list[tuple], x: float) -> float:
    s = 0.0
    for lo, hi, p in buckets:
        if lo >= x:
            s += p
        elif hi > x:
            hi_ = hi if math.isfinite(hi) else lo * 1.5
            s += p * (hi_ - x) / (hi_ - lo)
    return s


# ---------------------------------------------------------------- utvalg av markeder
def active(ms: list[dict], t: int) -> list[dict]:
    return [m for m in ms if api.ts(m["open_time"]) <= t < api.ts(m["close_time"])]


def by_event(ms: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for m in ms:
        out.setdefault(m["event_ticker"], []).append(m)
    return out


def spot_at(universe: dict, t: int) -> float | None:
    """BTC-pris: startprisen (CF BRTI) i 15-minuttersmarkedet som pågår ved t."""
    for m in active(universe.get("KXBTC15M", []), t):
        v = api.num(m.get("floor_strike"))
        if v:
            return v
    # Ved et kvarterskifte kan det mangle et aktivt marked: bruk sluttverdien i forrige kvarter.
    past = [m for m in universe.get("KXBTC15M", []) if api.ts(m["close_time"]) <= t]
    if past:
        m = max(past, key=lambda m: api.ts(m["close_time"]))
        if t - api.ts(m["close_time"]) <= 1800:
            return api.num(m.get("expiration_value")) or api.num(m.get("floor_strike"))
    return None


def snapshot(universe: dict, t: int, quote: QuoteFn) -> dict:
    """Ett øyeblikksbilde: nøkkeltall (row) og detaljer for siden (detail)."""
    spot = spot_at(universe, t)
    if not spot:
        raise RuntimeError("Fant ingen BTC-pris (KXBTC15M) for tidspunktet")
    row: dict = {"ts": api.iso(t), "date": api.iso(t)[:10], "spot": round(spot, 2)}
    detail: dict = {"ts": row["ts"], "spot": spot}

    # --- slutten av uken: fjerneste aktive KXBTCD-event innen 8 dager
    evs = {e: ms for e, ms in by_event(active(universe.get("KXBTCD", []), t)).items()
           if api.ts(ms[0]["close_time"]) - t <= 8 * 86400}
    if evs:
        ev, ms = max(evs.items(), key=lambda kv: api.ts(kv[1][0]["close_time"]))
        q = quote("KXBTCD", ev, ms, t)
        pts = sorted((api.num(m["floor_strike"]) + 0.01, q.get(m["ticker"])) for m in ms
                     if api.num(m.get("floor_strike")) and q.get(m["ticker"]) is not None)
        if len(pts) >= 8:
            K = [k for k, _ in pts]
            P, run = [], 1.0
            for _, p in pts:  # P(S >= K) må være fallende
                run = min(run, p)
                P.append(run)
            close = api.ts(ms[0]["close_time"])
            T = (close - t) / YEAR
            med = surv_quantile(K, P, 0.5)
            sig = fit_sigma(K, P, med)
            vol = sig / math.sqrt(T) if T > 0 else float("nan")
            row.update(wk_event=ev, wk_close=api.iso(close), wk_days=round((close - t) / 86400, 2),
                       wk_pup=round(interp(spot, K, P), 4), wk_median=round(med),
                       wk_q10=round(surv_quantile(K, P, 0.1)), wk_q90=round(surv_quantile(K, P, 0.9)),
                       wk_up5=round(interp(spot * 1.05, K, P), 4),
                       wk_dn5=round(1 - interp(spot * 0.95, K, P), 4),
                       vol=round(vol, 4))
            detail["week"] = {"event": ev, "close": api.iso(close),
                              "curve": [[round(k), round(p, 4)] for k, p in zip(K, P)]}

    # --- nyttår: nærmeste KXBTCY-event minst 3 dager frem
    evs = {e: ms for e, ms in by_event(active(universe.get("KXBTCY", []), t)).items()
           if api.ts(ms[0]["close_time"]) - t >= 3 * 86400}
    if evs:
        ev, ms = min(evs.items(), key=lambda kv: api.ts(kv[1][0]["close_time"]))
        q = quote("KXBTCY", ev, ms, t)
        b = []
        for m in ms:
            p = q.get(m["ticker"])
            if p is None:
                continue
            lo, hi = api.num(m.get("floor_strike")), api.num(m.get("cap_strike"))
            st = m.get("strike_type")
            if st == "less":
                lo, hi = 0.0, (hi or 0) + 0.01
            elif st == "greater":
                hi = math.inf
            else:
                hi = (hi or 0) + 0.01
            b.append((lo or 0.0, hi, p))
        tot = sum(p for *_, p in b)
        if len(b) >= 10 and tot > 0:
            b = sorted((lo, hi, p / tot) for lo, hi, p in b)
            close = api.ts(ms[0]["close_time"])
            med = bucket_quantile(b, 0.5)
            q10, q90 = bucket_quantile(b, 0.1), bucket_quantile(b, 0.9)
            row.update(ye_event=ev, ye_close=api.iso(close), ye_days=round((close - t) / 86400, 1),
                       ye_pup=round(bucket_p_above(b, spot), 4), ye_median=round(med),
                       ye_q10=round(q10), ye_q90=round(q90),
                       # Skjevhet: log-avstand opp minus ned fra medianen. Negativ = lengst hale ned.
                       ye_skew=round(math.log(q90 / med) - math.log(med / q10), 4))
            detail["year"] = {"event": ev, "close": api.iso(close),
                              "buckets": [[round(lo), None if not math.isfinite(hi) else round(hi), round(p, 4)]
                                          for lo, hi, p in b]}

    # --- berøringsmarkeder mot modell uten retning
    vol = row.get("vol")
    touches = []
    for series, kind, horizon in (("KXBTCMINMON", "ned", "måned"), ("KXBTCMAXMON", "opp", "måned"),
                                  ("KXBTCMINY", "ned", "år"), ("KXBTCMAXY", "opp", "år")):
        all_ev = by_event(universe.get(series, []))
        for ev, ms in by_event(active(universe.get(series, []), t)).items():
            # Et berøringsmarked stenger tidlig når nivået nås. Fristen er derfor
            # den seneste stengetiden blant alle markedene i eventet.
            deadline = max(api.ts(x["close_time"]) for x in all_ev.get(ev, ms))
            q = quote(series, ev, ms, t)
            for m in ms:
                p = q.get(m["ticker"])
                lvl = api.num(m.get("cap_strike") if kind == "ned" else m.get("floor_strike"))
                if p is None or not lvl:
                    continue
                if (kind == "ned" and lvl >= spot) or (kind == "opp" and lvl <= spot):
                    continue  # nivået er allerede passert
                close = deadline
                model = touch_model(lvl, spot, (close - t) / YEAR, vol) if vol else float("nan")
                touches.append({"serie": series, "retning": kind, "horisont": horizon,
                                "niva": round(lvl), "frist": api.iso(close)[:10],
                                "avstand": round(lvl / spot - 1, 4), "marked": round(p, 4),
                                "modell": None if math.isnan(model) else round(model, 4)})
    touches.sort(key=lambda r: (r["horisont"], r["retning"], abs(r["avstand"])))
    detail["touch"] = touches
    near_dn = [r for r in touches if r["horisont"] == "måned" and r["retning"] == "ned"]
    if near_dn:
        r = min(near_dn, key=lambda r: abs(r["avstand"]))
        row.update(mon_dn_level=r["niva"], mon_dn_p=r["marked"], mon_dn_model=r["modell"])
    yr_dn = [r for r in touches if r["horisont"] == "år" and r["retning"] == "ned"]
    if yr_dn:
        r = max(yr_dn, key=lambda r: abs(r["avstand"]))  # dypeste krakk-nivå
        row.update(crash_level=r["niva"], crash_p=r["marked"], crash_model=r["modell"])
    return {"row": row, "detail": detail}


# ---------------------------------------------------------------- kursfunksjoner
def live_quotes(series: str, event: str, ms: list, t: int) -> dict:
    return {m["ticker"]: api.live_mid(m) for m in ms}


def candle_quotes(series: str, event: str, ms: list, t: int) -> dict:
    """Midtpris ved t fra timescandles (brukes til tilbakefylling av historikk)."""
    window = 6 * 3600 if series == "KXBTCD" else 72 * 3600  # tynne markeder handles sjeldnere
    cs = api.event_candles(series, event, t - window, t, 60)
    return {tk: api.candle_mid(c, t) for tk, c in cs.items()}
