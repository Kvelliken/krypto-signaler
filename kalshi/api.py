"""Tynn klient mot Kalshis offentlige markedsdata-API (ingen innlogging trengs).

Kalshi deler data i et «live» og et «historisk» lag. Markeder som ble avgjort før
skillet (se /historical/cutoff) finnes bare i det historiske laget. Denne modulen
bruker bare live-laget, som dekker åpne markeder og markeder avgjort de siste
par månedene. Det holder for daglige øyeblikksbilder og for evaluering.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
log = logging.getLogger(__name__)
_session = requests.Session()


def _get(path: str, params: dict | None = None, tries: int = 12) -> dict:
    for i in range(tries):
        try:
            r = _session.get(BASE + path, params=params, timeout=40)
        except requests.RequestException as exc:
            log.warning("Kalshi %s feilet (%s), prøver igjen", path, exc)
            time.sleep(2 + i)
            continue
        if r.status_code == 429:  # rate limit: vent og prøv igjen
            time.sleep(1 + i)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi {path}: ga opp etter {tries} forsøk")


def ts(s: str) -> int:
    """ISO-tid fra API-et til Unix-sekunder."""
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def iso(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def markets(series: str, status: str | None = None, min_close_ts: int | None = None,
            max_close_ts: int | None = None, event_ticker: str | None = None) -> list[dict]:
    """Alle markeder i en serie (med paginering)."""
    out, cursor = [], None
    while True:
        p = {"series_ticker": series, "limit": 1000}
        if status:
            p["status"] = status
        if min_close_ts:
            p["min_close_ts"] = min_close_ts
        if max_close_ts:
            p["max_close_ts"] = max_close_ts
        if event_ticker:
            p["event_ticker"] = event_ticker
        if cursor:
            p["cursor"] = cursor
        d = _get("/markets", p)
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            return out
        time.sleep(0.3)


def event_candles(series: str, event: str, start: int, end: int, period: int = 60) -> dict[str, list]:
    """Candlesticks for alle markeder i et event: {ticker: [candle, ...]}."""
    d = _get(f"/series/{series}/events/{event}/candlesticks",
             {"start_ts": start, "end_ts": end, "period_interval": period})
    return dict(zip(d.get("market_tickers", []), d.get("market_candlesticks", [])))


def num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def live_mid(m: dict) -> float | None:
    """Midtpris fra markedets nåværende kjøps- og salgskurs."""
    b, a = num(m.get("yes_bid_dollars")), num(m.get("yes_ask_dollars"))
    if b is None or a is None or a <= 0 or a < b:
        return None
    return (b + a) / 2


def candle_mid(candles: list, t: int) -> float | None:
    """Midtpris fra siste candle som slutter før eller ved t."""
    best = None
    for c in candles or []:
        if c["end_period_ts"] <= t and (best is None or c["end_period_ts"] > best["end_period_ts"]):
            best = c
    if not best:
        return None
    b, a = num(best["yes_bid"].get("close_dollars")), num(best["yes_ask"].get("close_dollars"))
    if b is None or a is None or a <= 0 or a < b:
        return None
    return (b + a) / 2
