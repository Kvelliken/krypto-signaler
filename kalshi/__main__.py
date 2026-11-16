"""Kalshi-sentiment for BTC. Helt uavhengig av ksig-pakken.

  python -m kalshi snapshot              Øyeblikksbilde nå, evaluering og bygg av siden
  python -m kalshi backfill --since DATO Fyll inn daglige øyeblikksbilder bakover (kl. 23:40 UTC)
  python -m kalshi evaluate              Hent fasit for avsluttede ukemarkeder
  python -m kalshi build [--out MAPPE]   Bygg siden (standard: kalshi/public)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from . import api, outlook, store
from .site import build_site

SERIES = ["KXBTC15M", "KXBTCD", "KXBTCY", "KXBTCMINMON", "KXBTCMAXMON", "KXBTCMINY", "KXBTCMAXY"]
log = logging.getLogger("kalshi")


def live_universe() -> dict:
    u = {s: api.markets(s, status="open") for s in SERIES if s != "KXBTC15M"}
    u["KXBTC15M"] = api.markets("KXBTC15M", min_close_ts=int(time.time()) - 3600)
    return u


def cmd_snapshot() -> int:
    t = int(time.time()) // 60 * 60
    snap = outlook.snapshot(live_universe(), t, outlook.live_quotes)
    store.upsert(snap["row"])
    store.save_latest(snap["detail"])
    print("Øyeblikksbilde:", snap["row"])
    cmd_evaluate()
    build_site()
    return 0


def backfill_universe(since: int, until: int) -> dict:
    u = {s: api.markets(s, min_close_ts=since) for s in SERIES if s != "KXBTCD"}
    # KXBTCD har ~180 strikes per time. Hent bare eventene som stenger fredag 17:00 ET
    # (21:00 eller 22:00 UTC), som er ukemarkedene.
    wk = []
    day = datetime.fromtimestamp(since, timezone.utc).replace(hour=0, minute=0, second=0)
    end = datetime.fromtimestamp(until, timezone.utc) + timedelta(days=8)
    while day <= end:
        if day.weekday() == 4:
            lo, hi = int((day + timedelta(hours=20, minutes=30)).timestamp()), int((day + timedelta(hours=22, minutes=30)).timestamp())
            ms = api.markets("KXBTCD", min_close_ts=lo, max_close_ts=hi)
            # behold bare ukemarkedet (åpnet en uke før), ikke timemarkedene rundt
            evs = outlook.by_event(ms)
            if evs:
                ev = min(evs, key=lambda e: min(api.ts(m["open_time"]) for m in evs[e]))
                wk += evs[ev]
        day += timedelta(days=1)
    u["KXBTCD"] = wk
    return u


def cmd_backfill(since: str, hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    start = datetime.fromisoformat(since).replace(tzinfo=timezone.utc, hour=h, minute=m)
    now = datetime.now(timezone.utc)
    u = backfill_universe(int(start.timestamp()) - 86400, int(now.timestamp()))
    print("Markeder hentet:", {k: len(v) for k, v in u.items()})
    d, n = start, 0
    while d < now - timedelta(minutes=5):
        t = int(d.timestamp())
        try:
            snap = outlook.snapshot(u, t, outlook.candle_quotes)
            store.upsert(snap["row"])
            n += 1
            r = snap["row"]
            print(r["date"], r["spot"], r.get("wk_event"), r.get("wk_pup"), r.get("ye_median"), flush=True)
        except Exception as exc:  # noqa: BLE001 – en dag uten data skal ikke stoppe resten
            print(f"{d:%Y-%m-%d}: hoppet over ({exc})", file=sys.stderr)
        d += timedelta(days=1)
    print(f"{n} øyeblikksbilder lagret")
    cmd_evaluate()
    build_site()
    return 0


def cmd_evaluate() -> int:
    df = store.load()
    if df.empty or "wk_event" not in df:
        return 0
    now = time.time()
    if "wk_outcome" not in df:
        df["wk_outcome"] = None
    cache: dict[str, float | None] = {}
    for i, r in df.iterrows():
        ev = r.get("wk_event")
        if not isinstance(ev, str) or not ev or store.present(r.get("wk_outcome")):
            continue
        if api.ts(r["wk_close"]) > now - 600:
            continue  # ikke avgjort ennå
        if ev not in cache:
            cache[ev] = None
            for m in api.markets("KXBTCD", event_ticker=ev):
                v = api.num(m.get("expiration_value"))
                if v:
                    cache[ev] = v
                    break
        if cache[ev]:
            df.at[i, "wk_outcome"] = cache[ev]
    store.save(df)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m kalshi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    b = sub.add_parser("backfill")
    b.add_argument("--since", required=True)
    b.add_argument("--at", default="23:40", help="klokkeslett (UTC) for hvert daglige øyeblikksbilde")
    sub.add_parser("evaluate")
    o = sub.add_parser("build")
    o.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        return cmd_snapshot()
    if a.cmd == "backfill":
        return cmd_backfill(a.since, a.at)
    if a.cmd == "evaluate":
        cmd_evaluate()
        build_site()
        return 0
    build_site(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
