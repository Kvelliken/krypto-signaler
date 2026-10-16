"""Henter og vedlikeholder daglige priser.

Viktige prinsipper:
- Bare LUKKEDE døgn lagres (døgn slutter 00:00 UTC). Dagens uferdige lys
  ville ellers gitt signaler basert på en pris som ikke er endelig.
- Primærkilde er Yahoo Finance (yfinance). Coinbase Exchange brukes som
  reserve hvis Yahoo feiler eller ikke har publisert gårsdagens lys ennå.
- Overlappende dager hentes på nytt og nyeste verdi vinner, slik at
  etterkorrigeringer fra kilden kommer med.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR

log = logging.getLogger(__name__)
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def utc_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLUMNS, dtype=float,
                            index=pd.DatetimeIndex([], name="Date"))
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[[c for c in COLUMNS if c in df.columns]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Close"] > 0]
    return df


def fetch_yahoo(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(symbol, start=start.strftime("%Y-%m-%d"), interval="1d",
                              auto_adjust=False, progress=False, threads=False)
            df = _normalize(raw)
            if len(df):
                return df
            last_err = RuntimeError("tomt svar")
        except Exception as exc:  # noqa: BLE001 – vi prøver igjen / bytter kilde
            last_err = exc
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Yahoo feilet for {symbol}: {last_err}")


def fetch_coinbase(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    """Coinbase Exchange: maks 300 døgn per kall. symbol som 'BTC-USD'."""
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    end = utc_today() + pd.Timedelta(days=1)
    rows, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=299), end)
        resp = requests.get(url, params={
            "granularity": 86400,
            "start": cursor.strftime("%Y-%m-%dT00:00:00Z"),
            "end": chunk_end.strftime("%Y-%m-%dT00:00:00Z"),
        }, headers={"User-Agent": "krypto-signaler"}, timeout=30)
        resp.raise_for_status()
        rows.extend(resp.json())
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.4)
    if not rows:
        raise RuntimeError(f"Coinbase ga ingen data for {symbol}")
    df = pd.DataFrame(rows, columns=["t", "Low", "High", "Open", "Close", "Volume"])
    df.index = pd.to_datetime(df.pop("t"), unit="s")
    return _normalize(df)


def csv_path(symbol: str):
    return DATA_DIR / f"{symbol}.csv"


def load_prices(symbol: str) -> pd.DataFrame:
    path = csv_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Mangler data for {symbol}: {path}")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return _normalize(df)


@dataclass
class UpdateReport:
    symbol: str
    rows: int = 0
    first_date: str = ""
    last_date: str = ""
    sources: list[str] = field(default_factory=list)
    stale_days: int = 0
    missing_days: int = 0
    revisions: int = 0
    warnings: list[str] = field(default_factory=list)


def update_symbol(symbol: str, data_start: str, full: bool = False) -> UpdateReport:
    report = UpdateReport(symbol)
    path = csv_path(symbol)
    existing = load_prices(symbol) if path.exists() and not full else _normalize(None)
    today = utc_today()
    yesterday = today - pd.Timedelta(days=1)

    fetch_start = (existing.index[-1] - pd.Timedelta(days=10)) if len(existing) else pd.Timestamp(data_start)

    new = _normalize(None)
    try:
        new = fetch_yahoo(symbol, fetch_start)
        report.sources.append("yahoo")
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"Yahoo: {exc}")

    new = new[new.index < today]  # bare lukkede døgn
    if len(new) == 0 or new.index[-1] < yesterday:
        cb_start = new.index[-1] + pd.Timedelta(days=1) if len(new) else fetch_start
        try:
            cb = fetch_coinbase(symbol, cb_start)
            cb = cb[cb.index < today]
            if len(cb):
                new = pd.concat([new, cb[cb.index > (new.index[-1] if len(new) else cb.index[0] - pd.Timedelta(days=1))]])
                report.sources.append("coinbase")
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"Coinbase: {exc}")

    # Logg etterkorrigeringer på overlappende dager (> 0,5 % endring i sluttkurs)
    if len(existing) and len(new):
        overlap = existing.index.intersection(new.index)
        if len(overlap):
            diff = (new.loc[overlap, "Close"] / existing.loc[overlap, "Close"] - 1).abs()
            report.revisions = int((diff > 0.005).sum())
            if report.revisions:
                report.warnings.append(f"{report.revisions} dag(er) med korrigert sluttkurs fra kilden")

    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined[combined.index < today]

    if len(combined) == 0:
        report.warnings.append("Ingen data tilgjengelig")
        return report

    # Kvalitetskontroller
    full_range = pd.date_range(combined.index[0], combined.index[-1], freq="D")
    report.missing_days = int(len(full_range.difference(combined.index)))
    if report.missing_days:
        report.warnings.append(f"{report.missing_days} manglende dag(er) i serien")
    rets = combined["Close"].pct_change().abs()
    jumps = rets[rets > 0.6]
    for d, r in jumps.items():
        report.warnings.append(f"Uvanlig hopp {r:.0%} den {d.date()} – sjekk datakvalitet")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = combined.copy()
    out.index.name = "Date"
    out.to_csv(path, float_format="%.8g")

    report.rows = len(combined)
    report.first_date = str(combined.index[0].date())
    report.last_date = str(combined.index[-1].date())
    report.stale_days = int((yesterday - combined.index[-1]).days)
    return report


def update_all(cfg: dict, full: bool = False) -> list[UpdateReport]:
    reports = []
    for asset in cfg["assets"]:
        rep = update_symbol(asset["symbol"], cfg.get("data_start", "2014-01-01"), full=full)
        log.info("%s: %s rader, siste %s, kilder %s, advarsler %s",
                 rep.symbol, rep.rows, rep.last_date, rep.sources, rep.warnings)
        reports.append(rep)
    return reports


def synthetic_prices(seed: int, start: str, n: int, drift=0.0012, vol=0.035) -> pd.DataFrame:
    """Kun for tester: syntetiske priser med tykke haler og regimer."""
    rng = np.random.default_rng(seed)
    regime = np.sign(np.sin(np.arange(n) / 180)) * 0.002
    r = rng.standard_t(4, n) * vol / np.sqrt(2) + drift + regime
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"Open": close, "High": close * 1.02, "Low": close * 0.98,
                         "Close": close, "Volume": 1.0}, index=idx)
