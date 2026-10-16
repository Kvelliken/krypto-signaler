"""Kommandolinje.

  python -m ksig update-data [--full]   Hent/oppdater prisdata
  python -m ksig research                Kjør backtest og walk-forward
  python -m ksig daily                   Hele den daglige jobben (data, signaler, varsel, side)
  python -m ksig build                   Bygg siden uten å hente data eller varsle
  python -m ksig test-email              Send et testvarsel
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import RESULTS_DIR, STATE_DIR, load_config
from .data import update_all
from .notify import build_message, issue_markdown, send_email
from .research import run_research
from .signals import compute_signals, save_state
from .site import build_site


def _gh_output(**kv) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as fh:
            for k, v in kv.items():
                fh.write(f"{k}={v}\n")


def _research_or_cached(cfg: dict) -> dict:
    return run_research(cfg)


def cmd_daily(cfg: dict, notify: bool = True, fetch: bool = True) -> int:
    if fetch:
        reports = update_all(cfg)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "data_status.json").write_text(json.dumps([r.__dict__ for r in reports], indent=2, ensure_ascii=False))
        if any(r.rows == 0 for r in reports):
            print("Mangler data for minst én coin – avbryter.", file=sys.stderr)
            return 1
    research = _research_or_cached(cfg)
    sig = compute_signals(cfg)
    stale = sig["max_stale_days"] >= cfg.get("alerts", {}).get("stale_after_days", 2)

    send_reason = None
    if notify:
        if sig["any_change"]:
            send_reason = "change"
        elif stale:
            from .signals import load_state
            st = load_state() or {}
            if st.get("stale_alert_for") != max(a["date"] for a in sig["assets"]):
                send_reason = "stale"

    if send_reason:
        subject, text, html_body = build_message(sig, cfg["pages_url"], send_reason)
        try:
            send_email(subject, text, html_body)
        except Exception as exc:  # noqa: BLE001 – varslet skal ikke stoppe siden
            print(f"E-post feilet: {exc}", file=sys.stderr)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "last_alert.md").write_text(issue_markdown(subject, text))
        _gh_output(alert="true", alert_title=subject.replace("\n", " "))
    else:
        _gh_output(alert="false")

    if notify:
        save_state(sig)
        if send_reason == "stale":
            from .signals import STATE_FILE
            st = json.loads(STATE_FILE.read_text())
            st["stale_alert_for"] = max(a["date"] for a in sig["assets"])
            STATE_FILE.write_text(json.dumps(st, indent=2, ensure_ascii=False))
    build_site(research, sig)
    print(f"Ferdig. Endring: {sig['any_change']}. Varsel: {send_reason}.")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="ksig")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update-data"); u.add_argument("--full", action="store_true")
    sub.add_parser("research")
    sub.add_parser("daily")
    sub.add_parser("build")
    sub.add_parser("test-email")
    args = ap.parse_args(argv)
    cfg = load_config()

    if args.cmd == "update-data":
        for r in update_all(cfg, full=args.full):
            print(r)
        return 0
    if args.cmd == "research":
        res = run_research(cfg)
        print(f"Lagret {RESULTS_DIR / 'research.json'}")
        for a in res["assets"]:
            m = a["metrics"]
            print(f"{a['short']}: strategi CAGR {m['strategy']['cagr']:.1%} Sharpe {m['strategy']['sharpe']:.2f} "
                  f"MaxDD {m['strategy']['max_dd']:.1%} | kjøp&hold CAGR {m['buy_hold']['cagr']:.1%} "
                  f"MaxDD {m['buy_hold']['max_dd']:.1%}")
        return 0
    if args.cmd == "daily":
        return cmd_daily(cfg)
    if args.cmd == "build":
        return cmd_daily(cfg, notify=False, fetch=False)
    if args.cmd == "test-email":
        sig = compute_signals(cfg)
        subject, text, html_body = build_message(sig, cfg["pages_url"], "test")
        ok = send_email(subject, text, html_body)
        return 0 if ok else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
