"""E-postvarsling via SMTP.

Leser innstillinger fra miljøvariabler (GitHub Secrets):
  SMTP_HOST      f.eks. smtp.gmail.com
  SMTP_PORT      465 (SSL) eller 587 (STARTTLS)
  SMTP_USER      innloggingsbruker
  SMTP_PASSWORD  passord / app-passord
  MAIL_TO        mottaker(e), kommaseparert
  MAIL_FROM      valgfri avsender (standard: SMTP_USER)
"""
from __future__ import annotations

import html
import os
import smtplib
import ssl
from email.message import EmailMessage


def _pct(x):
    return "–" if x is None else f"{x:.0%}".replace("%", " %")


def build_message(sig: dict, pages_url: str, reason: str = "change") -> tuple[str, str, str]:
    changed = [r for r in sig["assets"] if r["changed"]]
    date = max(r["date"] for r in sig["assets"])
    if reason == "test":
        subject = f"Krypto-signaler: testvarsel ({date})"
    elif reason == "stale":
        subject = f"Krypto-signaler: data er ikke oppdatert ({date})"
    elif sig["first_run"]:
        subject = f"Krypto-signaler startet – gjeldende posisjoner ({date})"
    else:
        parts = [f"{r['short']}: {_pct(r['notified_target'])} → {_pct(r['target'])}" for r in changed]
        subject = f"Krypto-signal {date}: " + ", ".join(parts)

    lines = []
    if reason == "stale":
        lines += ["Dataene er ikke oppdatert som forventet. Signalene under kan være utdaterte.", ""]
    if sig.get("params_changed"):
        lines += [f"Strategien er endret til: {sig['params_label']}", ""]
    lines += [f"Signaler basert på døgnslutt {date} (00:00 UTC).", ""]
    for r in sig["assets"]:
        flag = "→ " if r["changed"] else "  "
        rsi = "–" if r["rsi"] is None else f"{r['rsi']:.0f}"
        dist = "–" if r["dist_sma"] is None else f"{r['dist_sma']:+.1%}"
        lines.append(f"{flag}{r['short']}: mål {_pct(r['target'])}  |  {r['action']}")
        lines.append(f"    Pris {r['close']:,.2f} USD · {dist} fra SMA · RSI {rsi}")
        for n in r["notes"]:
            lines.append(f"    • {n}")
        lines.append("")
    lines += [
        "Målprosenten gjelder andelen av den delen av porteføljen du har satt av til hver coin.",
        "Dette er regelbaserte signaler, ikke finansiell rådgivning.",
        "",
        f"Full oversikt: {pages_url}",
    ]
    text = "\n".join(lines)

    rows_html = "".join(
        f"<tr style='background:{'#fff6e0' if r['changed'] else 'transparent'}'>"
        f"<td style='padding:6px 10px'><b>{html.escape(r['short'])}</b></td>"
        f"<td style='padding:6px 10px'>{_pct(r['target'])}</td>"
        f"<td style='padding:6px 10px'>{html.escape(r['action'])}</td></tr>"
        for r in sig["assets"])
    notes_html = "".join(
        f"<li>{html.escape(r['short'])}: {html.escape(n)}</li>" for r in sig["assets"] for n in r["notes"])
    html_body = f"""<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#0b0b0b">
<p>Signaler basert på døgnslutt <b>{date}</b> (00:00 UTC).</p>
{'<p><b>Strategien er endret:</b> ' + html.escape(sig['params_label']) + '</p>' if sig.get('params_changed') else ''}
{'<p><b>Advarsel:</b> dataene er ikke oppdatert som forventet.</p>' if reason == 'stale' else ''}
<table style="border-collapse:collapse;border:1px solid #e1e0d9">
<tr style="background:#f0efec"><th style="padding:6px 10px;text-align:left">Coin</th>
<th style="padding:6px 10px;text-align:left">Mål</th><th style="padding:6px 10px;text-align:left">Handling</th></tr>
{rows_html}</table>
{'<ul>' + notes_html + '</ul>' if notes_html else ''}
<p style="color:#52514e;font-size:13px">Målprosenten gjelder andelen av den delen av porteføljen du har satt av til hver coin.
Regelbaserte signaler, ikke finansiell rådgivning.</p>
<p><a href="{html.escape(pages_url)}">Åpne full oversikt</a></p></div>"""
    return subject, text, html_body


def smtp_configured() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO"))


def send_email(subject: str, text: str, html_body: str) -> bool:
    if not smtp_configured():
        print("E-post er ikke satt opp (mangler SMTP-secrets) – hopper over.")
        return False
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT") or 465)
    user, password = os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("MAIL_FROM") or user
    msg["To"] = os.environ["MAIL_TO"]
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=30) as s:
            s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls(context=ctx)
            s.login(user, password)
            s.send_message(msg)
    print(f"E-post sendt: {subject}")
    return True


def issue_markdown(subject: str, text: str) -> str:
    return f"## {subject}\n\n```\n{text}\n```\n"
