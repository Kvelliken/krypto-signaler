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
    if x is None:
        return "–"
    v = round(x * 100, 1)
    return (f"{v:.0f}" if v == int(v) else f"{v:.1f}".replace(".", ",")) + " %"


def build_message(sig: dict, pages_url: str, reason: str = "change") -> tuple[str, str, str]:
    assets = sig["assets"]
    changed = [r for r in assets if r["changed"]]
    date = max(r["date"] for r in assets)
    cash = sig["portfolio"]["cash"]
    if reason == "test":
        subject = f"Krypto-signaler: testvarsel ({date})"
    elif reason == "stale":
        subject = f"Krypto-signaler: data er ikke oppdatert ({date})"
    elif sig.get("first_portfolio"):
        subject = f"Krypto-signaler: ny fordeling av porteføljen ({date})"
    else:
        parts = [f"{r['short']} {_pct(r['notified_share'])} → {_pct(r['share'])}" for r in changed]
        subject = f"Krypto-signal {date}: " + ", ".join(parts)

    lines = []
    if reason == "stale":
        lines += ["Dataene er ikke oppdatert som forventet. Signalene under kan være utdaterte.", ""]
    if sig.get("first_portfolio"):
        lines += ["Signalene gjelder nå andeler av HELE porteføljen (ikke en egen pott per coin).", ""]
    if sig.get("params_changed"):
        lines += ["Strategien er endret for minst én coin (se under).", ""]
    lines += [f"Fordeling basert på døgnslutt {date} (00:00 UTC):", ""]
    for r in assets:
        flag = "→ " if r["changed"] else "  "
        lines.append(f"{flag}{r['short']}: {_pct(r['hold_share'])} av porteføljen  |  {r['action']}")
        lines.append(f"    Pris {r['close']:,.2f} USD · regel {r['label']}: {_pct(r['target'])} av vekten "
                     f"{_pct(r['weight'])} (nr. {r['mom_rank']} på momentum)")
        for n in r["notes"]:
            lines.append(f"    • {n}")
        lines.append("")
    lines += [f"  Kontanter: {_pct(cash)}", "",
              "Handle bare de coinene som er markert med →. Små avvik fra prisbevegelser er normalt.",
              "Dette er regelbaserte signaler, ikke finansiell rådgivning.", "",
              f"Full oversikt og beløp i kroner: {pages_url}"]
    text = "\n".join(lines)

    rows_html = "".join(
        f"<tr style='background:{'#fff6e0' if r['changed'] else 'transparent'}'>"
        f"<td style='padding:6px 10px'><b>{html.escape(r['short'])}</b></td>"
        f"<td style='padding:6px 10px;text-align:right'><b>{_pct(r['hold_share'])}</b></td>"
        f"<td style='padding:6px 10px'>{html.escape(r['action'])}</td>"
        f"<td style='padding:6px 10px;color:#52514e'>{html.escape(r['label'])}</td></tr>"
        for r in assets)
    rows_html += (f"<tr><td style='padding:6px 10px'>Kontanter</td>"
                  f"<td style='padding:6px 10px;text-align:right'>{_pct(cash)}</td><td></td><td></td></tr>")
    notes_html = "".join(
        f"<li>{html.escape(r['short'])}: {html.escape(n)}</li>" for r in assets for n in r["notes"])
    html_body = f"""<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#0b0b0b">
<p>Fordeling av hele porteføljen basert på døgnslutt <b>{date}</b> (00:00 UTC).</p>
{'<p><b>Nytt:</b> signalene gjelder nå andeler av hele porteføljen.</p>' if sig.get('first_portfolio') else ''}
{'<p><b>Advarsel:</b> dataene er ikke oppdatert som forventet.</p>' if reason == 'stale' else ''}
<table style="border-collapse:collapse;border:1px solid #e1e0d9">
<tr style="background:#f0efec"><th style="padding:6px 10px;text-align:left">Coin</th>
<th style="padding:6px 10px;text-align:right">Andel</th><th style="padding:6px 10px;text-align:left">Handling</th>
<th style="padding:6px 10px;text-align:left">Regel</th></tr>
{rows_html}</table>
{'<ul>' + notes_html + '</ul>' if notes_html else ''}
<p style="color:#52514e;font-size:13px">Handle bare de coinene som er markert. Små avvik fra prisbevegelser er normalt.
Regelbaserte signaler, ikke finansiell rådgivning.</p>
<p><a href="{html.escape(pages_url)}">Åpne full oversikt med beløp i kroner</a></p></div>"""
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
