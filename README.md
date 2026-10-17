# Krypto-signaler

Daglige, regelbaserte posisjonssignaler for **BTC, ETH og SOL**, med backtest som tar hensyn til
kostnader, slippage, forsinket utførelse og overtilpasning.

**Nettside:** https://kvelliken.github.io/krypto-signaler/

> Regelbaserte signaler, ikke finansiell rådgivning.

## Slik fungerer det

Hver natt kl. 00:35 UTC kjører GitHub Actions denne jobben:

1. Henter døgnpriser (Yahoo Finance, med Coinbase som reserve). Bare **lukkede døgn** brukes.
2. Kjører backtest, robusthetstester og walk-forward-test.
3. Beregner dagens målposisjon per coin (0 %, 50 % eller 100 %).
4. Sammenligner med forrige varslede posisjon. Ved endring sendes **e-post** og det opprettes en **issue**.
5. Lagrer data og tilstand i repoet og publiserer nettsiden på GitHub Pages.

## Gjeldende strategi (egen regel per coin)

Satt under `[[assets]]` i `config.toml`. Alle tall er etter Firi-kostnader (0,7 % + 0,3 % påslag per handel).

| Coin | Regel | Kort fortalt |
|---|---|---|
| BTC | SMA150 ±2 %, daglig | Inn over SMA150 + 2 %, ut under SMA150 − 2 % |
| ETH | Trend × BTC-trend, ukentlig | Antall av SMA50/100/150/200 ETH er over × antall BTC er over / 16, i trinn på 12,5 %, sjekkes søndag |
| SOL | Trend × ETH-trend, lange snitt, ukentlig | Antall av SMA100/150/200/250/300 SOL er over × antall ETH er over / 25, i trinn på 12,5 %, sjekkes søndag |

ETH- og SOL-reglene er utformet i `research/eth_design.py` og `research/sol_design.py`. BTC-regelen ble valgt med `research/search.py` (320 varianter, walk-forward) og `research/classic.py`
(klassiske regler i alle delperioder, bootstrap). Se nettsiden for forbehold.

## Oppsett (én gang)

### 1. GitHub Pages
**Settings → Pages → Build and deployment → Source: «GitHub Actions».**

### 2. E-postvarsling
Under **Settings → Secrets and variables → Actions → New repository secret**, legg inn:

| Secret | Eksempel for Gmail |
|---|---|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `465` |
| `SMTP_USER` | din Gmail-adresse |
| `SMTP_PASSWORD` | et **app-passord** (ikke vanlig passord) |
| `MAIL_TO` | adressen varslene skal til |

Gmail krever totrinnsbekreftelse for å lage app-passord: Google-konto → Sikkerhet → App-passord.

Test oppsettet: **Actions → Daglige signaler → Run workflow → mode: test-email**.

### 3. Varsling fra GitHub (reserve)
Hvert varsel opprettes også som en issue. Med standard varslingsinnstillinger får du e-post fra GitHub
om nye issues i ditt eget repo.

## Kjøre lokalt

```bash
pip install -r requirements.txt
python -m ksig update-data     # hent priser
python -m ksig research        # backtest -> results/research.json
python -m ksig build           # bygg siden i site/ uten å varsle
```

## Struktur

| Sti | Innhold |
|---|---|
| `config.toml` | Strategi, kostnader, walk-forward og varsler |
| `ksig/data.py` | Datainnhenting og kvalitetskontroll |
| `ksig/strategy.py` | Indikatorer og posisjonsregler |
| `ksig/backtest.py` | Simulering med kostnader, drift og forsinkelse |
| `ksig/research.py` | Robusthet, walk-forward og portefølje |
| `ksig/signals.py` | Dagens signaler og endringsdeteksjon |
| `ksig/notify.py` | E-post |
| `data/` | Døgnpriser (oppdateres automatisk) |
| `state/` | Siste varslede posisjoner og signallogg |
| `results/` | Siste backtest-resultater |
