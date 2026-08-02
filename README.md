# Trading System — Architektura Mikroserwisowa

System tradingowy oparty na mikroserwisach: event-driven, Kubernetes-ready od dnia 1, z pełnym observability stackiem. Realizuje plan rozwoju z [`Plan_Rozwoju_Systemu_Tradingowego_2.md`](Plan_Rozwoju_Systemu_Tradingowego_2.md).

**Stan: 13 serwisów, wszystkie funkcjonalnie zaimplementowane** (brak szkieletów). Działa pełna pętla paper tradingu — od pobrania danych rynkowych, przez cechy, sygnały reguł i głos modelu ML, agregację, kontrolę ryzyka, aż po symulowane wypełnienia i sprzężenie zwrotne do portfela. Warstwa ML (trening, rejestr MLflow, serwowanie, dzienny monitoring driftu) jest kompletna; pierwszy trening na realnych danych uruchamia się dwiema komendami — patrz [Bootstrap danych i trening](#bootstrap-danych-i-pierwszy-trening).

> Kontekst projektu, historia decyzji i bieżące priorytety: [`CLAUDE.md`](CLAUDE.md).
> Architektura fazy ML (etykiety, walidacja, rejestr): [`docs/ml_integration_plan.md`](docs/ml_integration_plan.md).
> Bieżący plan poprawy przewidywania: [`docs/decisions/`](docs/decisions/).
> Mapa całej dokumentacji: [`docs/README.md`](docs/README.md).

---

## Spis treści

- [Jak to działa](#jak-to-działa)
- [Serwisy](#serwisy)
- [Zdarzenia i strumienie](#zdarzenia-i-strumienie)
- [Infrastruktura](#infrastruktura)
- [Shared Library](#shared-library)
- [Uruchamianie](#uruchamianie)
- [Bootstrap danych i pierwszy trening](#bootstrap-danych-i-pierwszy-trening)
- [Reguły ryzyka](#reguły-ryzyka)
- [Testowanie](#testowanie)
- [Git Workflow](#git-workflow)
- [CI/CD](#cicd)
- [Kubernetes / Helm](#kubernetes--helm)
- [Struktura plików](#struktura-plików)

---

## Jak to działa

Ścieżka decyzyjna jest w całości sterowana zdarzeniami (NATS JetStream); zapytania punktowe idą po HTTP.

```
  market-data ──market_data.updated──▶ feature-engine ──features.ready──┬──▶ strategy
       ▲                                     ▲                          │       │
       │                          fundamentals.updated                  │  signal.generated
       │                          company.classified                    │       │
       │                                     │                          ▼       ▼
  (Yahoo / Alpha Vantage)          fundamental-data              ml-pipeline    │
                                   company-classifier         ml.signal_generated
                                                                       │       │
                                                                       ▼       ▼
   macro-data ──macro.regime_changed──────────────────────▶  signal-aggregator
        │                                                              │
        │                                                     signal.aggregated
        │                                                              ▼
        └──────────────macro.regime_changed───────────────────▶   risk-mgmt
                                                                       │
                                                                order.requested
                                                                       ▼
                                                                  execution
                                                                       │
                                              order.filled ────────────┤
                                                                       ▼
                                    (HTTP: metryki portfela z powrotem do risk-mgmt)

  backtest ──backtest.strategy_revalidated──▶ strategy      notification ◀── 5 strumieni alertów
  dashboard ── (HTTP, tylko odczyt) ──▶ risk-mgmt · execution · notification · ml-pipeline
```

Kluczowe zasady, które ta ścieżka realizuje:

- **signal-aggregator jest węzłem decyzyjnym.** Sygnał ze `strategy` i głos ML to *komponenty*; agregator łączy je wagami adaptacyjnymi z reżimem makro i przepuszcza przez filtr kosztów. Dopiero `signal.aggregated` trafia do risk-mgmt.
- **ML nigdy nie handluje samo.** `ml.signal_generated` celowo nie zawiera poziomów SL/TP i nie jest agregowany bez komponentu strategii — to modulacja decyzji, nie decyzja.
- **Ryzyko jest bramą, nie sugestią.** Każdy sygnał przechodzi `RiskEnvelope`; wielkość pozycji liczy risk-mgmt (budżet zależny od obsunięcia, limity ekspozycji i sektorowe wg reżimu, maks. 5% na pozycję). Wyłącznik bezpieczeństwa jest uzbrojony 24/7.
- **Uczenie się z realnych wyników.** Dojrzały głos ML jest rozliczany tą samą regułą triple-barrier co trening; zrealizowany zwrot wraca do wag adaptacyjnych agregatora i do wykrywania degradacji modelu.

### Zasady komunikacji

| Typ | Protokół | Kiedy |
|-----|----------|-------|
| Asynchroniczna | NATS JetStream (durable, dedup po `Nats-Msg-Id`) | Zdarzenia rynkowe, sygnały, zlecenia, alerty |
| Synchroniczna | HTTP/REST (FastAPI) | Zapytania on-demand, dashboard → serwis, sprzężenie portfela |

---

## Serwisy

### Rdzeń (9)

| Serwis | Port | Rola |
|--------|------|------|
| [`market-data`](services/market-data/) | 8001 | Pobieranie OHLCV (Yahoo → Alpha Vantage), walidacja, TimescaleDB, cache Redis, publikacja zdarzeń |
| [`feature-engine`](services/feature-engine/) | 8002 | Cechy techniczne + wzbogacenie fundamentami/stylem, **rangi przekrojowe** (`/ranked`) |
| [`strategy`](services/strategy/) | 8003 | 5 reguł z registry (`trading_common.strategies`), `RiskEnvelope`, filtr kosztów, monitor degradacji per strategia |
| [`backtest`](services/backtest/) | 8004 | Silnik long/flat bez podglądania przyszłości, walk-forward, cotygodniowa rewalidacja |
| [`ml-pipeline`](services/ml-pipeline/) | 8005 | Trening (PyTorch), rejestr MLflow, serwowanie głosu ML, dzienny monitoring driftu |
| [`risk-mgmt`](services/risk-mgmt/) | 8006 | Wielkość pozycji, limity reżimowe i sektorowe, wyłącznik bezpieczeństwa, stan portfela |
| [`execution`](services/execution/) | 8007 | Paper trading: wypełnienia, ochronne wyjścia SL/TP, idempotencja, sprzężenie do risk-mgmt |
| [`notification`](services/notification/) | 8008 | Alerty z 5 strumieni: log, Slack, Telegram, e-mail (SMTP) |
| [`dashboard`](services/dashboard/) | 8501 | Backend-for-frontend (FastAPI) + strona `/ui` bez kroku budowania |

### Rozszerzenie ML/AI (4)

| Serwis | Port | Rola |
|--------|------|------|
| [`fundamental-data`](services/fundamental-data/) | 8009 | SEC EDGAR, sprawozdania roczne, pełny 9-sygnałowy F-Score Piotroskiego |
| [`macro-data`](services/macro-data/) | 8010 | FRED (krzywa, spready, PMI), regułowa detekcja reżimu rynkowego |
| [`company-classifier`](services/company-classifier/) | 8011 | Profil spółki → styl inwestycyjny i routing stosu modeli |
| [`signal-aggregator`](services/signal-aggregator/) | 8012 | Łączenie sygnałów (reguły + ML + makro) w jedną decyzję, wagi adaptacyjne |

Każdy serwis ma własny `Dockerfile`, `pyproject.toml`, testy i identyczną strukturę wewnętrzną — patrz [Struktura serwisu](#struktura-serwisu-wzorzec). Serwis A **nigdy** nie importuje z serwisu B; typy współdzielone mieszkają w `trading-common`.

---

## Zdarzenia i strumienie

Strumienie JetStream (tworzone idempotentnie przez `ensure_stream` przy starcie, więc kolejność uruchamiania serwisów nie ma znaczenia):

| Strumień | Subjects | Główne zdarzenia |
|----------|----------|------------------|
| `MARKET_DATA` | `market_data.>` | `market_data.updated` |
| `FEATURES` | `features.>` | `features.ready` |
| `SIGNALS` | `signal.>` | `signal.generated`, `signal.aggregated` |
| `ORDERS` | `order.>` | `order.requested`, `order.filled`, `order.rejected` |
| `RISK` | `risk.>` | `risk.circuit_breaker`, `risk.limit_breached` |
| `ML` | `ml.>` | `ml.signal_generated`, `ml.drift_detected`, `ml.model_trained` |
| `BACKTEST` | `backtest.>` | `backtest.completed`, `backtest.strategy_revalidated` |
| `STRATEGY` | `strategy.>` | `strategy.status_changed` |
| `MACRO` | `macro.>` | `macro.updated`, `macro.regime_changed` |
| `FUNDAMENTALS` | `fundamentals.>` | `fundamentals.updated` |
| `COMPANY` | `company.>` | `company.classified` |

Subskrypcje są **durable** z `max_deliver`; komunikat trwale błędny jest terminowany (`term`), przejściowy błąd → `nak` i ponowne dostarczenie.

---

## Infrastruktura

Pliki konfiguracyjne w [`infrastructure/`](infrastructure/).

| Komponent | Obraz | Port | Rola |
|-----------|-------|------|------|
| PostgreSQL + TimescaleDB | `timescale/timescaledb:latest-pg16` | 5432 | Baza; OHLCV jako hypertable |
| Redis | `redis:7-alpine` | 6379 | Cache + trwałość stanu (portfel, broker, cechy) |
| NATS | `nats:2-alpine` | 4222 / 8222 | Event bus (JetStream) |
| Traefik | `traefik:v3.0` | 80 / 8080 | API Gateway |
| Prometheus | `prom/prometheus` | 9090 | Metryki |
| Grafana | `grafana/grafana` | 3000 | Dashboardy |
| Loki | `grafana/loki` | 3100 | Logi strukturalne |

Schemat bazy inicjalizuje [`init-db.sql`](infrastructure/init-db.sql): schematy izolowane per-serwis, `market_data.ohlcv` jako hypertable z kluczem naturalnym `(symbol, interval, ts)` (umożliwia idempotentny upsert) i kompresją danych starszych niż 7 dni.

> **Produkcja:** [`docker-compose.prod.yml`](infrastructure/docker-compose.prod.yml) to warstwa nakładkowa — zdejmuje publikację portów na hosta (ruch wyłącznie przez Traefika z TLS) i wycisza logi do `INFO`. Używa znacznika `!reset`, bo Compose **scala** listy: samo `ports: []` niczego nie usuwa.

---

## Shared Library

[`shared/trading-common/`](shared/trading-common/) — biblioteka instalowalna przez `pip install -e`. Kontrakty definiuje się **tutaj, zanim** powstanie kod ich używający.

| Moduł | Zawartość |
|-------|-----------|
| [`schemas.py`](shared/trading-common/src/trading_common/schemas.py) | `OHLCVBar`, `TradingSignal` (wymusza `stop_loss`), `PortfolioMetrics`, `FeatureVector`, `CompanyProfile`, `FinancialStatements`, `MacroSnapshot` |
| [`events.py`](shared/trading-common/src/trading_common/events.py) | `EventType` (24 wartości) + klasy zdarzeń NATS |
| [`features.py`](shared/trading-common/src/trading_common/features.py) | `compute_feature_vector` — **wspólna** definicja cech dla treningu i serwowania |
| [`ranking.py`](shared/trading-common/src/trading_common/ranking.py) | `cross_sectional_rank` — percentylowa ranga przekrojowa (odporna na remisy) |
| [`risk_envelope.py`](shared/trading-common/src/trading_common/risk_envelope.py) | `RiskEnvelope` — brama, przez którą przechodzi każdy sygnał |
| [`cost_filter.py`](shared/trading-common/src/trading_common/cost_filter.py) | `CostAwareFilter` — odsiewa sygnały o przewadze mniejszej niż koszty |
| [`scheduler.py`](shared/trading-common/src/trading_common/scheduler.py) | `PeriodicTask` — zadania cykliczne w pętli asyncio, izolowane od wyjątków |
| [`constants.py`](shared/trading-common/src/trading_common/constants.py) | Porty, symbole domyślne, subjects, limity ryzyka |
| [`utils.py`](shared/trading-common/src/trading_common/utils.py) | `utcnow()`, `to_utc()`, `symbol_to_topic()` |

> `features.py` i `ranking.py` są współdzielone celowo: ml-pipeline liczy cechy do treningu **tym samym kodem**, którym feature-engine liczy je na produkcji. Zgodność trening/serwowanie jest strukturalna, nie deklaratywna.

```bash
pip install -e shared/trading-common          # runtime
pip install -e "shared/trading-common[dev]"   # + narzędzia testowe
```

---

## Uruchamianie

### Wymagania

Docker + Compose v2, ~15 GB dysku, `python3` (skrypty używają wyłącznie biblioteki standardowej). Python 3.12 tylko do testów lokalnych.

### Pierwsze uruchomienie

```bash
cp .env.example .env     # ustaw DB_PASSWORD i REDIS_PASSWORD
make build               # 13 obrazów; pierwszy raz 10–30 min
make up                  # infrastruktura + 13 serwisów
```

### Windows (bez `make`)

GNU Make nie jest częścią Windowsa. Repo zawiera [`make.ps1`](make.ps1) — te same cele w PowerShellu, bez instalowania czegokolwiek:

```powershell
Copy-Item .env.example .env        # uzupełnij hasła
.\make.ps1 build
.\make.ps1 up
.\make.ps1 bootstrap-universe --train --report-out reports/first-training.json
.\make.ps1 help                    # lista celów
```

> Jeśli PowerShell zablokuje skrypt (ExecutionPolicy), uruchom raz:
> `powershell -ExecutionPolicy Bypass -File .\make.ps1 up`, albo odblokuj na czas sesji:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Skrypt sam wykrywa interpreter Pythona (`python`, `py -3`, `python3`). Nic nie stoi też na przeszkodzie, by wołać narzędzia wprost — `make.ps1` jest tylko skrótem:

```powershell
docker compose -f infrastructure/docker-compose.yml --env-file .env up -d
python scripts/bootstrap-universe.py --train --report-out reports/first-training.json
```

> `.env` musi leżeć w katalogu root — Makefile przekazuje `--env-file .env` do Compose.
> **ml-pipeline potrzebuje ~2,5 min na start** (import torch + mlflow) i przez ten czas ma status `starting`. To poprawne, nie awaria — healthcheck ma na to zapas (`start_period: 300s`).

### Dostęp do usług (dev)

| Usługa | URL |
|--------|-----|
| **Dashboard** | http://localhost:8501 (przekierowuje na `/api/v1/dashboard/ui`) |
| API Gateway (Traefik) | http://localhost:80 · dashboard: http://localhost:8080 |
| market-data / feature-engine / strategy | :8001 · :8002 · :8003 (`/docs`) |
| backtest / ml-pipeline / risk-mgmt | :8004 · :8005 · :8006 (`/docs`) |
| execution / notification | :8007 · :8008 (`/docs`) |
| fundamental-data / macro-data | :8009 · :8010 (`/docs`) |
| company-classifier / signal-aggregator | :8011 · :8012 (`/docs`) |
| Grafana · Prometheus · NATS monitoring | :3000 · :9090 · :8222 |

Każdy serwis eksponuje `GET /health` (liveness), `GET /ready` (realne sprawdzenie zależności — DB/Redis/NATS/serwisy nadrzędne) i `GET /metrics` (Prometheus). Logi w JSON (structlog).

> **Uwaga na hasło do bazy.** `POSTGRES_PASSWORD` działa **wyłącznie przy pierwszym tworzeniu wolumenu**. Jeśli zmienisz `DB_PASSWORD` w `.env` po pierwszym starcie, baza dalej ma stare hasło, a serwisy wysyłają nowe — każdy zapis kończy się wtedy `password authentication failed for user "trader"`. Naprawa bez utraty danych:
> ```bash
> docker compose -f infrastructure/docker-compose.yml --env-file .env exec postgres \
>   psql -U trader -d trading_db -c "ALTER USER trader WITH PASSWORD 'hasło-z-.env';"
> ```
> Albo od zera (kasuje dane): `docker compose ... down -v` i ponowny `up`.

### Diagnostyka

Gdy coś nie działa, [`scripts/diagnose.py`](scripts/diagnose.py) zbiera cały obraz sytuacji w jednym przebiegu — stan kontenerów, `/health` i `/ready` wszystkich 13 serwisów, schemat bazy, Redis, strumienie NATS, próbny fetch z prawdziwym komunikatem błędu i ostatni traceback z market-data:

```bash
python scripts/diagnose.py
```

Używa wyłącznie biblioteki standardowej, nigdy nie drukuje haseł i przeżywa każdą awarię (martwy stack też da czytelny raport). Wynik nadaje się do wklejenia w zgłoszeniu.

### Pomocne komendy

```bash
make up / down / build         # cykl życia stacku
make build-market-data         # pojedynczy obraz
make logs / logs-ml-pipeline   # logi
make test                      # testy wszystkich komponentów
make verify-jetstream          # end-to-end NATS bez Dockera (spawnuje własny nats-server)
make bootstrap-universe        # backfill danych + opcjonalny trening (patrz niżej)
```

---

## Bootstrap danych i pierwszy trening

Skrypt [`scripts/bootstrap-universe.py`](scripts/bootstrap-universe.py) steruje **działającym** stackiem wyłącznie po HTTP — nie dotyka bazy i nie importuje `yfinance`. Pobieranie, walidacja, zapis i publikacja zdarzeń zostają po stronie market-data.

```bash
make bootstrap-universe ARGS="--train --report-out reports/first-training.json"
```

Co się dzieje po kolei:

1. **Backfill** — dla ~34 dużych spółek rozłożonych po sektorach GICS: 6 lat świec dziennych → walidacja → idempotentny upsert → jedno `market_data.updated` na symbol. Każde takie zdarzenie budzi żywą pętlę, więc po backfillu zobaczysz pierwsze **papierowe** pozycje na dashboardzie.
2. **Kontrola pokrycia** — odczyt zapisanych świec i sprawdzenie liczby sesji, zakresu i luk dłuższych niż 5 dni sesyjnych.
3. **Trening** (`--train`) — zbiór przekrojowy dla całego uniwersum, etykiety triple-barrier, purged walk-forward z embargiem, bramka aktywacji. Model trafia do MLflow **niezależnie od wyniku bramki**; baseline driftu rejestruje się automatycznie.
4. **Raport** (`--report-out`) — samowystarczalny JSON: pokrycie, kontekst zbioru (ile symboli miało historię, ile sesji, `positive_rate`) i pełny raport bramki z diagnostyką każdego foldu.

### Bramka aktywacji — sześć warunków

Sam Sharpe nie wystarcza i mamy na to dowód z własnego biegu: model o holdoutowym AUC **0.4865**
(poniżej rzutu monetą) przeszedł dawną bramkę, bo portfel long-only w rosnącym rynku zarobił na
becie — uniwersum ważone równo zrobiło wtedy Sharpe 1.36 przy 0.79 modelu. Bramka
([`core/gate.py`](services/ml-pipeline/src/core/gate.py)) sprawdza teraz sześć rzeczy i raportuje
każdą osobno wraz z liczbami:

| | Pytanie | Warunek |
|---|---|---|
| **G0** | Czy w ogóle powstał model? | fit poprawił się poza pierwszą epokę, predykcje nie są stałe, dość okien |
| **G1** | Czy ranking niesie informację? | **t-stat** średniego IC ≥ 2 (sam poziom IC nic nie znaczy bez błędu standardowego) |
| **G2** | Czy warstwa ML zarabia na siebie? | IC modelu **ze znakiem** wyższe niż IC najlepszej pojedynczej surowej cechy |
| **G3** | Czy zarabia i czy to jego zasługa? | Sharpe > 0,5 **i** active > 0 **i** lift > 0 **i** 2 z 3 ostatnich foldów |
| **G4** | Czy prawdopodobieństwa są uczciwe? | Brier ≤ wskaźnik bazowy **tego okna**, AUC > 0,5 |
| **G5** | Czy wynik przeżyje liczbę prób? | deflated Sharpe ≥ 0,90 na **sklejonej** krzywej OOS |

Próg G5 to świadoma decyzja: 0,95 (podręcznikowe) obowiązuje przed realnym kapitałem, a ta bramka
rządzi promocją do **papierowego** głosu — między nią a pieniędzmi stoi osobna reguła „30 dni
dodatniego Sharpe'a na papierze". Przy ~600 sesjach OOS i 10 próbach 0,90 i tak wymaga ok. 1,8
Sharpe'a rocznie.

### Sonda pojemności — „niedouczenie czy brak sygnału?"

```bash
make bootstrap-universe ARGS="--skip-backfill --capacity-probe"
```

Płaskie `auc_train ≈ 0,5` wygląda identycznie w dwóch przypadkach o przeciwnych rozwiązaniach:
problem optymalizacji albo brak sygnału w cechach. Sonda uczy **celowo przesadzony** model (bez
dropoutu, bez weight decay, bez early stoppingu) i porównuje train AUC na prawdziwych etykietach
z train AUC na etykietach **przetasowanych**. Kontrola jest tu istotą pomiaru — na czystym szumie
sonda i tak dochodzi do ~0,71 train AUC, więc bez niej wyglądałoby to na sukces. Liczy się
**różnica**: duża → struktura istnieje i trzeba naprawić trening; bliska zeru → to zapamiętywanie,
a więcej symboli i historii niczego nie zmieni.

Promocja jest **ręczna** — skrypt wypisuje gotową komendę tylko przy zdanej bramce:

```bash
curl -X POST localhost:8005/api/v1/ml-pipeline/models/versions/1/promote
```

Serwowanie przeładowuje się na gorąco (bez restartu), a dzienny monitor zaczyna rozliczać dojrzałe głosy modelu i pilnować driftu.

---

## Reguły ryzyka

Twarde, nienegocjowalne reguły wbudowane w kod (nie w dokumentację):

- Żadne zlecenie bez `stop_loss` — wymuszone walidacją `TradingSignal` i ponownie przez `RiskEnvelope`.
- Maks. **5%** portfela na pozycję, maks. **80%** łącznej ekspozycji.
- Wielkość pozycji zależna od obsunięcia: pełne 2% ryzyka do DD 5%, potem liniowo do zera przy DD 15%.
- Reżim makro steruje limitem ekspozycji (kryzys → 15%, kontrakcja → 35%) i dopuszczalnymi sektorami.
- Strata dzienna > 5% → wstrzymanie handlu do następnego dnia. Obsunięcie > 15% → zamknięcie pozycji.
- Wyłącznik bezpieczeństwa uzbrojony 24/7; jego stan przeżywa restart (odtwarzany z Redisa).
- Żadna strategia nie idzie na żywo bez walidacji walk-forward OOS i Sharpe'a > 0,5.

---

## Testowanie

```bash
make test                 # wszystkie komponenty
make test-market-data     # pojedynczy serwis
cd services/ml-pipeline && python -m pytest tests/ -v
```

Stan na 2026-08-02: **1378 testów, wszystkie zielone** na Pythonie 3.12; `ruff` + `ruff format` + `mypy` czyste (`--strict` dla `trading-common`). Rozkład:

| Komponent | Testów | Co jest testowane |
|-----------|:---:|-------------------|
| `ml-pipeline` | 321 | Etykiety triple-barrier, podziały purged, trening, bramka G0–G5, rejestr MLflow, serwowanie, monitoring, badania |
| `trading-common` | 308 | Kontrakty, zdarzenia, `RiskEnvelope`, `CostAwareFilter`, cechy, rangi, fundamenty, scheduler, **registry strategii** |
| `risk-mgmt` | 133 | Wielkość pozycji, limity reżimowe/sektorowe, wyłącznik z zatrzaskiem, trwałość stanu |
| `signal-aggregator` | 97 | Łączenie sygnałów per strategia, wagi adaptacyjne, bufory TTL, wybór poziomów, filtr kosztów |
| `market-data` | 71 | Fetchery, masowy upsert i deduplikacja, cache, harmonogram przyrostowy, wykrywanie restatementu |
| `strategy` | 60 | Reguły z registry (sygnał per strategia), brama ryzyka, degradacja, rewalidacja z backtestu |
| `backtest` | 58 | Punktacja ścieżki pozycji bez podglądania przyszłości, ocena reguły z registry, walk-forward |
| `fundamental-data` | 54 | F-Score Piotroskiego (9 sygnałów), klient EDGAR, panel point-in-time, czynniki §5 |
| `execution` | 60 | Wypełnienia, idempotencja, wyjścia ochronne, tylko long, trwałość, historia kapitału |
| `macro-data` | 54 | Detekcja reżimu, klient FRED/ALFRED, panel vintage, odczyt as-of |
| `feature-engine` | 38 | Orkiestracja cech, wzbogacanie atrybutami, magazyn, API rang |
| `notification` | 33 | Mapowanie zdarzeń na alerty, kanały, izolacja awarii |
| `scripts/` | 33 | Bootstrap uniwersum, kontrola pokrycia, diagnostyka, audyt zależności |
| `company-classifier` | 25 | Klasyfikacja stylu, routing modeli |
| `dashboard` | 34 | 6 sekcji, statystyki ryzyka, sonda zdrowia, tolerancja braku serwisu |

### Wzorzec konfiguracji testów

Serwisy z wymaganym `DB_PASSWORD` potrzebują env **przed** importem `src.*`:

```python
# tests/conftest.py
import os

os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("REDIS_PASSWORD", "test_redis")

from src.main import app  # dopiero tutaj
```

### Reguła

> Każdy nowy kod musi mieć testy. Nowy endpoint → test HTTP. Nowa funkcja biznesowa → test jednostkowy. Nowy event → test kontraktu. Zmiana ścieżki zdarzeniowej → weryfikacja na prawdziwym `nats-server`.

---

## Git Workflow

```
main                    ← gałąź główna; bezpośredni commit blokowany przez pre-commit
└── feat/… fix/… data/… ← praca; scalane do main po przejściu testów
```

```bash
git checkout -b feat/nazwa-funkcji
git add . && git commit -m "feat(zakres): opis"   # pre-commit: ruff + ruff-format
git push -u origin feat/nazwa-funkcji
```

Format commitów (Conventional Commits): `<typ>(<zakres>): <opis>`, gdzie typ to `feat | fix | chore | docs | test | refactor | perf`, a zakres to nazwa serwisu, `shared`, `infra` lub `ci`.

### Pre-commit hooks

| Hook | Akcja |
|------|-------|
| `trailing-whitespace`, `end-of-file-fixer` | Higiena plików |
| `check-yaml` (`--unsafe`), `check-toml` | Walidacja składni (`--unsafe` przepuszcza znaczniki Helma) |
| `check-merge-conflict` | Blokuje niedokończone scalenia |
| `check-added-large-files` | Blokuje pliki > 500 KB |
| `no-commit-to-branch` | Blokuje bezpośredni commit na `main` |
| `ruff`, `ruff-format` | Lint z auto-fixem i formatowanie |

```bash
python -m pre_commit install         # po sklonowaniu
python -m pre_commit run --all-files # ręcznie
```

---

## CI/CD

| Workflow | Wyzwalacz | Co robi |
|----------|-----------|---------|
| [`ci.yml`](.github/workflows/ci.yml) | push na `main`/`develop`, PR do `main` | Wykrywa zmienione serwisy (13 + `shared`) i uruchamia dla nich lint + mypy + pytest. Zmiana w `shared/**` testuje wszystko. |
| [`build-images.yml`](.github/workflows/build-images.yml) | merge do `main` | Buduje i publikuje 13 obrazów do `ghcr.io` z cache warstw |
| [`deploy.yml`](.github/workflows/deploy.yml) | po udanym buildzie | `helm upgrade` na klaster (staging domyślnie) |

> Gałęzie robocze **nie mają CI** dopóki nie powstanie PR do `main` — weryfikuj lokalnie przed wypchnięciem.

---

## Kubernetes / Helm

Chart jest **generyczny**: jeden szablon renderuje Deployment + Service dla wszystkich 13 serwisów na podstawie mapy `services:` w `values.yaml` (klucz = nazwa w k8s = nazwa w Compose).

```bash
make helm-template     # podgląd (dry-run)
make helm-install      # deploy
helm upgrade --install trading-system ./infrastructure/helm \
  -f infrastructure/helm/values.yaml -f infrastructure/helm/values-prod.yaml \
  --namespace trading-system --create-namespace
```

| Plik | Opis |
|------|------|
| [`helm/values.yaml`](infrastructure/helm/values.yaml) | Mapa 13 serwisów: obraz, port, env, `needsDb`, `startupSeconds` |
| [`helm/values-prod.yaml`](infrastructure/helm/values-prod.yaml) | Overrides produkcyjne (repliki tylko dla serwisów bez subskrypcji, logi `INFO`) |
| [`helm/templates/services.yaml`](infrastructure/helm/templates/services.yaml) | Generyczny Deployment + Service (sondy, adnotacje Prometheusa, sekrety) |
| [`helm/templates/ingress.yaml`](infrastructure/helm/templates/ingress.yaml) | 13 tras `/api/v1/{serwis}` — lustro etykiet Traefika z Compose |
| [`helm/templates/postgres-statefulset.yaml`](infrastructure/helm/templates/postgres-statefulset.yaml) | StatefulSet PostgreSQL |
| [`k8s/secrets.yaml.example`](infrastructure/k8s/secrets.yaml.example) | Wzorzec Secret (nie commituj prawdziwych wartości) |

`startupProbe` (per serwis, `startupSeconds`) wstrzymuje sondy liveness/readiness do końca startu — bez tego ml-pipeline z importem torcha wpadałby w crashloop. Replik > 1 używają tylko serwisy **bez** subskrypcji zdarzeń: konsumenci push nie rozkładają obciążenia, a risk-mgmt i execution są jedynymi pisarzami swojego stanu.

---

## Struktura plików

```
Market-App/
├── CLAUDE.md                        # Kontekst projektu, status, priorytety (czytaj najpierw)
├── Makefile                         # Skróty deweloperskie
├── .env.example                     # Szablon sekretów
│
├── docs/
│   ├── README.md                    # Mapa dokumentacji — który plik obowiązuje
│   ├── decisions/             # Dlaczego jest tak, a nie inaczej (7 plików + D1–D8)
│   ├── ml_integration_plan.md       # Architektura fazy ML (ML-0…ML-4)
│   ├── framework_supplement.md      # Referencja 12 komponentów + usunięty kod
│   └── archive/                     # Zamrożone: brief, audyt, backlog po audycie
│
├── .github/workflows/               # ci.yml · build-images.yml · deploy.yml
│
├── infrastructure/
│   ├── docker-compose.yml           # Środowisko dev (13 serwisów + infra)
│   ├── docker-compose.prod.yml      # Nakładka produkcyjna (!reset portów, TLS)
│   ├── init-db.sql                  # Inicjalizacja TimescaleDB
│   ├── helm/                        # Generyczny chart (values.yaml jako mapa serwisów)
│   ├── k8s/                         # namespace.yaml, secrets.yaml.example
│   └── monitoring/                  # prometheus.yml, alertmanager.yml, dashboardy Grafany
│
├── shared/trading-common/           # Biblioteka kontraktów (pip install -e)
│
├── services/                        # 13 mikroserwisów (9 rdzenia + 4 ML/AI)
│
└── scripts/
    ├── bootstrap-universe.py        # Backfill uniwersum + pierwszy trening + raport
    ├── verify-jetstream.py          # End-to-end NATS bez Dockera
    ├── run-all-tests.sh             # Testy wszystkich komponentów
    ├── setup-dev.sh                 # Zależności lokalnie
    └── seed-data.sh                 # Dane testowe
```

### Struktura serwisu (wzorzec)

```
services/{nazwa}/
├── Dockerfile              multi-stage (builder + runtime), użytkownik non-root
├── pyproject.toml          hatchling, Python 3.12+, extras [dev]
├── src/
│   ├── main.py             FastAPI + lifespan (połączenia, subskrypcje, scheduler)
│   ├── config.py           pydantic-settings
│   ├── api/                routery HTTP
│   ├── core/               logika biznesowa + observability.py (/health /ready /metrics)
│   ├── events/             publisher.py (NATS) + subscriber.py (durable)
│   └── models/             SQLAlchemy ORM + re-eksport schematów z trading-common
└── tests/
```

---

## Zmienne środowiskowe

Minimum w `.env`:

```env
DB_PASSWORD=...      # WYMAGANE — Compose nie wystartuje bez tej zmiennej
REDIS_PASSWORD=...   # WYMAGANE
```

Opcjonalne, włączające funkcje (brak klucza = funkcja wyłączona, serwis działa dalej):
`ALPHA_VANTAGE_API_KEY` (zapasowe źródło OHLCV) · `FRED_API_KEY` (makro) · `SEC_USER_AGENT` (EDGAR) ·
`SLACK_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `SMTP_HOST`+`EMAIL_FROM`+`EMAIL_TO` (kanały alertów).

Pełny szablon: [`.env.example`](.env.example).
