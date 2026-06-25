# Trading System — Architektura Mikroserwisowa

System tradingowy oparty na mikroserwisach: event-driven, Kubernetes-ready od dnia 1, z pełnym observability stackiem. Realizuje 24-tygodniowy plan rozwoju zawarty w [`Plan_Rozwoju_Systemu_Tradingowego_2.md`](Plan_Rozwoju_Systemu_Tradingowego_2.md).

---

## Spis treści

- [Architektura](#architektura)
- [Serwisy](#serwisy)
- [Infrastruktura](#infrastruktura)
- [Shared Library](#shared-library)
- [Uruchamianie](#uruchamianie)
- [Testowanie](#testowanie)
- [Git Workflow](#git-workflow)
- [CI/CD](#cicd)
- [Kubernetes / Helm](#kubernetes--helm)
- [Struktura plików](#struktura-plików)

---

## Architektura

```
                      ┌──────────────────────────────────┐
                      │   API GATEWAY (Traefik :80/443)  │
                      │  routing · rate-limit · auth      │
                      └──┬────┬────┬────┬────┬───────────┘
                         │    │    │    │    │
              ┌──────────┘  ┌─┘  ┌─┘  ┌─┘  └─────────┐
              ▼             ▼    ▼    ▼                ▼
        ┌──────────┐  ┌────────┐ ┌────────┐  ┌──────────┐
        │ market-  │  │feature-│ │strategy│  │ backtest │
        │ data     │  │engine  │ │        │  │          │
        └────┬─────┘  └───┬────┘ └───┬────┘  └────┬─────┘
             │            │          │              │
             └────────────┴──────────┴──────────────┘
                                   │
                     ┌─────────────▼──────────────┐
                     │   Message Broker (NATS)     │
                     │   event-driven pipeline     │
                     └────┬──────────┬─────────────┘
                          │          │
                ┌─────────▼──┐  ┌────▼──────┐
                │ml-pipeline │  │ risk-mgmt │
                └────────────┘  └───────────┘
                          │          │
                     ┌────▼──────────▼────┐
                     │  execution-svc     │
                     │ (paper / live)     │
                     └────────────────────┘
                                │
                      ┌─────────▼──────────┐
                      │  notification-svc  │
                      │ Telegram/email/... │
                      └────────────────────┘

  ┌────────────────────────────────────────────┐
  │  INFRASTRUKTURA WSPÓŁDZIELONA              │
  │  PostgreSQL + TimescaleDB · Redis · NATS   │
  │  Prometheus · Grafana · Loki · Traefik     │
  └────────────────────────────────────────────┘
```

### Zasady komunikacji

| Typ | Protokół | Kiedy |
|-----|----------|-------|
| Synchroniczna | HTTP/REST (FastAPI) | Zapytania on-demand, dashboard → serwis |
| Asynchroniczna | NATS JetStream | Sygnały, eventy rynkowe, pipeline ML |

---

## Serwisy

| Serwis | Port (dev) | Faza | Opis |
|--------|-----------|------|------|
| [`market-data`](services/market-data/) | 8001 | 1 – tydzień 2 | Pobieranie, walidacja i przechowywanie OHLCV |
| [`feature-engine`](services/feature-engine/) | 8002 | 1 – tydzień 3 | Wskaźniki techniczne i feature engineering |
| [`strategy`](services/strategy/) | 8003 | 2 – tydzień 5 | Definicja strategii, generowanie sygnałów |
| [`backtest`](services/backtest/) | 8004 | 2 – tydzień 7 | Silnik backtestingu, optymalizacja, walk-forward |
| [`ml-pipeline`](services/ml-pipeline/) | 8005 | 3 – tydzień 13 | Training, inference, model registry (MLflow) |
| [`risk-mgmt`](services/risk-mgmt/) | 8006 | 4 – tydzień 19 | Position sizing, portfolio optimization, VaR |
| [`execution`](services/execution/) | 8007 | 5 – tydzień 22 | Paper trading / live trading, order management |
| [`notification`](services/notification/) | 8008 | 5 – tydzień 23 | Alerty: Telegram, email, Slack |
| [`dashboard`](services/dashboard/) | 8501 | 4 – tydzień 21 | UI (Streamlit lub Next.js) |

Każdy serwis posiada identyczną strukturę wewnętrzną — patrz [Struktura serwisu](#struktura-serwisu).

---

## Infrastruktura

Pliki konfiguracyjne w [`infrastructure/`](infrastructure/).

### Komponenty

| Komponent | Obraz | Port | Rola |
|-----------|-------|------|------|
| PostgreSQL + TimescaleDB | `timescale/timescaledb:latest-pg16` | 5432 | Baza danych; OHLCV jako hypertable |
| Redis | `redis:7-alpine` | 6379 | Cache + pub/sub |
| NATS | `nats:2-alpine` | 4222 / 8222 | Event bus (JetStream) |
| Traefik | `traefik:v3.0` | 80 / 8080 | API Gateway, dashboard |
| Prometheus | `prom/prometheus` | 9090 | Metryki |
| Grafana | `grafana/grafana` | 3000 | Dashboardy |
| Loki | `grafana/loki` | 3100 | Logi strukturalne |

### Baza danych

Schemat inicjalizowany przez [`infrastructure/init-db.sql`](infrastructure/init-db.sql):

- Schematy izolowane per-serwis: `market_data`, `feature_engine`, `strategy`, `backtest`, `risk_mgmt`, `execution`
- `market_data.ohlcv` — TimescaleDB hypertable, kompresja danych > 7 dni
- `strategy.signals`, `backtest.results`, `risk_mgmt.positions`

---

## Shared Library

[`shared/trading-common/`](shared/trading-common/) — biblioteka Pydantic installowalna przez `pip install -e`.

### Moduły

| Moduł | Zawartość |
|-------|-----------|
| [`schemas.py`](shared/trading-common/src/trading_common/schemas.py) | `OHLCVBar`, `TradingSignal`, `PortfolioMetrics`, `Signal`, `Interval` |
| [`events.py`](shared/trading-common/src/trading_common/events.py) | Kontrakty NATS: `MarketDataUpdatedEvent`, `SignalGeneratedEvent`, `OrderFilledEvent`, `RiskLimitBreachedEvent` itd. |
| [`constants.py`](shared/trading-common/src/trading_common/constants.py) | Porty serwisów, symbole domyślne, NATS subjects, limity ryzyka |
| [`utils.py`](shared/trading-common/src/trading_common/utils.py) | `utcnow()`, `to_utc()`, `symbol_to_topic()` |

### Instalacja

```bash
pip install -e shared/trading-common          # runtime
pip install -e "shared/trading-common[dev]"   # + narzędzia testowe
```

### Import w serwisie

```python
from trading_common.schemas import OHLCVBar, TradingSignal
from trading_common.events import SignalGeneratedEvent, EventType
```

---

## Uruchamianie

### Wymagania

- Docker Desktop (z Compose v2)
- Python 3.12 (do lokalnych testów)

### Pierwsze uruchomienie

```bash
# 1. Skopiuj i uzupełnij sekrety
cp .env.example .env
# → edytuj .env: ustaw DB_PASSWORD i REDIS_PASSWORD

# 2. Uruchom stack (infrastruktura + serwisy fazy 1)
make up

# 3. Sprawdź logi
make logs
```

> **Uwaga:** `.env` musi być w katalogu root (`Market_App/`), nie w `infrastructure/`.
> Makefile przekazuje `--env-file .env` do compose automatycznie.

### Status infrastruktury (oczekiwany — weryfikowalny po `make up`; brak Dockera w sandbox/CI)

| Komponent | Status | Weryfikacja |
|-----------|--------|-------------|
| PostgreSQL + TimescaleDB | ✓ healthy | `pg_isready` + schematy z `init-db.sql` |
| Redis | ✓ healthy | `PONG` na ping |
| NATS + JetStream | ✓ healthy | `store_dir: /data/jetstream` aktywny |
| Prometheus | ✓ ready | `/‑/ready` → 200 |
| Grafana | ✓ running | `/api/health` → `ok` |
| Loki | ✓ ready | `/ready` → 200 |
| Traefik | ✓ running | `/api/overview` → 200 |

### Dostęp do usług (dev)

| Usługa | URL |
|--------|-----|
| API Gateway | http://localhost:80 |
| Traefik dashboard | http://localhost:8080 |
| market-data API | http://localhost:8001/docs |
| feature-engine API | http://localhost:8002/docs |
| strategy API | http://localhost:8003/docs |
| backtest API | http://localhost:8004/docs |
| Grafana | http://localhost:3000 (admin / wartość z `.env`) |
| Prometheus | http://localhost:9090 |
| NATS monitoring | http://localhost:8222 |

### Pomocne komendy

```bash
make up               # Uruchom wszystkie serwisy
make down             # Zatrzymaj
make build            # Zbuduj obrazy
make build-market-data  # Zbuduj konkretny serwis
make logs             # Śledź logi
make logs-market-data   # Logi konkretnego serwisu
make seed             # Załaduj dane testowe
```

---

## Testowanie

### Uruchom wszystkie testy

```bash
make test
# lub bezpośrednio:
bash scripts/run-all-tests.sh
```

### Testy konkretnego serwisu / biblioteki

```bash
make test-market-data
make test-shared       # (cd shared/trading-common && pytest)
```

### Testy lokalne bez Dockera

```bash
# Zainstaluj zależności (Windows: --user, Linux/Mac: wirtualne środowisko)
bash scripts/setup-dev.sh

# Uruchom testy jednego serwisu
cd services/market-data
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

> **Windows:** `pip install --user` jest wymagane jeśli Python jest zainstalowany w `C:\Program Files\` (brak uprawnień zapisu). Skrypt `setup-dev.sh` obsługuje to automatycznie.

### Co jest testowane

> Liczby testów są orientacyjne (rosną wraz z kodem) — źródłem prawdy jest `make test`.
> Stan na 2026-06-25: ~330 testów, wszystkie zielone na Pythonie 3.12.

| Komponent | Co jest testowane | Orientacyjnie |
|-----------|-------------------|:---:|
| `trading-common` | Kontrakty Pydantic (schemas), eventy NATS, `RiskEnvelope`, utils | ~126 |
| `market-data` | Szkielet: /health, /ready, /metrics, /ohlcv, /fetch, /symbols | 7 |
| `strategy` | `decay_monitor`, `cost_filter`, `adaptive_weights` + health | ~66 |
| `risk-mgmt` | `adaptive_sizing`, `regime_allocator` + health | ~57 |
| `feature-engine` | `vol_regime`, `earnings_decay`, `cross_asset` + health | ~50 |
| pozostałe serwisy | health + komponenty `framework_supplement` | rośnie |

### Znane wzorce konfiguracji testów

Serwisy z wymaganym `DB_PASSWORD` w `config.py` potrzebują ustawienia env przed importem:

```python
# tests/conftest.py — MUSI być przed importem src.*
import os
os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("REDIS_PASSWORD", "test_redis")

from src.main import app  # dopiero tutaj
```

### Reguła testowania

> Każdy nowy kod musi mieć testy. Nowy endpoint → test HTTP. Nowa funkcja biznesowa → test jednostkowy. Nowy event → test kontraktu.

---

## Git Workflow

### Strategia branchy

```
main        ← produkcja; chroniony przez hook (brak bezpośrednich commitów)
│
└── develop ← integracja; bieżąca gałąź robocza
    │
    ├── feat/market-data-fetcher   ← nowa funkcja
    ├── fix/ohlcv-validator-high-low
    └── chore/update-dependencies
```

### Codzienny cykl pracy

```bash
# 1. Utwórz branch z develop
git checkout develop
git checkout -b feat/nazwa-funkcji

# 2. Pracuj i commituj (pre-commit lint uruchomi się automatycznie)
git add services/market-data/src/...
git commit -m "feat(market-data): add OHLCV fetcher via yfinance"

# 3. Scal do develop po zakończeniu
git checkout develop
git merge --no-ff feat/nazwa-funkcji
git branch -d feat/nazwa-funkcji

# 4. Do main przez PR (GitHub) — CI musi przejść
```

### Format commitów (Conventional Commits)

```
<typ>(<zakres>): <opis>

typ:    feat | fix | chore | docs | test | refactor | perf
zakres: market-data | feature-engine | strategy | shared | infra | ci
```

Przykłady:
```
feat(market-data): implement OHLCV fetch from yfinance
fix(shared): replace deprecated datetime.utcnow with timezone-aware
test(market-data): add integration tests for DB layer
chore(deps): bump pydantic to 2.12
```

### Pre-commit hooks (automatyczne przy `git commit`)

| Hook | Akcja |
|------|-------|
| `trailing-whitespace` | Usuwa trailing whitespace |
| `end-of-file-fixer` | Zapewnia newline na końcu pliku |
| `check-yaml` / `check-toml` | Walidacja składni |
| `check-added-large-files` | Blokuje pliki > 500 KB |
| `no-commit-to-branch` | Blokuje bezpośredni commit na `main` |
| `ruff` | Lint + auto-fix |
| `ruff-format` | Formatowanie kodu |

Konfiguracja: [`.pre-commit-config.yaml`](.pre-commit-config.yaml)

```bash
# Ręczne uruchomienie na wszystkich plikach
python -m pre_commit run --all-files

# Reinstalacja hooków (np. po klonowaniu)
python -m pre_commit install
```

### Podłączenie remote (GitHub)

```bash
# 1. Utwórz repo na GitHub (bez inicjalizacji — repo jest już lokalnie)
# 2. Podłącz remote i wypchnij
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
git push -u origin develop
```

> Ustaw ochronę brancha `main` w GitHub: Settings → Branches → Add rule → wymagaj PR + CI pass.

---

## CI/CD

Pipeline GitHub Actions w [`.github/workflows/`](.github/workflows/).

### Workflow: CI ([`ci.yml`](.github/workflows/ci.yml))

Wyzwalany przy każdym push i pull request:

1. **detect-changes** — wykrywa które serwisy zmieniły się (nie testuje wszystkich przy zmianie jednego)
2. **test-shared** — lint + mypy + pytest dla `trading-common` (blokujące)
3. **test-services** — matrix: lint + mypy + pytest dla każdego zmienionego serwisu

### Workflow: Build ([`build-images.yml`](.github/workflows/build-images.yml))

Wyzwalany przy merge do `main`:
- Buduje i pushuje obrazy Docker do `ghcr.io`
- Cache warstw przez GitHub Actions cache

### Workflow: Deploy ([`deploy.yml`](.github/workflows/deploy.yml))

Wyzwalany po udanym `build-images.yml`:
- Helm upgrade do klastra K8s
- `staging` (domyślnie) lub `production`

---

## Kubernetes / Helm

### Helm chart

```bash
# Podgląd (dry-run)
make helm-template

# Deploy
make helm-install

# Deploy produkcyjny
helm upgrade --install trading-system ./infrastructure/helm \
  -f infrastructure/helm/values.yaml \
  -f infrastructure/helm/values-prod.yaml \
  --namespace trading-system --create-namespace
```

### Pliki

| Plik | Opis |
|------|------|
| [`helm/values.yaml`](infrastructure/helm/values.yaml) | Domyślne wartości (dev) |
| [`helm/values-prod.yaml`](infrastructure/helm/values-prod.yaml) | Produkcyjne overrides (repliki, logi) |
| [`helm/templates/market-data-deployment.yaml`](infrastructure/helm/templates/market-data-deployment.yaml) | Wzorzec deploymentu (kopiuj dla nowych serwisów) |
| [`helm/templates/postgres-statefulset.yaml`](infrastructure/helm/templates/postgres-statefulset.yaml) | StatefulSet dla PostgreSQL |
| [`k8s/secrets.yaml.example`](infrastructure/k8s/secrets.yaml.example) | Wzorzec K8s Secret (nie commituj z prawdziwymi wartościami!) |

---

## Struktura plików

```
Market_App/
├── .env.example                          # Szablon sekretów — skopiuj do .env
├── .gitignore
├── Makefile                              # Skróty deweloperskie
│
├── .github/workflows/
│   ├── ci.yml                            # Lint + test na każdym PR
│   ├── build-images.yml                  # Build + push Docker images
│   └── deploy.yml                        # Deploy do K8s
│
├── infrastructure/
│   ├── docker-compose.yml                # Dev environment
│   ├── docker-compose.prod.yml           # Production overrides
│   ├── init-db.sql                       # Inicjalizacja TimescaleDB
│   ├── helm/                             # Kubernetes Helm chart
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   ├── values-prod.yaml
│   │   └── templates/
│   ├── k8s/                              # Raw K8s manifesty
│   ├── monitoring/
│   │   ├── prometheus.yml
│   │   ├── alertmanager.yml
│   │   └── grafana-dashboards/
│   └── terraform/                        # IaC (przyszłość)
│
├── shared/
│   └── trading-common/                   # Pip-installable shared library
│       ├── src/trading_common/
│       │   ├── schemas.py                # Pydantic contracts
│       │   ├── events.py                 # NATS event definitions
│       │   ├── constants.py
│       │   └── utils.py
│       └── tests/
│
├── services/                             # 9 mikroserwisów
│   ├── market-data/      # port 8001
│   ├── feature-engine/   # port 8002
│   ├── strategy/         # port 8003
│   ├── backtest/         # port 8004
│   ├── ml-pipeline/      # port 8005
│   ├── risk-mgmt/        # port 8006
│   ├── execution/        # port 8007
│   ├── notification/     # port 8008
│   └── dashboard/        # port 8501
│
│   # Każdy serwis:
│   # ├── Dockerfile          (multi-stage, non-root user)
│   # ├── pyproject.toml      (hatchling, [dev] extras)
│   # ├── src/
│   # │   ├── main.py         (FastAPI + lifespan)
│   # │   ├── config.py       (pydantic-settings)
│   # │   ├── api/            (routers)
│   # │   ├── core/
│   # │   │   └── observability.py  (/health /ready /metrics)
│   # │   ├── models/
│   # │   └── events/
│   # └── tests/
│   #     ├── conftest.py     (AsyncClient fixture)
│   #     └── test_health.py
│
└── scripts/
    ├── setup-dev.sh          # Instalacja zależności lokalnie
    ├── run-all-tests.sh      # Testy wszystkich komponentów z raportem
    └── seed-data.sh          # Ładowanie danych testowych
```

### Struktura serwisu (wzorzec)

```
services/{nazwa}/
├── Dockerfile              multi-stage build (builder + runtime), non-root user
├── pyproject.toml          hatchling, wymaga Python 3.12+, [dev] extras
└── src/
    ├── main.py             FastAPI app, lifespan hooks (init/cleanup połączeń)
    ├── config.py           pydantic-settings: DB, Redis, NATS, service-specific
    ├── api/
    │   ├── __init__.py     Rejestracja routerów
    │   └── routes.py       HTTP endpoints
    ├── core/
    │   └── observability.py  setup_observability(app, name) → /health /ready /metrics
    ├── models/             SQLAlchemy modele + Pydantic schemas (per-serwis)
    └── events/             NATS publishers i subscribers
```

---

## Zmienne środowiskowe

Minimalne wymagane w `.env`:

```env
DB_PASSWORD=...      # WYMAGANE — docker compose nie wystartuje bez tej zmiennej
REDIS_PASSWORD=...   # WYMAGANE
```

Pełny szablon: [`.env.example`](.env.example)

---

## Obserwability

Każdy serwis eksponuje od dnia 1:

| Endpoint | Opis |
|----------|------|
| `GET /health` | Liveness probe — serwis żyje |
| `GET /ready` | Readiness probe — serwis gotowy (sprawdza DB/Redis/NATS) |
| `GET /metrics` | Prometheus metrics (prometheus-fastapi-instrumentator) |

Logi w formacie JSON (structlog) — zbierane przez Loki, wizualizowane w Grafanie.
