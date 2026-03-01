# Plan Rozwoju Systemu Tradingowego — Architektura Mikroserwisowa
## Kompleksowy przewodnik — 24 tygodnie

---

## Ocena i uzasadnienie zmian względem wersji oryginalnej

### Główne problemy oryginału

1. **Architektura monolityczna** — cały system projektowany jako jeden pakiet Python; Docker oznaczony jako „opcjonalny". Przy rosnącej złożoności (ML, real-time, dashboard) monolit staje się trudny do skalowania i deploymentu.
2. **Kubernetes dopiero po 24 tygodniach** — w sekcji „Next Steps". Przepisywanie monolitu na mikroserwisy post-factum wymaga znacznie więcej pracy niż projektowanie pod nie od początku.
3. **Brak event-driven architecture** — system tradingowy z natury jest event-driven (sygnały, ordery, market data). Oryginał nie przewiduje message brokera.
4. **Brak observability stack** — Prometheus/Grafana pojawiają się dopiero w tygodniu 24 produkcyjnym; bez nich nie da się skutecznie debugować mikroserwisów.
5. **Przestarzałe narzędzia** — `setup.py` zamiast `pyproject.toml`, `version: '3.8'` w compose (deprecated), `fillna(method='ffill')` (deprecated w pandas 2.x), outdated pinned versions.
6. **Konflikt portów** — kontenery `app` i `jupyter` oba eksponują port 8888.
7. **Hardcoded secrets** — hasło do bazy `secure_password` w docker-compose, brak secrets management.
8. **Brak API Gateway** — kluczowy element mikroserwisów do routingu, rate-limitingu i auth.
9. **Brak CI/CD do tygodnia 24** — w podejściu microservices CI/CD powinno działać od pierwszego dnia.
10. **Za dużo kodu, za mało architektury** — dokument zawiera pełne implementacje (5000+ linii kodu) zamiast skupić się na decyzjach architektonicznych i kontraktach między serwisami.

### Filozofia poprawionej wersji

- **Microservices-first**: każdy bounded context to osobny serwis z własnym Dockerfile
- **Docker-native od dnia 1**: docker-compose jako obowiązkowy sposób uruchamiania
- **Kubernetes-ready od tygodnia 1**: Helm chart, manifesty K8s — nawet jeśli początkowo uruchamiamy lokalnie przez `docker compose`
- **Event-driven core**: NATS/Redis Streams jako message broker od samego początku
- **Observability from day 1**: Prometheus + Grafana + structured logging
- **GitOps & CI/CD od początku**: GitHub Actions pipeline od pierwszego commita
- **Kontrakty > implementacja**: dokument definiuje interfejsy, protobuf/OpenAPI schemas, a implementację zostawia do kodowania z AI

---

## 📋 Spis Treści

### [FAZA 0: ARCHITEKTURA I INFRASTRUKTURA (Tydzień 0)](#faza-0-architektura-i-infrastruktura-tydzień-0)
- [Dekompozycja na mikroserwisy](#dekompozycja-na-mikroserwisy)
- [Strategia komunikacji między serwisami](#strategia-komunikacji-między-serwisami)
- [Observability stack](#observability-stack)

### [FAZA 1: FUNDAMENT (Tygodnie 1-4)](#faza-1-fundament-tygodnie-1-4)
- [Tydzień 1: Infrastruktura i DevOps baseline](#tydzień-1-infrastruktura-i-devops-baseline)
- [Tydzień 2: Serwis Market Data](#tydzień-2-serwis-market-data)
- [Tydzień 3: Serwis Feature Engineering](#tydzień-3-serwis-feature-engineering)
- [Tydzień 4: Pierwsza strategia + Backtest Runner](#tydzień-4-pierwsza-strategia--backtest-runner)

### [FAZA 2: STRATEGIE I BACKTESTING (Tygodnie 5-12)](#faza-2-strategie-i-backtesting-tygodnie-5-12)
- [Tydzień 5-6: Strategy Service Framework](#tydzień-5-6-strategy-service-framework)
- [Tydzień 7-8: Backtesting Engine Service](#tydzień-7-8-backtesting-engine-service)
- [Tydzień 9-10: Zaawansowane strategie](#tydzień-9-10-zaawansowane-strategie)
- [Tydzień 11-12: Optymalizacja i Walk-Forward](#tydzień-11-12-optymalizacja-i-walk-forward)

### [FAZA 3: MACHINE LEARNING (Tygodnie 13-18)](#faza-3-machine-learning-tygodnie-13-18)
- [Tydzień 13-14: ML Feature Pipeline Service](#tydzień-13-14-ml-feature-pipeline-service)
- [Tydzień 15-16: ML Model Training Service](#tydzień-15-16-ml-model-training-service)
- [Tydzień 17-18: Sentiment & Ensemble Service](#tydzień-17-18-sentiment--ensemble-service)

### [FAZA 4: RISK & PORTFOLIO (Tygodnie 19-21)](#faza-4-risk--portfolio-tygodnie-19-21)
- [Tydzień 19-20: Risk Management Service](#tydzień-19-20-risk-management-service)
- [Tydzień 21: Dashboard Service (Frontend)](#tydzień-21-dashboard-service-frontend)

### [FAZA 5: PRODUKCJA I LIVE TRADING (Tygodnie 22-24)](#faza-5-produkcja-i-live-trading-tygodnie-22-24)
- [Tydzień 22: Execution Service + Paper Trading](#tydzień-22-execution-service--paper-trading)
- [Tydzień 23: Alerting & Notification Service](#tydzień-23-alerting--notification-service)
- [Tydzień 24: Production Kubernetes Deployment](#tydzień-24-production-kubernetes-deployment)

### [DODATKI](#dodatki)

---

# FAZA 0: ARCHITEKTURA I INFRASTRUKTURA (Tydzień 0)

> Przed napisaniem pierwszej linii kodu — zaprojektuj system.

## Dekompozycja na mikroserwisy

### Mapa serwisów

```
┌─────────────────────────────────────────────────────────────────┐
│                        API GATEWAY (Traefik)                     │
│                    routing / rate-limit / auth                    │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │          │
  ┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼─────┐
  │Market  │ │Feature │ │Strategy│ │Backtest│ │Dashboard│
  │Data    │ │Engine  │ │Service │ │Engine  │ │(React/  │
  │Service │ │Service │ │        │ │        │ │Streamlit│
  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └─────────┘
      │          │          │          │
      │    ┌─────▼──────┐   │    ┌─────▼──────┐
      │    │ML Pipeline │   │    │Risk Mgmt   │
      │    │Service     │   │    │Service     │
      │    └────────────┘   │    └────────────┘
      │                     │
  ┌───▼─────────────────────▼───┐
  │    Message Broker            │
  │    (NATS / Redis Streams)    │
  └──────────┬──────────────────┘
             │
  ┌──────────▼──────────────────┐
  │  Execution Service           │
  │  (Paper Trading / Live)      │
  └─────────────────────────────┘
  
  ┌─────────────────────────────┐
  │  INFRASTRUKTURA (wspólna)    │
  │  PostgreSQL + TimescaleDB    │
  │  Redis (cache + pub/sub)     │
  │  NATS (event bus)            │
  │  Prometheus + Grafana        │
  │  Loki (logi)                 │
  └─────────────────────────────┘
```

### Bounded Contexts → Serwisy

| Serwis | Odpowiedzialność | Port | Technologia |
|--------|-----------------|------|-------------|
| `market-data-svc` | Pobieranie, walidacja i przechowywanie danych OHLCV | 8001 | FastAPI + async |
| `feature-engine-svc` | Obliczanie wskaźników technicznych i feature'ów | 8002 | FastAPI |
| `strategy-svc` | Definicja i ewaluacja strategii, generowanie sygnałów | 8003 | FastAPI |
| `backtest-svc` | Silnik backtestingu, optymalizacja, walk-forward | 8004 | FastAPI + Celery |
| `ml-pipeline-svc` | Training, inference, model registry | 8005 | FastAPI + MLflow |
| `risk-mgmt-svc` | Position sizing, portfolio optimization, risk metrics | 8006 | FastAPI |
| `execution-svc` | Paper/live trading, order management | 8007 | FastAPI + WebSocket |
| `notification-svc` | Alerty: Telegram, email, Slack | 8008 | FastAPI |
| `dashboard-svc` | UI: Streamlit lub React frontend | 8501 | Streamlit / Next.js |
| `api-gateway` | Routing, auth, rate limiting | 80/443 | Traefik |

### Konwencja nazewnictwa

```
trading-system/
├── services/
│   ├── market-data/          # Każdy serwis to osobny kontekst
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py       # FastAPI app entry point
│   │   │   ├── api/          # HTTP endpoints (routers)
│   │   │   ├── core/         # Business logic
│   │   │   ├── models/       # Pydantic schemas + DB models
│   │   │   ├── events/       # Event publishers/subscribers
│   │   │   └── config.py     # Serwis-specific config
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── README.md
│   ├── feature-engine/
│   ├── strategy/
│   ├── backtest/
│   ├── ml-pipeline/
│   ├── risk-mgmt/
│   ├── execution/
│   ├── notification/
│   └── dashboard/
├── infrastructure/
│   ├── docker-compose.yml        # Dev environment
│   ├── docker-compose.prod.yml   # Production
│   ├── helm/                     # Kubernetes Helm chart
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   ├── values-prod.yaml
│   │   └── templates/
│   │       ├── market-data-deployment.yaml
│   │       ├── market-data-service.yaml
│   │       ├── ... (per service)
│   │       ├── postgres-statefulset.yaml
│   │       ├── redis-deployment.yaml
│   │       ├── nats-deployment.yaml
│   │       ├── prometheus-config.yaml
│   │       └── ingress.yaml
│   ├── k8s/                      # Raw manifesty (alternatywa dla Helm)
│   ├── terraform/                # IaC (opcjonalnie)
│   └── monitoring/
│       ├── prometheus.yml
│       ├── grafana-dashboards/
│       └── alertmanager.yml
├── shared/                       # Shared libraries (pip installable)
│   ├── trading-common/
│   │   ├── src/trading_common/
│   │   │   ├── schemas.py        # Pydantic models shared across services
│   │   │   ├── events.py         # Event definitions
│   │   │   ├── constants.py
│   │   │   └── utils.py
│   │   └── pyproject.toml
│   └── trading-proto/            # Protobuf/OpenAPI definitions (opcjonalnie)
├── scripts/
│   ├── setup-dev.sh
│   ├── run-all-tests.sh
│   └── seed-data.sh
├── .github/
│   └── workflows/
│       ├── ci.yml                # Lint + test na każdym PR
│       ├── build-images.yml      # Build + push Docker images
│       └── deploy.yml            # Deploy to K8s
├── .env.example
├── .gitignore
├── Makefile                      # Developer shortcuts
└── README.md
```

## Strategia komunikacji między serwisami

### Synchroniczna (request/response) — HTTP/gRPC

Używaj do: zapytań o dane, on-demand kalkulacji.

```
Dashboard → GET /api/v1/market-data/ohlcv/AAPL → market-data-svc
Dashboard → GET /api/v1/risk/portfolio/metrics   → risk-mgmt-svc
```

### Asynchroniczna (event-driven) — NATS / Redis Streams

Używaj do: sygnałów tradingowych, alertów, pipeline ML.

```python
# shared/trading-common/src/trading_common/events.py
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Optional

class EventType(str, Enum):
    MARKET_DATA_UPDATED = "market_data.updated"
    FEATURES_COMPUTED = "features.computed"
    SIGNAL_GENERATED = "signal.generated"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    RISK_LIMIT_BREACHED = "risk.limit_breached"
    MODEL_TRAINED = "ml.model_trained"
    ALERT_TRIGGERED = "alert.triggered"

class BaseEvent(BaseModel):
    event_type: EventType
    timestamp: datetime
    source_service: str
    correlation_id: Optional[str] = None

class MarketDataUpdatedEvent(BaseEvent):
    event_type: EventType = EventType.MARKET_DATA_UPDATED
    symbol: str
    interval: str
    rows_count: int

class SignalGeneratedEvent(BaseEvent):
    event_type: EventType = EventType.SIGNAL_GENERATED
    symbol: str
    strategy_name: str
    signal: str          # "BUY" | "SELL" | "HOLD"
    confidence: float
    price: float
    metadata: dict = {}
```

### Flow przykładowy: od danych do egzekucji

```
1. market-data-svc pobiera nowe dane AAPL
2. → PUBLISH event: MarketDataUpdatedEvent(symbol="AAPL")
3. feature-engine-svc SUBSCRIBE → oblicza wskaźniki
4. → PUBLISH event: FeaturesComputedEvent(symbol="AAPL")
5. strategy-svc SUBSCRIBE → generuje sygnał BUY
6. → PUBLISH event: SignalGeneratedEvent(signal="BUY", symbol="AAPL")
7. risk-mgmt-svc SUBSCRIBE → waliduje position sizing
8. → PUBLISH event: OrderApprovedEvent(...)
9. execution-svc SUBSCRIBE → składa zlecenie
10. → PUBLISH event: OrderFilledEvent(...)
11. notification-svc SUBSCRIBE → wysyła alert na Telegram
```

## Observability stack

### Od dnia 1 — nie od tygodnia 24

```yaml
# Każdy serwis FastAPI zawiera:
# 1. Prometheus metrics endpoint  /metrics
# 2. Health check endpoint        /health
# 3. Structured JSON logging      (structlog)
```

```python
# Wspólny pattern dla każdego serwisu — src/core/observability.py
import structlog
from prometheus_client import Counter, Histogram, Gauge, Info
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

def setup_observability(app: FastAPI, service_name: str):
    """Konfiguracja observability — wywołaj w main.py każdego serwisu."""
    
    # Structured logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if app.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )
    
    # Prometheus auto-instrumentation
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    
    # Health endpoint
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": service_name}
    
    @app.get("/ready")
    async def readiness():
        # Tu sprawdź zależności (DB, Redis, NATS)
        return {"status": "ready", "service": service_name}
```

---

# FAZA 1: FUNDAMENT (Tygodnie 1-4)

## Tydzień 1: Infrastruktura i DevOps baseline

### 🎯 Cel tygodnia
Działający dev environment z docker-compose, CI pipeline, observability stack i pierwszym serwisem (health check).

### Dzień 1: Struktura repo + Shared library

**Makefile (developer shortcuts):**
```makefile
.PHONY: up down build test lint

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

test:
	docker compose run --rm test-runner pytest

lint:
	docker compose run --rm test-runner ruff check .

logs:
	docker compose logs -f --tail=100

seed:
	docker compose exec market-data python -m scripts.seed_data

# Per-service builds
build-%:
	docker compose build $*

# K8s helpers
helm-install:
	helm upgrade --install trading-system ./infrastructure/helm -f ./infrastructure/helm/values.yaml

helm-template:
	helm template trading-system ./infrastructure/helm -f ./infrastructure/helm/values.yaml
```

**Shared library — `shared/trading-common/pyproject.toml`:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "trading-common"
version = "0.1.0"
description = "Shared schemas, events, and utilities for trading system"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
]

[tool.hatch.build.targets.wheel]
packages = ["src/trading_common"]
```

**Shared schemas — `shared/trading-common/src/trading_common/schemas.py`:**
```python
"""
Shared Pydantic models — kontrakt między serwisami.
Każdy serwis importuje te modele: pip install -e ../../shared/trading-common
"""
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional
from enum import Enum

class Interval(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"
    W1 = "1wk"

class OHLCVBar(BaseModel):
    symbol: str
    timestamp: datetime
    interval: Interval
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    source: Optional[str] = None

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v, info):
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v

class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradingSignal(BaseModel):
    symbol: str
    strategy: str
    signal: Signal
    confidence: float = Field(ge=0, le=1)
    price: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict = {}

class PortfolioMetrics(BaseModel):
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    daily_pnl: float
    daily_pnl_pct: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
```

### Dzień 1-2: Docker Compose (dev)

**`infrastructure/docker-compose.yml`:**
```yaml
# NIE MA 'version:' — deprecated od Docker Compose v2
name: trading-system

services:
  # ==================== INFRASTRUCTURE ====================
  postgres:
    image: timescale/timescaledb:latest-pg16
    container_name: ts-postgres
    environment:
      POSTGRES_DB: ${DB_NAME:-trading_db}
      POSTGRES_USER: ${DB_USER:-trader}
      POSTGRES_PASSWORD: ${DB_PASSWORD:?DB_PASSWORD is required}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-trader} -d ${DB_NAME:-trading_db}"]
      interval: 5s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: ts-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: >
      redis-server
      --appendonly yes
      --requirepass ${REDIS_PASSWORD:-dev_redis_pass}
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD:-dev_redis_pass}", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  nats:
    image: nats:2-alpine
    container_name: ts-nats
    ports:
      - "4222:4222"   # Client
      - "8222:8222"   # Monitoring
    command: >
      --jetstream
      --store_dir /data
      --http_port 8222
    volumes:
      - nats_data:/data
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8222/healthz"]
      interval: 5s
      timeout: 3s
      retries: 5

  # ==================== OBSERVABILITY ====================
  prometheus:
    image: prom/prometheus:latest
    container_name: ts-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./infrastructure/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"

  grafana:
    image: grafana/grafana:latest
    container_name: ts-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_INSTALL_PLUGINS: grafana-clock-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infrastructure/monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

  loki:
    image: grafana/loki:latest
    container_name: ts-loki
    ports:
      - "3100:3100"
    volumes:
      - loki_data:/loki

  # ==================== APPLICATION SERVICES ====================
  market-data:
    build:
      context: .
      dockerfile: services/market-data/Dockerfile
    container_name: ts-market-data
    environment:
      SERVICE_NAME: market-data
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: ${DB_NAME:-trading_db}
      DB_USER: ${DB_USER:-trader}
      DB_PASSWORD: ${DB_PASSWORD:?DB_PASSWORD is required}
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_PASSWORD: ${REDIS_PASSWORD:-dev_redis_pass}
      NATS_URL: nats://nats:4222
      LOG_LEVEL: ${LOG_LEVEL:-DEBUG}
    ports:
      - "8001:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      nats:
        condition: service_healthy
    volumes:
      - ./services/market-data/src:/app/src    # Hot reload w dev
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  feature-engine:
    build:
      context: .
      dockerfile: services/feature-engine/Dockerfile
    container_name: ts-feature-engine
    environment:
      SERVICE_NAME: feature-engine
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: ${DB_NAME:-trading_db}
      DB_USER: ${DB_USER:-trader}
      DB_PASSWORD: ${DB_PASSWORD:?DB_PASSWORD is required}
      REDIS_HOST: redis
      REDIS_PASSWORD: ${REDIS_PASSWORD:-dev_redis_pass}
      NATS_URL: nats://nats:4222
    ports:
      - "8002:8000"
    depends_on:
      postgres:
        condition: service_healthy
      nats:
        condition: service_healthy

  strategy:
    build:
      context: .
      dockerfile: services/strategy/Dockerfile
    container_name: ts-strategy
    environment:
      SERVICE_NAME: strategy
      NATS_URL: nats://nats:4222
      REDIS_HOST: redis
      REDIS_PASSWORD: ${REDIS_PASSWORD:-dev_redis_pass}
    ports:
      - "8003:8000"
    depends_on:
      nats:
        condition: service_healthy

  backtest:
    build:
      context: .
      dockerfile: services/backtest/Dockerfile
    container_name: ts-backtest
    environment:
      SERVICE_NAME: backtest
      DB_HOST: postgres
      DB_NAME: ${DB_NAME:-trading_db}
      DB_USER: ${DB_USER:-trader}
      DB_PASSWORD: ${DB_PASSWORD:?DB_PASSWORD is required}
      REDIS_HOST: redis
      REDIS_PASSWORD: ${REDIS_PASSWORD:-dev_redis_pass}
      NATS_URL: nats://nats:4222
    ports:
      - "8004:8000"
    depends_on:
      postgres:
        condition: service_healthy

  # Kolejne serwisy (ml-pipeline, risk-mgmt, execution, notification, dashboard)
  # dodawane w odpowiednich tygodniach — ta sama konwencja

  # ==================== API GATEWAY ====================
  traefik:
    image: traefik:v3.0
    container_name: ts-gateway
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entrypoints.web.address=:80"
    ports:
      - "80:80"
      - "8080:8080"    # Traefik dashboard
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro

volumes:
  postgres_data:
  redis_data:
  nats_data:
  prometheus_data:
  grafana_data:
  loki_data:
```

**.env.example:**
```env
# WYMAGANE — docker compose nie wystartuje bez nich
DB_PASSWORD=change_me_in_production_123!
REDIS_PASSWORD=change_me_redis_456!

# Opcjonalne
DB_NAME=trading_db
DB_USER=trader
LOG_LEVEL=DEBUG
GRAFANA_PASSWORD=admin

# API Keys (per-service)
ALPHA_VANTAGE_API_KEY=
ANTHROPIC_API_KEY=

# Trading
INITIAL_CAPITAL=100000.0
PAPER_TRADING=true
```

### Dzień 2: Bazowy Dockerfile (multi-stage)

**`services/market-data/Dockerfile` (wzorzec dla wszystkich serwisów):**
```dockerfile
# ============ BUILD STAGE ============
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps (np. dla TA-Lib — tylko jeśli potrzebne)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C library (tylko dla serwisów które tego potrzebują)
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ && ./configure --prefix=/usr && make -j$(nproc) && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Shared library
COPY shared/trading-common /build/trading-common

# Service dependencies
COPY services/market-data/pyproject.toml /build/service/
RUN pip install --no-cache-dir --prefix=/install \
    /build/trading-common \
    && cd /build/service \
    && pip install --no-cache-dir --prefix=/install .

# ============ RUNTIME STAGE ============
FROM python:3.12-slim AS runtime

# TA-Lib shared lib
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
RUN ldconfig

# Python packages
COPY --from=builder /install /usr/local

WORKDIR /app
COPY services/market-data/src /app/src

# Non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**`services/market-data/pyproject.toml`:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "market-data-service"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "trading-common",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sqlalchemy>=2.0",
    "asyncpg>=0.30",
    "psycopg[binary]>=3.2",
    "redis>=5.0",
    "nats-py>=2.7",
    "yfinance>=0.2.40",
    "aiohttp>=3.10",
    "tenacity>=8.3",
    "structlog>=24.1",
    "prometheus-client>=0.21",
    "prometheus-fastapi-instrumentator>=7.0",
    "pydantic-settings>=2.2",
    "pandas>=2.2",
    "numpy>=1.26",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "httpx>=0.27",      # For TestClient
    "ruff>=0.5",
    "mypy>=1.10",
]
```

### Dzień 3: CI/CD Pipeline

**`.github/workflows/ci.yml`:**
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      services: ${{ steps.changes.outputs.services }}
    steps:
      - uses: actions/checkout@v4
      - id: changes
        uses: dorny/paths-filter@v3
        with:
          filters: |
            market-data:
              - 'services/market-data/**'
              - 'shared/**'
            feature-engine:
              - 'services/feature-engine/**'
              - 'shared/**'
            strategy:
              - 'services/strategy/**'
              - 'shared/**'
            backtest:
              - 'services/backtest/**'
              - 'shared/**'

  test:
    needs: detect-changes
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [market-data, feature-engine, strategy, backtest]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install shared lib
        run: pip install -e shared/trading-common

      - name: Install service deps
        run: |
          cd services/${{ matrix.service }}
          pip install -e ".[dev]"

      - name: Lint
        run: ruff check services/${{ matrix.service }}/

      - name: Type check
        run: mypy services/${{ matrix.service }}/src/ --ignore-missing-imports

      - name: Test
        run: |
          cd services/${{ matrix.service }}
          pytest tests/ -v --cov=src --cov-report=xml

  build-and-push:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [market-data, feature-engine, strategy, backtest]
    steps:
      - uses: actions/checkout@v4

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: services/${{ matrix.service }}/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/${{ matrix.service }}:latest
            ghcr.io/${{ github.repository }}/${{ matrix.service }}:${{ github.sha }}
```

### Dzień 3-4: Konfiguracja centralna per-service

**Wzorzec konfiguracji (każdy serwis):**
```python
# services/market-data/src/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class ServiceSettings(BaseSettings):
    """Konfiguracja market-data-svc. Wartości z env vars."""

    # Identity
    SERVICE_NAME: str = "market-data"
    LOG_LEVEL: str = "INFO"

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "trading_db"
    DB_USER: str = "trader"
    DB_PASSWORD: str  # WYMAGANE — brak domyślnej

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    # NATS
    NATS_URL: str = "nats://localhost:4222"

    # Service-specific
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    DEFAULT_FETCH_INTERVAL: str = "1d"
    MAX_CONCURRENT_FETCHES: int = 5
    CACHE_TTL_SECONDS: int = 3600

    model_config = {"env_file": ".env", "case_sensitive": True}

settings = ServiceSettings()
```

### Dzień 4-5: Pierwszy serwis — market-data (skeleton)

**`services/market-data/src/main.py`:**
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import structlog
from src.config import settings
from src.core.observability import setup_observability
from src.api import router as api_router

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting market-data service", log_level=settings.LOG_LEVEL)
    # TODO: init DB pool, NATS connection, Redis
    yield
    logger.info("Shutting down market-data service")
    # TODO: cleanup

app = FastAPI(
    title="Market Data Service",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
```

**`services/market-data/src/api/__init__.py`:**
```python
from fastapi import APIRouter
from .routes import router as ohlcv_router

router = APIRouter()
router.include_router(ohlcv_router, prefix="/market-data", tags=["market-data"])
```

**`services/market-data/src/api/routes.py`:**
```python
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date
from trading_common.schemas import OHLCVBar, Interval
import structlog

logger = structlog.get_logger()
router = APIRouter()

@router.get("/ohlcv/{symbol}", response_model=list[OHLCVBar])
async def get_ohlcv(
    symbol: str,
    interval: Interval = Interval.D1,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(default=100, le=5000),
):
    """Pobierz dane OHLCV dla symbolu."""
    logger.info("Fetching OHLCV", symbol=symbol, interval=interval)
    # TODO: implementacja w tygodniu 2
    raise HTTPException(501, "Not implemented yet")

@router.post("/fetch/{symbol}")
async def trigger_fetch(symbol: str, interval: Interval = Interval.D1):
    """Trigger pobrania nowych danych (async — publish event po zakończeniu)."""
    logger.info("Triggering fetch", symbol=symbol, interval=interval)
    # TODO: implementacja w tygodniu 2
    return {"status": "accepted", "symbol": symbol}

@router.get("/symbols")
async def list_symbols():
    """Lista dostępnych symboli."""
    return {"symbols": ["AAPL", "MSFT", "GOOGL", "SPY"]}  # placeholder
```

### Dzień 5: Kubernetes Helm Chart (scaffold)

**`infrastructure/helm/Chart.yaml`:**
```yaml
apiVersion: v2
name: trading-system
description: Microservices trading system
version: 0.1.0
appVersion: "0.1.0"
```

**`infrastructure/helm/values.yaml`:**
```yaml
global:
  imagePullPolicy: IfNotPresent
  env: development

# Per-service config
marketData:
  replicaCount: 1
  image:
    repository: ghcr.io/your-org/trading-system/market-data
    tag: latest
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "500m"
  env:
    LOG_LEVEL: INFO

featureEngine:
  replicaCount: 1
  image:
    repository: ghcr.io/your-org/trading-system/feature-engine
    tag: latest

# Infrastructure
postgres:
  enabled: true
  storage: 10Gi

redis:
  enabled: true

nats:
  enabled: true

prometheus:
  enabled: true

grafana:
  enabled: true
```

**Przykładowy template — `infrastructure/helm/templates/market-data-deployment.yaml`:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: market-data
  labels:
    app: market-data
    component: service
spec:
  replicas: {{ .Values.marketData.replicaCount }}
  selector:
    matchLabels:
      app: market-data
  template:
    metadata:
      labels:
        app: market-data
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: market-data
          image: "{{ .Values.marketData.image.repository }}:{{ .Values.marketData.image.tag }}"
          imagePullPolicy: {{ .Values.global.imagePullPolicy }}
          ports:
            - containerPort: 8000
          env:
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: trading-secrets
                  key: db-password
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: trading-secrets
                  key: redis-password
            - name: NATS_URL
              value: "nats://nats:4222"
            - name: DB_HOST
              value: "postgres"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            {{- toYaml .Values.marketData.resources | nindent 12 }}
---
apiVersion: v1
kind: Service
metadata:
  name: market-data
spec:
  selector:
    app: market-data
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

### ✅ Checklist Tydzień 1

- [ ] Repo zainicjalizowane z pełną strukturą mikroserwisów
- [ ] `shared/trading-common` zainstalowane i importowalne
- [ ] `docker compose up` startuje: postgres, redis, nats, prometheus, grafana, traefik
- [ ] Serwis `market-data` odpowiada na `/health` i `/metrics`
- [ ] Prometheus scrape'uje metryki z serwisu
- [ ] Grafana dostępna na `localhost:3000`
- [ ] CI pipeline (lint + test) przechodzi na GitHub Actions
- [ ] Helm chart renderuje się poprawnie (`helm template`)
- [ ] `.env` z sekretami — NIE commitowany do repo
- [ ] `Makefile` z komendami `up`, `down`, `build`, `test`

---

## Tydzień 2: Serwis Market Data

### 🎯 Cel tygodnia
Pełna implementacja `market-data-svc`: pobieranie danych (Yahoo, Alpha Vantage), walidacja, zapis do TimescaleDB, cache w Redis, publikacja eventów do NATS.

### Architektura serwisu

```
market-data-svc
├── src/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── routes.py          # GET /ohlcv/{symbol}, POST /fetch/{symbol}
│   │   └── deps.py            # FastAPI dependencies (DB session, etc.)
│   ├── core/
│   │   ├── fetchers/
│   │   │   ├── base.py        # ABC z retry, rate-limit, validation
│   │   │   ├── yahoo.py       # yfinance wrapper
│   │   │   └── alpha_vantage.py
│   │   ├── storage.py         # SQLAlchemy async + TimescaleDB
│   │   ├── cache.py           # Redis cache layer
│   │   └── observability.py   # Shared pattern
│   ├── events/
│   │   ├── publisher.py       # NATS publish: MarketDataUpdatedEvent
│   │   └── subscriber.py      # (opcjonalnie) reaguj na żądania fetch
│   └── models/
│       ├── db.py              # SQLAlchemy ORM models
│       └── schemas.py         # Re-export z trading_common + local
└── tests/
    ├── test_fetchers.py
    ├── test_storage.py
    └── test_api.py
```

### Kluczowe wzorce implementacyjne

**Base Fetcher (async, z retry + rate limit):**
```python
# services/market-data/src/core/fetchers/base.py
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import asyncio
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()

class BaseFetcher(ABC):
    """
    Abstract fetcher z:
    - async/await first
    - tenacity retry z exponential backoff
    - rate limiting (token bucket)
    - data validation (delegowana do Pydantic schemas)
    """

    def __init__(self, rate_limit_per_sec: float = 5.0, timeout: int = 30):
        self._rate_limit = rate_limit_per_sec
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(int(rate_limit_per_sec))
        self._logger = logger.bind(fetcher=self.__class__.__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def fetch(self, symbol: str, **kwargs) -> pd.DataFrame:
        async with self._semaphore:
            self._logger.info("Fetching", symbol=symbol, params=kwargs)
            data = await self._fetch_impl(symbol, **kwargs)
            validated = self._validate(data, symbol)
            self._logger.info("Fetched OK", symbol=symbol, rows=len(validated))
            return validated

    @abstractmethod
    async def _fetch_impl(self, symbol: str, **kwargs) -> pd.DataFrame:
        ...

    def _validate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df.empty:
            raise ValueError(f"Empty data for {symbol}")

        df.columns = df.columns.str.lower()
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Fix NaN — pandas 2.x way (bez deprecated method= argument)
        df = df.ffill().bfill()

        # OHLC sanity
        mask = (df["high"] < df["low"])
        if mask.any():
            self._logger.warning("Invalid OHLC rows", count=int(mask.sum()))
            df.loc[mask, ["high", "low"]] = df.loc[mask, ["close", "close"]].values

        # Deduplicate
        df = df[~df.index.duplicated(keep="last")]

        return df[list(required)]
```

**NATS Event Publisher:**
```python
# services/market-data/src/events/publisher.py
import nats
import json
import structlog
from trading_common.events import MarketDataUpdatedEvent

logger = structlog.get_logger()

class EventPublisher:
    def __init__(self, nats_url: str):
        self._nats_url = nats_url
        self._nc = None

    async def connect(self):
        self._nc = await nats.connect(self._nats_url)
        logger.info("Connected to NATS", url=self._nats_url)

    async def publish_market_data_updated(self, symbol: str, interval: str, rows: int):
        event = MarketDataUpdatedEvent(
            source_service="market-data",
            symbol=symbol,
            interval=interval,
            rows_count=rows,
        )
        await self._nc.publish(
            f"market_data.updated.{symbol}",
            event.model_dump_json().encode(),
        )
        logger.info("Published event", event_type=event.event_type, symbol=symbol)

    async def close(self):
        if self._nc:
            await self._nc.drain()
```

### ✅ Checklist Tydzień 2

- [ ] YahooFetcher pobiera dane i zwraca walidowany DataFrame
- [ ] AlphaVantageFetcher działa (wymaga API key)
- [ ] Dane zapisywane do TimescaleDB (hypertable)
- [ ] Redis cache działa (TTL = 1h dla daily data)
- [ ] Event `MarketDataUpdatedEvent` publikowany do NATS po każdym fetch
- [ ] API endpoints: GET `/api/v1/market-data/ohlcv/{symbol}`, POST `/api/v1/market-data/fetch/{symbol}`
- [ ] Testy jednostkowe: fetcher validation, storage CRUD
- [ ] Metryki Prometheus: `fetch_duration_seconds`, `fetch_errors_total`

---

## Tydzień 3: Serwis Feature Engineering

### 🎯 Cel tygodnia
`feature-engine-svc` — oblicza wskaźniki techniczne na żądanie (HTTP) lub reaktywnie (subskrypcja NATS na `market_data.updated.*`).

### Kluczowe endpointy

```
GET  /api/v1/features/{symbol}/indicators?indicators=sma_20,rsi_14,bb_20
GET  /api/v1/features/{symbol}/all                # Pełny zestaw 30+ wskaźników
POST /api/v1/features/compute                     # Batch computation
```

### Event-driven flow

```
NATS subscribe: market_data.updated.*
→ Oblicz wskaźniki dla zaktualizowanego symbolu
→ Cache wyniki w Redis
→ PUBLISH: features.computed.{symbol}
```

### Wskaźniki do implementacji (Faza 1)

Trend: SMA (10, 20, 50, 200), EMA (12, 26), MACD, ADX, Aroon. Momentum: RSI, Stochastic, CCI, Williams %R, ROC. Volatility: Bollinger Bands, ATR, Keltner Channel. Volume: OBV, VWAP, A/D Line, MFI. Pattern: rozpoznawanie formacji świecowych (via TA-Lib).

### ✅ Checklist Tydzień 3

- [ ] 30+ wskaźników technicznych obliczanych poprawnie
- [ ] Subskrypcja NATS → automatyczne przeliczanie po nowych danych
- [ ] Cache wskaźników w Redis (z invalidacją)
- [ ] API endpoint zwraca wskaźniki w formacie JSON
- [ ] Testy porównujące wyniki z reference values

---

## Tydzień 4: Pierwsza strategia + Backtest Runner

### 🎯 Cel tygodnia
`strategy-svc` z prostą strategią SMA Crossover + `backtest-svc` z podstawowym silnikiem wektorowym.

### Strategy Service — kontrakt

```python
# Interfejs strategii (w shared lib)
class StrategyResult(BaseModel):
    signals: list[TradingSignal]
    metadata: dict = {}

# API
POST /api/v1/strategy/evaluate
Body: { "strategy_name": "sma_cross", "symbol": "AAPL", "params": {"fast": 10, "slow": 50} }
Response: StrategyResult
```

### Backtest Service — kontrakt

```python
# API
POST /api/v1/backtest/run
Body: {
    "strategy_name": "sma_cross",
    "symbols": ["AAPL"],
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000,
    "commission": 0.001,
    "params": {"fast": 10, "slow": 50}
}
Response: {
    "total_return": 0.45,
    "sharpe_ratio": 1.23,
    "max_drawdown": -0.12,
    "trades_count": 48,
    "win_rate": 0.56,
    "equity_curve": [...],
    "trades": [...]
}
```

### ✅ Checklist Tydzień 4

- [ ] SMA Crossover strategy generuje sygnały BUY/SELL
- [ ] Backtest engine oblicza: total return, Sharpe, max DD, win rate
- [ ] Equity curve generowana i zwracana przez API
- [ ] Strategia porównana z buy-and-hold benchmark
- [ ] Event flow: market_data → features → strategy → sygnały

---

# FAZA 2: STRATEGIE I BACKTESTING (Tygodnie 5-12)

## Tydzień 5-6: Strategy Service Framework

### Cel
Rozbudowa `strategy-svc` o pluggable strategy pattern: każda strategia to osobny moduł rejestrowany w registry.

### Strategie do implementacji

1. **SMA/EMA Crossover** (już z tygodnia 4)
2. **RSI + Bollinger Bands** — mean reversion
3. **MACD Divergence** — momentum
4. **Breakout (Donchian Channel)** — trend following
5. **Pair Trading** — statistical arbitrage

### Pattern: Strategy Registry

```python
# services/strategy/src/core/registry.py
from typing import Protocol, Type

class StrategyProtocol(Protocol):
    name: str
    def generate_signals(self, data: pd.DataFrame, params: dict) -> list[TradingSignal]: ...

_registry: dict[str, Type[StrategyProtocol]] = {}

def register(cls: Type[StrategyProtocol]):
    _registry[cls.name] = cls
    return cls

def get_strategy(name: str) -> StrategyProtocol:
    if name not in _registry:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_registry.keys())}")
    return _registry[name]()
```

## Tydzień 7-8: Backtesting Engine Service

### Cel
Rozbudowa `backtest-svc`: wektorowy engine (szybki) + event-driven (dokładny), metryki, wizualizacje, commission/slippage modeling.

### Metryki backtestingu

Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown (depth + duration), Win Rate, Profit Factor, Average Win/Loss, Expectancy, Recovery Factor, VaR 95%, CVaR 95%.

### Optymalizacja wydajności

Backtesty to compute-intensive → osobne repliki, Celery/async workers. Długie joby → status polling:

```
POST /api/v1/backtest/run   → 202 Accepted, { "job_id": "abc123" }
GET  /api/v1/backtest/status/abc123  → { "status": "running", "progress": 45 }
GET  /api/v1/backtest/result/abc123  → pełne wyniki
```

## Tydzień 9-10: Zaawansowane strategie

### Cel
Mean reversion, multi-timeframe analysis, pair trading.

### Multi-timeframe w mikroserwisach

`feature-engine-svc` oblicza wskaźniki na wielu interwałach (1h, 4h, 1d). `strategy-svc` konsumuje features z wielu timeframe'ów — logika agregacji po stronie serwisu strategii.

## Tydzień 11-12: Optymalizacja i Walk-Forward

### Cel
Grid search, Bayesian optimization (Optuna), walk-forward analysis — wszystko w `backtest-svc`.

### Walk-Forward jako long-running job

```
POST /api/v1/backtest/walk-forward
Body: {
    "strategy_name": "rsi_bb",
    "symbol": "AAPL",
    "in_sample_days": 252,
    "out_of_sample_days": 63,
    "param_grid": { "rsi_period": [10, 14, 20], "bb_period": [15, 20, 25] },
    "optimization_metric": "sharpe_ratio"
}
→ 202 Accepted, job_id
```

### ✅ Checklist Faza 2

- [ ] 5+ strategii zarejestrowanych w registry
- [ ] Backtesting engine: wektorowy + event-driven mode
- [ ] 15+ metryk backtestingu
- [ ] Optymalizacja parametrów (grid search + Optuna)
- [ ] Walk-forward analysis z out-of-sample walidacją
- [ ] Long-running jobs z progress tracking
- [ ] Commission + slippage realistycznie modelowane

---

# FAZA 3: MACHINE LEARNING (Tygodnie 13-18)

## Tydzień 13-14: ML Feature Pipeline Service

### Cel
`ml-pipeline-svc` — feature engineering pipeline (100+ features), feature selection, transformacje.

### Architektura

```
ml-pipeline-svc
├── src/
│   ├── core/
│   │   ├── feature_builder.py    # Buduje feature matrix z OHLCV + indicators
│   │   ├── feature_selector.py   # Mutual information, feature importance
│   │   ├── transformers.py       # Scaling, encoding, lag features
│   │   └── splitter.py           # Time-series aware train/val/test split
│   ├── models/
│   │   ├── registry.py           # MLflow model registry
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lstm.py
│   │   └── ensemble.py
│   └── api/
│       ├── train_routes.py       # POST /train
│       └── predict_routes.py     # POST /predict
```

### Ważne: Time-series split, NIE random split

```python
# Purged K-Fold Cross-Validation (Lopez de Prado)
# NIGDY sklearn.model_selection.train_test_split na danych czasowych!
```

## Tydzień 15-16: ML Model Training Service

### Modele

1. **Random Forest** — baseline, feature importance
2. **XGBoost/LightGBM** — gradient boosting, najlepsza wydajność na tabelarycznych danych
3. **LSTM** — deep learning na sekwencjach (PyTorch, nie TensorFlow — lżejszy, łatwiejszy w K8s)

### MLflow Integration

```
POST /api/v1/ml/train
Body: {
    "model_type": "xgboost",
    "symbol": "AAPL",
    "features": ["sma_20", "rsi_14", "volume_ratio", ...],
    "target": "return_5d_direction",
    "params": { "n_estimators": 500, "max_depth": 6 }
}
→ 202 Accepted, { "run_id": "mlflow-run-123" }
```

### GPU w Kubernetes

Jeśli training LSTM wymaga GPU → oddzielny node pool z GPU + tolerations/affinity w Helm values.

## Tydzień 17-18: Sentiment & Ensemble Service

### FinBERT Sentiment

Osobny model serwowany jako inference endpoint (FastAPI + model załadowany do pamięci). W K8s: osobny deployment z większym limitem RAM.

### Ensemble

Stacking / voting — łączy sygnały z wielu modeli. Wagi ensemble'a optymalizowane walk-forward.

### ✅ Checklist Faza 3

- [ ] 100+ features obliczanych przez pipeline
- [ ] Random Forest, XGBoost, LSTM wytrenowane
- [ ] MLflow tracking experiments
- [ ] Feature importance raport
- [ ] FinBERT sentiment scores
- [ ] Ensemble model z lepszym Sharpe niż pojedyncze modele
- [ ] Purged K-Fold Cross-Validation (nie random split!)
- [ ] Modele serwowane jako API endpoints

---

# FAZA 4: RISK & PORTFOLIO (Tygodnie 19-21)

## Tydzień 19-20: Risk Management Service

### Cel
`risk-mgmt-svc` — position sizing, risk limits, portfolio optimization.

### Kluczowe funkcje

1. **Position Sizing**: Fixed fractional, Kelly Criterion, Volatility-adjusted, Risk parity
2. **Risk Limits**: max position size, max portfolio risk, max drawdown limit, max correlation, sector limits
3. **Portfolio Optimization**: Mean-Variance (Markowitz), Minimum Variance, Maximum Sharpe, Risk Parity, Black-Litterman, HRP (Hierarchical Risk Parity)
4. **Risk Metrics**: VaR (parametric, historical, Monte Carlo), CVaR, Beta, Tracking Error

### Event-driven risk gate

```
strategy-svc → PUBLISH SignalGeneratedEvent
risk-mgmt-svc → SUBSCRIBE → walidacja → PUBLISH OrderApprovedEvent / OrderRejectedEvent
execution-svc → SUBSCRIBE OrderApprovedEvent → execute
```

To zapewnia, że żaden order nie przejdzie bez risk checku.

## Tydzień 21: Dashboard Service (Frontend)

### Opcja A: Streamlit (szybciej)

Osobny kontener, odpytuje inne serwisy przez internal API.

### Opcja B: React/Next.js + FastAPI BFF (lepsze UX)

Frontend w React, Backend-for-Frontend aggreguje dane z mikroserwisów.

### Sekcje dashboardu

1. Portfolio Overview (equity curve, P&L, positions)
2. Risk Metrics (VaR, drawdown, correlation matrix)
3. Strategy Performance (per-strategy attribution)
4. Backtest Results (interactive charts)
5. ML Model Performance (accuracy, feature importance)
6. System Health (uptime, latency, error rates — z Grafana embed)

---

# FAZA 5: PRODUKCJA I LIVE TRADING (Tygodnie 22-24)

## Tydzień 22: Execution Service + Paper Trading

### Cel
`execution-svc` — paper trading na live data, z realistycznym modelowaniem fills.

### Broker Abstraction

```python
class BrokerProtocol(Protocol):
    async def submit_order(self, order: Order) -> Fill: ...
    async def get_positions(self) -> list[Position]: ...
    async def get_account(self) -> AccountInfo: ...

class PaperBroker(BrokerProtocol):
    """Symulacja — slippage, partial fills, market hours."""

class AlpacaBroker(BrokerProtocol):
    """Alpaca API — paper + live."""

class IBKRBroker(BrokerProtocol):
    """Interactive Brokers TWS API."""
```

## Tydzień 23: Alerting & Notification Service

### Cel
`notification-svc` — multi-channel (Telegram, email, Slack), deduplikacja, rate limiting alertów.

### Subskrypcje NATS

```
risk.limit_breached   → CRITICAL alert
order.filled          → INFO alert
ml.model_trained      → INFO alert
system.error          → ERROR alert
```

## Tydzień 24: Production Kubernetes Deployment

### Cel
Deploy na K8s cluster (managed: GKE/EKS/AKS lub self-hosted: k3s/microk8s).

### Production checklist

- [ ] Helm chart z `values-prod.yaml` (wyższe repliki, limity, HPA)
- [ ] Secrets w K8s Secrets lub External Secrets Operator (Vault/AWS SM)
- [ ] Ingress z TLS (cert-manager + Let's Encrypt)
- [ ] HPA (Horizontal Pod Autoscaler) dla compute-intensive serwisów (backtest, ml)
- [ ] PVC dla PostgreSQL (StatefulSet)
- [ ] NetworkPolicies (serwisy rozmawiają tylko z NATS, DB, Redis)
- [ ] Prometheus + Grafana + AlertManager w osobnym namespace `monitoring`
- [ ] Loki + Promtail dla logów
- [ ] Backup CronJob dla PostgreSQL (pg_dump → S3/GCS)
- [ ] CI/CD: GitHub Actions → build images → helm upgrade --install
- [ ] Rollback: `helm rollback trading-system <revision>`
- [ ] Resource quotas per namespace
- [ ] Pod Disruption Budgets dla stateful serwisów

### Deployment pipeline (final)

```yaml
# .github/workflows/deploy.yml
name: Deploy to K8s

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v4
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Helm upgrade
        run: |
          helm upgrade --install trading-system \
            ./infrastructure/helm \
            -f ./infrastructure/helm/values-prod.yaml \
            --set global.imageTag=${{ github.sha }} \
            --namespace trading \
            --create-namespace \
            --wait --timeout 5m
```

---

# DODATKI

## Porównanie oryginału vs nowa wersja

| Aspekt | Oryginał | Nowa wersja |
|--------|----------|-------------|
| Architektura | Monolit (1 pakiet Python) | 9+ mikroserwisów |
| Docker | Opcjonalny (tydzień 1, dzień 3) | Obowiązkowy od dnia 1 |
| Kubernetes | "Next Steps" po 24 tygodniach | Helm chart od tygodnia 1 |
| Message broker | Brak | NATS JetStream od dnia 1 |
| Event-driven | Brak | Core architecture pattern |
| Observability | Prometheus w tygodniu 24 | Prometheus + Grafana + Loki od dnia 1 |
| CI/CD | Tydzień 24 | GitHub Actions od tygodnia 1 |
| API Gateway | Brak | Traefik od dnia 1 |
| Secrets | Hardcoded w docker-compose | Env vars + K8s Secrets |
| `setup.py` | Tak (deprecated) | `pyproject.toml` |
| Python | 3.11 | 3.12 |
| Shared code | Brak mechanizmu | `trading-common` pip package |
| Testy | Pytest wspomniane | Per-service tests + CI |
| Dependency management | 1 `requirements.txt` na cały projekt | Per-service `pyproject.toml` |
| Skalowanie | Brak | HPA, repliki, worker pools |

## Jak efektywnie korzystać z Claude

### Prompty architektoniczne (nowe — microservices-specific)

```
Zaprojektuj event schema (Pydantic) dla komunikacji między
market-data-svc a feature-engine-svc. Użyj NATS jako transport.
Uwzględnij: correlation_id, versioning, error handling.
```

```
Review mój Dockerfile dla serwisu backtest.
Optymalizuj: multi-stage build, layer caching, security (non-root).
Sprawdź czy obraz jest < 500MB.
```

```
Napisz Helm template dla StatefulSet PostgreSQL z:
- PVC 20Gi
- Backup CronJob (pg_dump co noc)
- Resource limits
- Liveness/readiness probes
```

```
Zaimplementuj circuit breaker pattern w feature-engine-svc
gdy market-data-svc jest niedostępny. Użyj tenacity + Redis cache jako fallback.
```

### Prompty implementacyjne (standardowe)

```
Implement [functionality] w serwisie [service-name].

Requirements:
- FastAPI endpoint
- Pydantic schemas (in/out)
- Async database access (SQLAlchemy + asyncpg)
- NATS event publish po zakończeniu
- Prometheus metrics
- Structured logging (structlog)
- Unit test z pytest-asyncio
```

## Checklisty tygodniowe (podsumowanie)

### Faza 0-1 (Tydzień 0-4): Infrastruktura + Market Data + Features + First Strategy
- [ ] Docker compose z pełnym stackiem infra
- [ ] CI/CD pipeline działa
- [ ] Helm chart renderuje się
- [ ] market-data-svc: fetch, store, cache, publish events
- [ ] feature-engine-svc: 30+ wskaźników, event-driven
- [ ] strategy-svc: SMA Crossover + registry pattern
- [ ] backtest-svc: podstawowy silnik z metrykami
- [ ] Prometheus + Grafana dashboard ze wszystkimi serwisami

### Faza 2 (Tydzień 5-12): Strategie + Backtesting
- [ ] 5+ strategii w registry
- [ ] Walk-forward analysis
- [ ] Optymalizacja parametrów (Optuna)
- [ ] Long-running backtest jobs z progress API

### Faza 3 (Tydzień 13-18): ML
- [ ] Feature pipeline (100+ features)
- [ ] RF, XGBoost, LSTM wytrenowane
- [ ] MLflow tracking
- [ ] Ensemble model
- [ ] FinBERT sentiment

### Faza 4 (Tydzień 19-21): Risk + Dashboard
- [ ] Position sizing (Kelly, vol-adjusted)
- [ ] Portfolio optimization (Markowitz, HRP)
- [ ] Risk gate (event-driven pre-trade check)
- [ ] Dashboard z real-time metrics

### Faza 5 (Tydzień 22-24): Produkcja
- [ ] Paper trading na live data
- [ ] Multi-channel notifications
- [ ] K8s deployment z Helm
- [ ] TLS, secrets, HPA, backup

## Rozwiązywanie problemów (microservices-specific)

### Problem: Serwis nie widzi NATS
```bash
# Sprawdź czy NATS działa
docker compose logs nats
# Sprawdź connectivity z kontenera
docker compose exec market-data nats-cli pub test "hello"
```

### Problem: Eventy nie docierają
```bash
# Monitor NATS subjects
docker compose exec nats nats-cli sub ">"
# → pokaże wszystkie eventy w systemie
```

### Problem: Slow backtest jobs
- Użyj wektorowego engine zamiast event-driven
- Skaluj repliki: `docker compose up --scale backtest=3`
- W K8s: HPA na CPU usage
- Cache features w Redis (nie przeliczaj za każdym razem)

### Problem: Database connection exhaustion
- Każdy serwis: `pool_size=5, max_overflow=10` (nie 20+)
- Użyj pgBouncer przed PostgreSQL w produkcji
- Health check: `SELECT 1` z timeout 2s

## Zasoby i dalszy rozwój

### Polecane materiały (microservices-focused)

**Książki:**
- "Building Microservices" — Sam Newman
- "Designing Data-Intensive Applications" — Martin Kleppmann
- "Kubernetes in Action" — Marko Lukša
- "Advances in Financial Machine Learning" — Marcos Lopez de Prado

**Narzędzia:**
- NATS: https://nats.io — lekki message broker, JetStream dla persistence
- Helm: https://helm.sh — package manager dla K8s
- Traefik: https://traefik.io — API gateway / reverse proxy
- MLflow: https://mlflow.org — ML experiment tracking
- Optuna: https://optuna.org — Bayesian hyperparameter optimization

### Next Steps (po 24 tygodniach)

1. **Service mesh** (Istio/Linkerd) — mTLS, traffic management
2. **Kafka** zamiast NATS — jeśli potrzeba heavy stream processing
3. **Feature store** (Feast) — centralne zarządzanie ML features
4. **A/B testing framework** — porównywanie strategii na live data
5. **Reinforcement Learning** — RL agents jako strategy plugins
6. **Multi-cloud deployment** — Terraform + ArgoCD
7. **Chaos Engineering** — Litmus/Chaos Monkey dla resiliency testing

---

## Podsumowanie

Ten plan to **microservices-first** podejście do budowy systemu tradingowego. Kluczowe różnice względem monolitu:

- Każdy serwis można **deployować niezależnie** — zmiana w strategii nie wymaga redeployu ML pipeline
- **Skalowanie granularne** — 10 replik backtest engine, 1 replika notification service
- **Resilience** — awaria jednego serwisu nie kładzie całego systemu (circuit breaker, retry, fallback)
- **Technologiczna wolność** — serwis ML może używać PyTorch, dashboard React, backtest czysty numpy — nie ma jednego requirements.txt
- **Team-ready** — każdy serwis może rozwijać inna osoba/zespół

Podejście wymaga **więcej pracy infrastrukturalnej na starcie** (Docker, NATS, Helm), ale zwraca się szybko przy rosnącej złożoności systemu.

---

*Dokument zaktualizowany | Architektura mikroserwisowa | Marzec 2026*
