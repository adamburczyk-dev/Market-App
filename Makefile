.PHONY: up down build test lint logs seed help \
        helm-install helm-template helm-diff

# .env leży w root projektu; compose file w infrastructure/ —
# przekazujemy --env-file jawnie, bo Compose v2 szuka .env
# relatywnie do compose file, nie do CWD.
COMPOSE = docker compose -f infrastructure/docker-compose.yml --env-file .env

# ============================================================
# Developer shortcuts
# ============================================================

up:          ## Uruchom wszystkie serwisy (dev)
	$(COMPOSE) up -d

down:        ## Zatrzymaj wszystkie serwisy
	$(COMPOSE) down

build:       ## Zbuduj obrazy wszystkich serwisów
	$(COMPOSE) build

build-%:     ## Zbuduj obraz konkretnego serwisu: make build-market-data
	$(COMPOSE) build $*

test:        ## Uruchom testy wszystkich serwisów
	bash scripts/run-all-tests.sh

test-%:      ## Uruchom testy konkretnego serwisu: make test-market-data
	cd services/$* && python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:        ## Lintuj cały kod przez pre-commit (własne venv, brak problemów z PATH)
	python -m pre_commit run ruff --all-files
	python -m pre_commit run ruff-format --all-files

lint-%:      ## Lintuj konkretny serwis: make lint-market-data
	python -m pre_commit run ruff --files $$(find services/$* -name "*.py")

type-%:      ## Sprawdź typy w serwisie: make type-market-data
	python -m mypy services/$*/src/ --ignore-missing-imports

logs:        ## Śledź logi (ostatnie 100 linii)
	$(COMPOSE) logs -f --tail=100

logs-%:      ## Śledź logi konkretnego serwisu: make logs-market-data
	$(COMPOSE) logs -f --tail=100 $*

seed:        ## Załaduj dane testowe
	bash scripts/seed-data.sh

setup:       ## Skonfiguruj środowisko deweloperskie
	bash scripts/setup-dev.sh

# ============================================================
# Kubernetes / Helm
# ============================================================

helm-install:   ## Deploy do K8s przez Helm
	helm upgrade --install trading-system ./infrastructure/helm \
	  -f ./infrastructure/helm/values.yaml

helm-template:  ## Renderuj Helm templates (dry-run)
	helm template trading-system ./infrastructure/helm \
	  -f ./infrastructure/helm/values.yaml

helm-diff:      ## Pokaż diff przed deployem (wymaga helm-diff plugin)
	helm diff upgrade trading-system ./infrastructure/helm \
	  -f ./infrastructure/helm/values.yaml

help:           ## Pokaż tę pomoc
	@grep -E '^[a-zA-Z_%/-]+:.*?##.*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
