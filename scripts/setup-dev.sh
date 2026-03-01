#!/usr/bin/env bash
# setup-dev.sh — konfiguracja środowiska deweloperskiego
set -euo pipefail

echo "==> Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }

echo "==> Setting up .env..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    .env created from .env.example — fill in the required values!"
else
  echo "    .env already exists — skipping"
fi

echo "==> Installing shared library..."
pip install --user -e "shared/trading-common[dev]"

echo "==> Installing service dependencies (dev mode)..."
SERVICES="market-data feature-engine strategy backtest ml-pipeline risk-mgmt execution notification dashboard"
for svc in $SERVICES; do
  echo "    -> $svc"
  pip install --user -e "services/$svc[dev]"
done

echo "==> Done! Run 'make up' to start the stack."
