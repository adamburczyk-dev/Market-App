#!/usr/bin/env bash
# run-all-tests.sh — uruchamia testy dla shared library i wszystkich serwisów
set -euo pipefail

FAILED=()
PASSED=()

run_tests() {
  local name=$1
  local path=$2

  echo ""
  echo "======================================"
  echo "  Testing: $name"
  echo "======================================"

  if (cd "$path" && python -m pytest tests/ -v --tb=short 2>&1); then
    PASSED+=("$name")
  else
    FAILED+=("$name")
  fi
}

# Shared library
run_tests "trading-common" "shared/trading-common"

# Serwisy
SERVICES="market-data feature-engine strategy backtest ml-pipeline risk-mgmt execution notification dashboard"
for svc in $SERVICES; do
  run_tests "$svc" "services/$svc"
done

# Raport końcowy
echo ""
echo "======================================"
echo "  SUMMARY"
echo "======================================"
echo "PASSED (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "FAILED (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "======================================"

if [ ${#FAILED[@]} -gt 0 ]; then
  exit 1
fi
