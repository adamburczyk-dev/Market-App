#!/usr/bin/env bash
# seed-data.sh — ładuje przykładowe dane do bazy przez market-data service
set -euo pipefail

MARKET_DATA_URL="${MARKET_DATA_URL:-http://localhost:8001}"
SYMBOLS="${SYMBOLS:-AAPL MSFT GOOGL SPY}"

echo "==> Seeding market data from $MARKET_DATA_URL..."
for symbol in $SYMBOLS; do
  echo "    -> Triggering fetch for $symbol (1d)"
  curl -sf -X POST "$MARKET_DATA_URL/api/v1/market-data/fetch/$symbol?interval=1d" \
    -H "Content-Type: application/json" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'      status: {d[\"status\"]}')"
done

echo "==> Done!"
