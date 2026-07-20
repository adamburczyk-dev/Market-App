#!/usr/bin/env python3
"""Real-data bootstrap: backfill the trading universe and (optionally) train.

Drives the RUNNING services over HTTP only — start the stack first
(`make up`). No direct DB access, no yfinance import here: market-data owns
fetching/validation/storage/eventing; this script just orchestrates it.

Usage:
  python scripts/bootstrap-universe.py
      Backfill ~6 years of daily OHLCV for the default universe into
      market-data (each fetch also publishes market_data.updated, so a running
      feature-engine builds ranked vectors along the way).

  python scripts/bootstrap-universe.py --train
      ...then run one full training pass on the backfilled history and print
      the activation-gate report. Promotion stays a manual sign-off — the
      script prints the exact command when a version is produced.

  python scripts/bootstrap-universe.py --symbols AAPL,MSFT --years 5
  python scripts/bootstrap-universe.py --symbols @my-universe.txt

Environment (flags win): MARKET_DATA_URL (default http://localhost:8001),
ML_PIPELINE_URL (default http://localhost:8005).

Exit code 0 when every requested symbol backfilled (and training, if
requested, completed — a FAILED gate is still a completed, honest result);
1 when any symbol fails or a requested step errors out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta

# ~34 liquid US large caps across GICS sectors (cross-sectional learning needs
# sector breadth, not just tech). Equities only — no ETFs in the model universe.
DEFAULT_UNIVERSE = [
    # Information Technology
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "ORCL",
    "CRM",
    # Communication Services
    "GOOGL",
    "META",
    "NFLX",
    # Consumer Discretionary
    "AMZN",
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    # Financials
    "JPM",
    "BAC",
    "GS",
    # Health Care
    "UNH",
    "JNJ",
    "LLY",
    "PFE",
    # Consumer Staples
    "PG",
    "KO",
    "PEP",
    "WMT",
    "COST",
    # Energy
    "XOM",
    "CVX",
    # Industrials
    "CAT",
    "HON",
    "UPS",
    # Materials / Utilities / Real Estate
    "LIN",
    "NEE",
    "PLD",
]

MIN_SESSIONS_FOR_TRAINING = 945  # holdout 126 + train 756 + test 63 (TrainingParams)


def _request(
    method: str, url: str, payload: dict | None = None, timeout: float = 120.0
) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read() or b"{}")
        except json.JSONDecodeError:
            body = {}
        return exc.code, body


def _check_service(base_url: str, name: str) -> None:
    try:
        status, _ = _request("GET", f"{base_url}/health", timeout=10)
    except OSError as exc:
        sys.exit(f"{name} unreachable at {base_url} ({exc}) — run `make up` first.")
    if status != 200:
        sys.exit(
            f"{name} unhealthy at {base_url} (HTTP {status}) — run `make up` first."
        )


def backfill(
    market_url: str, symbols: list[str], start: date, end: date, pause_s: float
) -> dict[str, int]:
    """POST /fetch per symbol; returns rows stored per successful symbol."""
    rows_by_symbol: dict[str, int] = {}
    for i, symbol in enumerate(symbols):
        query = urllib.parse.urlencode(
            {
                "interval": "1d",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
        )
        url = f"{market_url}/api/v1/market-data/fetch/{symbol}?{query}"
        try:
            status, body = _request("POST", url, timeout=300)
        except OSError as exc:
            print(f"  [{i + 1:>2}/{len(symbols)}] {symbol:<6} FETCH ERROR: {exc}")
            continue
        if status == 200:
            rows_by_symbol[symbol] = int(body.get("rows", 0))
            print(
                f"  [{i + 1:>2}/{len(symbols)}] {symbol:<6} {rows_by_symbol[symbol]:>5} rows"
            )
        else:
            detail = body.get("detail", "")
            print(f"  [{i + 1:>2}/{len(symbols)}] {symbol:<6} HTTP {status}: {detail}")
        if pause_s and i + 1 < len(symbols):
            time.sleep(pause_s)  # politeness toward the upstream data source
    return rows_by_symbol


def validate_coverage(
    market_url: str, symbols: list[str], start: date
) -> dict[str, dict]:
    """Read back stored bars and sanity-check span + gaps (>5 business days)."""
    report: dict[str, dict] = {}
    for symbol in symbols:
        query = urllib.parse.urlencode(
            {"interval": "1d", "start_date": start.isoformat(), "limit": 5000}
        )
        try:
            status, bars = _request(
                "GET",
                f"{market_url}/api/v1/market-data/ohlcv/{symbol}?{query}",
                timeout=120,
            )
        except OSError:
            status, bars = 0, []
        if status != 200 or not isinstance(bars, list) or not bars:
            report[symbol] = {"sessions": 0, "ok": False, "note": "no stored bars"}
            continue
        stamps = [datetime.fromisoformat(b["timestamp"]).date() for b in bars]
        max_gap = max(
            ((b - a).days for a, b in zip(stamps, stamps[1:], strict=False)), default=0
        )
        note = []
        if len(stamps) < MIN_SESSIONS_FOR_TRAINING:
            note.append(
                f"only {len(stamps)} sessions (<{MIN_SESSIONS_FOR_TRAINING} for training)"
            )
        if max_gap > 7:  # 5 business days ≈ 7 calendar days
            note.append(f"max gap {max_gap} calendar days")
        report[symbol] = {
            "sessions": len(stamps),
            "first": stamps[0].isoformat(),
            "last": stamps[-1].isoformat(),
            "ok": not note,
            "note": "; ".join(note),
        }
    return report


def run_training(ml_url: str, symbols: list[str], limit: int, timeout_s: float) -> int:
    print(f"\nTraining on {len(symbols)} symbols (sync — can take minutes)...")
    try:
        status, body = _request(
            "POST",
            f"{ml_url}/api/v1/ml-pipeline/models/train",
            {"symbols": symbols, "interval": "1d", "limit": limit},
            timeout=timeout_s,
        )
    except OSError as exc:
        print(f"Training request failed: {exc}")
        return 1
    if status != 200:
        print(f"Training failed: HTTP {status}: {body.get('detail', body)}")
        return 1

    gate = body.get("gate", {})
    holdout = gate.get("holdout", {})
    print(f"\nModel: {body.get('model_id')}  (samples: {body.get('samples')})")
    print(f"Gate PASSED: {gate.get('passed')}")
    for reason in gate.get("reasons", []):
        print(f"  - {reason}")
    print(
        f"  holdout: sharpe {holdout.get('sharpe')}  auc {holdout.get('auc')}  "
        f"brier {holdout.get('brier')}  (n_test {holdout.get('n_test')})"
    )
    for fold in gate.get("folds", []):
        print(
            f"  {fold['name']:<8} sharpe {fold['sharpe']:>8}  auc {fold['auc']:>7}  "
            f"brier {fold['brier']:>7}"
        )

    version = body.get("version")
    if version is not None:
        print(
            f"\nVersion v{version} logged to MLflow (drift baseline auto-registered)."
        )
    if not gate.get("passed"):
        print(
            "Gate FAILED — an honest result, not an error. Do NOT promote; "
            "revisit universe/history depth or wait for more data."
        )
    elif version is not None:
        print(
            "Promotion is a MANUAL sign-off. Review the gate report above, then:"
            f"\n  curl -X POST {ml_url}/api/v1/ml-pipeline/models/versions/{version}/promote"
            "\nServing hot-reloads on promotion — the model votes on the next features.ready."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated tickers, or @file with one per line (default: built-in universe)",
    )
    parser.add_argument(
        "--years", type=float, default=6.0, help="History depth (default 6)"
    )
    parser.add_argument(
        "--pause", type=float, default=1.0, help="Seconds between fetches"
    )
    parser.add_argument(
        "--train", action="store_true", help="Run a training pass after backfill"
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=2000,
        help="Bars per symbol for training fetch",
    )
    parser.add_argument(
        "--train-timeout", type=float, default=1800.0, help="Training HTTP timeout (s)"
    )
    parser.add_argument(
        "--market-data-url",
        default=os.environ.get("MARKET_DATA_URL", "http://localhost:8001"),
    )
    parser.add_argument(
        "--ml-pipeline-url",
        default=os.environ.get("ML_PIPELINE_URL", "http://localhost:8005"),
    )
    args = parser.parse_args()

    if args.symbols is None:
        symbols = list(DEFAULT_UNIVERSE)
    elif args.symbols.startswith("@"):
        with open(args.symbols[1:], encoding="utf-8") as fh:
            symbols = [line.strip().upper() for line in fh if line.strip()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        sys.exit("empty universe")

    market_url = args.market_data_url.rstrip("/")
    end = datetime.now(UTC).date()
    start = end - timedelta(days=round(args.years * 365.25))

    _check_service(market_url, "market-data")
    print(
        f"Backfilling {len(symbols)} symbols, {start} → {end} (daily) via {market_url}"
    )
    rows = backfill(market_url, symbols, start, end, args.pause)

    failed = [s for s in symbols if s not in rows]
    print(
        f"\nBackfilled {len(rows)}/{len(symbols)} symbols, {sum(rows.values())} rows total."
    )
    if failed:
        print(f"FAILED: {', '.join(failed)}")

    print("\nCoverage check (stored bars):")
    coverage = validate_coverage(market_url, list(rows), start)
    for symbol, info in coverage.items():
        flag = "ok " if info["ok"] else "WARN"
        span = f"{info.get('first', '—')} → {info.get('last', '—')}"
        note = f"  ({info['note']})" if info["note"] else ""
        print(f"  {flag} {symbol:<6} {info['sessions']:>5} sessions  {span}{note}")

    exit_code = 1 if failed else 0
    if args.train:
        trainable = [s for s, info in coverage.items() if info["sessions"] > 0]
        ml_url = args.ml_pipeline_url.rstrip("/")
        _check_service(ml_url, "ml-pipeline")
        exit_code = max(
            exit_code,
            run_training(ml_url, trainable, args.train_limit, args.train_timeout),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
