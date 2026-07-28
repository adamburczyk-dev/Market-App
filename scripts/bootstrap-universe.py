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
import itertools
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
        max_gap = max(((b - a).days for a, b in itertools.pairwise(stamps)), default=0)
        note = []
        closes = [float(b["close"]) for b in bars]
        adjs = [
            float(b["adj_close"])
            if b.get("adj_close") not in (None, "")
            else float(b["close"])
            for b in bars
        ]
        missing_adj = sum(1 for b in bars if b.get("adj_close") in (None, ""))

        # Which series matters: features and labels run on the ADJUSTED close,
        # so that is the one whose extremes can poison a model. Raw-vs-adjusted
        # disagreement is the discriminator the first version of this check
        # lacked — it flagged NFLX's real -35% earnings crash of 2022-04-20 as a
        # corporate action, because a raw move alone cannot tell the two apart.
        suspect: list[tuple[str, float]] = []
        notable: list[tuple[str, float]] = []
        actions: list[tuple[str, float, float]] = []
        for i in range(len(bars) - 1):
            if closes[i] <= 0 or adjs[i] <= 0:
                continue
            raw_ret = closes[i + 1] / closes[i] - 1.0
            adj_ret = adjs[i + 1] / adjs[i] - 1.0
            day = stamps[i + 1].isoformat()
            if adj_ret <= -0.45 or adj_ret >= 0.80:
                # Beyond what a large cap does on news in one session — almost
                # certainly an unadjusted split or a bad tick IN THE SERIES WE
                # MEASURE RETURNS ON. This is the only real data error here.
                suspect.append((day, adj_ret))
            elif adj_ret <= -0.25 or adj_ret >= 0.50:
                notable.append((day, adj_ret))
            if abs(raw_ret - adj_ret) > 0.10:
                actions.append((day, raw_ret, adj_ret))

        if suspect:
            note.append(
                "SUSPECT adjusted move (split or bad tick?): "
                + ", ".join(f"{d} {r:+.0%}" for d, r in suspect[:3])
            )
        if missing_adj:
            note.append(
                f"{missing_adj}/{len(bars)} bars without adj_close "
                "(returns exclude dividends — re-run the backfill)"
            )
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
            "adj_close_coverage": round(1.0 - missing_adj / len(bars), 4),
            # Real single-session moves (earnings, guidance) — data, not defects.
            "notable_moves": [
                {"date": d, "return": round(r, 3)} for d, r in notable[:5]
            ],
            # Sessions where raw and adjusted disagree: the adjustment doing its
            # job on a split or dividend. Their ABSENCE would be the warning.
            "corporate_actions": [
                {"date": d, "raw": round(a, 3), "adjusted": round(b, 3)}
                for d, a, b in actions[:5]
            ],
            "ok": not note,
            "note": "; ".join(note),
        }
    return report


def _num(value: object, width: int = 7) -> str:
    """Format a metric, keeping a missing one visibly missing (never as 0)."""
    if isinstance(value, int | float):
        return f"{value:>{width}.4f}"
    return f"{'n/a':>{width}}"


FOLD_HEADER = (
    f"  {'fold':<8} {'IC':>8} {'ICIR':>7} {'net':>7} {'active':>7} "
    f"{'AUCval':>7} {'AUCtr':>7} {'lift':>7} {'predσ':>7}"
)


def _print_fold(fold: dict) -> None:
    print(
        f"  {fold['name']:<8} {_num(fold.get('ic_mean'), 8)} {_num(fold.get('icir'))} "
        f"{_num(fold.get('sharpe_net'))} {_num(fold.get('sharpe_active'))} "
        f"{_num(fold.get('auc'))} {_num(fold.get('auc_train'))} "
        f"{_num(fold.get('lift'))} {_num(fold.get('pred_std'))}"
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _collect(folds: list[dict], key: str) -> list[float]:
    return [f[key] for f in folds if isinstance(f.get(key), int | float)]


def _print_verdict(gate: dict, holdout: dict) -> None:
    """State what the numbers imply — underfit, overfit, or no signal.

    The first real run failed the gate, and the gate alone could not say WHY:
    a fold Sharpe of 3.85 came with a NEGATIVE lift (the market rose, the model
    picked worse than average). These three readings are the reason Tier 0 added
    train-AUC, IC and the relative metrics.
    """
    folds = gate.get("folds", [])
    auc_val = _mean(_collect(folds, "auc") + _collect([holdout], "auc"))
    auc_train = _mean(_collect(folds, "auc_train") + _collect([holdout], "auc_train"))
    icir = _mean(_collect(folds, "icir") + _collect([holdout], "icir"))
    ic = _mean(_collect(folds, "ic_mean") + _collect([holdout], "ic_mean"))
    spread = _mean(
        _collect(folds, "pred_std_pre_calibration")
        + _collect([holdout], "pred_std_pre_calibration")
    )
    baseline_keys = {k for f in folds for k in f if k.startswith("baseline_ic_")}
    baseline = {k: _mean(_collect(folds, k)) for k in sorted(baseline_keys)}

    print("\nReading:")
    if auc_train is None or auc_val is None:
        print("  - not enough fold data to judge fit quality")
    elif auc_train < 0.55:
        spread_txt = (
            f" (pre-calibration pred σ {spread:.4f})" if spread is not None else ""
        )
        print(
            f"  - train AUC {auc_train:.3f} ≈ coin flip{spread_txt}: the model does not fit"
            "\n    even the data it was SHOWN. Two causes look identical from this number:"
            "\n    (a) optimization — capacity / lr / scaling;"
            "\n    (b) the features carry no signal at all, so there is nothing to fit."
            "\n    Run --capacity-probe to decide: it fits an over-parameterized model on real"
            "\n    and on SHUFFLED labels. A small gap means (b), and more symbols or more"
            "\n    history will not help."
        )
    elif auc_val <= 0.52:
        print(
            f"  - NO TRANSFERABLE SIGNAL: train AUC {auc_train:.3f} but val AUC"
            f" {auc_val:.3f}."
            "\n    The model can memorize and cannot generalize — the features carry no"
            "\n    edge at this horizon. Fix the features/universe, not the optimizer."
        )
    else:
        print(f"  - fit is real: train AUC {auc_train:.3f}, val AUC {auc_val:.3f}")

    if ic is not None:
        if ic < -0.01:
            verdict = "ranks BACKWARDS (a consistently negative IC is not an edge)"
        elif abs(ic) < 0.01:
            verdict = "no rank edge (|IC| < 0.01 is noise)"
        else:
            verdict = "some rank edge"
        icir_txt = f", ICIR {icir:.2f}" if icir is not None else ""
        print(f"  - IC {ic:+.4f}{icir_txt} → {verdict}")
    for key, value in baseline.items():
        if value is None:
            continue
        feature = key.removeprefix("baseline_ic_")
        # Signed, not absolute: a model with IC −0.02 does not "beat" a baseline
        # of +0.003 — it ranks the universe upside down.
        beaten = ic is not None and ic > value and ic > 0
        print(
            f"  - baseline IC of raw {feature}: {value:+.4f} → the model"
            f" {'beats' if beaten else 'does NOT beat'} a single feature's rank"
        )
    print(
        "  - net = long-only Sharpe after costs; active = vs the equal-weight universe."
        "\n    A high net with a low active means the MARKET paid, not the model."
    )


def run_capacity_probe(
    ml_url: str,
    symbols: list[str],
    limit: int,
    timeout_s: float,
    report: dict | None = None,
) -> int:
    """Fit a big unregularized model on real and on shuffled labels.

    The one experiment that separates "the training is underperforming" from
    "there is nothing in these features to learn" — a flat train AUC looks
    identical in both cases, and the responses are opposite.
    """
    print(f"\nCapacity probe on {len(symbols)} symbols (sync — can take minutes)...")
    try:
        status, body = _request(
            "POST",
            f"{ml_url}/api/v1/ml-pipeline/models/capacity-probe",
            {"symbols": symbols, "interval": "1d", "limit": limit},
            timeout=timeout_s,
        )
    except OSError as exc:
        print(f"Capacity probe failed: {exc}")
        if report is not None:
            report["capacity_probe_error"] = str(exc)
        return 1
    if status != 200:
        print(f"Capacity probe failed: HTTP {status}: {body.get('detail', body)}")
        if report is not None:
            report["capacity_probe_error"] = (
                f"HTTP {status}: {body.get('detail', body)}"
            )
        return 1
    if report is not None:
        report["capacity_probe"] = body

    probe = body.get("probe", {})
    runs = ", ".join(f"{v:.4f}" for v in probe.get("real_runs", []))
    controls = ", ".join(f"{v:.4f}" for v in probe.get("shuffled_runs", []))
    print(
        f"  rows {probe.get('n_rows'):,}, {probe.get('n_features')} features\n"
        f"  train AUC  real {probe.get('auc_train_real'):.4f} [{runs}]\n"
        f"             shuffled {probe.get('auc_train_shuffled'):.4f} [{controls}]\n"
        f"             production config {probe.get('auc_train_production'):.4f}\n"
        f"  gap (mean real − mean shuffled): {probe.get('gap'):+.4f}\n"
        f"  margin (worst real − best control): {probe.get('margin'):+.4f} "
        f"→ {'separated' if probe.get('separated') else 'OVERLAPPING'}"
    )
    print(f"\n  {probe.get('verdict')}")
    return 0


def run_sweep(
    ml_url: str,
    symbols: list[str],
    limit: int,
    timeout_s: float,
    report: dict | None = None,
) -> int:
    """Compare training configurations on the walk-forward folds."""
    print(
        f"\nConfiguration sweep on {len(symbols)} symbols (sync — several minutes)..."
    )
    try:
        status, body = _request(
            "POST",
            f"{ml_url}/api/v1/ml-pipeline/models/tune",
            {"symbols": symbols, "interval": "1d", "limit": limit},
            timeout=timeout_s,
        )
    except OSError as exc:
        print(f"Sweep failed: {exc}")
        if report is not None:
            report["sweep_error"] = str(exc)
        return 1
    if status != 200:
        print(f"Sweep failed: HTTP {status}: {body.get('detail', body)}")
        if report is not None:
            report["sweep_error"] = f"HTTP {status}: {body.get('detail', body)}"
        return 1
    if report is not None:
        report["sweep"] = body

    sweep = body.get("sweep", {})
    print(
        f"  {'config':<20} {'IC':>9} {'t':>6} {'AUCval':>8} {'AUCtr':>8} {'lift':>8} {'predσ':>8}"
    )
    for candidate in sweep.get("candidates", []):
        print(
            f"  {candidate['name']:<20} {candidate['ic_mean']:>9.5f} "
            f"{candidate['ic_tstat']:>6.2f} {candidate['auc_val_mean']:>8.4f} "
            f"{candidate['auc_train_mean']:>8.4f} {candidate['lift_mean']:>8.4f} "
            f"{candidate['pred_std_mean']:>8.4f}"
        )
    print(
        f"\n  best by IC t-stat: {sweep.get('best')}  |  configurations tried: "
        f"{sweep.get('n_trials')}"
    )
    print(
        "  Selection used folds only — the holdout is still untouched. Feed n_trials to the\n"
        "  gate (G5) so the deflated Sharpe knows how many times we looked."
    )
    return 0


def run_training(
    ml_url: str,
    symbols: list[str],
    limit: int,
    timeout_s: float,
    report: dict | None = None,
) -> int:
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
        if report is not None:
            report["training_error"] = str(exc)
        return 1
    if status != 200:
        print(f"Training failed: HTTP {status}: {body.get('detail', body)}")
        if report is not None:
            report["training_error"] = f"HTTP {status}: {body.get('detail', body)}"
        return 1
    if report is not None:
        report["training"] = body  # full response — the reviewable artifact

    gate = body.get("gate", {})
    holdout = gate.get("holdout", {})
    data = body.get("dataset", {})
    print(f"\nModel: {body.get('model_id')}  (samples: {body.get('samples')})")
    print(
        f"  data: {data.get('symbols_with_rows')}/{data.get('symbols_requested')} symbols, "
        f"{data.get('sessions')} sessions {data.get('first_session', '')[:10]} → "
        f"{data.get('last_session', '')[:10]}, "
        f"positive rate {data.get('positive_rate')}, {data.get('n_features')} features"
    )
    if data.get("symbols_missing"):
        print(f"  symbols without usable history: {', '.join(data['symbols_missing'])}")
    ess = body.get("effective_sample_size", {})
    if ess:
        effective = ess.get("n_effective_samples")
        effective_txt = (
            f"{effective:,.0f}" if isinstance(effective, int | float) else "n/a"
        )
        print(
            f"  effective sample: {effective_txt} of {ess.get('n_samples'):,} raw "
            f"(≈{ess.get('n_symbols_effective')} independent names of "
            f"{ess.get('n_symbols')}, avg pair corr "
            f"{ess.get('avg_pairwise_correlation')}) — this, not the row count, "
            "is what the metrics stand on"
        )
    dropped = body.get("dropped_zero_variance") or []
    if dropped:
        print(f"  dropped constant features: {', '.join(dropped)}")
    print(f"Gate PASSED: {gate.get('passed')}")
    for condition in gate.get("conditions", []):
        mark = "PASS" if condition.get("passed") else "FAIL"
        print(
            f"  [{mark}] {condition.get('id')} {condition.get('name')}: {condition.get('detail')}"
        )
    if not gate.get("conditions"):  # older report shape
        for reason in gate.get("reasons", []):
            print(f"  - {reason}")
    print(FOLD_HEADER)
    _print_fold({"name": "holdout", **holdout})
    for fold in gate.get("folds", []):
        _print_fold(fold)
    _print_verdict(gate, holdout)

    version = body.get("version")
    if version is not None:
        print(
            f"\nVersion v{version} logged to MLflow (drift baseline auto-registered)."
        )
    else:
        # A run nobody can inspect or promote later is a wasted run — say so.
        print(
            "\n!! NIE ZAPISANO w MLflow (version: null) — rejestr modeli jest "
            "niedostepny, wiec tego biegu nie da sie pozniej obejrzec ani "
            "promowac.\n   Sprawdz: curl localhost:8005/ready  -> model_registry"
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
        "--capacity-probe",
        action="store_true",
        help=(
            "Fit a deliberately over-parameterized model on real and on SHUFFLED "
            "labels and compare train AUCs — separates 'training underperforms' "
            "from 'nothing here to learn'. Diagnostic only; registers nothing."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Score several training configurations on the walk-forward folds "
            "(never the holdout) and report the winner plus how many were tried."
        ),
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help=(
            "Train on the bars already stored in market-data (re-running a "
            "training pass with new diagnostics does not need a re-fetch). "
            "The coverage read-back still runs — a symbol with no stored bars "
            "is reported as failed."
        ),
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
    parser.add_argument(
        "--report-out",
        default=None,
        help=(
            "Write a self-contained JSON report (coverage + full gate/diagnostics) "
            "to this path — the artifact to share for an off-machine review"
        ),
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

    report: dict = {
        "report_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "universe": symbols,
        "years": args.years,
        "range": {"start": start.isoformat(), "end": end.isoformat()},
    }

    _check_service(market_url, "market-data")
    if args.skip_backfill:
        print(
            f"Skipping fetch — using bars already stored in market-data ({market_url})"
        )
        rows = {}
        to_check = symbols
    else:
        print(
            f"Backfilling {len(symbols)} symbols, {start} → {end} (daily) via {market_url}"
        )
        rows = backfill(market_url, symbols, start, end, args.pause)
        print(
            f"\nBackfilled {len(rows)}/{len(symbols)} symbols, "
            f"{sum(rows.values())} rows total."
        )
        failed_fetch = [s for s in symbols if s not in rows]
        if failed_fetch:
            print(f"FAILED: {', '.join(failed_fetch)}")
        to_check = list(rows)

    print("\nCoverage check (stored bars):")
    coverage = validate_coverage(market_url, to_check, start)
    # A symbol with no usable stored history is a failure either way: it cannot
    # be fetched (backfill) or it is not there to train on (--skip-backfill).
    failed = [s for s in symbols if coverage.get(s, {}).get("sessions", 0) == 0]
    for symbol, info in coverage.items():
        flag = "ok " if info["ok"] else "WARN"
        span = f"{info.get('first', '—')} → {info.get('last', '—')}"
        note = f"  ({info['note']})" if info["note"] else ""
        print(f"  {flag} {symbol:<6} {info['sessions']:>5} sessions  {span}{note}")

    report["backfill"] = {
        "rows_by_symbol": rows,
        "failed": failed,
        "skipped": bool(args.skip_backfill),
    }
    report["coverage"] = coverage

    exit_code = 1 if failed else 0
    if args.capacity_probe:
        trainable = [s for s, info in coverage.items() if info["sessions"] > 0]
        ml_url = args.ml_pipeline_url.rstrip("/")
        _check_service(ml_url, "ml-pipeline")
        exit_code = max(
            exit_code,
            run_capacity_probe(
                ml_url, trainable, args.train_limit, args.train_timeout, report=report
            ),
        )
    if args.tune:
        trainable = [s for s, info in coverage.items() if info["sessions"] > 0]
        ml_url = args.ml_pipeline_url.rstrip("/")
        _check_service(ml_url, "ml-pipeline")
        exit_code = max(
            exit_code,
            run_sweep(
                ml_url, trainable, args.train_limit, args.train_timeout, report=report
            ),
        )
    if args.train:
        trainable = [s for s, info in coverage.items() if info["sessions"] > 0]
        ml_url = args.ml_pipeline_url.rstrip("/")
        _check_service(ml_url, "ml-pipeline")
        exit_code = max(
            exit_code,
            run_training(
                ml_url, trainable, args.train_limit, args.train_timeout, report=report
            ),
        )

    if args.report_out:
        path = os.path.abspath(args.report_out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=False)
        print(f"\nReport written to {path}")
        print("Share this file (commit it or paste it) for a review of the run.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
