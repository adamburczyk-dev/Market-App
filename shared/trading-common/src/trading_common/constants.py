"""Stałe współdzielone przez wszystkie serwisy."""

# Porty serwisów (wewnętrzne kontenery)
SERVICE_PORTS = {
    "market-data": 8001,
    "feature-engine": 8002,
    "strategy": 8003,
    "backtest": 8004,
    "ml-pipeline": 8005,
    "risk-mgmt": 8006,
    "execution": 8007,
    "notification": 8008,
    "dashboard": 8501,
}

# Domyślne symbole
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]

# Largest window a single OHLCV read may ask for.
#
# The ceiling is SHARED because it is a contract between whoever asks for bars
# and whoever serves them. Declared twice at two values, it surfaces only as a
# 422 on the longest run — that is, on the one run that mattered. Which is how
# the measurement campaign died: ml-pipeline accepted `limit <= 10_000`,
# market-data served at most 5_000, and a 20-year request (5040 sessions + 253
# warm-up bars = 5293) fell exactly in between.
#
# The value is ~40 years of daily sessions, so any realistic backfill fits
# whole. A ceiling still exists: an unbounded read is one request away from
# exhausting the service's memory.
MAX_OHLCV_LIMIT = 10_000

# NATS subjects
NATS_SUBJECTS = {
    "market_data": "market_data.updated",
    "features": "features.computed",
    "signal": "signal.generated",
    "order_submitted": "order.submitted",
    "order_filled": "order.filled",
    "risk_breach": "risk.limit_breached",
    "circuit_breaker": "risk.circuit_breaker",
    "alert": "alert.triggered",
    "backtest_done": "backtest.completed",
    "model_drift": "ml.drift_detected",
    "model_retrained": "ml.model_retrained",
    "model_trained": "ml.model_trained",
    "order_rejected": "order.rejected",
    "strategy_status": "strategy.status_changed",
    # ML/AI extension (serwisy 10-13)
    "fundamentals": "fundamentals.updated",
    "macro": "macro.updated",
    "regime_changed": "macro.regime_changed",
    "sentiment": "sentiment.updated",
    "company_classified": "company.classified",
    "features_ready": "features.ready",
    "signal_aggregated": "signal.aggregated",
}

# Risk defaults (Layer 2 — full risk-mgmt-svc)
# NOTE: Layer 2 drawdown (0.20) is intentionally higher than Layer 1 RiskEnvelope (0.15).
# Layer 1 is a conservative pre-trade gate; Layer 2 is the full risk management system.
DEFAULT_MAX_POSITION_PCT = 0.05  # max 5% portfela na jedną pozycję
DEFAULT_MAX_DRAWDOWN_PCT = 0.20  # stop trading przy 20% drawdown
DEFAULT_VAR_CONFIDENCE = 0.95

# Risk defaults (Layer 1 — RiskEnvelope, first-line defense)
DEFAULT_MAX_PORTFOLIO_EXPOSURE_PCT = 0.80
DEFAULT_MAX_SINGLE_LOSS_PCT = 0.02
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_MAX_CORRELATED_POSITIONS = 3
