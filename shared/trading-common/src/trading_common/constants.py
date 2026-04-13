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
    "strategy_status": "strategy.status_changed",
}

# Risk defaults (Layer 2 — full risk-mgmt-svc)
DEFAULT_MAX_POSITION_PCT = 0.05  # max 5% portfela na jedną pozycję
DEFAULT_MAX_DRAWDOWN_PCT = 0.20  # stop trading przy 20% drawdown
DEFAULT_VAR_CONFIDENCE = 0.95

# Risk defaults (Layer 1 — RiskEnvelope, first-line defense)
DEFAULT_MAX_PORTFOLIO_EXPOSURE_PCT = 0.80
DEFAULT_MAX_SINGLE_LOSS_PCT = 0.02
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_MAX_CORRELATED_POSITIONS = 3
