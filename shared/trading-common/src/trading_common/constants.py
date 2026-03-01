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
    "alert": "alert.triggered",
    "backtest_done": "backtest.completed",
}

# Risk defaults
DEFAULT_MAX_POSITION_PCT = 0.05   # max 5% portfela na jedną pozycję
DEFAULT_MAX_DRAWDOWN_PCT = 0.20   # stop trading przy 20% drawdown
DEFAULT_VAR_CONFIDENCE = 0.95
