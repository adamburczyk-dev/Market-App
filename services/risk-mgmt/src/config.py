from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "risk-mgmt"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # NATS JetStream — source (signals) + own streams (orders + risk events)
    NATS_SOURCE_STREAM: str = "SIGNALS"
    NATS_SOURCE_SUBJECT: str = "signal.generated"
    NATS_DURABLE: str = "risk-mgmt"
    NATS_MAX_DELIVER: int = 5
    NATS_ORDERS_STREAM: str = "ORDERS"
    NATS_ORDERS_SUBJECTS: str = "order.>"
    NATS_RISK_STREAM: str = "RISK"
    NATS_RISK_SUBJECTS: str = "risk.>"

    # Position sizing (drawdown-adaptive)
    BASE_RISK_PER_TRADE: float = 0.02
    DD_SCALING_START: float = 0.05
    DD_SCALING_END: float = 0.15
    MAX_POSITION_PCT: float = 0.05

    # Circuit breaker thresholds
    DRAWDOWN_WARN_PCT: float = 0.08  # > 8% drawdown → YELLOW
    DAILY_LOSS_HALT_PCT: float = 0.05  # > 5% daily loss → RED (halt)
    DRAWDOWN_FLATTEN_PCT: float = 0.15  # > 15% drawdown → BLACK (flatten)

    # Initial portfolio state (placeholder until execution feeds real state)
    PORTFOLIO_VALUE: float = 100_000.0

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
