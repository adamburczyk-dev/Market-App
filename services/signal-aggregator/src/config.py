from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "signal-aggregator"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # NATS JetStream — publishes signal.aggregated (shares the SIGNALS stream / signal.>)
    NATS_SIGNALS_STREAM: str = "SIGNALS"
    NATS_SIGNALS_SUBJECTS: str = "signal.>"
    NATS_MAX_DELIVER: int = 5

    # Consume strategy (rule-based) signals — the primary per-symbol component
    NATS_SIGNAL_SUBJECT: str = "signal.generated"
    NATS_SIGNAL_DURABLE: str = "signal-aggregator-signals"
    # Consume macro regime changes — the market-wide directional bias
    NATS_MACRO_STREAM: str = "MACRO"
    NATS_MACRO_SUBJECT: str = "macro.regime_changed"
    NATS_MACRO_DURABLE: str = "signal-aggregator-regime"

    # Signal sources combined (rules-based + ML + macro regime)
    SIGNAL_SOURCES: str = "strategy,ml,macro"
    BUY_THRESHOLD: float = 0.2  # weighted-score magnitude for BUY/SELL
    BASE_EDGE_BPS: float = 200.0  # confidence → expected edge for the cost gate

    # Adaptive weighting
    WEIGHT_LOOKBACK_DAYS: int = 60
    WEIGHT_MIN: float = 0.05
    WEIGHT_MAX: float = 0.60

    @property
    def sources(self) -> list[str]:
        return [s.strip() for s in self.SIGNAL_SOURCES.split(",") if s.strip()]

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
