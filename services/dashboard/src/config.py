from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "dashboard"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # Upstream services the dashboard aggregates (HTTP, read-only apart from
    # the on-demand backtest, which the user triggers explicitly)
    RISK_MGMT_URL: str = "http://risk-mgmt:8000"
    EXECUTION_URL: str = "http://execution:8000"
    NOTIFICATION_URL: str = "http://notification:8000"
    ML_PIPELINE_URL: str = "http://ml-pipeline:8000"
    STRATEGY_URL: str = "http://strategy:8000"
    SIGNAL_AGGREGATOR_URL: str = "http://signal-aggregator:8000"
    MARKET_DATA_URL: str = "http://market-data:8000"
    BACKTEST_URL: str = "http://backtest:8000"
    FEATURE_ENGINE_URL: str = "http://feature-engine:8000"
    FUNDAMENTAL_DATA_URL: str = "http://fundamental-data:8000"
    MACRO_DATA_URL: str = "http://macro-data:8000"
    COMPANY_CLASSIFIER_URL: str = "http://company-classifier:8000"

    # Health probes get their own, shorter budget: a probe that takes as long
    # as a data query is not reporting health, it is reporting a hang.
    HEALTH_TIMEOUT_SECONDS: float = 2.0

    @property
    def health_urls(self) -> dict[str, str]:
        """Every service the System Health section probes.

        Built from the URLs above rather than a second hard-coded list — a
        service added here cannot be missing from the health grid, which is the
        one place its absence would be invisible.
        """
        return {
            "market-data": self.MARKET_DATA_URL,
            "feature-engine": self.FEATURE_ENGINE_URL,
            "strategy": self.STRATEGY_URL,
            "backtest": self.BACKTEST_URL,
            "ml-pipeline": self.ML_PIPELINE_URL,
            "risk-mgmt": self.RISK_MGMT_URL,
            "execution": self.EXECUTION_URL,
            "notification": self.NOTIFICATION_URL,
            "fundamental-data": self.FUNDAMENTAL_DATA_URL,
            "macro-data": self.MACRO_DATA_URL,
            "company-classifier": self.COMPANY_CLASSIFIER_URL,
            "signal-aggregator": self.SIGNAL_AGGREGATOR_URL,
        }

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
