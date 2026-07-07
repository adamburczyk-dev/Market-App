from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "feature-engine"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # Where to query OHLCV (HTTP, per architecture: events in, queries over HTTP)
    MARKET_DATA_URL: str = "http://market-data:8000"
    # Where to query fundamentals announced by fundamentals.updated events
    FUNDAMENTAL_DATA_URL: str = "http://fundamental-data:8000"

    # NATS JetStream — source (subscribe) + own stream (publish)
    NATS_SOURCE_STREAM: str = "MARKET_DATA"
    NATS_SOURCE_SUBJECT: str = "market_data.updated"
    NATS_DURABLE: str = "feature-engine"
    NATS_MAX_DELIVER: int = 5
    NATS_FEATURES_STREAM: str = "FEATURES"
    NATS_FEATURES_SUBJECTS: str = "features.>"
    # Tier-2 attribute sources (fundamentals + company style)
    NATS_FUNDAMENTALS_STREAM: str = "FUNDAMENTALS"
    NATS_FUNDAMENTALS_SUBJECT: str = "fundamentals.updated"
    NATS_FUNDAMENTALS_DURABLE: str = "feature-engine-fundamentals"
    NATS_COMPANY_STREAM: str = "COMPANY"
    NATS_COMPANY_SUBJECT: str = "company.classified"
    NATS_COMPANY_DURABLE: str = "feature-engine-company"

    # Feature computation
    FEATURE_LOOKBACK: int = 250
    FEATURE_MIN_BARS: int = 20

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
