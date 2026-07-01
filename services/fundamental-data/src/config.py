from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "fundamental-data"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # NATS JetStream — fundamental-data publishes fundamentals.updated
    NATS_FUNDAMENTALS_STREAM: str = "FUNDAMENTALS"
    NATS_FUNDAMENTALS_SUBJECTS: str = "fundamentals.>"

    # SEC EDGAR (optional — without a User-Agent the service relies on posted statements).
    # SEC requires a descriptive UA, e.g. "trading-system fundamentals contact@example.com".
    SEC_USER_AGENT: str | None = None
    SEC_BASE_URL: str = "https://data.sec.gov"
    SEC_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
