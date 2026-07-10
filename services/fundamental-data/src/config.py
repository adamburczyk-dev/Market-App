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

    # Scheduled EDGAR refresh of a configured symbol universe (runs only when
    # SEC_USER_AGENT is set AND REFRESH_SYMBOLS is non-empty).
    SCHEDULE_REFRESH_ENABLED: bool = True
    REFRESH_SYMBOLS: str = ""  # csv, e.g. "AAPL,MSFT,NVDA"
    REFRESH_INTERVAL_S: float = 604_800.0  # weekly — annual filings move slowly
    REFRESH_INITIAL_DELAY_S: float = 60.0  # first run shortly after boot
    REFRESH_SYMBOL_PAUSE_S: float = 1.0  # politeness gap between symbols

    @property
    def refresh_symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.REFRESH_SYMBOLS.split(",") if s.strip()]

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
