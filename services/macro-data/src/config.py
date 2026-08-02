from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "macro-data"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # NATS JetStream — macro-data publishes macro.updated + macro.regime_changed
    NATS_MACRO_STREAM: str = "MACRO"
    NATS_MACRO_SUBJECTS: str = "macro.>"

    # Vintage panel (P2-4). Without it the service still classifies the LIVE
    # regime; it just cannot answer "what was the regime on 2015-03-01", which
    # is the question training asks.
    DB_ENABLED: bool = True
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "trading"
    DB_USER: str = "trading"
    DB_PASSWORD: str = ""

    # FRED (optional — without a key the service relies on manually-posted indicators)
    FRED_API_KEY: str | None = None
    FRED_BASE_URL: str = "https://api.stlouisfed.org/fred"

    # Scheduled FRED refresh (runs only when FRED_API_KEY is set). The regime
    # publish path is transition-safe: RegimeChangedEvent fires only on a real
    # change, so a periodic refresh is idempotent for downstream consumers.
    SCHEDULE_REFRESH_ENABLED: bool = True
    REFRESH_INTERVAL_S: float = 21_600.0  # 6h — FRED series update at most daily
    REFRESH_INITIAL_DELAY_S: float = 0.0  # populate the regime right at boot

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
