from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "market-data"
    LOG_LEVEL: str = "INFO"

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "trading_db"
    DB_USER: str = "trader"
    DB_PASSWORD: str  # WYMAGANE

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    # NATS
    NATS_URL: str = "nats://localhost:4222"
    NATS_STREAM_NAME: str = "MARKET_DATA"
    NATS_STREAM_SUBJECTS: str = "market_data.>"

    # Service-specific
    ALPHA_VANTAGE_API_KEY: str | None = None
    DEFAULT_FETCH_INTERVAL: str = "1d"
    MAX_CONCURRENT_FETCHES: int = 5
    CACHE_TTL_SECONDS: int = 3600

    # --- Scheduled incremental pull -------------------------------------
    # market-data is the root of the event chain: every downstream service
    # reacts to market_data.updated. Without this the system cannot run a
    # single day unattended, which is what the "30 days of paper trading"
    # rule requires before any real capital.
    SCHEDULE_FETCH_ENABLED: bool = True
    # A stack that is not up 24/7 never reaches FETCH_AT_HOUR_UTC, so the
    # aligned daily schedule would never fire even once — not late, never. On
    # boot, run the pull immediately if none is recorded for today.
    FETCH_CATCHUP_ON_START: bool = True
    # Comma-separated. Empty = nothing to pull, and the job stays down rather
    # than inventing a universe.
    FETCH_SYMBOLS: str = ""
    # Hour (UTC) to run at. 23:00 is after the US close, so the session that
    # just ended is available. Firing "24h after the container started" would
    # drift into the middle of a session.
    FETCH_AT_HOUR_UTC: int = 23
    FETCH_INTERVAL_S: float = 86_400.0
    # Re-request this many days before the newest stored bar. Cheap (the upsert
    # is idempotent) and it is what makes a restated adj_close detectable.
    FETCH_OVERLAP_DAYS: int = 5
    # Used only for a symbol we hold nothing for.
    FETCH_INITIAL_HISTORY_DAYS: int = 365 * 6
    FETCH_SYMBOL_PAUSE_S: float = 1.0
    # Weekends are skipped: a gap-based pull on a closed day just returns
    # nothing. Exchange holidays are not modelled — same harmless outcome.
    FETCH_SKIP_WEEKENDS: bool = True

    @property
    def fetch_symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.FETCH_SYMBOLS.split(",") if s.strip()]

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
