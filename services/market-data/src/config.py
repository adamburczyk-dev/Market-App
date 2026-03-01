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

    # Service-specific
    ALPHA_VANTAGE_API_KEY: str | None = None
    DEFAULT_FETCH_INTERVAL: str = "1d"
    MAX_CONCURRENT_FETCHES: int = 5
    CACHE_TTL_SECONDS: int = 3600

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()  # type: ignore[call-arg]
