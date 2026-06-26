from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "execution"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # Push portfolio state back to risk-mgmt after fills (HTTP)
    RISK_MGMT_URL: str = "http://risk-mgmt:8000"
    # Query latest prices to mark open positions (HTTP)
    MARKET_DATA_URL: str = "http://market-data:8000"

    # NATS JetStream — ORDERS stream (consume order.requested, publish order.filled)
    NATS_ORDERS_STREAM: str = "ORDERS"
    NATS_ORDERS_SUBJECTS: str = "order.>"
    NATS_SOURCE_SUBJECT: str = "order.requested"
    NATS_DURABLE: str = "execution"
    NATS_MAX_DELIVER: int = 5

    # Subscribe to market-data updates to re-mark open positions
    NATS_MARKET_STREAM: str = "MARKET_DATA"
    NATS_MARKET_SUBJECT: str = "market_data.updated"
    NATS_MARKET_DURABLE: str = "execution-marks"

    # Paper broker
    INITIAL_CASH: float = 100_000.0
    SLIPPAGE_BPS: float = 0.0  # paper: fill at the requested price by default

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
