from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "notification"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"
    NATS_MAX_DELIVER: int = 5

    # Source streams/subjects consumed (each owned by another service)
    NATS_RISK_STREAM: str = "RISK"
    NATS_RISK_SUBJECT: str = "risk.circuit_breaker"
    NATS_ORDERS_STREAM: str = "ORDERS"
    NATS_ORDERS_SUBJECT: str = "order.filled"
    NATS_BACKTEST_STREAM: str = "BACKTEST"
    NATS_BACKTEST_SUBJECT: str = "backtest.strategy_revalidated"
    NATS_ML_STREAM: str = "ML"
    NATS_ML_SUBJECT: str = "ml.drift_detected"
    NATS_STRATEGY_STREAM: str = "STRATEGY"
    NATS_STRATEGY_SUBJECT: str = "strategy.status_changed"

    # Alert routing
    MIN_SEVERITY: str = "info"  # info | warning | critical

    # Optional channels (enabled only when configured; otherwise log-only)
    SLACK_WEBHOOK_URL: str | None = None
    TELEGRAM_BOT_TOKEN: str | None = None
    TELEGRAM_CHAT_ID: str | None = None
    # Email/SMTP (enabled when host + from + recipients are all set)
    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_STARTTLS: bool = True
    EMAIL_FROM: str | None = None
    EMAIL_TO: str = ""  # csv of recipients

    @property
    def email_recipients(self) -> list[str]:
        return [addr.strip() for addr in self.EMAIL_TO.split(",") if addr.strip()]

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
