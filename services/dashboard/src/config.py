from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "dashboard"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # Upstream services the dashboard aggregates (HTTP, read-only)
    RISK_MGMT_URL: str = "http://risk-mgmt:8000"
    EXECUTION_URL: str = "http://execution:8000"
    NOTIFICATION_URL: str = "http://notification:8000"
    ML_PIPELINE_URL: str = "http://ml-pipeline:8000"

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
