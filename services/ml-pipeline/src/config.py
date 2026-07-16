from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "ml-pipeline"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # NATS JetStream — ml-pipeline publishes drift/retrain/training events
    NATS_ML_STREAM: str = "ML"
    NATS_ML_SUBJECTS: str = "ml.>"

    # Training (plan ML-1): history source + MLflow local-backend registry
    MARKET_DATA_URL: str = "http://market-data:8000"
    MLFLOW_TRACKING_URI: str = "sqlite:///mlruns/mlflow.db"
    MODEL_NAME: str = "global_v1"

    # Serving (plan ML-2): features.ready → infer → ml.signal_generated
    FEATURE_ENGINE_URL: str = "http://feature-engine:8000"
    MACRO_DATA_URL: str = "http://macro-data:8000"
    NATS_FEATURES_STREAM: str = "FEATURES"
    NATS_FEATURES_SUBJECT: str = "features.ready"
    NATS_FEATURES_DURABLE: str = "ml-pipeline-features"
    NATS_MAX_DELIVER: int = 5
    SERVE_INTERVAL: str = "1d"
    BUY_PROBABILITY: float = 0.55  # dead zone between the two thresholds is silent
    SELL_PROBABILITY: float = 0.45
    LABEL_HORIZON_DAYS: int = 10

    # Daily monitoring loop (plan ML-3): resolve matured outcomes → drift check
    SIGNAL_AGGREGATOR_URL: str = "http://signal-aggregator:8000"
    MONITOR_INTERVAL_S: float = 86_400.0  # daily, per the monitoring requirements
    MONITOR_INITIAL_DELAY_S: float = 3_600.0  # first run 1h after boot
    INFERENCE_LOG_MAXLEN: int = 2000
    OUTCOME_DROP_AFTER_DAYS: int = 42  # unresolved votes dropped past ~3× horizon

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
