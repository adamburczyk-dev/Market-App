from pydantic_settings import BaseSettings

# The horizon is a property of the TRAINED MODEL, not of the deployment, so it
# is read from the label definition rather than declared again here. Deliberately
# NOT surfaced in docker-compose or Helm: an operator knob that can disagree with
# the model being served is a train/serve divergence with no symptom.
from src.core.inference_log import retention_for
from src.core.labels import LABEL_HORIZON, outcome_drop_after_days


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
    # Point-in-time fundamentals panel (P2-3), joined into training on request
    FUNDAMENTAL_DATA_URL: str = "http://fundamental-data:8000"
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
    LABEL_HORIZON_DAYS: int = LABEL_HORIZON

    # Daily monitoring loop (plan ML-3): resolve matured outcomes → drift check
    SIGNAL_AGGREGATOR_URL: str = "http://signal-aggregator:8000"
    MONITOR_INTERVAL_S: float = 86_400.0  # daily, per the monitoring requirements
    MONITOR_INITIAL_DELAY_S: float = 3_600.0  # first run 1h after boot
    # Derived: universe x horizon, doubled. At 2000 the log retained under 5
    # sessions of a 414-name universe and evicted votes BEFORE they matured —
    # oldest first, which is exactly the ones about to resolve.
    INFERENCE_LOG_MAXLEN: int = retention_for()
    # Derived, not typed: a literal here silently outlives the next horizon
    # change, and this particular literal going stale kills the whole ML-3 loop
    # without logging anything (votes resolve as label=None forever).
    OUTCOME_DROP_AFTER_DAYS: int = outcome_drop_after_days()

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
