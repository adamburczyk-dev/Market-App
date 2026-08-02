from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "strategy"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # Where to query computed features (HTTP — events in, queries over HTTP)
    FEATURE_ENGINE_URL: str = "http://feature-engine:8000"
    # Where to read live portfolio state for RiskEnvelope (HTTP; falls back to placeholder)
    RISK_MGMT_URL: str = "http://risk-mgmt:8000"

    # NATS JetStream — source (features.ready) + own streams (signals, strategy status)
    NATS_SOURCE_STREAM: str = "FEATURES"
    NATS_SOURCE_SUBJECT: str = "features.ready"
    NATS_DURABLE: str = "strategy"
    NATS_MAX_DELIVER: int = 5
    NATS_SIGNALS_STREAM: str = "SIGNALS"
    NATS_SIGNALS_SUBJECTS: str = "signal.>"
    # StrategyStatusChangedEvent (strategy.status_changed) lands here
    NATS_STRATEGY_STREAM: str = "STRATEGY"
    NATS_STRATEGY_SUBJECTS: str = "strategy.>"
    # Consume backtest walk-forward revalidations (R7 — closes the backtest→strategy loop)
    NATS_BACKTEST_STREAM: str = "BACKTEST"
    NATS_BACKTEST_SUBJECT: str = "backtest.strategy_revalidated"
    NATS_BACKTEST_DURABLE: str = "strategy-revalidation"

    # Which registered rules this instance runs. Empty = every rule registered
    # in trading_common.strategies. Naming a subset is how a rule is retired
    # without a deployment; an UNKNOWN name fails startup rather than being
    # skipped, because a silently-not-running strategy looks exactly like one
    # that ran and found nothing.
    ENABLED_STRATEGIES: str = ""
    # Per-rule parameter overrides as JSON, e.g.
    # STRATEGY_PARAMS='{"momentum_rank": {"buy_rank": 0.85}}'.
    # Overrides are merged over the rule's own defaults at call time — the
    # registry keeps ONE instance per name, so a tuned copy cannot drift into
    # being a second strategy answering to the same name.
    STRATEGY_PARAMS: dict[str, dict[str, float]] = {}

    # Fallback stop distance (fraction of price) for a symbol too young to have
    # an ATR. Rules normally size their stop in ATR multiples — see
    # StrategyService._protective_levels.
    STOP_LOSS_PCT: float = 0.05

    # Cost filter
    EXPECTED_EDGE_BPS: float = 200.0  # baseline expected edge at full confidence
    MARKET_CAP_TIER: str = "large"

    # RiskEnvelope placeholder portfolio (until risk-mgmt provides real state)
    PORTFOLIO_VALUE: float = 100_000.0
    CURRENT_EXPOSURE_PCT: float = 0.0
    CURRENT_DRAWDOWN_PCT: float = 0.0
    DAILY_LOSS_PCT: float = 0.0

    @property
    def enabled_strategies(self) -> list[str]:
        return [s.strip() for s in self.ENABLED_STRATEGIES.split(",") if s.strip()]

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
