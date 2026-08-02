from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "backtest"
    LOG_LEVEL: str = "INFO"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    NATS_URL: str = "nats://localhost:4222"

    # Where to pull historical OHLCV (HTTP — events for notifications, queries via HTTP)
    MARKET_DATA_URL: str = "http://market-data:8000"

    # NATS JetStream — backtest publishes results + revalidation recommendations
    NATS_BACKTEST_STREAM: str = "BACKTEST"
    NATS_BACKTEST_SUBJECTS: str = "backtest.>"

    # Execution cost only. The strategy's own thresholds belong to the rule
    # (trading_common.strategies) — the service-level lookback/entry_momentum
    # knobs were parameters of a price-momentum engine that no longer exists;
    # keeping them would have implied this service still defines a strategy.
    BACKTEST_COST_BPS: float = 5.0
    BACKTEST_DEFAULT_LIMIT: int = 500

    # Walk-forward revalidation windows (trading days) + degradation gate
    OOS_WINDOW_DAYS: int = 126  # ~6 months out-of-sample
    IS_WINDOW_DAYS: int = 252  # ~1 year in-sample
    DEGRADATION_THRESHOLD: float = 0.40  # OOS Sharpe drop >= 40% → probation

    # Scheduled weekly revalidation (Saturday per the monitoring requirements).
    # OPT-IN: the published StrategyRevalidatedEvent drives the live strategy
    # status (R7), so enable only with the strategy's real activation-time
    # OOS-Sharpe baseline configured below.
    SCHEDULE_REVALIDATION_ENABLED: bool = False
    REVALIDATION_WEEKDAY: int = 5  # Monday=0 … Saturday=5
    REVALIDATION_HOUR_UTC: int = 6
    # NOT momentum_rank: it reads cross-sectional ranks, which a single-symbol
    # backtest cannot produce, so this job would now refuse it outright. It used
    # to "succeed" by grading a price-momentum proxy instead — i.e. by
    # revalidating a strategy that was not the one running. Revalidating
    # momentum_rank needs a universe backtest (open decision D7).
    REVALIDATION_STRATEGY: str = "sma_ema_crossover"
    REVALIDATION_SYMBOL: str = "AAPL"
    REVALIDATION_INTERVAL: str = "1d"
    REVALIDATION_ORIGINAL_OOS_SHARPE: float = 1.0  # activation-time OOS Sharpe

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
