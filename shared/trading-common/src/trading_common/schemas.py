"""
Shared Pydantic models — kontrakt między serwisami.
Każdy serwis importuje: from trading_common.schemas import OHLCVBar
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class Interval(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"
    W1 = "1wk"


class OHLCVBar(BaseModel):
    symbol: str
    timestamp: datetime
    interval: Interval
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    source: str | None = None

    @model_validator(mode="after")
    def check_high_low(self) -> Self:
        """OHLC invariant: high must be >= low.

        Uses a model-level validator (mode="after") because field validators
        only see *previously* validated fields — when ``high`` is validated
        ``low`` is not yet available, so a per-field check on ``high`` is dead
        code. A model validator sees the full, validated bar.
        """
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class Signal(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignal(BaseModel):
    symbol: str
    strategy: str
    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    price: float = Field(gt=0)
    timestamp: datetime
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    metadata: dict = Field(default_factory=dict)


class PortfolioMetrics(BaseModel):
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    daily_pnl: float
    daily_pnl_pct: float
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    current_drawdown: float | None = None
    var_95: float | None = None
    cvar_95: float | None = None


# ============================================================
# ML/AI extension contracts (serwisy 10-13).
# Initial, intentionally minimal contracts — defined here ("contracts first")
# so that fundamental-data / macro-data / company-classifier / signal-aggregator
# can be built against a stable shared shape. Refine as those services mature.
# ============================================================


class MacroRegime(StrEnum):
    """Market regimes — values aligned with risk-mgmt RegimeAllocator keys."""

    EXPANSION = "expansion"
    RECOVERY = "recovery"
    SLOWDOWN = "slowdown"
    CONTRACTION = "contraction"
    CRISIS = "crisis"


class CompanyProfile(BaseModel):
    """Company metadata — drives model-stack routing (company-classifier-svc)."""

    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    market_cap: float | None = Field(default=None, ge=0)
    style: str | None = None  # "growth" | "value" | "blend"
    model_stack: str | None = None  # assigned ML model-stack id
    as_of: datetime | None = None


class FinancialStatements(BaseModel):
    """Periodic fundamentals from SEC EDGAR (10-Q/10-K) + derived Piotroski F-Score."""

    symbol: str
    period_end: date
    fiscal_period: str  # "Q1".."Q4" | "FY"
    revenue: float | None = None
    net_income: float | None = None
    total_assets: float | None = Field(default=None, ge=0)
    total_liabilities: float | None = Field(default=None, ge=0)
    operating_cash_flow: float | None = None
    eps: float | None = None
    piotroski_f_score: int | None = Field(default=None, ge=0, le=9)
    source: str | None = None
    filed_at: datetime | None = None


class MacroSnapshot(BaseModel):
    """Macro state from FRED + derived market regime (macro-data-svc)."""

    timestamp: datetime
    regime: MacroRegime | None = None
    yield_curve_10y_2y: float | None = None
    credit_spread_baa_10y: float | None = None
    pmi: float | None = None
    cpi_yoy: float | None = None
    unemployment_rate: float | None = None
    fed_funds_rate: float | None = None


class SentimentSnapshot(BaseModel):
    """News / social sentiment for a symbol."""

    symbol: str
    timestamp: datetime
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    news_count: int = Field(default=0, ge=0)
    social_volume: int | None = Field(default=None, ge=0)
    source: str | None = None


class FeatureVector(BaseModel):
    """Computed features for one symbol/timestamp (feature-engine-svc).

    Feature values should be cross-sectional percentile ranks where applicable
    (López de Prado), not raw values.
    """

    symbol: str
    timestamp: datetime
    interval: Interval
    features: dict[str, float] = Field(default_factory=dict)
    tier: int | None = Field(default=None, ge=1, le=3)
    rank_transformed: bool = False
