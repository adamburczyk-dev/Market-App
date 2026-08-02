"""
Shared Pydantic models — kontrakt między serwisami.
Każdy serwis importuje: from trading_common.schemas import OHLCVBar
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Self

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
    # Dividend- and split-adjusted close. The raw OHLC above is what an order
    # would have paid; THIS is what a return is measured on. Without it a
    # cross-sectional model systematically under-rates dividend payers, which
    # is precisely the value/quality axis a ranking model is supposed to see.
    # Optional so bars stored before 2026-07-28 stay valid — consumers fall
    # back to `close` and lose only the dividend component.
    adj_close: float | None = Field(default=None, gt=0)
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
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_stop_loss_for_orders(self) -> Self:
        """Risk rule (non-negotiable): no order (BUY/SELL) without a stop_loss.

        HOLD is exempt — it places no order.
        """
        if self.signal in (Signal.BUY, Signal.SELL) and self.stop_loss is None:
            raise ValueError("stop_loss is required for BUY/SELL signals")
        return self


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
    # Gross profitability (Novy-Marx 2013) needs revenue minus cost of revenue.
    # Both are carried because filers report one or the other: `gross_profit`
    # direct where it exists, otherwise derived from `cost_of_revenue`. A factor
    # present on half the universe is not a factor, so coverage decides here.
    gross_profit: float | None = None
    cost_of_revenue: float | None = None
    net_income: float | None = None
    total_assets: float | None = Field(default=None, ge=0)
    total_liabilities: float | None = Field(default=None, ge=0)
    # balance-sheet detail for the liquidity / dilution Piotroski signals
    current_assets: float | None = Field(default=None, ge=0)
    current_liabilities: float | None = Field(default=None, ge=0)
    shares_outstanding: float | None = Field(default=None, ge=0)
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


class MacroObservation(BaseModel):
    """One macro reading, dated on BOTH axes that matter (macro-data-svc).

    `observation_date` is the period the number describes; `realtime_start` is
    the date from which that number was the published value. The two are not the
    same and the gap is where look-ahead lives: unemployment for March 2015 is
    published in April and then **revised** for years afterwards. Asking FRED
    today what the March-2015 rate was returns the revised figure — a number
    nobody could have traded on. ALFRED's vintage API answers the honest
    question instead, and this contract is the shape of that answer.

    `realtime_start` is optional so a non-vintage fetch is still representable,
    but such a row is deliberately INVISIBLE to as-of reads: a fact we cannot
    date cannot be used point-in-time. Same rule as `filed_at` on fundamentals.
    """

    series: str
    observation_date: date
    value: float
    realtime_start: date | None = None
    source: str = "fred"


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
