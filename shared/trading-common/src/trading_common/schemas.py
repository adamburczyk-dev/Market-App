"""
Shared Pydantic models — kontrakt między serwisami.
Każdy serwis importuje: from trading_common.schemas import OHLCVBar
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Interval(str, Enum):
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
    source: Optional[str] = None

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info: object) -> float:
        data = getattr(info, "data", {})
        if "low" in data and v < data["low"]:
            raise ValueError("high must be >= low")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_high(cls, v: float, info: object) -> float:
        data = getattr(info, "data", {})
        if "high" in data and v > data["high"]:
            raise ValueError("low must be <= high")
        return v


class Signal(str, Enum):
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
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    metadata: dict = Field(default_factory=dict)


class PortfolioMetrics(BaseModel):
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    daily_pnl: float
    daily_pnl_pct: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
