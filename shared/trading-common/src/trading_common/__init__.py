"""trading-common — shared contracts for the trading system."""

from trading_common.events import BaseEvent, EventType, MarketDataUpdatedEvent, SignalGeneratedEvent
from trading_common.schemas import (
    Interval,
    OHLCVBar,
    PortfolioMetrics,
    Signal,
    TradingSignal,
)

__all__ = [
    "Interval",
    "OHLCVBar",
    "Signal",
    "TradingSignal",
    "PortfolioMetrics",
    "EventType",
    "BaseEvent",
    "MarketDataUpdatedEvent",
    "SignalGeneratedEvent",
]
