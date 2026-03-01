"""trading-common — shared contracts for the trading system."""

from trading_common.schemas import (
    Interval,
    OHLCVBar,
    Signal,
    TradingSignal,
    PortfolioMetrics,
)
from trading_common.events import EventType, BaseEvent, MarketDataUpdatedEvent, SignalGeneratedEvent

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
