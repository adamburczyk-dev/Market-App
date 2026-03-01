"""Testy event definitions — weryfikacja kontraktów NATS."""

from datetime import datetime

from trading_common.events import (
    BaseEvent,
    EventType,
    MarketDataUpdatedEvent,
    SignalGeneratedEvent,
)


class TestBaseEvent:
    def make_event(self, **kwargs):
        defaults = {
            "event_type": EventType.MARKET_DATA_UPDATED,
            "source_service": "market-data",
        }
        return {**defaults, **kwargs}

    def test_event_id_auto_generated(self):
        e = BaseEvent(**self.make_event())
        assert e.event_id is not None
        assert len(e.event_id) > 0

    def test_two_events_have_different_ids(self):
        e1 = BaseEvent(**self.make_event())
        e2 = BaseEvent(**self.make_event())
        assert e1.event_id != e2.event_id

    def test_timestamp_auto_set(self):
        e = BaseEvent(**self.make_event())
        assert isinstance(e.timestamp, datetime)

    def test_correlation_id_optional(self):
        e = BaseEvent(**self.make_event())
        assert e.correlation_id is None

    def test_correlation_id_can_be_set(self):
        e = BaseEvent(**self.make_event(correlation_id="trace-123"))
        assert e.correlation_id == "trace-123"

    def test_subject_returns_event_type_value(self):
        e = MarketDataUpdatedEvent(symbol="AAPL", interval="1d", rows_count=100)
        assert e.subject() == "market_data.updated"


class TestMarketDataUpdatedEvent:
    def test_valid_event(self):
        e = MarketDataUpdatedEvent(symbol="AAPL", interval="1d", rows_count=250)
        assert e.symbol == "AAPL"
        assert e.rows_count == 250
        assert e.event_type == EventType.MARKET_DATA_UPDATED
        assert e.source_service == "market-data"

    def test_serialization_roundtrip(self):
        e = MarketDataUpdatedEvent(symbol="MSFT", interval="1h", rows_count=100)
        data = e.model_dump()
        assert data["symbol"] == "MSFT"
        assert data["event_type"] == "market_data.updated"


class TestSignalGeneratedEvent:
    def test_valid_signal_event(self):
        e = SignalGeneratedEvent(
            symbol="AAPL",
            strategy_name="sma_crossover",
            signal="BUY",
            confidence=0.85,
            price=153.5,
        )
        assert e.signal == "BUY"
        assert e.confidence == 0.85
        assert e.source_service == "strategy"

    def test_metadata_defaults_empty(self):
        e = SignalGeneratedEvent(
            symbol="AAPL",
            strategy_name="rsi",
            signal="SELL",
            confidence=0.7,
            price=150.0,
        )
        assert e.metadata == {}


class TestEventTypes:
    def test_all_event_types_have_dot_notation(self):
        for event_type in EventType:
            assert "." in event_type.value, f"{event_type} should use dot notation"

    def test_event_types_unique(self):
        values = [e.value for e in EventType]
        assert len(values) == len(set(values)), "EventType values must be unique"
