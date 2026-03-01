"""Testy kontraktów Pydantic — walidacja danych między serwisami."""

import pytest
from datetime import datetime, timezone
from trading_common.schemas import (
    Interval,
    OHLCVBar,
    Signal,
    TradingSignal,
    PortfolioMetrics,
)


def make_bar(**kwargs) -> dict:
    defaults = {
        "symbol": "AAPL",
        "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        "interval": Interval.D1,
        "open": 150.0,
        "high": 155.0,
        "low": 149.0,
        "close": 153.0,
        "volume": 1_000_000.0,
    }
    return {**defaults, **kwargs}


class TestOHLCVBar:
    def test_valid_bar(self):
        bar = OHLCVBar(**make_bar())
        assert bar.symbol == "AAPL"
        assert bar.interval == Interval.D1

    def test_high_gte_low_valid(self):
        bar = OHLCVBar(**make_bar(high=155.0, low=149.0))
        assert bar.high > bar.low

    def test_high_less_than_low_raises(self):
        with pytest.raises(ValueError):
            OHLCVBar(**make_bar(high=148.0, low=149.0))

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            OHLCVBar(**make_bar(open=-1.0))

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            OHLCVBar(**make_bar(close=0.0))

    def test_negative_volume_raises(self):
        with pytest.raises(ValueError):
            OHLCVBar(**make_bar(volume=-100.0))

    def test_zero_volume_allowed(self):
        bar = OHLCVBar(**make_bar(volume=0.0))
        assert bar.volume == 0.0

    def test_optional_source(self):
        bar = OHLCVBar(**make_bar(source="yfinance"))
        assert bar.source == "yfinance"

    def test_interval_enum_values(self):
        assert Interval.M1 == "1m"
        assert Interval.D1 == "1d"

    def test_serialization_roundtrip(self):
        bar = OHLCVBar(**make_bar())
        restored = OHLCVBar.model_validate(bar.model_dump())
        assert restored == bar


class TestTradingSignal:
    def make_signal(self, **kwargs) -> dict:
        defaults = {
            "symbol": "AAPL",
            "strategy": "sma_crossover",
            "signal": Signal.BUY,
            "confidence": 0.75,
            "price": 153.0,
            "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        }
        return {**defaults, **kwargs}

    def test_valid_signal(self):
        sig = TradingSignal(**self.make_signal())
        assert sig.signal == Signal.BUY
        assert sig.confidence == 0.75

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            TradingSignal(**self.make_signal(confidence=1.5))

    def test_negative_confidence_raises(self):
        with pytest.raises(ValueError):
            TradingSignal(**self.make_signal(confidence=-0.1))

    def test_confidence_boundary_values(self):
        TradingSignal(**self.make_signal(confidence=0.0))
        TradingSignal(**self.make_signal(confidence=1.0))

    def test_all_signal_values(self):
        for sig in [Signal.BUY, Signal.SELL, Signal.HOLD]:
            s = TradingSignal(**self.make_signal(signal=sig))
            assert s.signal == sig

    def test_optional_stop_loss_take_profit(self):
        sig = TradingSignal(**self.make_signal(stop_loss=145.0, take_profit=165.0))
        assert sig.stop_loss == 145.0
        assert sig.take_profit == 165.0

    def test_metadata_default_empty(self):
        sig = TradingSignal(**self.make_signal())
        assert sig.metadata == {}


class TestPortfolioMetrics:
    def test_valid_metrics(self):
        m = PortfolioMetrics(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            total_value=105_000.0,
            cash=50_000.0,
            positions_value=55_000.0,
            daily_pnl=500.0,
            daily_pnl_pct=0.005,
            sharpe_ratio=1.5,
            max_drawdown=-0.08,
        )
        assert m.total_value == 105_000.0
        assert m.sharpe_ratio == 1.5

    def test_optional_fields_default_none(self):
        m = PortfolioMetrics(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            total_value=100_000.0,
            cash=100_000.0,
            positions_value=0.0,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
        )
        assert m.sharpe_ratio is None
        assert m.var_95 is None
