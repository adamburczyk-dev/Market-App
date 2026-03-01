"""Testy utility functions."""

from datetime import datetime, timezone, timedelta
from trading_common.utils import utcnow, to_utc, symbol_to_topic


class TestUtcNow:
    def test_returns_aware_datetime(self):
        dt = utcnow()
        assert dt.tzinfo is not None

    def test_returns_utc(self):
        dt = utcnow()
        assert dt.tzinfo == timezone.utc

    def test_close_to_current_time(self):
        dt = utcnow()
        now = datetime.now(timezone.utc)
        assert abs((now - dt).total_seconds()) < 1.0


class TestToUtc:
    def test_naive_datetime_gets_utc_tzinfo(self):
        naive = datetime(2024, 1, 1, 12, 0)
        aware = to_utc(naive)
        assert aware.tzinfo == timezone.utc

    def test_already_aware_datetime_converted(self):
        eastern = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 1, 12, 0, tzinfo=eastern)
        utc = to_utc(dt)
        assert utc.tzinfo == timezone.utc
        assert utc.hour == 17  # 12:00 EST = 17:00 UTC

    def test_utc_datetime_unchanged(self):
        dt = datetime(2024, 6, 15, 9, 30, tzinfo=timezone.utc)
        assert to_utc(dt) == dt


class TestSymbolToTopic:
    def test_simple_symbol(self):
        assert symbol_to_topic("AAPL") == "aapl"

    def test_slash_replaced(self):
        assert symbol_to_topic("BTC/USD") == "btc_usd"

    def test_dash_replaced(self):
        assert symbol_to_topic("BTC-USD") == "btc_usd"

    def test_lowercase(self):
        assert symbol_to_topic("SPY") == "spy"
