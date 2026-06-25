"""Testy fetcherów — bez sieci (yfinance podmieniony, Alpha Vantage parsowany z próbki)."""

import pandas as pd
import pytest
from trading_common.schemas import Interval

from src.core.fetchers.alpha_vantage import AlphaVantageFetcher
from src.core.fetchers.base import FallbackFetcher, FetchError
from src.core.fetchers.yahoo import YahooFetcher

from .conftest import FakeFetcher, make_bar


def _yahoo_df(multiindex: bool = False) -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    idx.name = "Date"
    data = {
        "Open": [100.0, 101.0],
        "High": [105.0, 106.0],
        "Low": [99.0, 100.0],
        "Close": [103.0, 104.0],
        "Volume": [1_000_000.0, 1_100_000.0],
    }
    df = pd.DataFrame(data, index=idx)
    if multiindex:
        df.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in df.columns])
    return df


@pytest.mark.asyncio
async def test_yahoo_maps_dataframe_to_bars(monkeypatch):
    monkeypatch.setattr("src.core.fetchers.yahoo.yf.download", lambda *a, **k: _yahoo_df())
    bars = await YahooFetcher().fetch("AAPL", Interval.D1)
    assert len(bars) == 2
    assert bars[0].close == 103.0
    assert bars[0].source == "yahoo"
    assert bars[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_yahoo_handles_multiindex_columns(monkeypatch):
    df = _yahoo_df(multiindex=True)
    monkeypatch.setattr("src.core.fetchers.yahoo.yf.download", lambda *a, **k: df)
    bars = await YahooFetcher().fetch("AAPL", Interval.D1)
    assert len(bars) == 2
    assert bars[1].high == 106.0


@pytest.mark.asyncio
async def test_yahoo_empty_dataframe_returns_empty(monkeypatch):
    monkeypatch.setattr("src.core.fetchers.yahoo.yf.download", lambda *a, **k: pd.DataFrame())
    assert await YahooFetcher().fetch("AAPL", Interval.D1) == []


def test_alpha_vantage_parses_daily_payload():
    payload = {
        "Meta Data": {"2. Symbol": "AAPL"},
        "Time Series (Daily)": {
            "2024-01-03": {
                "1. open": "100.0",
                "2. high": "105.0",
                "3. low": "99.0",
                "4. close": "103.0",
                "5. volume": "1000000",
            },
            "2024-01-02": {
                "1. open": "98.0",
                "2. high": "101.0",
                "3. low": "97.0",
                "4. close": "100.0",
                "5. volume": "900000",
            },
        },
    }
    bars = AlphaVantageFetcher._parse("AAPL", Interval.D1, payload, None, None)
    assert len(bars) == 2
    # posortowane rosnąco po czasie
    assert [b.close for b in bars] == [100.0, 103.0]
    assert bars[0].source == "alpha_vantage"


def test_alpha_vantage_raises_on_error_payload():
    with pytest.raises(FetchError):
        AlphaVantageFetcher._parse("AAPL", Interval.D1, {"Error Message": "invalid"}, None, None)


@pytest.mark.asyncio
async def test_fallback_returns_first_non_empty():
    chain = FallbackFetcher([FakeFetcher([]), FakeFetcher([make_bar(close=42, day=1)])])
    bars = await chain.fetch("AAPL", Interval.D1)
    assert len(bars) == 1
    assert bars[0].close == 42


class _BrokenFetcher(FakeFetcher):
    async def fetch(self, symbol, interval, start=None, end=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("source down")


@pytest.mark.asyncio
async def test_fallback_raises_when_all_fail():
    chain = FallbackFetcher([_BrokenFetcher([])])
    with pytest.raises(FetchError):
        await chain.fetch("AAPL", Interval.D1)
