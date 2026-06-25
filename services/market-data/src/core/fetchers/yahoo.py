"""Yahoo Finance fetcher (yfinance). Primary OHLCV source."""

import asyncio
from datetime import datetime
from typing import Any

import pandas as pd
import structlog
import yfinance as yf
from trading_common.schemas import Interval, OHLCVBar

from .base import Fetcher

logger = structlog.get_logger()

# When no explicit start is given, fetch a reasonable trailing window.
_DEFAULT_PERIOD = "1y"


class YahooFetcher(Fetcher):
    name = "yahoo"

    async def fetch(
        self,
        symbol: str,
        interval: Interval,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OHLCVBar]:
        # yfinance is synchronous — run it off the event loop.
        return await asyncio.to_thread(self._fetch_sync, symbol, interval, start, end)

    def _fetch_sync(
        self,
        symbol: str,
        interval: Interval,
        start: datetime | None,
        end: datetime | None,
    ) -> list[OHLCVBar]:
        df = yf.download(
            symbol,
            interval=interval.value,
            start=start,
            end=end,
            period=None if start else _DEFAULT_PERIOD,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            logger.warning("Yahoo returned no data", symbol=symbol, interval=interval.value)
            return []
        return self._to_bars(symbol, interval, df)

    @staticmethod
    def _to_bars(symbol: str, interval: Interval, df: pd.DataFrame) -> list[OHLCVBar]:
        df = df.reset_index()
        # Newer yfinance returns MultiIndex columns even for a single ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        ts_col = "Datetime" if "Datetime" in df.columns else "Date"
        bars: list[OHLCVBar] = []
        for record in df.to_dict("records"):
            row: dict[str, Any] = record
            try:
                bar = OHLCVBar(
                    symbol=symbol,
                    timestamp=pd.Timestamp(row[ts_col]).to_pydatetime(),
                    interval=interval,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    source="yahoo",
                )
            except (ValueError, KeyError, TypeError) as exc:
                # Skip malformed / NaN rows rather than failing the whole fetch.
                logger.warning("Skipping invalid Yahoo row", symbol=symbol, error=str(exc))
                continue
            bars.append(bar)
        return bars
