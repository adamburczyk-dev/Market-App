"""Alpha Vantage fetcher (HTTP/JSON). Secondary OHLCV source / fallback."""

from datetime import UTC, datetime
from typing import Any

import aiohttp
import structlog
from trading_common.schemas import Interval, OHLCVBar

from .base import Fetcher, FetchError

logger = structlog.get_logger()

_BASE_URL = "https://www.alphavantage.co/query"

# Map our intervals to Alpha Vantage intraday "interval" values.
_INTRADAY = {
    Interval.M1: "1min",
    Interval.M5: "5min",
    Interval.M15: "15min",
    Interval.H1: "60min",
}


class AlphaVantageFetcher(Fetcher):
    name = "alpha_vantage"

    def __init__(self, api_key: str, timeout_s: float = 15.0) -> None:
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)

    def _params(self, symbol: str, interval: Interval) -> dict[str, str]:
        if interval in _INTRADAY:
            return {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": _INTRADAY[interval],
                "outputsize": "full",
                "apikey": self._api_key,
            }
        function = "TIME_SERIES_WEEKLY" if interval == Interval.W1 else "TIME_SERIES_DAILY"
        return {
            "function": function,
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self._api_key,
        }

    async def fetch(
        self,
        symbol: str,
        interval: Interval,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OHLCVBar]:
        params = self._params(symbol, interval)
        async with (
            aiohttp.ClientSession(timeout=self._timeout) as session,
            session.get(_BASE_URL, params=params) as resp,
        ):
            if resp.status != 200:
                raise FetchError(f"Alpha Vantage HTTP {resp.status} for {symbol}")
            payload = await resp.json()
        return self._parse(symbol, interval, payload, start, end)

    @staticmethod
    def _parse(
        symbol: str,
        interval: Interval,
        payload: dict[str, Any],
        start: datetime | None,
        end: datetime | None,
    ) -> list[OHLCVBar]:
        if "Error Message" in payload or "Information" in payload:
            msg = payload.get("Error Message") or payload.get("Information")
            raise FetchError(f"Alpha Vantage error: {msg}")

        # The time-series key varies by function ("Time Series (Daily)", etc.).
        series_key = next((k for k in payload if "Time Series" in k or "Weekly" in k), None)
        if series_key is None:
            raise FetchError(f"Alpha Vantage: no time series in response for {symbol}")

        bars: list[OHLCVBar] = []
        for ts_str, values in payload[series_key].items():
            ts = _parse_timestamp(ts_str)
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            try:
                bar = OHLCVBar(
                    symbol=symbol,
                    timestamp=ts,
                    interval=interval,
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["4. close"]),
                    volume=float(values.get("5. volume", 0.0)),
                    source="alpha_vantage",
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping invalid Alpha Vantage row", symbol=symbol, error=str(exc))
                continue
            bars.append(bar)
        bars.sort(key=lambda b: b.timestamp)
        return bars


def _parse_timestamp(ts_str: str) -> datetime:
    """Alpha Vantage uses 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."""
    fmt = "%Y-%m-%d %H:%M:%S" if " " in ts_str else "%Y-%m-%d"
    return datetime.strptime(ts_str, fmt).replace(tzinfo=UTC)
