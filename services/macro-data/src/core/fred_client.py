"""FRED / ALFRED (Federal Reserve Economic Data) HTTP client.

Two different questions, two different requests:

- **"What is it now?"** — the latest observation. This is what serving needs and
  what the client has always done.
- **"What did we know on day D?"** — the VINTAGE question, which needs ALFRED.
  FRED revises series backwards: today's answer for March 2015 is the revised
  figure, which nobody could have traded on. Passing `realtime_start` /
  `realtime_end` on the same endpoint makes FRED return each value together
  with the window during which it was the published number, and that is the
  only form in which a macro series can become a model feature without
  smuggling the future into the past.

Requires ``FRED_API_KEY``; without one the client is disabled and every fetch
returns nothing (the service then relies on manually-posted indicators).
Values reported as "." by FRED (missing) are normalized to ``None``.
"""

import asyncio
from datetime import date, datetime
from typing import Protocol

import httpx
import structlog
from trading_common.schemas import MacroObservation

logger = structlog.get_logger()

# Indicator name → FRED series id. All are published directly by FRED.
DEFAULT_SERIES = {
    "yield_curve_10y_2y": "T10Y2Y",  # 10Y minus 2Y Treasury spread
    "credit_spread_baa_10y": "BAA10Y",  # Moody's BAA minus 10Y Treasury
    "unemployment_rate": "UNRATE",
    "fed_funds_rate": "FEDFUNDS",
}

# ALFRED's "give me every vintage" window. 1776-07-04 is FRED's own documented
# floor for realtime_start, not a joke of ours.
ALFRED_EPOCH = "1776-07-04"
ALFRED_HORIZON = "9999-12-31"


class MacroFetcher(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def fetch_indicators(self) -> dict[str, float | None]: ...

    async def fetch_vintage_history(
        self, series_id: str, start: date | None = None, end: date | None = None
    ) -> list[MacroObservation]: ...

    async def aclose(self) -> None: ...


class FredClient:
    def __init__(
        self,
        api_key: str | None,
        series: dict[str, str] | None = None,
        base_url: str = "https://api.stlouisfed.org/fred",
        timeout_s: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._series = series or dict(DEFAULT_SERIES)
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def latest(self, series_id: str) -> float | None:
        if not self._api_key:
            return None
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            resp = await self._client.get(f"{self._base}/series/observations", params=params)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
        except httpx.HTTPError as exc:
            logger.warning("FRED fetch failed", series_id=series_id, error=str(exc))
            return None
        if not obs:
            return None
        value = obs[0].get("value")
        if value in (None, ".", ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def fetch_indicators(self) -> dict[str, float | None]:
        """Fetch every configured series concurrently → indicator name → value/None."""
        names = list(self._series)
        values = await asyncio.gather(*(self.latest(self._series[n]) for n in names))
        return dict(zip(names, values, strict=True))

    async def fetch_vintage_history(
        self, series_id: str, start: date | None = None, end: date | None = None
    ) -> list[MacroObservation]:
        """Every VINTAGE of every observation in the window (ALFRED).

        Widening the realtime window is what makes FRED return one row per
        (period, revision) instead of one row per period carrying its latest
        revised value. Without it a 20-year backfill would look complete and be
        wrong in exactly the way that is hardest to notice: plausible numbers,
        none of which were knowable at the time they are attached to.

        `realtime_start` is carried through verbatim. An observation whose
        vintage FRED does not report is returned with `realtime_start=None`,
        which makes it invisible to as-of reads rather than silently dated.
        """
        if not self._api_key:
            return []
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "realtime_start": ALFRED_EPOCH,
            "realtime_end": ALFRED_HORIZON,
            "sort_order": "asc",
        }
        if start is not None:
            params["observation_start"] = start.isoformat()
        if end is not None:
            params["observation_end"] = end.isoformat()

        try:
            resp = await self._client.get(f"{self._base}/series/observations", params=params)
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
        except httpx.HTTPError as exc:
            logger.warning("ALFRED vintage fetch failed", series_id=series_id, error=str(exc))
            return []

        out: list[MacroObservation] = []
        for row in observations:
            value = _as_float(row.get("value"))
            observed = _as_date(row.get("date"))
            if value is None or observed is None:
                continue
            out.append(
                MacroObservation(
                    series=series_id,
                    observation_date=observed,
                    value=value,
                    realtime_start=_as_date(row.get("realtime_start")),
                    source="alfred",
                )
            )
        logger.info(
            "Vintage history fetched",
            series_id=series_id,
            rows=len(out),
            undated=sum(1 for o in out if o.realtime_start is None),
        )
        return out

    async def aclose(self) -> None:
        await self._client.aclose()


def _as_float(value: object) -> float | None:
    if value in (None, ".", ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_date(value: object) -> date | None:
    """FRED dates are ISO; anything else is treated as unknown, not guessed."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
