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

# FRED refuses a JSON response covering more vintage dates than this, with a
# 400 naming the count. Daily series are revised every business day, so the cap
# is reached in about eight years — which is why the realtime window is sliced.
MAX_VINTAGES_PER_REQUEST = 2000
# FRED's wording when the requested realtime window is entirely before the
# series entered ALFRED. It arrives as a 400 and means "nothing here", which is
# a different thing from "the request was wrong".
ALFRED_MISSING_SERIES = "does not exist in ALFRED"
# Years per slice. A daily series produces ~250 vintages/year, so five years is
# ~1250 — comfortably under the cap even if a series is revised more than once
# a day, and few enough requests that the backfill stays quick.
VINTAGE_SLICE_YEARS = 5
ALFRED_HORIZON = "9999-12-31"


class MacroFetcher(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def fetch_indicators(self) -> dict[str, float | None]: ...

    async def fetch_vintage_history(
        self, series_id: str, start: date | None = None, end: date | None = None
    ) -> list[MacroObservation]: ...

    async def aclose(self) -> None: ...


def _vintage_slices(
    first_year: int = 1990, slice_years: int = VINTAGE_SLICE_YEARS
) -> list[tuple[str, str]]:
    """Consecutive realtime windows covering everything up to the far future.

    Starts at `first_year` rather than at ALFRED_EPOCH: slicing 1776 onwards in
    five-year steps would mean forty pointless requests before any series
    existed. The final slice runs to ALFRED_HORIZON so a vintage published
    tomorrow is still inside the last window.
    """
    slices: list[tuple[str, str]] = [(ALFRED_EPOCH, f"{first_year - 1}-12-31")]
    year = first_year
    today_year = date.today().year
    while year <= today_year:
        end_year = min(year + slice_years - 1, today_year)
        upper = ALFRED_HORIZON if end_year >= today_year else f"{end_year}-12-31"
        slices.append((f"{year}-01-01", upper))
        year = end_year + 1
    return slices


def _redact(text: str, secret: str | None) -> str:
    """Never let the API key reach a log line.

    httpx puts the full request URL in the exception message, and the key is a
    query parameter — so the first upstream failure wrote it in plaintext to
    the container log, where it is picked up by log shipping and kept.
    """
    return text.replace(secret, "***") if secret else text


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

    async def _fetch_vintage_slice(
        self,
        series_id: str,
        realtime_start: str,
        realtime_end: str,
        start: date | None,
        end: date | None,
    ) -> list[dict] | None:
        """One realtime window. None means the request FAILED, [] means empty.

        The distinction carries weight: an empty window is normal (the series
        did not exist yet), a failed one means the vintage history has a hole,
        and a hole in a vintage panel is indistinguishable from "this value was
        never revised" — which is a statement about the past, made by accident.
        """
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._api_key or "",
            "file_type": "json",
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            "sort_order": "asc",
        }
        if start is not None:
            params["observation_start"] = start.isoformat()
        if end is not None:
            params["observation_end"] = end.isoformat()

        try:
            resp = await self._client.get(f"{self._base}/series/observations", params=params)
            # FRED explains itself in the BODY, and that is the only place the
            # reason appears: the status alone reads "Bad Request" for a missing
            # series, a rejected key and an over-wide vintage window alike.
            # Discarding it is how "backfill returned 400" stayed unexplained.
            if resp.status_code >= 400:
                try:
                    detail = str(resp.json().get("error_message", ""))
                except ValueError:
                    detail = resp.text[:200]
                # A window that predates the series in ALFRED is EMPTY, not
                # broken — FRED just reports it with a 400. Treating it as a
                # failure aborted the whole series on its first slice, which is
                # how a calculated series with vintages only since ~2013 looked
                # like a series with no vintages at all.
                if ALFRED_MISSING_SERIES in detail:
                    return []
                logger.warning(
                    "ALFRED vintage fetch failed",
                    series_id=series_id,
                    realtime_window=f"{realtime_start}..{realtime_end}",
                    status=resp.status_code,
                    reason=_redact(detail, self._api_key),
                )
                return None
            return list(resp.json().get("observations", []))
        except httpx.HTTPError as exc:
            logger.warning(
                "ALFRED vintage fetch failed",
                series_id=series_id,
                realtime_window=f"{realtime_start}..{realtime_end}",
                error=_redact(str(exc), self._api_key),
            )
            return None

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

        The realtime window is fetched in SLICES because FRED caps a single
        JSON response at `MAX_VINTAGES_PER_REQUEST` vintage dates. A monthly
        series stays under it for decades; a DAILY one is revised every
        business day, so T10Y2Y over 20 years has ~3100 vintages and the whole
        request is refused with a 400. That is not an edge case — the two
        series the regime classifier actually reads are both daily, so before
        slicing a 20-year backfill returned HTTP 200 having stored nothing the
        classifier could use.
        """
        if not self._api_key:
            return []

        observations: list[dict] = []
        for slice_start, slice_end in _vintage_slices():
            page = await self._fetch_vintage_slice(series_id, slice_start, slice_end, start, end)
            if page is None:
                return []  # a failed slice would leave a hole shaped like a revision
            observations.extend(page)

        out: list[MacroObservation] = []
        seen: set[tuple[date, date | None]] = set()
        for row in observations:
            value = _as_float(row.get("value"))
            observed = _as_date(row.get("date"))
            if value is None or observed is None:
                continue
            # Slices share boundaries, so the same (period, vintage) can arrive
            # twice. The store's natural key would absorb it, but the reported
            # row count would then overstate what was actually learned.
            key = (observed, _as_date(row.get("realtime_start")))
            if key in seen:
                continue
            seen.add(key)
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
