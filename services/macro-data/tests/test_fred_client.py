"""Tests for the FRED HTTP client (via httpx MockTransport)."""

import httpx
import pytest

from src.core.fred_client import FredClient


def obs_response(value: str) -> httpx.Response:
    return httpx.Response(200, json={"observations": [{"date": "2026-06-01", "value": value}]})


def client_with(handler, api_key="KEY"):  # type: ignore[no-untyped-def]
    fc = FredClient(api_key)
    fc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return fc


@pytest.mark.asyncio
async def test_disabled_without_api_key():
    fc = FredClient(None)
    assert fc.enabled is False
    assert await fc.latest("T10Y2Y") is None
    assert await fc.fetch_indicators() == {
        "treasury_10y": None,
        "treasury_2y": None,
        "corporate_baa": None,
        "unemployment_rate": None,
        "fed_funds_rate": None,
    }
    await fc.aclose()


@pytest.mark.asyncio
async def test_latest_parses_value_and_sends_key():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return obs_response("0.42")

    fc = client_with(handler)
    assert await fc.latest("T10Y2Y") == 0.42
    assert captured["params"]["series_id"] == "T10Y2Y"
    assert captured["params"]["api_key"] == "KEY"
    assert captured["params"]["sort_order"] == "desc"
    await fc.aclose()


@pytest.mark.asyncio
async def test_missing_value_dot_becomes_none():
    fc = client_with(lambda r: obs_response("."))
    assert await fc.latest("BAA10Y") is None
    await fc.aclose()


@pytest.mark.asyncio
async def test_http_error_returns_none():
    fc = client_with(lambda r: httpx.Response(500))
    assert await fc.latest("UNRATE") is None
    await fc.aclose()


@pytest.mark.asyncio
async def test_empty_observations_returns_none():
    fc = client_with(lambda r: httpx.Response(200, json={"observations": []}))
    assert await fc.latest("FEDFUNDS") is None
    await fc.aclose()


@pytest.mark.asyncio
async def test_fetch_indicators_maps_all_series():
    def handler(request: httpx.Request) -> httpx.Response:
        sid = request.url.params["series_id"]
        return obs_response(
            {"DGS10": "4.2", "DGS2": "3.7", "BAA": "6.0", "UNRATE": "4.0", "FEDFUNDS": "5.25"}[sid]
        )

    fc = client_with(handler)
    result = await fc.fetch_indicators()
    assert result == {
        "treasury_10y": 4.2,
        "treasury_2y": 3.7,
        "corporate_baa": 6.0,
        "unemployment_rate": 4.0,
        "fed_funds_rate": 5.25,
    }
    await fc.aclose()


# --- derived spreads (2026-08-04) -----------------------------------------


def test_spreads_are_derived_from_source_series():
    """FRED's own definitions: T10Y2Y = DGS10 - DGS2, BAA10Y = BAA - DGS10.

    The calculated series exist in FRED but ALFRED only archives their vintages
    from 2014-01-27, so an as-of read before then found nothing — nine years of
    a twenty-year panel with no macro regime. The inputs carry vintages back to
    2005, so we store those and subtract at read time.
    """
    from src.core.fred_client import derive_spreads

    derived = derive_spreads({"DGS10": 4.2, "DGS2": 3.7, "BAA": 6.0})
    assert derived["T10Y2Y"] == pytest.approx(0.5)
    assert derived["BAA10Y"] == pytest.approx(1.8)
    # The source values survive — the map is extended, not replaced.
    assert derived["DGS10"] == pytest.approx(4.2)


def test_a_missing_leg_yields_no_spread_rather_than_half_of_one():
    """A one-legged spread is a number with no meaning, and it would classify."""
    from src.core.fred_client import derive_spreads

    assert "T10Y2Y" not in derive_spreads({"DGS10": 4.2})
    assert "BAA10Y" not in derive_spreads({"BAA": 6.0})


def test_a_stored_calculated_series_is_left_alone():
    """Databases written before this change hold real T10Y2Y vintages.

    Recomputing over them would silently replace a value FRED published with
    one we reconstructed — close, but not the number that was public.
    """
    from src.core.fred_client import derive_spreads

    derived = derive_spreads({"T10Y2Y": 0.42, "DGS10": 4.2, "DGS2": 3.7})
    assert derived["T10Y2Y"] == pytest.approx(0.42)


def test_the_backfill_route_defaults_to_the_fetchers_own_series():
    """Two lists of series drifted apart, and the drift reported success.

    The route carried its own copy naming FRED's CALCULATED series. After the
    fetcher moved to source series the copy still said T10Y2Y/BAA10Y, so a
    backfill kept storing precisely what the change existed to stop using, and
    returned HTTP 200 with a healthy-looking row count.
    """
    from src.api.routes import BackfillRequest
    from src.core.fred_client import DEFAULT_SERIES

    assert BackfillRequest().series == DEFAULT_SERIES
