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
        "yield_curve_10y_2y": None,
        "credit_spread_baa_10y": None,
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
            {"T10Y2Y": "0.5", "BAA10Y": "1.8", "UNRATE": "4.0", "FEDFUNDS": "5.25"}[sid]
        )

    fc = client_with(handler)
    result = await fc.fetch_indicators()
    assert result == {
        "yield_curve_10y_2y": 0.5,
        "credit_spread_baa_10y": 1.8,
        "unemployment_rate": 4.0,
        "fed_funds_rate": 5.25,
    }
    await fc.aclose()
