"""Tests for DashboardService.overview aggregation + partial tolerance."""

import pytest

from .conftest import FakeSource, build_service


@pytest.mark.asyncio
async def test_overview_full_availability():
    service = build_service(FakeSource())
    ov = await service.overview()
    assert ov["sources"] == {
        "risk-mgmt": "ok",
        "execution": "ok",
        "notification": "ok",
        "ml-pipeline": "ok",
    }
    assert ov["portfolio"]["value"] == 100000.0
    assert ov["positions"]["AAPL"]["quantity"] == 50
    assert len(ov["recent_alerts"]) == 1
    assert ov["models"] == ["m1"]


@pytest.mark.asyncio
async def test_overview_partial_when_ml_down():
    service = build_service(FakeSource(ml=None))
    ov = await service.overview()
    assert ov["sources"]["ml-pipeline"] == "unavailable"
    assert ov["sources"]["risk-mgmt"] == "ok"
    assert ov["models"] == []  # missing source → empty, not a crash


@pytest.mark.asyncio
async def test_risk_source_needs_both_portfolio_and_breaker():
    # breaker down but portfolio up → risk-mgmt reported unavailable (incomplete)
    service = build_service(FakeSource(cb=None))
    ov = await service.overview()
    assert ov["sources"]["risk-mgmt"] == "unavailable"
    assert ov["circuit_breaker"] is None
    assert ov["portfolio"] is not None  # the part that loaded is still surfaced


@pytest.mark.asyncio
async def test_execution_source_needs_both_portfolio_and_positions():
    service = build_service(FakeSource(pos=None))
    ov = await service.overview()
    assert ov["sources"]["execution"] == "unavailable"
    assert ov["positions"] == {}


@pytest.mark.asyncio
async def test_overview_all_down():
    service = build_service(FakeSource(rp=None, cb=None, ep=None, pos=None, al=None, ml=None))
    ov = await service.overview()
    assert set(ov["sources"].values()) == {"unavailable"}
    assert ov["portfolio"] is None
    assert ov["positions"] == {}
    assert ov["recent_alerts"] == []
    assert ov["models"] == []
