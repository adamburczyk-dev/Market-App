"""Tests for company-classifier HTTP routes."""

import pytest
from httpx import AsyncClient

from src.core.service import CompanyClassifierService

from .conftest import profile


def classify_body(**metrics: float) -> dict:
    body: dict = {"profile": profile().model_dump(mode="json")}
    if metrics:
        body["metrics"] = metrics
    return body


@pytest.mark.asyncio
async def test_status_ok(client: AsyncClient):
    resp = await client.get("/api/v1/company-classifier/status")
    assert resp.status_code == 200
    assert resp.json()["service"] == "company-classifier"


@pytest.mark.asyncio
async def test_classify_then_get(wired: tuple[AsyncClient, CompanyClassifierService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/company-classifier/classify",
        json=classify_body(pe_ratio=33, revenue_growth=0.25, dividend_yield=0.0),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["style"] == "growth"
    assert body["model_stack"] == "growth_largecap_v1"
    assert body["cap_tier"] == "mega"
    assert body["profile"]["style"] == "growth"

    got = await client.get("/api/v1/company-classifier/companies/AAPL")
    assert got.status_code == 200
    assert got.json()["style"] == "growth"

    listing = await client.get("/api/v1/company-classifier/companies")
    assert listing.json()["symbols"] == ["AAPL"]


@pytest.mark.asyncio
async def test_classify_without_metrics_uses_sector(
    wired: tuple[AsyncClient, CompanyClassifierService],
):
    client, _ = wired
    resp = await client.post("/api/v1/company-classifier/classify", json=classify_body())
    assert resp.status_code == 200
    # IT sector prior → growth
    assert resp.json()["style"] == "growth"
    assert resp.json()["basis"] == "sector"


@pytest.mark.asyncio
async def test_get_unknown_404(wired: tuple[AsyncClient, CompanyClassifierService]):
    client, _ = wired
    resp = await client.get("/api/v1/company-classifier/companies/TSLA")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_classify_503_when_unwired(client: AsyncClient):
    resp = await client.post("/api/v1/company-classifier/classify", json=classify_body())
    assert resp.status_code == 503
