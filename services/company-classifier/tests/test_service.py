"""Tests for CompanyClassifierService — enrich, store, publish."""

import pytest
from trading_common.events import EventType

from src.core.classifier import ValuationMetrics
from src.events.publisher import NullPublisher

from .conftest import build_service, profile


@pytest.mark.asyncio
async def test_classify_enriches_profile_and_publishes():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    metrics = ValuationMetrics(pe_ratio=32, revenue_growth=0.25, dividend_yield=0.0)
    enriched, result = await service.classify(profile(), metrics)
    assert enriched.style == "growth"
    assert enriched.model_stack == "growth_largecap_v1"
    assert enriched.as_of is not None
    assert result.cap_tier == "mega"
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.COMPANY_CLASSIFIED
    assert event.symbol == "AAPL"
    assert event.style == "growth"
    assert event.model_stack == "growth_largecap_v1"
    assert event.source_service == "company-classifier"


@pytest.mark.asyncio
async def test_classify_stores_latest_by_symbol():
    service = build_service()
    await service.classify(
        profile(), ValuationMetrics(pe_ratio=10, pb_ratio=1.0, dividend_yield=0.05)
    )
    record = service.get("AAPL")
    assert record is not None
    assert record[0].style == "value"
    assert service.symbols() == ["AAPL"]


@pytest.mark.asyncio
async def test_get_is_case_insensitive():
    service = build_service()
    await service.classify(profile(), None)
    assert service.get("aapl") is not None


@pytest.mark.asyncio
async def test_classify_without_metrics_uses_sector_prior():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    enriched, result = await service.classify(profile(sector="Utilities", market_cap=8e9), None)
    assert enriched.style == "value"
    assert result.basis == "sector"


@pytest.mark.asyncio
async def test_reclassify_updates_stored_record():
    service = build_service()
    await service.classify(profile(), ValuationMetrics(pe_ratio=40, revenue_growth=0.3))
    assert service.get("AAPL")[0].style == "growth"
    # new metrics flip it to value
    await service.classify(
        profile(), ValuationMetrics(pe_ratio=8, pb_ratio=1.0, dividend_yield=0.05)
    )
    assert service.get("AAPL")[0].style == "value"
    assert len(service.symbols()) == 1
