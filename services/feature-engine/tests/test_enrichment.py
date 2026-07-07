"""Tests for Tier-2 attribute enrichment (fundamentals + company style).

fundamentals.updated / company.classified → per-symbol attribute store →
merged into feature vectors at read time (features + cross-sectional ranks).
"""

import httpx
import pytest
from trading_common.events import CompanyClassifiedEvent, FundamentalsUpdatedEvent
from trading_common.schemas import Interval

from src.core.attributes import InMemoryAttributeStore
from src.core.enrichment import fundamental_features, style_features
from src.core.fundamental_client import HttpFundamentalsClient
from src.core.service import FeatureEngineService
from src.core.store import InMemoryFeatureStore
from src.events.publisher import NullPublisher

from .conftest import FakeMarketDataClient

# --- pure derivation ---


def payload(f_score=6, revenue=1000.0, net_income=100.0, assets=2000.0, liabilities=800.0):
    return {
        "f_score": f_score,
        "statement": {
            "revenue": revenue,
            "net_income": net_income,
            "total_assets": assets,
            "total_liabilities": liabilities,
        },
    }


def test_fundamental_features_full():
    feats = fundamental_features(payload())
    assert feats["f_score"] == 6.0
    assert feats["fund_net_margin"] == pytest.approx(0.1)
    assert feats["fund_roa"] == pytest.approx(0.05)
    assert feats["fund_leverage"] == pytest.approx(0.4)


def test_fundamental_features_conservative_on_missing():
    feats = fundamental_features({"f_score": None, "statement": {"net_income": 5.0}})
    assert feats == {}  # no revenue/assets → no ratios; no score → no f_score


def test_zero_revenue_yields_no_margin():
    feats = fundamental_features(payload(revenue=0.0))
    assert "fund_net_margin" not in feats
    assert "fund_roa" in feats  # assets intact


def test_style_encoding():
    assert style_features("growth") == {"style_growth": 1.0, "style_value": 0.0}
    assert style_features("value") == {"style_growth": 0.0, "style_value": 1.0}
    assert style_features("blend") == {"style_growth": 0.5, "style_value": 0.5}
    assert style_features("weird") == {}


# --- attribute store ---


@pytest.mark.asyncio
async def test_attribute_store_merges_disjoint_writers():
    store = InMemoryAttributeStore()
    await store.put("AAPL", {"f_score": 6.0})
    await store.put("aapl", {"style_growth": 1.0})  # case-insensitive symbol key
    merged = await store.get("AAPL")
    assert merged == {"f_score": 6.0, "style_growth": 1.0}
    assert await store.get("GHOST") == {}


# --- HTTP client ---


def fundamentals_http(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fund:8000")
    return HttpFundamentalsClient("http://fund:8000", client=http)


@pytest.mark.asyncio
async def test_client_returns_payload():
    client = fundamentals_http(lambda req: httpx.Response(200, json=payload()))
    data = await client.get_fundamentals("aapl")
    assert data["f_score"] == 6


@pytest.mark.asyncio
async def test_client_404_is_none():
    client = fundamentals_http(lambda req: httpx.Response(404, json={"detail": "none"}))
    assert await client.get_fundamentals("GHOST") is None


@pytest.mark.asyncio
async def test_client_transport_error_raises_for_redelivery():
    def boom(request):
        raise httpx.ConnectError("fundamental-data down")

    client = fundamentals_http(boom)
    with pytest.raises(httpx.ConnectError):
        await client.get_fundamentals("AAPL")


# --- event handlers + read-time merge ---


class FakeFundamentalsClient:
    def __init__(self, data):
        self.data = data

    async def get_fundamentals(self, symbol):
        return self.data

    async def aclose(self):
        return None


def build_service(fundamentals_data=None):
    return FeatureEngineService(
        FakeMarketDataClient(n=30),
        InMemoryFeatureStore(),
        NullPublisher(),
        min_bars=20,
        attributes=InMemoryAttributeStore(),
        fundamentals_client=FakeFundamentalsClient(fundamentals_data),
    )


def fundamentals_event(symbol="AAPL"):
    return FundamentalsUpdatedEvent(symbol=symbol, period_end="2024-09-28", fiscal_period="FY")


def classified_event(symbol="AAPL", style="growth"):
    return CompanyClassifiedEvent(symbol=symbol, style=style, model_stack=f"{style}_largecap_v1")


@pytest.mark.asyncio
async def test_fundamentals_event_enriches_features():
    service = build_service(fundamentals_data=payload())
    await service.compute_for_symbol("AAPL", Interval.D1)  # technical vector first
    await service.handle_fundamentals_event(fundamentals_event().model_dump_json().encode())
    fv = await service.get_features("AAPL", Interval.D1)
    assert fv.features["f_score"] == 6.0
    assert fv.features["fund_net_margin"] == pytest.approx(0.1)
    assert "momentum_20" in fv.features  # technicals intact


@pytest.mark.asyncio
async def test_classified_event_enriches_features():
    service = build_service()
    await service.compute_for_symbol("AAPL", Interval.D1)
    await service.handle_company_classified_event(classified_event().model_dump_json().encode())
    fv = await service.get_features("AAPL", Interval.D1)
    assert fv.features["style_growth"] == 1.0
    assert fv.features["style_value"] == 0.0


@pytest.mark.asyncio
async def test_missing_fundamentals_payload_is_noop():
    service = build_service(fundamentals_data=None)  # 404 path
    await service.compute_for_symbol("AAPL", Interval.D1)
    await service.handle_fundamentals_event(fundamentals_event().model_dump_json().encode())
    fv = await service.get_features("AAPL", Interval.D1)
    assert "f_score" not in fv.features


@pytest.mark.asyncio
async def test_attributes_without_technical_vector_stay_invisible():
    service = build_service(fundamentals_data=payload())
    await service.handle_fundamentals_event(fundamentals_event().model_dump_json().encode())
    assert await service.get_features("AAPL", Interval.D1) is None  # no bars → no vector


@pytest.mark.asyncio
async def test_attribute_updates_do_not_publish_features_ready():
    publisher = NullPublisher()
    service = FeatureEngineService(
        FakeMarketDataClient(n=30),
        InMemoryFeatureStore(),
        publisher,
        min_bars=20,
        attributes=InMemoryAttributeStore(),
        fundamentals_client=FakeFundamentalsClient(payload()),
    )
    await service.handle_fundamentals_event(fundamentals_event().model_dump_json().encode())
    await service.handle_company_classified_event(classified_event().model_dump_json().encode())
    assert publisher.published == []  # no strategy re-evaluation on attribute refresh


@pytest.mark.asyncio
async def test_f_score_ranks_cross_sectionally():
    service = build_service()
    await service.compute_for_symbol("AAPL", Interval.D1)
    await service.compute_for_symbol("MSFT", Interval.D1)
    strong, weak = payload(f_score=8), payload(f_score=2)
    service._fundamentals = FakeFundamentalsClient(strong)
    await service.handle_fundamentals_event(fundamentals_event("AAPL").model_dump_json().encode())
    service._fundamentals = FakeFundamentalsClient(weak)
    await service.handle_fundamentals_event(fundamentals_event("MSFT").model_dump_json().encode())

    ranked = {fv.symbol: fv for fv in await service.ranked_universe(Interval.D1)}
    assert ranked["AAPL"].features["f_score"] > ranked["MSFT"].features["f_score"]
    assert 0.0 <= ranked["MSFT"].features["f_score"] <= 1.0  # percentile, not raw score


@pytest.mark.asyncio
async def test_service_without_attribute_store_unchanged():
    service = FeatureEngineService(
        FakeMarketDataClient(n=30), InMemoryFeatureStore(), NullPublisher(), min_bars=20
    )
    await service.compute_for_symbol("AAPL", Interval.D1)
    await service.handle_fundamentals_event(fundamentals_event().model_dump_json().encode())
    await service.handle_company_classified_event(classified_event().model_dump_json().encode())
    fv = await service.get_features("AAPL", Interval.D1)
    assert "f_score" not in fv.features  # enrichment disabled, technicals intact
