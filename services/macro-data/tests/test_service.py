"""Tests for MacroDataService — refresh, classify, publish."""

import pytest
from trading_common.events import EventType
from trading_common.schemas import MacroRegime

from src.events.publisher import NullPublisher

from .conftest import FakeFetcher, build_service


@pytest.mark.asyncio
async def test_refresh_classifies_and_stores_snapshot():
    service = build_service()
    snap = await service.refresh(
        {"yield_curve_10y_2y": 1.5, "credit_spread_baa_10y": 1.2, "pmi": 57}
    )
    assert snap.regime == MacroRegime.EXPANSION
    assert service.snapshot is snap
    assert service.regime == MacroRegime.EXPANSION


@pytest.mark.asyncio
async def test_refresh_publishes_macro_updated_always():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.refresh({"pmi": 57})
    assert len(publisher.published) == 1
    assert publisher.published[0].event_type == EventType.MACRO_UPDATED
    assert publisher.published[0].regime == "expansion"


@pytest.mark.asyncio
async def test_no_regime_change_event_on_baseline():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.refresh({"pmi": 57})  # first classification → no change event
    assert all(e.event_type == EventType.MACRO_UPDATED for e in publisher.published)


@pytest.mark.asyncio
async def test_regime_change_publishes_event():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.refresh({"yield_curve_10y_2y": 1.5, "credit_spread_baa_10y": 1.2, "pmi": 57})
    await service.refresh({"credit_spread_baa_10y": 3.5})  # → crisis
    changes = [e for e in publisher.published if e.event_type == EventType.REGIME_CHANGED]
    assert len(changes) == 1
    assert changes[0].old_regime == "expansion"
    assert changes[0].new_regime == "crisis"


@pytest.mark.asyncio
async def test_no_change_event_when_regime_stable():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.refresh({"pmi": 57})
    await service.refresh({"pmi": 58})  # still expansion
    assert not any(e.event_type == EventType.REGIME_CHANGED for e in publisher.published)


@pytest.mark.asyncio
async def test_overrides_take_precedence_over_fetched():
    # fetcher reports a benign curve; override forces a deep inversion + weak pmi → crisis
    fetcher = FakeFetcher({"yield_curve_10y_2y": 1.0, "credit_spread_baa_10y": 1.0})
    service = build_service(fetcher=fetcher)
    snap = await service.refresh({"yield_curve_10y_2y": -0.8, "pmi": 44})
    assert snap.regime == MacroRegime.CRISIS
    assert snap.yield_curve_10y_2y == -0.8


@pytest.mark.asyncio
async def test_fetched_values_used_when_no_override():
    fetcher = FakeFetcher({"yield_curve_10y_2y": 1.5, "credit_spread_baa_10y": 1.2})
    service = build_service(fetcher=fetcher)
    # only pmi supplied manually; curve/spread come from the fetcher
    snap = await service.refresh({"pmi": 57})
    assert snap.regime == MacroRegime.EXPANSION
    assert snap.credit_spread_baa_10y == 1.2


@pytest.mark.asyncio
async def test_none_overrides_do_not_wipe_fetched():
    fetcher = FakeFetcher({"pmi": 57})
    service = build_service(fetcher=fetcher)
    # route always sends all keys; None ones must not clobber the fetched pmi
    snap = await service.refresh(
        {"pmi": None, "yield_curve_10y_2y": None, "credit_spread_baa_10y": None}
    )
    assert snap.pmi == 57
    assert snap.regime == MacroRegime.EXPANSION
