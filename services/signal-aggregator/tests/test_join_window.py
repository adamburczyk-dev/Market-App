"""N2, aggregator half: components of one decision are coalesced, not raced.

`features.ready` fans out to strategy and ml-pipeline in parallel; the rule path
is a comparison and the ML path an inference, so strategy always arrives first.
Deciding per component published two decisions where the domain has one. The
window merges them; risk-mgmt's ledger is the second line of defence for a
component that arrives after the window closed.
"""

import asyncio

import pytest

from src.events.publisher import NullPublisher

from .conftest import build_service
from .test_live_events import ml_event, signal_event


@pytest.mark.asyncio
async def test_window_merges_components_into_one_decision():
    publisher = NullPublisher()
    service = build_service(publisher=publisher, join_window_s=0.05)
    await service.handle_signal_generated(signal_event(side="BUY").model_dump_json().encode())
    await service.handle_ml_signal(ml_event(side="BUY").model_dump_json().encode())
    assert publisher.published == []
    await service.drain_pending()
    assert len(publisher.published) == 1
    assert set(publisher.published[0].components_present) == {"strategy:momentum_rank", "ml"}


@pytest.mark.asyncio
async def test_component_after_the_window_decides_again():
    """A late component is new information — the aggregator re-decides, and
    risk-mgmt refuses the duplicate order (the ledger, not silence, is the
    guard). Suppressing it here would also suppress a genuine regime change."""
    publisher = NullPublisher()
    service = build_service(publisher=publisher, join_window_s=0.01)
    await service.handle_signal_generated(signal_event(side="BUY").model_dump_json().encode())
    await service.drain_pending()
    assert len(publisher.published) == 1

    await service.handle_ml_signal(ml_event(side="BUY").model_dump_json().encode())
    await service.drain_pending()
    assert len(publisher.published) == 2
    assert set(publisher.published[-1].components_present) == {"strategy:momentum_rank", "ml"}


@pytest.mark.asyncio
async def test_zero_window_decides_immediately():
    publisher = NullPublisher()
    service = build_service(publisher=publisher, join_window_s=0.0)
    await service.handle_signal_generated(signal_event(side="BUY").model_dump_json().encode())
    assert len(publisher.published) == 1


@pytest.mark.asyncio
async def test_repeated_components_share_one_pending_decision():
    publisher = NullPublisher()
    service = build_service(publisher=publisher, join_window_s=0.05)
    for _ in range(5):
        await service.handle_signal_generated(signal_event(side="BUY").model_dump_json().encode())
    await service.drain_pending()
    assert len(publisher.published) == 1


@pytest.mark.asyncio
async def test_failed_decision_clears_the_pending_slot():
    # One symbol blowing up must neither kill the subscriber loop nor wedge the
    # symbol permanently (a stuck pending entry would swallow every later event).
    publisher = NullPublisher()
    service = build_service(publisher=publisher, join_window_s=0.01)

    async def boom(symbol: str) -> None:
        raise RuntimeError("aggregation failed")

    service.aggregate_symbol = boom  # type: ignore[method-assign]
    await service.handle_signal_generated(signal_event(side="BUY").model_dump_json().encode())
    await service.drain_pending()
    assert service._pending == {}


@pytest.mark.asyncio
async def test_drain_pending_is_a_noop_without_work():
    service = build_service(join_window_s=0.05)
    await asyncio.wait_for(service.drain_pending(), timeout=1.0)
