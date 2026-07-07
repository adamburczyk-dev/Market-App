"""Tests for R8 — sector enrichment from company-classifier.

The aggregator attaches the symbol's sector to every aggregate so risk-mgmt
can apply regime-aware sector caps. Lookup failures degrade to None (sizing
then skips the sector gate).
"""

import httpx
import pytest

from src.core.company_client import HttpCompanyClient, NullCompanyClient
from src.events.publisher import NullPublisher

from .conftest import build_service
from .test_live_events import signal_event

# --- HttpCompanyClient ---


def make_client(handler) -> tuple[HttpCompanyClient, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(counting), base_url="http://classifier:8000"
    )
    return HttpCompanyClient("http://classifier:8000", client=http), calls


@pytest.mark.asyncio
async def test_sector_from_stored_profile():
    client, _ = make_client(
        lambda req: httpx.Response(
            200, json={"profile": {"symbol": "AAPL", "sector": "Information Technology"}}
        )
    )
    assert await client.get_sector("aapl") == "Information Technology"  # case-normalized


@pytest.mark.asyncio
async def test_unclassified_symbol_is_none():
    client, _ = make_client(lambda req: httpx.Response(404, json={"detail": "not classified"}))
    assert await client.get_sector("GHOST") is None


@pytest.mark.asyncio
async def test_profile_without_sector_is_none():
    client, _ = make_client(lambda req: httpx.Response(200, json={"profile": {"symbol": "X"}}))
    assert await client.get_sector("X") is None


@pytest.mark.asyncio
async def test_connection_error_degrades_to_none():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("classifier down")

    client, _ = make_client(boom)
    assert await client.get_sector("AAPL") is None


@pytest.mark.asyncio
async def test_positive_answers_are_cached():
    client, calls = make_client(
        lambda req: httpx.Response(200, json={"profile": {"sector": "Utilities"}})
    )
    assert await client.get_sector("NEE") == "Utilities"
    assert await client.get_sector("NEE") == "Utilities"
    assert len(calls) == 1  # second lookup served from cache


@pytest.mark.asyncio
async def test_negative_answers_are_not_cached():
    responses = iter(
        [
            httpx.Response(404, json={"detail": "not classified"}),
            httpx.Response(200, json={"profile": {"sector": "Health Care"}}),
        ]
    )
    client, calls = make_client(lambda req: next(responses))
    assert await client.get_sector("JNJ") is None
    assert await client.get_sector("JNJ") == "Health Care"  # re-queried once classified
    assert len(calls) == 2


# --- live-path enrichment ---


class FakeCompanyClient:
    def __init__(self, sector: str | None) -> None:
        self.sector = sector

    async def get_sector(self, symbol: str) -> str | None:
        return self.sector

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_aggregated_event_carries_sector():
    publisher = NullPublisher()
    service = build_service(
        publisher=publisher, company_client=FakeCompanyClient("Information Technology")
    )
    await service.handle_signal_generated(signal_event().model_dump_json().encode())
    event = publisher.published[-1]
    assert event.final_signal == "BUY"
    assert event.sector == "Information Technology"


@pytest.mark.asyncio
async def test_no_client_means_no_sector():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)  # company_client omitted
    await service.handle_signal_generated(signal_event().model_dump_json().encode())
    assert publisher.published[-1].sector is None


@pytest.mark.asyncio
async def test_null_client_means_no_sector():
    publisher = NullPublisher()
    service = build_service(publisher=publisher, company_client=NullCompanyClient())
    await service.handle_signal_generated(signal_event().model_dump_json().encode())
    assert publisher.published[-1].sector is None
