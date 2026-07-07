from collections.abc import AsyncIterator
from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient
from trading_common.schemas import FinancialStatements

from src.api.deps import get_service
from src.core.service import FundamentalDataService
from src.events.publisher import NullPublisher


def stmt(
    period_end: date,
    revenue: float | None = None,
    net_income: float | None = None,
    total_assets: float | None = None,
    total_liabilities: float | None = None,
    operating_cash_flow: float | None = None,
    current_assets: float | None = None,
    current_liabilities: float | None = None,
    shares_outstanding: float | None = None,
    symbol: str = "AAPL",
) -> FinancialStatements:
    return FinancialStatements(
        symbol=symbol,
        period_end=period_end,
        fiscal_period="FY",
        revenue=revenue,
        net_income=net_income,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        operating_cash_flow=operating_cash_flow,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        shares_outstanding=shares_outstanding,
    )


def improving_pair() -> tuple[FinancialStatements, FinancialStatements]:
    """A clearly-improving, profitable company → all 9 signals pass (score 9)."""
    # current ratio 2.5 vs 1.6; buyback 95 vs 100 shares
    current = stmt(date(2024, 12, 31), 1200, 200, 1000, 400, 250, 500, 200, 95)
    prior = stmt(date(2023, 12, 31), 1000, 100, 1000, 500, 120, 400, 250, 100)
    return current, prior


def deteriorating_pair() -> tuple[FinancialStatements, FinancialStatements]:
    """A loss-making, deteriorating company → all 9 signals fail (score 0)."""
    # current ratio 1.0 vs 1.6; dilution 110 vs 100 shares
    current = stmt(date(2024, 12, 31), 800, -50, 1000, 600, -60, 300, 300, 110)
    prior = stmt(date(2023, 12, 31), 1000, 100, 1000, 400, 120, 400, 250, 100)
    return current, prior


class FakeFetcher:
    """FundamentalsFetcher double — returns configured statements."""

    def __init__(
        self, statements: list[FinancialStatements] | None = None, enabled: bool = False
    ) -> None:
        self._statements = statements or []
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def latest_statements(self, symbol: str, count: int = 2) -> list[FinancialStatements]:
        return self._statements[:count]

    async def aclose(self) -> None:
        return None


def build_service(fetcher: FakeFetcher | None = None, publisher=None):  # type: ignore[no-untyped-def]
    return FundamentalDataService(fetcher or FakeFetcher(), publisher or NullPublisher())


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, FundamentalDataService]]:
    from src.main import app

    service = build_service(publisher=NullPublisher())
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, service
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
