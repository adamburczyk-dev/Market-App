"""FundamentalDataService — assemble fundamentals, score them, publish updates."""

import asyncio
from collections.abc import Sequence

import structlog
from trading_common.events import FundamentalsUpdatedEvent
from trading_common.schemas import FinancialStatements

from src.core.edgar_client import FundamentalsFetcher
from src.core.piotroski import FScoreBreakdown, compute_f_score
from src.events.publisher import Publisher

logger = structlog.get_logger()


class FundamentalDataService:
    def __init__(self, fetcher: FundamentalsFetcher, publisher: Publisher) -> None:
        self._fetcher = fetcher
        self._publisher = publisher
        # latest scored statement + its F-score breakdown, per symbol
        self._latest: dict[str, tuple[FinancialStatements, FScoreBreakdown]] = {}

    def get(self, symbol: str) -> tuple[FinancialStatements, FScoreBreakdown] | None:
        return self._latest.get(symbol.upper())

    def symbols(self) -> list[str]:
        return sorted(self._latest)

    async def _process(
        self, current: FinancialStatements, prior: FinancialStatements | None
    ) -> tuple[FinancialStatements, FScoreBreakdown]:
        breakdown = compute_f_score(current, prior)
        scored = current.model_copy(update={"piotroski_f_score": breakdown.score})
        self._latest[scored.symbol.upper()] = (scored, breakdown)
        await self._publisher.publish(
            FundamentalsUpdatedEvent(
                symbol=scored.symbol,
                period_end=scored.period_end.isoformat(),
                fiscal_period=scored.fiscal_period,
            )
        )
        logger.info(
            "Fundamentals updated",
            symbol=scored.symbol,
            period_end=scored.period_end.isoformat(),
            f_score=breakdown.score,
        )
        return scored, breakdown

    async def refresh(self, symbol: str) -> tuple[FinancialStatements, FScoreBreakdown] | None:
        """Pull the latest two annual filings from EDGAR, score, and publish."""
        statements = await self._fetcher.latest_statements(symbol, count=2)
        if not statements:
            logger.warning("No fundamentals available", symbol=symbol)
            return None
        current = statements[0]
        prior = statements[1] if len(statements) > 1 else None
        return await self._process(current, prior)

    async def ingest(
        self, current: FinancialStatements, prior: FinancialStatements | None = None
    ) -> tuple[FinancialStatements, FScoreBreakdown]:
        """Score and publish manually-provided statements (no SEC access required)."""
        return await self._process(current, prior)

    async def refresh_universe(self, symbols: Sequence[str], pause_s: float = 1.0) -> int:
        """Refresh each symbol from EDGAR (scheduled path); returns the refreshed count.

        ``pause_s`` spaces the per-symbol fetches out of politeness to SEC's
        rate limits. A symbol without data is skipped (already logged by
        ``refresh``); transport errors degrade to "no data" inside the fetcher.
        """
        refreshed = 0
        for i, symbol in enumerate(symbols):
            if i and pause_s > 0:
                await asyncio.sleep(pause_s)
            if await self.refresh(symbol) is not None:
                refreshed += 1
        logger.info("Universe refresh finished", requested=len(symbols), refreshed=refreshed)
        return refreshed
