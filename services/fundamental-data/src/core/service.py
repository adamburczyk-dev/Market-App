"""FundamentalDataService — assemble fundamentals, score them, publish updates."""

import asyncio
from collections.abc import Sequence
from datetime import date, datetime

import structlog
from trading_common.events import FundamentalsUpdatedEvent
from trading_common.fundamentals import session_cutoff
from trading_common.schemas import FinancialStatements

from src.core.edgar_client import FundamentalsFetcher
from src.core.piotroski import FScoreBreakdown, compute_f_score
from src.core.repository import FundamentalsStore, NullFundamentalsStore
from src.events.publisher import Publisher

logger = structlog.get_logger()


class FundamentalDataService:
    def __init__(
        self,
        fetcher: FundamentalsFetcher,
        publisher: Publisher,
        store: FundamentalsStore | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._publisher = publisher
        # latest scored statement + its F-score breakdown, per symbol. This is a
        # cache of "what do we know now"; the PANEL (store) is what answers
        # "what was known on date D", which is the only form training can use.
        self._latest: dict[str, tuple[FinancialStatements, FScoreBreakdown]] = {}
        self._store = store or NullFundamentalsStore()

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
        # Persist BOTH periods: the prior year is a panel row in its own right,
        # and a panel that only ever holds the newest filing cannot answer an
        # as-of question about last year.
        await self._store.save([s for s in (scored, prior) if s is not None])
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

    async def refresh_history(self, symbol: str, periods: int = 24) -> int:
        """Store EVERY available annual period, not just the newest two.

        `refresh` answers "what do we know now", which is the serving question,
        and it is the only path that ever wrote to the panel — so the panel
        could hold at most two years per symbol and a point-in-time join over
        twenty years was impossible by construction. The XBRL response already
        contains the full history; it was being sliced away.

        Each period is scored against its own predecessor (an F-Score compares
        consecutive years, so scoring 2012 against 2025 would be meaningless).
        Only the newest publishes an event: `fundamentals.updated` announces
        that current knowledge changed, and replaying twenty years of history
        would wake every downstream consumer twenty times for one symbol.
        """
        statements = await self._fetcher.latest_statements(symbol, count=periods)
        if not statements:
            logger.warning("No fundamentals available", symbol=symbol)
            return 0
        # newest first from the fetcher; score each against the year before it
        scored: list[FinancialStatements] = []
        for index, current in enumerate(statements):
            prior = statements[index + 1] if index + 1 < len(statements) else None
            breakdown = compute_f_score(current, prior)
            scored.append(current.model_copy(update={"piotroski_f_score": breakdown.score}))
            if index == 0:
                self._latest[current.symbol.upper()] = (scored[0], breakdown)
        await self._store.save(scored)
        await self._publisher.publish(
            FundamentalsUpdatedEvent(
                symbol=scored[0].symbol,
                period_end=scored[0].period_end.isoformat(),
                fiscal_period=scored[0].fiscal_period,
            )
        )
        logger.info(
            "Fundamentals history stored",
            symbol=scored[0].symbol,
            periods=len(scored),
            oldest=scored[-1].period_end.isoformat(),
            newest=scored[0].period_end.isoformat(),
        )
        return len(scored)

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

    async def available_before(self, symbol: str, cutoff: datetime) -> FinancialStatements | None:
        """Point-in-time read: the latest filing published strictly before `cutoff`."""
        return await self._store.available_before(symbol, cutoff)

    async def as_of_session(self, symbol: str, day: date) -> FinancialStatements | None:
        """What was knowable about `symbol` when session `day` opened."""
        return await self._store.available_before(symbol, session_cutoff(day))

    async def panel(self, symbols: Sequence[str]) -> list[FinancialStatements]:
        """The whole stored history for these symbols — training's single fetch."""
        return await self._store.panel([s.upper() for s in symbols])

    async def store_ready(self) -> bool:
        return await self._store.ping()

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
