"""HTTP client for the fundamentals PANEL (events for news, HTTP for queries).

Training fetches the whole history once and does the as-of join locally, using
the same rule fundamental-data applies in SQL
(``trading_common.fundamentals.latest_available_before``). One request beats
symbols x sessions round trips, and having both sides call the same shared
function is what keeps the two from drifting into different answers.
"""

from typing import Protocol

import httpx
import structlog
from trading_common.schemas import FinancialStatements

logger = structlog.get_logger()


class FundamentalsClient(Protocol):
    async def panel(self, symbols: list[str]) -> dict[str, list[FinancialStatements]]: ...

    async def aclose(self) -> None: ...


class HttpFundamentalsClient:
    def __init__(self, base_url: str, timeout_s: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def panel(self, symbols: list[str]) -> dict[str, list[FinancialStatements]]:
        """symbol → every stored period. Degrades to {} rather than failing the run.

        A missing panel is a legitimate state (no database, no EDGAR access) and
        the caller reports the resulting coverage; an exception here would turn a
        data gap into a crashed training run.
        """
        if not symbols:
            return {}
        url = f"{self._base}/api/v1/fundamental-data/panel"
        try:
            resp = await self._client.get(url, params={"symbols": ",".join(symbols)})
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Fundamentals panel unavailable", error=str(exc))
            return {}

        out: dict[str, list[FinancialStatements]] = {}
        for raw in body.get("statements", []):
            statement = FinancialStatements.model_validate(raw)
            out.setdefault(statement.symbol.upper(), []).append(statement)
        undated = int(body.get("rows_without_filed_at", 0))
        if undated:
            # Not an error, but it must be said: an undated row is invisible to
            # every point-in-time read, so it is stored history that cannot be
            # used and would otherwise look like plain missing data.
            logger.warning("Panel rows without filed_at are unusable", rows=undated)
        logger.info(
            "Fetched fundamentals panel",
            symbols=len(out),
            rows=sum(len(v) for v in out.values()),
        )
        return out

    async def aclose(self) -> None:
        await self._client.aclose()
