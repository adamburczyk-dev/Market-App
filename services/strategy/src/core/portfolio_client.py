"""HTTP client for reading live portfolio state from risk-mgmt."""

from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger()


class PortfolioClient(Protocol):
    async def get_portfolio(self) -> dict | None: ...


class HttpPortfolioClient:
    """Reads risk-mgmt's live portfolio; returns None (→ caller falls back) on error."""

    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self._url = base_url.rstrip("/") + "/api/v1/risk-mgmt/portfolio"
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def get_portfolio(self) -> dict | None:
        try:
            resp = await self._client.get(self._url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Could not fetch portfolio from risk-mgmt", error=str(exc))
            return None

    async def aclose(self) -> None:
        await self._client.aclose()
