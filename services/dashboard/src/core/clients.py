"""Upstream HTTP sources the dashboard aggregates.

A backend-for-frontend: read-only GETs against the other services' APIs. Every
call degrades gracefully — a failed or unreachable upstream yields ``None`` rather
than failing the whole overview, so the dashboard reflects partial availability.
"""

from typing import Any, Protocol

import httpx
import structlog

logger = structlog.get_logger()


class DashboardSource(Protocol):
    async def risk_portfolio(self) -> dict | None: ...
    async def circuit_breaker(self) -> dict | None: ...
    async def execution_portfolio(self) -> dict | None: ...
    async def positions(self) -> dict | None: ...
    async def recent_alerts(self) -> dict | None: ...
    async def models(self) -> dict | None: ...
    async def aclose(self) -> None: ...


class HttpDashboardSource:
    def __init__(
        self,
        risk_url: str,
        execution_url: str,
        notification_url: str,
        ml_url: str,
        timeout_s: float = 5.0,
    ) -> None:
        self._risk = risk_url.rstrip("/")
        self._execution = execution_url.rstrip("/")
        self._notification = notification_url.rstrip("/")
        self._ml = ml_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def _get(self, url: str) -> Any | None:
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Upstream unavailable", url=url, error=str(exc))
            return None

    async def risk_portfolio(self) -> dict | None:
        return await self._get(f"{self._risk}/api/v1/risk-mgmt/portfolio")

    async def circuit_breaker(self) -> dict | None:
        return await self._get(f"{self._risk}/api/v1/risk-mgmt/circuit-breaker")

    async def execution_portfolio(self) -> dict | None:
        return await self._get(f"{self._execution}/api/v1/execution/portfolio")

    async def positions(self) -> dict | None:
        return await self._get(f"{self._execution}/api/v1/execution/positions")

    async def recent_alerts(self) -> dict | None:
        return await self._get(f"{self._notification}/api/v1/notification/alerts/recent")

    async def models(self) -> dict | None:
        return await self._get(f"{self._ml}/api/v1/ml-pipeline/models")

    async def aclose(self) -> None:
        await self._client.aclose()
