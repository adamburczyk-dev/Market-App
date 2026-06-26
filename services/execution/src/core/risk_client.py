"""HTTP client for pushing portfolio state back to risk-mgmt."""

from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger()


class RiskClient(Protocol):
    async def push_portfolio(self, metrics: dict) -> None: ...


class NullRiskClient:
    """No-op client. Used in tests and when risk-mgmt push is disabled."""

    def __init__(self) -> None:
        self.pushed: list[dict] = []

    async def push_portfolio(self, metrics: dict) -> None:
        self.pushed.append(metrics)


class HttpRiskClient:
    """Best-effort POST of portfolio metrics to risk-mgmt /portfolio."""

    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self._url = base_url.rstrip("/") + "/api/v1/risk-mgmt/portfolio"
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def push_portfolio(self, metrics: dict) -> None:
        try:
            await self._client.post(self._url, json=metrics)
        except httpx.HTTPError as exc:
            logger.warning("Could not push portfolio to risk-mgmt", error=str(exc))

    async def aclose(self) -> None:
        await self._client.aclose()
