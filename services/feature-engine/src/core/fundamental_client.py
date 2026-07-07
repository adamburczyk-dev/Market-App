"""HTTP client for fundamental-data (events in, queries over HTTP).

``FundamentalsUpdatedEvent`` only announces *that* a symbol has fresh
fundamentals — the payload (statement + F-score) is queried back here.
A 404 (symbol unknown) returns None — redelivery won't help; transport
errors raise so the subscriber NAKs and retries the event.
"""

from typing import Any, Protocol

import httpx
import structlog

logger = structlog.get_logger()


class FundamentalsClient(Protocol):
    async def get_fundamentals(self, symbol: str) -> dict[str, Any] | None: ...

    async def aclose(self) -> None: ...


class HttpFundamentalsClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)

    async def get_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        resp = await self._client.get(f"/api/v1/fundamental-data/fundamentals/{symbol.upper()}")
        if resp.status_code == 404:
            logger.warning("No fundamentals stored for symbol", symbol=symbol)
            return None
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    async def aclose(self) -> None:
        await self._client.aclose()
