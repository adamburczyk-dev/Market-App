"""Sector lookup from company-classifier (HTTP query, graceful degradation).

The aggregator enriches each actionable aggregate with the symbol's GICS-style
sector so risk-mgmt can apply regime-aware sector caps (R8). A missing profile,
a down classifier, or a profile without a sector all yield None — sizing then
skips the sector gate rather than blocking the order.
"""

from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger()


class CompanyClient(Protocol):
    async def get_sector(self, symbol: str) -> str | None: ...

    async def aclose(self) -> None: ...


class NullCompanyClient:
    """No classifier configured — every symbol is unclassified."""

    async def get_sector(self, symbol: str) -> str | None:
        return None

    async def aclose(self) -> None:
        return None


class HttpCompanyClient:
    """Query company-classifier's stored profile; cache positive answers.

    Sectors effectively never change intraday, so a per-symbol in-memory cache
    avoids one HTTP round-trip per aggregation. Negative answers are NOT cached
    — the symbol may simply not be classified yet.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)
        self._cache: dict[str, str] = {}

    async def get_sector(self, symbol: str) -> str | None:
        key = symbol.upper()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            resp = await self._client.get(f"/api/v1/company-classifier/companies/{key}")
            resp.raise_for_status()
            sector = resp.json().get("profile", {}).get("sector")
        except httpx.HTTPError as exc:
            logger.warning("Sector lookup failed", symbol=key, error=str(exc))
            return None
        if isinstance(sector, str) and sector:
            self._cache[key] = sector
            return sector
        return None

    async def aclose(self) -> None:
        await self._client.aclose()
