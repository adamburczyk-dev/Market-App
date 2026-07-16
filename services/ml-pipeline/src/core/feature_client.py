"""HTTP client for the RANKED feature vectors from feature-engine.

Serving consumes exactly what the model was trained on: the cross-sectional
rank transform of the merged (Tier-1 + Tier-2) vector. 404 → None (symbol not
in the universe yet); transport errors raise so the subscriber NAKs.
"""

from typing import Protocol

import httpx
import structlog
from trading_common.schemas import FeatureVector, Interval

logger = structlog.get_logger()


class FeatureClient(Protocol):
    async def get_ranked(self, symbol: str, interval: Interval) -> FeatureVector | None: ...

    async def aclose(self) -> None: ...


class HttpFeatureClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def get_ranked(self, symbol: str, interval: Interval) -> FeatureVector | None:
        resp = await self._client.get(
            f"{self._base}/api/v1/feature-engine/ranked/{symbol}",
            params={"interval": interval.value},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return FeatureVector.model_validate(resp.json())

    async def aclose(self) -> None:
        await self._client.aclose()
