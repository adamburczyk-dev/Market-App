"""HTTP client for querying features from the feature-engine service."""

from typing import Protocol

import httpx
import structlog
from trading_common.schemas import FeatureVector, Interval

logger = structlog.get_logger()


class FeatureClient(Protocol):
    async def get_features(self, symbol: str, interval: Interval) -> FeatureVector | None: ...
    async def get_ranked(self, symbol: str, interval: Interval) -> FeatureVector | None: ...


class HttpFeatureClient:
    """Queries feature-engine over HTTP for raw and rank-transformed features."""

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def _get(self, path: str, interval: Interval) -> FeatureVector | None:
        resp = await self._client.get(f"{self._base}{path}", params={"interval": interval.value})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return FeatureVector.model_validate(resp.json())

    async def get_features(self, symbol: str, interval: Interval) -> FeatureVector | None:
        return await self._get(f"/api/v1/feature-engine/features/{symbol}", interval)

    async def get_ranked(self, symbol: str, interval: Interval) -> FeatureVector | None:
        return await self._get(f"/api/v1/feature-engine/ranked/{symbol}", interval)

    async def aclose(self) -> None:
        await self._client.aclose()
