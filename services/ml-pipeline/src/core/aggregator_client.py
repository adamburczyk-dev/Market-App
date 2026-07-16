"""HTTP client feeding realized ML outcomes to the signal-aggregator.

``POST /outcomes`` drives the AdaptiveWeightOptimizer — the aggregator's
"ml" weight learns from REALIZED triple-barrier results, not hopes. Failures
degrade gracefully (the outcome still counts for local decay metrics; the
weight update is retried implicitly by the next day's resolutions).
"""

from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger()


class AggregatorClient(Protocol):
    async def record_outcome(self, source: str, daily_return: float) -> bool: ...

    async def aclose(self) -> None: ...


class NullAggregatorClient:
    async def record_outcome(self, source: str, daily_return: float) -> bool:
        return False

    async def aclose(self) -> None:
        return None


class HttpAggregatorClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def record_outcome(self, source: str, daily_return: float) -> bool:
        try:
            resp = await self._client.post(
                f"{self._base}/api/v1/signal-aggregator/outcomes",
                json={"source": source, "daily_return": daily_return},
            )
            resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("Outcome push to aggregator failed", error=str(exc))
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
