"""Upstream HTTP sources the dashboard aggregates.

A backend-for-frontend: read-only GETs against the other services' APIs. Every
call degrades gracefully — a failed or unreachable upstream yields ``None`` rather
than failing the whole overview, so the dashboard reflects partial availability.

The health fan-out is the one call that measures rather than reads: it times
each service's ``/health`` so the System Health section can report latency, and
a timeout there is a RESULT, not an error to swallow silently.
"""

import time
from typing import Any, Protocol

import httpx
import structlog

logger = structlog.get_logger()


def _decode(resp: httpx.Response) -> dict[str, Any]:
    """Body as a dict, KEEPING the text when it is not JSON.

    An unhandled upstream error answers with Starlette's plain-text
    "Internal Server Error", and calling `.json()` on that raises a
    JSONDecodeError — which is not an `httpx.HTTPError`, so it escapes the
    error handling entirely and turns an upstream 500 into a 500 from THIS
    service. Discarding a non-JSON body is also how a real failure once became
    the literal report `HTTP 500: {}`.
    """
    if not resp.content:
        return {"detail": f"upstream {resp.status_code} with an empty body"}
    try:
        body = resp.json()
    except ValueError:
        return {"detail": resp.text[:500]}
    return body if isinstance(body, dict) else {"detail": str(body)[:500]}


class DashboardSource(Protocol):
    async def risk_portfolio(self) -> dict | None: ...
    async def circuit_breaker(self) -> dict | None: ...
    async def execution_portfolio(self) -> dict | None: ...
    async def positions(self) -> dict | None: ...
    async def equity_curve(self) -> dict | None: ...
    async def recent_alerts(self) -> dict | None: ...
    async def models(self) -> dict | None: ...
    async def ml_runs(self) -> dict | None: ...
    async def ml_run(self, operation: str) -> dict | None: ...
    async def ml_serving(self) -> dict | None: ...
    async def strategies(self) -> dict | None: ...
    async def signal_weights(self) -> dict | None: ...
    async def ohlcv(self, symbol: str, limit: int) -> list[dict] | None: ...
    async def health_all(self) -> dict[str, dict[str, Any]]: ...
    async def run_backtest(self, strategy: str, symbol: str, limit: int) -> tuple[int, Any]: ...
    async def aclose(self) -> None: ...


class HttpDashboardSource:
    def __init__(
        self,
        risk_url: str,
        execution_url: str,
        notification_url: str,
        ml_url: str,
        strategy_url: str = "",
        aggregator_url: str = "",
        market_data_url: str = "",
        backtest_url: str = "",
        health_urls: dict[str, str] | None = None,
        timeout_s: float = 5.0,
        health_timeout_s: float = 2.0,
    ) -> None:
        self._risk = risk_url.rstrip("/")
        self._execution = execution_url.rstrip("/")
        self._notification = notification_url.rstrip("/")
        self._ml = ml_url.rstrip("/")
        self._strategy = strategy_url.rstrip("/")
        self._aggregator = aggregator_url.rstrip("/")
        self._market_data = market_data_url.rstrip("/")
        self._backtest = backtest_url.rstrip("/")
        self._health_urls = health_urls or {}
        # A separate, shorter budget: a health probe that takes as long as a
        # data query is not reporting health, it is reporting a hang.
        self._health_timeout_s = health_timeout_s
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def _get(self, url: str) -> Any | None:
        if not url:
            return None
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

    async def equity_curve(self) -> dict | None:
        return await self._get(f"{self._execution}/api/v1/execution/equity?limit=500")

    async def recent_alerts(self) -> dict | None:
        return await self._get(f"{self._notification}/api/v1/notification/alerts/recent")

    async def models(self) -> dict | None:
        return await self._get(f"{self._ml}/api/v1/ml-pipeline/models")

    async def ml_runs(self) -> dict | None:
        return await self._get(f"{self._ml}/api/v1/ml-pipeline/runs")

    async def ml_run(self, operation: str) -> dict | None:
        """The full payload of one completed run.

        404 here means "that operation has not finished in this container",
        which `_get` turns into None — the same value an unreachable
        ml-pipeline yields. The section that reads this distinguishes the two
        by whether ANY ml-pipeline call answered, so a stopped service is never
        reported as a measurement that came back empty.
        """
        return await self._get(f"{self._ml}/api/v1/ml-pipeline/runs/{operation}")

    async def ml_serving(self) -> dict | None:
        return await self._get(f"{self._ml}/api/v1/ml-pipeline/serving")

    async def strategies(self) -> dict | None:
        return await self._get(f"{self._strategy}/api/v1/strategy/status")

    async def signal_weights(self) -> dict | None:
        return await self._get(f"{self._aggregator}/api/v1/signal-aggregator/weights")

    async def ohlcv(self, symbol: str, limit: int = 120) -> list[dict] | None:
        data = await self._get(
            f"{self._market_data}/api/v1/market-data/ohlcv/{symbol}?interval=1d&limit={limit}"
        )
        return data if isinstance(data, list) else None

    async def health_all(self) -> dict[str, dict[str, Any]]:
        """Probe every configured service's /health, timing each one.

        Latency is measured around the request, so a service that answers
        slowly is visibly different from one that is down — the two look
        identical in a plain up/down check and mean very different things.
        """
        results: dict[str, dict[str, Any]] = {}
        for name, base in self._health_urls.items():
            started = time.perf_counter()
            try:
                resp = await self._client.get(
                    f"{base.rstrip('/')}/health", timeout=self._health_timeout_s
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                results[name] = {
                    "status": "up" if resp.status_code == 200 else "degraded",
                    "http_status": resp.status_code,
                    "latency_ms": round(elapsed_ms, 1),
                }
            except httpx.HTTPError as exc:
                results[name] = {
                    "status": "down",
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "error": type(exc).__name__,
                }
        return results

    async def run_backtest(self, strategy: str, symbol: str, limit: int = 500) -> tuple[int, Any]:
        """Run a backtest on demand and pass BACKTEST'S OWN STATUS back.

        Deliberately not swallowed into ``None`` like the read paths: backtest
        answers 404 for an unknown strategy and 422 for one that needs a
        universe, and collapsing both into "unavailable" would tell the user
        the service is down when it is in fact explaining what they asked for.
        """
        if not self._backtest:
            return 503, {"detail": "backtest URL not configured"}
        try:
            resp = await self._client.post(
                f"{self._backtest}/api/v1/backtest/run",
                json={"strategy_name": strategy, "symbol": symbol, "limit": limit},
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            logger.warning("Backtest unreachable", error=str(exc))
            return 502, {"detail": f"backtest unreachable: {type(exc).__name__}"}
        return resp.status_code, _decode(resp)

    async def aclose(self) -> None:
        await self._client.aclose()
