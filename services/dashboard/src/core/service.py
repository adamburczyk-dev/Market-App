"""DashboardService — aggregate upstream service state into one overview."""

import asyncio
from typing import Any

import structlog

from src.core.clients import DashboardSource

logger = structlog.get_logger()


class DashboardService:
    def __init__(self, source: DashboardSource) -> None:
        self._source = source

    async def overview(self) -> dict[str, Any]:
        """Fan out to every upstream concurrently; compose a single view.

        Each upstream is independent — a missing one is reported in ``sources``
        as "unavailable" while the rest of the overview still renders.
        """
        risk_portfolio, breaker, exec_portfolio, positions, alerts, models = await asyncio.gather(
            self._source.risk_portfolio(),
            self._source.circuit_breaker(),
            self._source.execution_portfolio(),
            self._source.positions(),
            self._source.recent_alerts(),
            self._source.models(),
        )

        sources = {
            "risk-mgmt": _status(risk_portfolio is not None and breaker is not None),
            "execution": _status(exec_portfolio is not None and positions is not None),
            "notification": _status(alerts is not None),
            "ml-pipeline": _status(models is not None),
        }

        return {
            "portfolio": risk_portfolio,
            "circuit_breaker": breaker,
            "execution": exec_portfolio,
            "positions": (positions or {}).get("positions", {}),
            "recent_alerts": (alerts or {}).get("alerts", []),
            "models": (models or {}).get("models", []),
            "sources": sources,
        }


def _status(ok: bool) -> str:
    return "ok" if ok else "unavailable"
