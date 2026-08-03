"""DashboardService — compose the six sections of the specified dashboard.

Sections (Plan_Rozwoju, Week 21): portfolio, risk, strategy attribution,
backtest, ML, system health. Each is assembled from upstream HTTP reads and
tolerates a missing source: an unreachable service costs its own section, never
the page.

**Derived risk statistics are computed here, from `trading_common.risk_metrics`.**
That is deliberate. execution owns the equity history but not the risk
semantics; risk-mgmt owns the limits and the breaker but has no series. Putting
the arithmetic in the shared module and calling it from the consumer that
displays it keeps exactly one definition of "drawdown" in the system, which is
the property that matters — a second one would eventually disagree about
whether a limit was breached.
"""

import asyncio
from typing import Any

import structlog
from trading_common.risk_metrics import (
    annualized_sharpe,
    average_pairwise_correlation,
    conditional_var,
    correlation_matrix,
    drawdown_series,
    historical_var,
    max_drawdown,
    returns_from_equity,
)

from src.core.clients import DashboardSource

logger = structlog.get_logger()

# How many held names to pull price history for when building the correlation
# grid. The matrix is O(n²) to read and to look at; beyond this it stops being
# something a human can use and starts being a load test on market-data.
MAX_CORRELATION_SYMBOLS = 12
CORRELATION_LOOKBACK = 120


class DashboardService:
    def __init__(self, source: DashboardSource) -> None:
        self._source = source

    async def overview(self) -> dict[str, Any]:
        """Fan out to every upstream concurrently; compose a single view.

        Each upstream is independent — a missing one is reported in ``sources``
        as "unavailable" while the rest of the overview still renders.
        """
        (
            risk_portfolio,
            breaker,
            exec_portfolio,
            positions,
            alerts,
            models,
        ) = await asyncio.gather(
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

    # --- section 1: portfolio overview ------------------------------------

    async def portfolio_section(self) -> dict[str, Any]:
        """Equity curve, P&L and positions — the curve is the new part.

        Nothing in the system retained an equity series until execution started
        recording one per session, so this section could previously show a
        number but never a shape.
        """
        curve, exec_portfolio, positions = await asyncio.gather(
            self._source.equity_curve(),
            self._source.execution_portfolio(),
            self._source.positions(),
        )
        points = (curve or {}).get("points", [])
        equity = [float(p["equity"]) for p in points]
        returns = returns_from_equity(equity)

        # Both ends or neither: a P&L computed against a missing baseline is
        # not a smaller number, it is a different quantity.
        pnl_abs: float | None = None
        pnl_pct: float | None = None
        if equity and equity[0] > 0:
            pnl_abs = equity[-1] - equity[0]
            pnl_pct = equity[-1] / equity[0] - 1.0

        return {
            "curve": points,
            "labels": [p["date"] for p in points],
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "sessions": len(points),
            "sharpe": annualized_sharpe(returns),
            "broker": exec_portfolio,
            "positions": (positions or {}).get("positions", {}),
            "available": bool(points),
        }

    # --- section 2: risk metrics ------------------------------------------

    async def risk_section(self) -> dict[str, Any]:
        """VaR, drawdown path and the correlation grid of what is actually held.

        Every statistic can come back ``None``: `trading_common.risk_metrics`
        refuses a sample too small to support it, and a chart that draws a VaR
        from twelve observations is worse than a chart that says it cannot.
        """
        curve, breaker, risk_portfolio, positions = await asyncio.gather(
            self._source.equity_curve(),
            self._source.circuit_breaker(),
            self._source.risk_portfolio(),
            self._source.positions(),
        )
        points = (curve or {}).get("points", [])
        equity = [float(p["equity"]) for p in points]
        returns = returns_from_equity(equity)

        held = sorted((positions or {}).get("positions", {}))[:MAX_CORRELATION_SYMBOLS]
        matrix = await self._correlations(held)

        return {
            "var_95": historical_var(returns, 0.95),
            "cvar_95": conditional_var(returns, 0.95),
            "max_drawdown": max_drawdown(equity) if equity else None,
            "drawdown_curve": drawdown_series(equity),
            "labels": [p["date"] for p in points],
            "samples": len(returns),
            "correlation": matrix.as_dict(),
            "avg_correlation": average_pairwise_correlation(matrix),
            # An empty grid because market-data is unreachable looks exactly
            # like an empty grid because nothing is held. These two numbers are
            # what tells them apart, and "0 of 3 have price history" is a very
            # different message from "nothing to correlate".
            "held_symbols": held,
            "correlated_symbols": matrix.symbols,
            "circuit_breaker": breaker,
            "limits": risk_portfolio,
        }

    async def _correlations(self, symbols: list[str]) -> Any:
        """Correlation grid from market-data closes for the names held."""
        if not symbols:
            return correlation_matrix({})
        series = await asyncio.gather(
            *(self._source.ohlcv(s, CORRELATION_LOOKBACK) for s in symbols)
        )
        returns_by_symbol: dict[str, list[float]] = {}
        for symbol, bars in zip(symbols, series, strict=True):
            if not bars:
                continue
            # Adjusted close where the bar carries one — the same definition
            # `trading_common.prices` uses, so a dividend payer is not scored as
            # having dropped on the ex-date.
            closes = [float(b.get("adj_close") or b["close"]) for b in bars]
            returns_by_symbol[symbol] = returns_from_equity(closes)
        return correlation_matrix(returns_by_symbol)

    # --- section 3: strategy attribution ----------------------------------

    async def strategy_section(self) -> dict[str, Any]:
        """Per-strategy status and its learned weight in the decision.

        Only possible since the aggregator started keying signals by
        (symbol, strategy): before that every rule was one "strategy" source
        and there was nothing to attribute anything to.
        """
        statuses, weights = await asyncio.gather(
            self._source.strategies(), self._source.signal_weights()
        )
        rules = (statuses or {}).get("strategies", [])
        weight_map = (weights or {}).get("weights", {})

        rows = [
            {
                "name": rule["name"],
                "status": rule.get("status"),
                "required_features": rule.get("required_features", []),
                "required_ranks": rule.get("required_ranks", []),
                # The aggregator prefixes strategy sources; an unweighted rule
                # is reported as None rather than 0.0, which would read as
                # "measured and worthless" instead of "not yet measured".
                "weight": weight_map.get(f"strategy:{rule['name']}"),
            }
            for rule in rules
        ]
        non_strategy = {k: v for k, v in weight_map.items() if not k.startswith("strategy:")}
        return {
            "strategies": rows,
            "other_sources": non_strategy,
            "available": bool(rows),
        }

    # --- section 4: backtest results --------------------------------------

    async def backtest_section(
        self, strategy: str, symbol: str, limit: int = 500
    ) -> tuple[int, dict[str, Any]]:
        """Run a backtest on demand. NOT polled — this one costs real work.

        The upstream status is passed through rather than flattened: backtest
        answers 404 for an unknown strategy and 422 for one that reads
        cross-sectional ranks, and both are answers the user needs to read.
        """
        status, body = await self._source.run_backtest(strategy, symbol, limit)
        return status, body if isinstance(body, dict) else {"detail": str(body)}

    # --- section 5: ML model performance ----------------------------------

    async def ml_section(self) -> dict[str, Any]:
        models, runs, serving, trained, study = await asyncio.gather(
            self._source.models(),
            self._source.ml_runs(),
            self._source.ml_serving(),
            self._source.ml_run("train"),
            self._source.ml_run("feature-importance"),
        )
        return {
            "models": (models or {}).get("models", []),
            "runs": (runs or {}).get("runs", []),
            "serving": serving,
            "available": models is not None,
            "importance": _importance(trained, study, reachable=models is not None),
        }

    # --- section 6: system health -----------------------------------------

    async def health_section(self) -> dict[str, Any]:
        """Up/down and latency per service, plus how much of the stack answered."""
        probes = await self._source.health_all()
        up = sum(1 for v in probes.values() if v.get("status") == "up")
        latencies = [v["latency_ms"] for v in probes.values() if v.get("status") == "up"]
        return {
            "services": probes,
            "up": up,
            "total": len(probes),
            "slowest_ms": max(latencies) if latencies else None,
        }


def _importance(
    trained: dict[str, Any] | None, study: dict[str, Any] | None, reachable: bool
) -> dict[str, Any]:
    """The feature-importance table plus WHERE it came from.

    Two runs can produce one, and they are not interchangeable: the training
    run measures the model the system would actually serve, while the
    standalone study fits a diagnostic model carrying a planted noise column
    that serving cannot produce. The training run therefore wins when both
    exist, and the source travels with the table — an importance table read
    against the wrong model is worse than none.

    Every "no table" case names itself. "Nobody has measured this", "the model
    was trained with the measurement switched off" and "ml-pipeline is down"
    are three different situations, and a blank panel says the same nothing
    for all three.
    """
    from_training = ((trained or {}).get("result") or {}).get("gate", {}).get("importance")
    from_study = ((study or {}).get("result") or {}).get("importance")
    if from_training:
        return {
            "table": from_training,
            "source": "training run — the model this system would serve",
            "measured_at": (trained or {}).get("completed_at"),
        }
    if from_study:
        return {
            "table": from_study,
            "source": (
                "feature-importance study — a DIAGNOSTIC model with a planted noise "
                "column, not the served one"
            ),
            "measured_at": (study or {}).get("completed_at"),
        }
    if not reachable:
        return {"table": None, "source": None, "reason": "ml-pipeline unavailable"}
    if trained is not None:
        return {
            "table": None,
            "source": None,
            "reason": (
                "the last training run carries no importance table — it ran with the "
                "measurement switched off, or its holdout was too short to pair"
            ),
        }
    return {
        "table": None,
        "source": None,
        "reason": (
            "no training run or importance study has completed in this container — "
            "not measured, which is not the same as no feature mattering"
        ),
    }


def _status(ok: bool) -> str:
    return "ok" if ok else "unavailable"
