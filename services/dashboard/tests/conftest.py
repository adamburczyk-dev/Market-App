from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_service
from src.core.service import DashboardService

# 25 sessions: enough for the VaR floor (20) so the risk section has something
# to measure, with a real drawdown in the middle.
_EQUITY = [
    100_000.0,
    101_000.0,
    102_500.0,
    101_800.0,
    103_000.0,
    104_200.0,
    103_100.0,
    99_800.0,
    97_500.0,
    98_900.0,
    100_400.0,
    101_100.0,
    102_900.0,
    104_800.0,
    103_700.0,
    105_100.0,
    106_300.0,
    105_200.0,
    107_000.0,
    108_400.0,
    107_100.0,
    109_000.0,
    110_200.0,
    109_400.0,
    111_000.0,
]


def _row(name: str, drop: float, t: float, twin: str | None = None) -> dict:
    return {
        "feature": name,
        "members": [name],
        "ic_drop": drop,
        "ic_drop_se": abs(drop / t) if t else 0.0,
        "t": t,
        "auc_drop": drop / 2,
        "max_abs_correlation": 0.95 if twin else 0.31,
        "most_correlated_with": twin or "volume_ratio",
        "redundant": twin is not None,
    }


_IMPORTANCE = {
    "n_rows": 5040,
    "n_sessions": 126,
    "base_ic": 0.0121,
    "base_auc": 0.531,
    "n_repeats": 3,
    "tstat_bar": 2.93,
    "features": [
        _row("return_20d", 0.0041, 4.1),
        _row("momentum_12_1", 0.0009, 1.2, twin="return_20d"),
        _row("rsi_14", -0.0002, -0.4),
    ],
    "groups": [_row("momentum", 0.0055, 5.2)],
    "noise_control": None,
    "verdict": "1 of 3 features clear the corrected bar (|t| > 2.93): return_20d.",
}

_DEFAULTS: dict[str, dict] = {
    "rp": {
        "value": 100000.0,
        "exposure_pct": 0.2,
        "drawdown_pct": 0.03,
        "daily_loss_pct": 0.0,
        "regime": "expansion",
    },
    "cb": {"level": "none", "tripped": False},
    "ep": {"cash": 95000.0, "equity": 100000.0, "exposure_pct": 0.05},
    "pos": {"positions": {"AAPL": {"quantity": 50, "last_price": 100.0}}},
    "al": {"alerts": [{"severity": "critical", "title": "Circuit breaker RED"}]},
    "ml": {"models": ["m1"]},
    "eq": {
        "points": [
            {"date": f"2024-01-{d:02d}", "equity": e}
            for d, e in zip(range(1, 26), _EQUITY, strict=True)
        ],
        "count": 25,
    },
    # /runs is an INDEX of {operation, completed_at} — a LIST, as ml-pipeline
    # really answers. The fixture used to be a map, which is how the UI came to
    # render array positions in the Operation column.
    "runs": {"runs": [{"operation": "train", "completed_at": "2026-08-03T09:00:00+00:00"}]},
    "run_train": {
        "operation": "train",
        "completed_at": "2026-08-03T09:00:00+00:00",
        "result": {"gate": {"importance": _IMPORTANCE}},
    },
    "serving": {"model": "m1", "paused": False},
    "strat": {
        "strategies": [
            {
                "name": "momentum_rank",
                "status": "active",
                "required_features": ["rsi_14"],
                "required_ranks": ["momentum_20"],
            },
            {
                "name": "donchian_breakout",
                "status": "probation",
                "required_features": ["donchian_pos_20"],
                "required_ranks": [],
            },
        ]
    },
    "weights": {"weights": {"strategy:momentum_rank": 0.4, "ml": 0.35, "macro": 0.25}},
}


class FakeSource:
    """DashboardSource double — healthy by default; pass key=None to mark a source down."""

    def __init__(self, **overrides: dict | None) -> None:
        # an override (even None) replaces the default → key=None simulates an unavailable upstream
        self._data: dict[str, dict | None] = {**_DEFAULTS, **overrides}

    async def risk_portfolio(self) -> dict | None:
        return self._data["rp"]

    async def circuit_breaker(self) -> dict | None:
        return self._data["cb"]

    async def execution_portfolio(self) -> dict | None:
        return self._data["ep"]

    async def positions(self) -> dict | None:
        return self._data["pos"]

    async def recent_alerts(self) -> dict | None:
        return self._data["al"]

    async def models(self) -> dict | None:
        return self._data["ml"]

    async def equity_curve(self) -> dict | None:
        return self._data["eq"]

    async def ml_runs(self) -> dict | None:
        return self._data["runs"]

    async def ml_run(self, operation: str) -> dict | None:
        # absent = "that operation has not completed here", which is exactly
        # what a 404 from ml-pipeline becomes on the real client
        return self._data.get(f"run_{operation}")

    async def ml_serving(self) -> dict | None:
        return self._data["serving"]

    async def strategies(self) -> dict | None:
        return self._data["strat"]

    async def signal_weights(self) -> dict | None:
        return self._data["weights"]

    async def ohlcv(self, symbol: str, limit: int = 120) -> list[dict] | None:
        """Deterministic per-symbol closes, long enough to correlate."""
        seed = sum(ord(c) for c in symbol)
        return [
            {"timestamp": f"day{i}", "close": 100.0 + (i * 0.5) + ((i + seed) % 7) * 0.3}
            for i in range(60)
        ]

    async def health_all(self) -> dict[str, dict]:
        return {
            "market-data": {"status": "up", "latency_ms": 4.2, "http_status": 200},
            "execution": {"status": "up", "latency_ms": 11.5, "http_status": 200},
            "ml-pipeline": {"status": "down", "latency_ms": 2000.0, "error": "ConnectError"},
        }

    async def run_backtest(self, strategy: str, symbol: str, limit: int = 500):
        if strategy == "nie_ma":
            return 404, {"detail": "unknown strategy (known: ['sma_ema_crossover'])"}
        if strategy == "momentum_rank":
            return 422, {"detail": "reads cross-sectional ranks ['momentum_20']"}
        return 200, {
            "strategy_name": strategy,
            "symbol": symbol,
            "total_return": 0.21,
            "sharpe_ratio": 1.3,
            "max_drawdown": 0.08,
            "n_trades": 12,
            "n_bars": 480,
            "equity_curve": [1.0, 1.05, 1.12, 1.21],
        }

    async def aclose(self) -> None:
        return None


def build_service(source: FakeSource | None = None) -> DashboardService:
    return DashboardService(source or FakeSource())


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, DashboardService]]:
    from src.main import app

    service = build_service()
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, service
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
