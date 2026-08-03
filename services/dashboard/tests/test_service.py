"""Tests for DashboardService.overview aggregation + partial tolerance."""

import pytest

from .conftest import FakeSource, build_service


@pytest.mark.asyncio
async def test_overview_full_availability():
    service = build_service(FakeSource())
    ov = await service.overview()
    assert ov["sources"] == {
        "risk-mgmt": "ok",
        "execution": "ok",
        "notification": "ok",
        "ml-pipeline": "ok",
    }
    assert ov["portfolio"]["value"] == 100000.0
    assert ov["positions"]["AAPL"]["quantity"] == 50
    assert len(ov["recent_alerts"]) == 1
    assert ov["models"] == ["m1"]


@pytest.mark.asyncio
async def test_overview_partial_when_ml_down():
    service = build_service(FakeSource(ml=None))
    ov = await service.overview()
    assert ov["sources"]["ml-pipeline"] == "unavailable"
    assert ov["sources"]["risk-mgmt"] == "ok"
    assert ov["models"] == []  # missing source → empty, not a crash


@pytest.mark.asyncio
async def test_risk_source_needs_both_portfolio_and_breaker():
    # breaker down but portfolio up → risk-mgmt reported unavailable (incomplete)
    service = build_service(FakeSource(cb=None))
    ov = await service.overview()
    assert ov["sources"]["risk-mgmt"] == "unavailable"
    assert ov["circuit_breaker"] is None
    assert ov["portfolio"] is not None  # the part that loaded is still surfaced


@pytest.mark.asyncio
async def test_execution_source_needs_both_portfolio_and_positions():
    service = build_service(FakeSource(pos=None))
    ov = await service.overview()
    assert ov["sources"]["execution"] == "unavailable"
    assert ov["positions"] == {}


@pytest.mark.asyncio
async def test_overview_all_down():
    service = build_service(FakeSource(rp=None, cb=None, ep=None, pos=None, al=None, ml=None))
    ov = await service.overview()
    assert set(ov["sources"].values()) == {"unavailable"}
    assert ov["portfolio"] is None
    assert ov["positions"] == {}
    assert ov["recent_alerts"] == []
    assert ov["models"] == []


# --- the six sections -----------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_section_derives_pnl_from_the_curve():
    d = await build_service().portfolio_section()
    assert d["available"] is True
    assert d["sessions"] == 25
    assert d["pnl_abs"] == pytest.approx(11_000.0)
    assert d["pnl_pct"] == pytest.approx(0.11)
    assert d["labels"][0] == "2024-01-01"
    assert d["sharpe"] is not None


@pytest.mark.asyncio
async def test_portfolio_section_says_so_when_there_is_no_history():
    """Before execution recorded a series this section had a number and no
    shape; an empty chart must read as 'no history', not as a flat line."""
    d = await build_service(FakeSource(eq={"points": [], "count": 0})).portfolio_section()
    assert d["available"] is False
    assert d["curve"] == []
    assert d["pnl_pct"] is None
    assert d["sharpe"] is None


@pytest.mark.asyncio
async def test_risk_section_measures_var_only_with_enough_samples():
    d = await build_service().risk_section()
    assert d["samples"] == 24  # 25 points → 24 returns
    assert d["var_95"] is not None and d["var_95"] > 0
    assert d["cvar_95"] >= d["var_95"]
    # peak 104_200 → trough 97_500 = 6700/104200
    assert d["max_drawdown"] == pytest.approx(6_700 / 104_200, abs=1e-6)
    assert len(d["drawdown_curve"]) == 25


@pytest.mark.asyncio
async def test_risk_section_reports_unavailable_rather_than_a_made_up_var():
    short = {"points": [{"date": f"2024-01-{i:02d}", "equity": 100.0 + i} for i in range(1, 6)]}
    d = await build_service(FakeSource(eq=short)).risk_section()
    assert d["var_95"] is None
    assert d["cvar_95"] is None
    assert d["samples"] == 4


@pytest.mark.asyncio
async def test_risk_section_correlates_only_what_is_actually_held():
    d = await build_service().risk_section()
    assert d["correlation"]["symbols"] == ["AAPL"]
    assert d["correlation"]["matrix"] == [[1.0]]
    # One name has no pairs, so there is no average to report.
    assert d["avg_correlation"] is None


@pytest.mark.asyncio
async def test_correlation_grid_covers_several_held_names():
    held = ("AAPL", "MSFT", "XOM")
    positions = {"positions": {s: {"quantity": 1, "last_price": 10.0} for s in held}}
    d = await build_service(FakeSource(pos=positions)).risk_section()
    assert d["correlation"]["symbols"] == ["AAPL", "MSFT", "XOM"]
    assert d["avg_correlation"] is not None
    assert d["correlation"]["coverage"] == 1.0


@pytest.mark.asyncio
async def test_strategy_section_joins_status_with_the_learned_weight():
    d = await build_service().strategy_section()
    by_name = {r["name"]: r for r in d["strategies"]}
    assert by_name["momentum_rank"]["weight"] == pytest.approx(0.4)
    assert by_name["momentum_rank"]["required_ranks"] == ["momentum_20"]
    # A rule with no recorded outcomes is None, NOT 0.0 — "not measured" and
    # "measured and worthless" are different claims.
    assert by_name["donchian_breakout"]["weight"] is None
    assert by_name["donchian_breakout"]["status"] == "probation"
    assert d["other_sources"] == {"ml": 0.35, "macro": 0.25}


@pytest.mark.asyncio
async def test_ml_section_carries_runs_and_serving():
    d = await build_service().ml_section()
    assert d["available"] is True
    # the run index is a LIST of {operation, completed_at}, as ml-pipeline answers
    assert [r["operation"] for r in d["runs"]] == ["train"]
    assert d["serving"]["paused"] is False


@pytest.mark.asyncio
async def test_ml_section_shows_the_importance_table_and_names_its_source():
    d = (await build_service().ml_section())["importance"]
    assert [row["feature"] for row in d["table"]["features"]] == [
        "return_20d",
        "momentum_12_1",
        "rsi_14",
    ]
    assert d["table"]["groups"][0]["feature"] == "momentum"
    # An importance table read against the wrong model is worse than none, so
    # the model it describes travels with it.
    assert "training run" in d["source"]
    assert d["measured_at"] == "2026-08-03T09:00:00+00:00"


@pytest.mark.asyncio
async def test_the_study_is_used_only_when_the_training_run_has_no_table():
    """The study fits a DIAGNOSTIC model carrying a planted noise column."""
    study = {
        "operation": "feature-importance",
        "completed_at": "2026-08-03T10:00:00+00:00",
        "result": {"importance": {"features": [], "groups": [], "noise_control": {"t": 0.4}}},
    }
    both = await build_service(FakeSource(**{"run_feature-importance": study})).ml_section()
    assert "training run" in both["importance"]["source"]  # production model wins

    only_study = await build_service(
        FakeSource(**{"run_train": None, "run_feature-importance": study})
    ).ml_section()
    assert "DIAGNOSTIC" in only_study["importance"]["source"]
    assert only_study["importance"]["table"]["noise_control"]["t"] == 0.4


@pytest.mark.asyncio
async def test_every_missing_table_says_which_kind_of_missing_it_is():
    """ "nobody measured", "measurement switched off" and "service down" are
    three situations; a blank panel says the same nothing for all three."""
    nothing_ran = await build_service(FakeSource(**{"run_train": None})).ml_section()
    assert nothing_ran["importance"]["table"] is None
    assert "no training run" in nothing_ran["importance"]["reason"]

    measured_off = await build_service(
        FakeSource(**{"run_train": {"completed_at": "x", "result": {"gate": {}}}})
    ).ml_section()
    assert "switched off" in measured_off["importance"]["reason"]

    down = await build_service(FakeSource(ml=None, **{"run_train": None})).ml_section()
    assert down["importance"]["reason"] == "ml-pipeline unavailable"


@pytest.mark.asyncio
async def test_health_section_counts_up_and_finds_the_slowest():
    d = await build_service().health_section()
    assert d["total"] == 3
    assert d["up"] == 2  # ml-pipeline is down
    assert d["slowest_ms"] == pytest.approx(11.5)  # among the ones that answered


@pytest.mark.asyncio
async def test_a_dead_upstream_costs_its_own_section_not_the_page():
    source = FakeSource(strat=None, weights=None)
    service = build_service(source)
    assert (await service.strategy_section())["available"] is False
    # ...while the others still render.
    assert (await service.portfolio_section())["available"] is True


@pytest.mark.asyncio
async def test_an_unreachable_market_data_is_not_reported_as_zero_correlation():
    """An empty grid because market-data is down looks exactly like an empty
    grid because nothing is held. These two counts are what tells them apart."""

    class NoPrices(FakeSource):
        async def ohlcv(self, symbol: str, limit: int = 120):
            return None

    positions = {"positions": {s: {"quantity": 1, "last_price": 10.0} for s in ("AAPL", "MSFT")}}
    d = await build_service(NoPrices(pos=positions)).risk_section()
    assert d["held_symbols"] == ["AAPL", "MSFT"]
    assert d["correlated_symbols"] == []
    assert d["avg_correlation"] is None


@pytest.mark.asyncio
async def test_a_non_json_upstream_body_is_kept_not_dropped():
    """An unhandled upstream error answers with plain text; calling .json() on
    it raises a JSONDecodeError that is NOT an httpx error, so it escapes the
    handling entirely and turns an upstream 500 into a 500 from this service."""
    import httpx

    from src.core.clients import _decode

    plain = httpx.Response(500, text="Internal Server Error")
    assert _decode(plain)["detail"] == "Internal Server Error"
    assert _decode(httpx.Response(204))["detail"].startswith("upstream 204")
    assert _decode(httpx.Response(200, json={"ok": True})) == {"ok": True}
    assert _decode(httpx.Response(200, json=[1, 2]))["detail"] == "[1, 2]"
