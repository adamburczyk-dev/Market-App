"""Realistic trading costs (P5-2) — spread + impact instead of a flat 5 bps.

Two things need pinning here, and they are different in kind. The impact term
has real content: it depends on inputs we measure well (dollar volume,
volatility) and on the book size, so its scaling laws are testable. The spread
estimator does not identify a level from daily bars — it orders names — so what
is pinned is the ordering, the aggregation, and the two bugs that made it read
a spread where there was none.
"""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from trading_common.schemas import Interval, OHLCVBar

from src.core.costs import (
    MIN_HALF_SPREAD_BPS,
    CostParams,
    corwin_schultz_half_spread,
    cost_summary,
    cost_table_bps,
    estimate_costs,
    estimate_symbol_cost,
)

D0 = datetime(2024, 1, 2, tzinfo=UTC)


def diffusion_bars(
    symbol: str = "AAA",
    n: int = 63,
    seed: int = 0,
    spread: float = 0.0,
    daily_vol: float = 0.02,
    steps: int = 390,
    gap_vol: float = 0.0,
    volume: float = 5_000_000.0,
    price: float = 100.0,
) -> list[OHLCVBar]:
    """Bars whose high/low are the running extremes of an intraday diffusion.

    Corwin-Schultz assumes exactly this process, so it is the only construction
    on which the estimator can be checked against a KNOWN spread: `spread` is
    injected by widening each day's observed high and low by half of it.
    """
    rng = np.random.default_rng(seed)
    step_vol = daily_vol / math.sqrt(steps)
    bars: list[OHLCVBar] = []
    for i in range(n):
        if gap_vol:
            price *= math.exp(float(rng.normal(0, gap_vol)))
        path = price * np.exp(np.cumsum(rng.normal(0, step_vol, steps)))
        close = float(path[-1])
        bars.append(
            OHLCVBar(
                symbol=symbol,
                timestamp=D0 + timedelta(days=i),
                interval=Interval.D1,
                open=float(path[0]),
                high=float(path.max()) * (1 + spread / 2),
                low=float(path.min()) * (1 - spread / 2),
                close=close,
                volume=volume / close,  # keep dollar volume at `volume`
                adj_close=close,
            )
        )
        price = close
    return bars


def hl_arrays(bars: list[OHLCVBar]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([b.high for b in bars]),
        np.array([b.low for b in bars]),
        np.array([b.close for b in bars]),
    )


# --- the spread estimator: what it can and cannot do -----------------------


def test_a_zero_spread_series_is_not_read_as_a_wide_spread():
    """The bug this replaced: aggregating the estimator per two-day observation
    and taking the median of only the POSITIVE values throws away the negative
    half of a symmetric noise distribution, so a series with NO spread at all
    reported ~65 bps. Averaging beta and gamma before the transform — the form
    the estimator's identity actually holds in — leaves it near zero.
    """
    estimates = [
        corwin_schultz_half_spread(*hl_arrays(diffusion_bars(seed=s, spread=0.0))) for s in range(9)
    ]
    assert abs(float(np.median(estimates))) < 25.0, estimates
    # and the old aggregation, reproduced here, is what it must not do
    highs, lows, _ = hl_arrays(diffusion_bars(seed=0, spread=0.0))
    log_hl = np.log(highs / lows)
    beta = log_hl[:-1] ** 2 + log_hl[1:] ** 2
    gamma = np.log(np.maximum(highs[:-1], highs[1:]) / np.minimum(lows[:-1], lows[1:])) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    per_observation = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    dropped_negatives = float(np.median(per_observation[per_observation > 0])) / 2 * 10_000
    assert dropped_negatives > 40.0, "the bug no longer reproduces; the test has lost its point"


def test_the_estimator_orders_names_by_their_spread():
    """The property the cost model actually relies on. The LEVEL is not
    identified from daily bars — the same zero-spread series reads −9.9 bps at
    390 intraday observations and +9.1 at 4000 — but the ordering survives, so
    a wider-spread name must estimate higher than a tighter one.
    """

    def median_estimate(injected: float) -> float:
        return float(
            np.median(
                [
                    corwin_schultz_half_spread(*hl_arrays(diffusion_bars(seed=s, spread=injected)))
                    for s in range(9)
                ]
            )
        )

    tight = median_estimate(0.0)
    wide = median_estimate(0.005)  # 50 bps quoted → 25 bps half
    wider = median_estimate(0.01)
    assert tight < wide < wider
    # roughly unit slope: 25 bps more injected half-spread should move it by a
    # comparable amount, not by a token one
    assert wide - tight > 10.0


def test_the_overnight_adjustment_is_what_makes_gapping_names_usable():
    """An overnight gap enters the two-day range but not the one-day ranges,
    which inflates gamma and drags the estimate down — far enough that a normal
    equity would look like it had a negative spread. Pinning the fix means
    pinning that the adjusted estimate is materially higher than the raw one.
    """
    bars = [diffusion_bars(seed=s, spread=0.005, gap_vol=0.015) for s in range(9)]
    unadjusted = float(np.median([corwin_schultz_half_spread(*hl_arrays(b)[:2]) for b in bars]))
    adjusted = float(np.median([corwin_schultz_half_spread(*hl_arrays(b)) for b in bars]))
    assert adjusted > unadjusted + 10.0, (adjusted, unadjusted)


def test_degenerate_inputs_return_no_estimate_rather_than_a_number():
    assert corwin_schultz_half_spread(np.array([10.0]), np.array([9.0]), None) == 0.0
    assert corwin_schultz_half_spread(np.array([1.0, -1.0, 2.0]), np.array([1.0, 1.0, 1.0])) == 0.0


# --- impact: the term with real content ------------------------------------


def test_impact_follows_the_square_root_of_book_size():
    """The square-root law is the whole reason an AUM has to be stated: four
    times the money is twice the impact, not four times, and a flat rate says
    it is neither.
    """
    bars = diffusion_bars(volume=200_000_000.0)
    small = estimate_symbol_cost("AAA", bars, CostParams(aum_usd=1_000_000))
    large = estimate_symbol_cost("AAA", bars, CostParams(aum_usd=4_000_000))
    assert small.tradeable and large.tradeable
    assert large.impact_bps == pytest.approx(2 * small.impact_bps, rel=1e-6)


def test_impact_follows_the_square_root_of_illiquidity():
    """Same money, a name that trades a quarter as much → twice the impact."""
    liquid = estimate_symbol_cost("AAA", diffusion_bars(volume=400_000_000.0))
    thin = estimate_symbol_cost("BBB", diffusion_bars(symbol="BBB", volume=100_000_000.0))
    assert thin.impact_bps == pytest.approx(2 * liquid.impact_bps, rel=1e-6)
    assert thin.median_dollar_volume < liquid.median_dollar_volume


def test_a_volatile_name_costs_more_to_push_through():
    calm = estimate_symbol_cost("AAA", diffusion_bars(daily_vol=0.01, volume=200_000_000.0))
    wild = estimate_symbol_cost(
        "BBB", diffusion_bars(symbol="BBB", daily_vol=0.04, volume=200_000_000.0)
    )
    assert wild.impact_bps > 2.5 * calm.impact_bps


def test_a_size_the_name_cannot_absorb_is_flagged_not_extrapolated():
    """Past the participation cap the square-root law stops being a reasonable
    extrapolation. The honest output is "not tradeable at this size", and the
    impact number must stop growing rather than keep producing a comfortable
    finite figure.
    """
    thin = diffusion_bars(volume=1_000_000.0)  # $1M/day
    at_size = estimate_symbol_cost("AAA", thin, CostParams(aum_usd=100_000_000))
    assert not at_size.tradeable
    assert at_size.participation > 0.10
    huge = estimate_symbol_cost("AAA", thin, CostParams(aum_usd=1_000_000_000))
    assert huge.impact_bps == pytest.approx(at_size.impact_bps), "impact grew past the cap"


def test_too_little_history_is_untradeable_rather_than_free():
    short = estimate_symbol_cost("AAA", diffusion_bars(n=5))
    assert not short.tradeable
    assert not short.spread_identified
    assert short.half_spread_bps > 50.0, "a name we cannot measure must not look cheap"


def test_the_spread_floor_is_reported_as_a_floor():
    """A floored estimate and a small measurement are the same number; only the
    flag distinguishes them, and the summary has to carry it or the spread
    column reads as data when it is a default.
    """
    cost = estimate_symbol_cost("AAA", diffusion_bars(spread=0.0, gap_vol=0.02))
    assert cost.half_spread_bps == MIN_HALF_SPREAD_BPS
    assert not cost.spread_identified


# --- the summary the report reads ------------------------------------------


def universe_bars() -> dict[str, list[OHLCVBar]]:
    return {
        "BIG": diffusion_bars("BIG", seed=1, volume=500_000_000.0, daily_vol=0.015),
        "MID": diffusion_bars("MID", seed=2, volume=50_000_000.0, daily_vol=0.02),
        "THIN": diffusion_bars("THIN", seed=3, volume=300_000.0, daily_vol=0.05),
    }


def test_summary_says_how_wrong_the_flat_rate_was_and_at_what_size():
    costs = estimate_costs(universe_bars(), CostParams(aum_usd=20_000_000))
    summary = cost_summary(costs, CostParams(aum_usd=20_000_000))
    assert summary["symbols"] == 3
    assert summary["aum_usd"] == 20_000_000
    assert summary["max_position_usd"] == 1_000_000
    # $1M into a name trading $300k/day is not a trade
    assert summary["untradeable"] == ["THIN"]
    assert summary["untradeable_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["cost_ratio_vs_flat"] > 1.0, (
        "5 bps flat was optimistic and must be shown as such"
    )
    assert 0.0 <= summary["spread_identified_share"] <= 1.0
    assert "sqrt(AUM)" in summary["note"]


def test_the_same_universe_is_cheap_small_and_expensive_large():
    """The statement a flat rate cannot make."""
    bars = universe_bars()
    small = cost_summary(
        estimate_costs(bars, CostParams(aum_usd=500_000)), CostParams(aum_usd=500_000)
    )
    large = cost_summary(
        estimate_costs(bars, CostParams(aum_usd=50_000_000)), CostParams(aum_usd=50_000_000)
    )
    assert large["total_bps"]["median"] > small["total_bps"]["median"]
    assert large["untradeable_share"] >= small["untradeable_share"]


def test_empty_input_reports_nothing_rather_than_zero_cost():
    assert cost_summary({}, CostParams())["symbols"] == 0
    assert estimate_costs({}) == {}
    assert cost_table_bps({}) == {}


# --- the capacity curve: the statement a flat rate cannot make -------------


def test_the_capacity_curve_shows_where_size_stops_working():
    """The deliverable. Impact grows with sqrt(size) while the edge does not
    grow at all, so the honest output is a curve: cheap and fully tradeable
    small, expensive and partly untradeable large.
    """
    from src.core.cost_study import run_cost_study

    report = run_cost_study(
        universe_bars(),
        aums=(250_000.0, 5_000_000.0, 100_000_000.0),
        base_params=CostParams(aum_usd=5_000_000.0),
    )
    curve = report["capacity_curve"]
    assert [p["aum_usd"] for p in curve] == [250_000.0, 5_000_000.0, 100_000_000.0]
    costs = [p["median_total_bps"] for p in curve]
    assert costs == sorted(costs), "cost must be non-decreasing in book size"
    assert curve[-1]["untradeable_share"] >= curve[0]["untradeable_share"]
    # the per-name table is sorted worst-first — that is the list to read
    per_symbol = report["per_symbol"]
    assert per_symbol[0]["symbol"] == "THIN"
    assert [r["total_bps"] for r in per_symbol] == sorted(
        (r["total_bps"] for r in per_symbol), reverse=True
    )
    assert "flat 5 bps" in report["verdict"]


def test_the_curve_converts_cost_into_a_number_comparable_to_a_sharpe():
    """bps per trade is not a unit anyone can weigh against a Sharpe of 0.8.
    Given the measured turnover, the report has to state the annual drag —
    and it must be worse than the flat-rate drag it replaces."""
    from src.core.cost_study import run_cost_study

    report = run_cost_study(
        universe_bars(),
        aums=(10_000_000.0,),
        base_params=CostParams(aum_usd=10_000_000.0),
        turnover_daily=0.05,
    )
    point = report["capacity_curve"][0]
    assert point["annual_cost_drag"] > point["annual_cost_drag_flat"] > 0


def test_a_cost_study_without_history_refuses_rather_than_reporting_zero():
    from src.core.cost_study import run_cost_study

    with pytest.raises(ValueError, match="no history"):
        run_cost_study({})


def test_the_spread_window_is_longer_than_the_liquidity_window_on_purpose():
    """The spread is a slow-moving property and the estimator is noisy, so the
    two inputs do not want the same window. Sharing the 63-session one showed up
    as a mega-cap's total cost swinging 2.5 -> 8.9 bps between two runs on the
    same universe — noise reported as cost.

    Pinned as a dispersion: across independent paths with the SAME injected
    spread, the long window must scatter materially less than the short one.
    """
    short, long_ = [], []
    for seed in range(24):
        bars = diffusion_bars(n=252, seed=seed, spread=0.0005, daily_vol=0.012, volume=1e9)
        short.append(
            estimate_symbol_cost("AAA", bars, CostParams(spread_window=63)).half_spread_bps
        )
        long_.append(
            estimate_symbol_cost("AAA", bars, CostParams(spread_window=252)).half_spread_bps
        )
    assert float(np.std(long_)) < 0.7 * float(np.std(short)), (np.std(short), np.std(long_))
    assert CostParams().spread_window > CostParams().window


def test_impact_is_the_term_that_orders_a_large_cap_universe():
    """Given the spread noise, the ordering the cost model can actually defend
    is the impact one — deterministic in inputs we measure well. A universe
    spanning a decade of liquidity must order by impact without exception.
    """
    liquidity = {"MEGA": 2e9, "LARG": 4e8, "MIDD": 8e7, "SMAL": 1.2e7, "MICR": 9e5}
    bars = {
        name: diffusion_bars(name, n=252, seed=i, volume=dv)
        for i, (name, dv) in enumerate(liquidity.items())
    }
    costs = estimate_costs(bars, CostParams(aum_usd=5_000_000))
    impacts = [costs[name].impact_bps for name in liquidity]
    assert impacts == sorted(impacts), impacts
    assert not costs["MICR"].tradeable and costs["MEGA"].tradeable
