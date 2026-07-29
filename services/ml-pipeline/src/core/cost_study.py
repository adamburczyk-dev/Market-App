"""What the strategy costs, and at what size it stops working (P5-2).

`costs.py` prices one name at one book size. This asks the question that
actually decides something: **how large can this book get before costs eat the
edge?** The answer is a curve, not a number, because impact grows with the
square root of size while the edge does not grow at all.

The output is deliberately shaped as a comparison against the flat 5 bps every
evaluation so far assumed. That rate is not wrong everywhere — at a small book
in liquid names it is roughly right, which is exactly why it survived unnoticed.
It becomes wrong quietly, as size rises, and the point of the curve is to show
where.

Model-free: no model is fitted, nothing is registered, `n_trials` is untouched.
"""

from typing import Any

import structlog
from trading_common.schemas import OHLCVBar

from src.core.costs import CostParams, cost_summary, estimate_costs

logger = structlog.get_logger()

DEFAULT_AUMS = (250_000.0, 1_000_000.0, 5_000_000.0, 25_000_000.0, 100_000_000.0)
FLAT_BASELINE_BPS = 5.0


def run_cost_study(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    aums: tuple[float, ...] = DEFAULT_AUMS,
    base_params: CostParams | None = None,
    turnover_daily: float | None = None,
) -> dict[str, Any]:
    """Cost profile of the universe across book sizes.

    ``turnover_daily`` — if the last training run's measured turnover is passed
    in, the report converts bps into the only unit that matters for a decision:
    annual Sharpe-equivalent drag on the strategy. Without it the cost is a
    number in a table that nobody can weigh against a Sharpe of 0.8.
    """
    if not bars_by_symbol:
        raise ValueError("no history for any requested symbol")
    base = base_params or CostParams()

    curve: list[dict[str, Any]] = []
    for aum in aums:
        params = CostParams(
            aum_usd=aum,
            impact_coefficient=base.impact_coefficient,
            window=base.window,
            spread_window=base.spread_window,
            max_position_weight=base.max_position_weight,
            max_participation=base.max_participation,
        )
        costs = estimate_costs(bars_by_symbol, params)
        summary = cost_summary(costs, params, FLAT_BASELINE_BPS)
        point: dict[str, Any] = {
            "aum_usd": aum,
            "median_total_bps": summary["total_bps"]["median"],
            "p90_total_bps": summary["total_bps"]["p90"],
            "median_impact_bps": summary["impact_bps"]["median"],
            "cost_ratio_vs_flat": summary["cost_ratio_vs_flat"],
            "untradeable_share": summary["untradeable_share"],
            "n_untradeable": len(summary["untradeable"]),
        }
        if turnover_daily is not None:
            point["annual_cost_drag"] = round(
                summary["total_bps"]["median"] / 10_000.0 * turnover_daily * 252, 4
            )
            point["annual_cost_drag_flat"] = round(
                FLAT_BASELINE_BPS / 10_000.0 * turnover_daily * 252, 4
            )
        curve.append(point)

    # the per-name table at the reference size the caller asked for
    reference_costs = estimate_costs(bars_by_symbol, base)
    reference = cost_summary(reference_costs, base, FLAT_BASELINE_BPS)
    by_symbol = sorted(
        (c.as_dict() for c in reference_costs.values()),
        key=lambda row: float(row["total_bps"]),
        reverse=True,
    )

    report: dict[str, Any] = {
        "symbols": len(bars_by_symbol),
        "window": base.window,
        "reference": reference,
        "capacity_curve": curve,
        "per_symbol": by_symbol,
        "verdict": _verdict(curve, reference),
    }
    logger.info(
        "Cost study finished",
        symbols=len(bars_by_symbol),
        reference_aum=base.aum_usd,
        median_total_bps=reference["total_bps"]["median"],
        ratio_vs_flat=reference["cost_ratio_vs_flat"],
    )
    return report


def _verdict(curve: list[dict[str, Any]], reference: dict[str, Any]) -> str:
    """One sentence on what the curve means — the part a table cannot say."""
    ratio = float(reference["cost_ratio_vs_flat"])
    identified = float(reference.get("spread_identified_share", 0.0))
    # the largest size where costs stay within ~2x the assumed flat rate and
    # essentially the whole universe is still tradeable
    viable = [
        point["aum_usd"]
        for point in curve
        if float(point["cost_ratio_vs_flat"]) <= 2.0 and float(point["untradeable_share"]) <= 0.05
    ]
    if viable:
        headroom = (
            f"Costs stay within 2x the assumed flat rate up to about "
            f"${max(viable):,.0f}; past that the square-root term takes over."
        )
    else:
        headroom = (
            "Even the smallest book size tested already costs more than twice the "
            "flat 5 bps every past evaluation charged — those results were optimistic."
        )
    caveat = (
        "Spread levels are mostly the floor rather than measurements "
        f"({identified:.0%} identified), so the total is impact-dominated and, if anything, "
        "understated for thin names."
        if identified < 0.5
        else f"Spreads were identified for {identified:.0%} of names."
    )
    return f"At the reference size costs are {ratio:.1f}x the flat 5 bps. {headroom} {caveat}"


__all__ = ["DEFAULT_AUMS", "FLAT_BASELINE_BPS", "run_cost_study"]
