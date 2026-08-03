"""The benchmark leg an excess label is measured against.

An excess label asks "did this name beat the others", so it needs a number for
what "the others" did. Three call sites need that number and they must all get
the SAME one: ``build_dataset`` (training labels), ``target_study`` (the
measurement that chooses the target), and ``OutcomeResolver`` (the realized
label the drift monitor scores against). A private helper in any one of them is
a train/serve divergence waiting to be written.

Two properties carry the whole module, and the old helper had neither.

**Keyed by session, not by array position.** The previous version took
``max(lengths)`` and kept only series of exactly that length, then guarded each
label with ``len(market) == n``. On a real panel with heterogeneous listing
dates that is a benchmark built from the names with the longest history —
survivors — while every shorter name silently fell through to the ABSOLUTE
label inside what was reported as an excess candidate. Two label kinds mixed in
one measurement, and nothing said so.

**Median of DAILY returns, cumulated — not the median of cumulative ratios.**
Rebasing each series to its own first bar and taking the median across them
makes the index depend on the start date and lets long-history names dominate.
Taking each session's median return across the names present THAT session is a
rebalanced equal-weight index: survivorship-honest by construction, because a
name contributes only to the days it actually traded.

The median rather than the mean throughout: one name doubling should not
redefine "the market" for a 34-name cross-section.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import pairwise

import numpy as np
import structlog
from trading_common.prices import adjusted_ohlc
from trading_common.schemas import OHLCVBar

from src.core.universe import Universe

logger = structlog.get_logger()

# A level is only as meaningful as the cross-section behind it. Below this the
# "median" is one or two names, which is not a market — the level carries
# forward instead, and `thin_sessions` says how often that happened.
MIN_CROSS_SECTION = 3


def market_levels(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    universe: Universe | None = None,
) -> dict[datetime, float]:
    """Benchmark level per session, rebased to 1.0 on the first session.

    The date axis is the union of every symbol's bars, so any symbol's own
    timestamp always has a level — which is what lets the result be projected
    onto each symbol's index without a length guard.

    ``universe`` restricts which names CONTRIBUTE to each session's median
    (P3-1), not which dates exist. Restricting the axis too would make the
    benchmark disappear on days the point-in-time universe happens to be empty,
    and every label spanning such a day would vanish with it.
    """
    adjusted: dict[str, dict[datetime, float]] = {}
    for symbol, bars in bars_by_symbol.items():
        ordered = sorted(bars, key=lambda b: b.timestamp)
        _, _, _, adj_close = adjusted_ohlc(ordered)
        adjusted[symbol] = {
            bar.timestamp: float(close)
            for bar, close in zip(ordered, adj_close, strict=True)
            if close > 0
        }

    sessions = sorted({ts for by_date in adjusted.values() for ts in by_date})
    if not sessions:
        return {}

    levels: dict[datetime, float] = {sessions[0]: 1.0}
    level = 1.0
    thin = 0
    for previous, current in pairwise(sessions):
        eligible = universe.members_on(current.date()) if universe is not None else None
        returns = [
            np.log(by_date[current] / by_date[previous])
            for symbol, by_date in adjusted.items()
            if (eligible is None or symbol in eligible)
            and current in by_date
            and previous in by_date
        ]
        if len(returns) < MIN_CROSS_SECTION:
            # Carry rather than invent: with one or two names the median is a
            # single stock, and calling that "the market" would make its own
            # excess return zero by construction.
            thin += 1
        else:
            level *= float(np.exp(np.median(returns)))
        levels[current] = level

    if thin:
        logger.info(
            "Benchmark carried through thin sessions",
            thin_sessions=thin,
            total_sessions=len(sessions),
            min_cross_section=MIN_CROSS_SECTION,
        )
    return levels


def project_levels(
    levels: Mapping[datetime, float],
    ordered_bars: Sequence[OHLCVBar],
) -> np.ndarray:
    """The benchmark aligned to one symbol's own bar index.

    ``excess_barrier_label`` indexes the market array with the symbol's bar
    index, so the two must share an index basis. Built from `market_levels`
    over the same panel every timestamp resolves; a timestamp that somehow does
    not carries the previous level rather than a zero, since a zero would be
    read as a price and make the log return meaningless.
    """
    out = np.empty(len(ordered_bars), dtype=float)
    last = 1.0
    for i, bar in enumerate(ordered_bars):
        last = float(levels.get(bar.timestamp, last))
        out[i] = last
    return out


__all__ = ["MIN_CROSS_SECTION", "market_levels", "project_levels"]
