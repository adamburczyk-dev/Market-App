"""Deciding what to fetch — resume from the last stored bar, not from yesterday.

A scheduled pull that asks for "today" is only correct while it never misses a
run. Miss a day (container down, provider outage, a bug) and that day is gone
permanently: nothing in the system would ever go back for it, and the gap would
surface much later as a hole in a feature window. So the window starts from
what we actually hold, which makes the schedule self-healing — after a week of
downtime the next run simply asks for a week.

Two things beyond the obvious arithmetic:

**Overlap.** The window reaches back a few sessions before the newest stored
bar. Providers restate recent bars (late prints, corrected volume), and the
last bar we stored may have been captured mid-session. Re-fetching a handful of
bars costs nothing because the upsert is idempotent.

**Retroactive adjustment.** This is the one that silently corrupts. `adj_close`
is not a property of a bar, it is a property of the bar *plus every corporate
action after it*. When a split or dividend occurs, the provider rewrites
adj_close for the ENTIRE history — so a purely incremental fetch leaves every
older bar holding a pre-event figure, while new bars hold post-event ones. The
series stays plausible and is wrong exactly at the join, and since features and
labels are computed on adjusted prices (P0-1), the damage lands in the model
rather than in an error log. The overlap window is what makes this detectable:
if the adjustment factor of a bar we already store has changed, the symbol's
whole history is stale and has to be re-fetched.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from trading_common.schemas import OHLCVBar

logger = structlog.get_logger()

# Relative change in adj_close/close big enough to mean a corporate action
# rather than floating-point noise or a provider rounding a cent differently.
ADJUSTMENT_TOLERANCE = 1e-4


@dataclass(frozen=True)
class FetchPlan:
    start: datetime
    end: datetime
    # "full" when we hold nothing (or are repairing a restated history),
    # "incremental" when resuming from the newest stored bar.
    mode: str

    @property
    def is_full(self) -> bool:
        return self.mode == "full"


def plan_fetch(
    latest_stored: datetime | None,
    now: datetime,
    overlap_days: int = 5,
    initial_history_days: int = 365 * 6,
) -> FetchPlan:
    """Window to request for one symbol.

    Nothing stored → a full history. Otherwise from ``overlap_days`` before the
    newest stored bar up to now, so a missed run is repaired by the next one
    instead of leaving a permanent hole.
    """
    end = now
    if latest_stored is None:
        return FetchPlan(now - timedelta(days=initial_history_days), end, "full")
    start = latest_stored - timedelta(days=max(overlap_days, 0))
    return FetchPlan(min(start, end), end, "incremental")


def full_history_plan(
    now: datetime,
    initial_history_days: int = 365 * 6,
    earliest_stored: datetime | None = None,
) -> FetchPlan:
    """A window covering everything we hold, not just the default depth.

    ``earliest_stored`` is load-bearing when this is used to REPAIR a restated
    history: a corporate action rewrites adj_close all the way back, so a repair
    limited to the default depth would leave every older bar on the pre-event
    scale. Measured on a real 5000-bar history repaired with a 4000-bar window:
    1000 bars silently kept the old factor. With a 20-year backfill and a 6-year
    default that is fourteen years of quietly wrong returns.
    """
    start = now - timedelta(days=initial_history_days)
    if earliest_stored is not None:
        start = min(start, earliest_stored)
    return FetchPlan(start, now, "full")


def _factor(bar: OHLCVBar) -> float | None:
    """adj_close / close — how much the whole series has been restated."""
    if bar.adj_close is None or bar.close <= 0:
        return None
    return bar.adj_close / bar.close


def adjustment_drifted(
    stored: list[OHLCVBar],
    fetched: list[OHLCVBar],
    tolerance: float = ADJUSTMENT_TOLERANCE,
) -> bool:
    """Did a corporate action restate the history we already hold?

    Compares the adjustment factor of bars present on BOTH sides, matched by
    timestamp. A bar whose adj_close changed while its raw close did not is the
    signature of a split or dividend applied retroactively — and it means every
    older bar we hold is now on the wrong scale.

    Bars without adj_close (written before the column existed) carry no factor
    and are skipped rather than treated as a change: a legacy NULL is missing
    information, not evidence of a corporate action.
    """
    fresh = {bar.timestamp: bar for bar in fetched}
    for old in stored:
        new = fresh.get(old.timestamp)
        if new is None:
            continue
        old_factor, new_factor = _factor(old), _factor(new)
        if old_factor is None or new_factor is None or old_factor <= 0:
            continue
        if abs(new_factor - old_factor) / old_factor > tolerance:
            logger.info(
                "Adjustment factor changed — history restated",
                symbol=old.symbol,
                at=old.timestamp.isoformat(),
                stored=round(old_factor, 6),
                fetched=round(new_factor, 6),
            )
            return True
    return False


def is_weekend(moment: datetime) -> bool:
    """Saturday or Sunday in UTC.

    Only an optimization: a gap-based fetch on a closed day simply returns
    nothing new. Skipping the obvious closures avoids pointless calls to the
    provider; exchange holidays are not modelled and resolve the same harmless
    way, which is why no market-calendar dependency is pulled in for this.
    """
    return moment.astimezone(UTC).weekday() >= 5


__all__ = [
    "ADJUSTMENT_TOLERANCE",
    "FetchPlan",
    "adjustment_drifted",
    "full_history_plan",
    "is_weekend",
    "plan_fetch",
]
