"""The scheduled pull must resume from what we hold, not from yesterday.

Two failure modes are worth more than the happy path. A missed run that leaves
a permanent hole — nothing in the system ever goes back for a day it skipped.
And a corporate action restating adj_close, which is the dangerous one: it
produces no error anywhere, just a series that is silently on two different
scales either side of the split, feeding features and labels that are computed
on adjusted prices.
"""

from datetime import UTC, datetime, timedelta

import pytest
from trading_common.schemas import Interval, OHLCVBar

from src.core.incremental import (
    adjustment_drifted,
    full_history_plan,
    is_weekend,
    plan_fetch,
)

NOW = datetime(2026, 7, 30, 23, 0, tzinfo=UTC)


_SAME = object()  # "adj_close equals close" — distinct from an explicit None


def bar(day: datetime, close: float = 100.0, adj: float | None | object = _SAME) -> OHLCVBar:
    return OHLCVBar(
        symbol="AAPL",
        timestamp=day,
        interval=Interval.D1,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        adj_close=close if adj is _SAME else adj,  # type: ignore[arg-type]
        volume=1_000_000.0,
    )


# --- what window to ask for ------------------------------------------------


def test_an_empty_symbol_gets_a_full_history():
    plan = plan_fetch(None, NOW, overlap_days=5, initial_history_days=2000)
    assert plan.is_full
    assert plan.end == NOW
    assert (NOW - plan.start).days == 2000


def test_a_stored_symbol_resumes_from_its_newest_bar():
    latest = NOW - timedelta(days=3)
    plan = plan_fetch(latest, NOW, overlap_days=5)
    assert plan.mode == "incremental"
    assert plan.start == latest - timedelta(days=5)
    assert plan.end == NOW


def test_a_missed_week_is_repaired_by_the_next_run():
    """The requirement the whole design exists for: a run that did not happen
    must not cost those days permanently. A week of downtime becomes a
    week-wide window, not a one-day one."""
    latest = NOW - timedelta(days=7)
    plan = plan_fetch(latest, NOW, overlap_days=5)
    assert (plan.end - plan.start).days == 12  # 7 missed + 5 overlap
    # ...and specifically NOT a single day, which is what a naive "fetch today"
    # scheduler would have asked for
    assert (plan.end - plan.start).days > 1


def test_the_overlap_reaches_back_past_the_newest_bar():
    """Providers restate recent bars, and the newest one we hold may have been
    captured mid-session. Re-asking costs nothing — the upsert is idempotent."""
    latest = NOW - timedelta(days=1)
    plan = plan_fetch(latest, NOW, overlap_days=5)
    assert plan.start < latest


def test_a_future_timestamp_cannot_invert_the_window():
    """Clock skew or a provider stamping tomorrow must not produce start > end,
    which would be an empty or nonsensical request."""
    plan = plan_fetch(NOW + timedelta(days=10), NOW, overlap_days=0)
    assert plan.start <= plan.end


def test_full_history_plan_is_anchored_on_now():
    plan = full_history_plan(NOW, initial_history_days=100)
    assert plan.is_full and plan.end == NOW and (NOW - plan.start).days == 100


# --- the silent corruption -------------------------------------------------


def test_a_split_that_restated_the_history_is_detected():
    """A 2:1 split halves adj_close for every earlier bar while raw close is
    untouched at those dates. Incrementally fetched, our old bars keep the
    pre-split figure and the series is wrong exactly at the join."""
    day = NOW - timedelta(days=2)
    stored = [bar(day, close=100.0, adj=100.0)]
    after_split = [bar(day, close=100.0, adj=50.0)]
    assert adjustment_drifted(stored, after_split)


def test_a_dividend_sized_restatement_is_detected():
    day = NOW - timedelta(days=2)
    stored = [bar(day, close=100.0, adj=100.0)]
    assert adjustment_drifted(stored, [bar(day, close=100.0, adj=99.4)])


def test_an_unchanged_history_is_not_re_fetched():
    """The check must be quiet in the normal case, or every run would drag the
    full history for 486 symbols."""
    days = [NOW - timedelta(days=n) for n in (2, 3, 4)]
    stored = [bar(d, close=100.0 + n, adj=99.0 + n) for n, d in enumerate(days)]
    assert not adjustment_drifted(stored, list(stored))


def test_floating_point_noise_is_not_a_corporate_action():
    day = NOW - timedelta(days=2)
    stored = [bar(day, close=100.0, adj=98.7)]
    assert not adjustment_drifted(stored, [bar(day, close=100.0, adj=98.7000001)])


def test_a_price_move_alone_is_not_a_restatement():
    """The discriminator is the FACTOR, not the price. A bar whose close and
    adj_close moved together is a market move; only their ratio changing means
    the history was restated."""
    day = NOW - timedelta(days=2)
    stored = [bar(day, close=100.0, adj=98.0)]
    corrected = [bar(day, close=200.0, adj=196.0)]  # same 0.98 factor
    assert not adjustment_drifted(stored, corrected)


def test_legacy_rows_without_adj_close_are_skipped_not_flagged():
    """A NULL adj_close is missing information, not evidence of a split.
    Treating it as drift would re-fetch the whole universe on the first run
    after the column was added."""
    day = NOW - timedelta(days=2)
    stored = [bar(day, close=100.0, adj=None)]
    assert not adjustment_drifted(stored, [bar(day, close=100.0, adj=50.0)])
    assert not adjustment_drifted([bar(day, 100.0, 100.0)], [bar(day, 100.0, None)])


def test_bars_we_do_not_hold_cannot_be_compared():
    """Only dates present on both sides carry evidence; new bars are just new."""
    stored = [bar(NOW - timedelta(days=10), 100.0, 100.0)]
    fetched = [bar(NOW - timedelta(days=1), 100.0, 50.0)]
    assert not adjustment_drifted(stored, fetched)
    assert not adjustment_drifted([], fetched)


def test_a_zero_price_cannot_reach_the_comparison_at_all():
    """The divide-by-zero guard in `_factor` is defence in depth — the contract
    rejects a non-positive price before a bar can ever be built, which is the
    protection that actually matters."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        bar(NOW, close=0.0, adj=0.0)


# --- calendar --------------------------------------------------------------


def test_weekends_are_recognised():
    assert is_weekend(datetime(2026, 8, 1, 12, tzinfo=UTC))  # Saturday
    assert is_weekend(datetime(2026, 8, 2, 12, tzinfo=UTC))  # Sunday
    assert not is_weekend(datetime(2026, 7, 31, 12, tzinfo=UTC))  # Friday
    assert not is_weekend(datetime(2026, 8, 3, 12, tzinfo=UTC))  # Monday


def test_weekend_is_judged_in_utc():
    """A Friday-evening moment in a positive offset is already Saturday UTC;
    the schedule runs on UTC, so the check must agree with it."""
    from datetime import timezone

    friday_late = datetime(2026, 7, 31, 22, tzinfo=timezone(timedelta(hours=-5)))
    assert is_weekend(friday_late) == (friday_late.astimezone(UTC).weekday() >= 5)


@pytest.mark.parametrize("overlap", [0, 1, 30])
def test_any_overlap_keeps_the_window_valid(overlap: int):
    plan = plan_fetch(NOW - timedelta(days=2), NOW, overlap_days=overlap)
    assert plan.start <= plan.end
