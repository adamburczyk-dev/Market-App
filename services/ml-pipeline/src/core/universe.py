"""Point-in-time universe selection (P3-1) — and an honest survivorship report.

Today's 34 names are a list of winners chosen after the fact. A model trained
on it learns "large technology companies go up", because in that data that is
the only thing there is to learn. The fix has two halves and only one of them
is code:

**The mechanism (here).** Membership is decided on a rebalance date from data
available BEFORE it — trailing median dollar volume, top N, held until the next
rebalance. A name that qualified in 2012 is in the 2012 cross-sections whether
or not it still exists today, so the selection itself introduces no hindsight.

**The candidate list (not here).** If the tickers fed in are the ones that
survived, no selection rule recovers the ones that did not. So this module
refuses to pretend: `survivorship_report` measures how many names actually
enter and leave over the window and says plainly when the answer is "none",
which means the universe is a survivor list and the metrics computed on it are
optimistic by an amount nobody can estimate from the inside.

That second part is the same discipline as `share_neutralized_against_peers`
in the sector study: report the precondition, because a number computed on data
that cannot support it looks exactly like a number that can.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import structlog
from trading_common.features import FULL_HISTORY
from trading_common.schemas import OHLCVBar

logger = structlog.get_logger()


@dataclass(frozen=True)
class UniverseParams:
    """How membership is decided. Every value is a liquidity/limits choice, not
    a performance one — nothing here looks at returns, so the selection cannot
    accidentally become a strategy."""

    top_n: int = 200
    # Quarterly. Rebalancing more often chases noise in the ranking; less often
    # lets a name that collapsed in liquidity stay in the book for a year.
    rebalance_days: int = 63
    # Trailing window for the liquidity measure. The MEDIAN over it, not the
    # mean: one earnings-day volume spike should not buy a place in the universe.
    liquidity_window: int = 63
    # A name must have enough history to produce a full feature vector, or it
    # would enter the cross-section with half its columns neutral-filled.
    min_history: int = FULL_HISTORY
    min_dollar_volume: float = 0.0  # optional hard floor, in dollars


@dataclass(frozen=True)
class Universe:
    """Membership per rebalance date, plus what it took to get there."""

    members_by_date: dict[date, tuple[str, ...]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def members_on(self, session: date) -> frozenset[str]:
        """Who was in the universe on this session (the last rebalance at or
        before it). Before the first rebalance the universe is empty — not
        'everyone', which would quietly restore the survivor list."""
        eligible = [d for d in self.members_by_date if d <= session]
        if not eligible:
            return frozenset()
        return frozenset(self.members_by_date[max(eligible)])

    @property
    def all_members(self) -> frozenset[str]:
        return frozenset(s for members in self.members_by_date.values() for s in members)


def _as_date(moment: datetime | date) -> date:
    return moment.date() if isinstance(moment, datetime) else moment


def _trailing_dollar_volume(bars: list[OHLCVBar], upto: int, window: int) -> float | None:
    """Median dollar volume over the `window` bars ending at `upto` (inclusive).

    Raw close x raw volume: this is money that actually changed hands, and
    adjusting the price would inflate historical turnover by every split since.
    """
    start = max(0, upto - window + 1)
    if upto - start + 1 < window // 2:  # not enough of the window to judge
        return None
    values = [b.close * b.volume for b in bars[start : upto + 1]]
    return float(np.median(values)) if values else None


def build_universe(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    params: UniverseParams | None = None,
) -> Universe:
    """Select the top-N most liquid eligible names on each rebalance date."""
    p = params or UniverseParams()
    ordered = {s: sorted(b, key=lambda x: x.timestamp) for s, b in bars_by_symbol.items()}
    index_by_date = {
        s: {_as_date(b.timestamp): i for i, b in enumerate(bars)} for s, bars in ordered.items()
    }
    all_sessions = sorted({_as_date(b.timestamp) for bars in ordered.values() for b in bars})
    if not all_sessions:
        return Universe(diagnostics={"sessions": 0, "rebalances": 0})

    rebalance_dates = all_sessions[:: p.rebalance_days]
    members_by_date: dict[date, tuple[str, ...]] = {}
    sizes: list[int] = []
    churn: list[float] = []
    previous: frozenset[str] = frozenset()

    for rebalance in rebalance_dates:
        scored: list[tuple[float, str]] = []
        for symbol, bars in ordered.items():
            i = index_by_date[symbol].get(rebalance)
            if i is None or i + 1 < p.min_history:
                continue
            liquidity = _trailing_dollar_volume(bars, i, p.liquidity_window)
            if liquidity is None or liquidity < p.min_dollar_volume:
                continue
            scored.append((liquidity, symbol))
        # descending liquidity, symbol as the tiebreak so the result is deterministic
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        members = tuple(symbol for _, symbol in scored[: p.top_n])
        members_by_date[rebalance] = members
        sizes.append(len(members))
        current = frozenset(members)
        if previous:
            changed = len(current ^ previous) / max(len(current | previous), 1)
            churn.append(changed)
        previous = current

    diagnostics = {
        "sessions": len(all_sessions),
        "first_session": all_sessions[0].isoformat(),
        "last_session": all_sessions[-1].isoformat(),
        "rebalances": len(rebalance_dates),
        "rebalance_days": p.rebalance_days,
        "top_n": p.top_n,
        "universe_size_median": float(np.median(sizes)) if sizes else 0.0,
        "universe_size_min": min(sizes) if sizes else 0,
        "universe_size_max": max(sizes) if sizes else 0,
        "turnover_mean": round(float(np.mean(churn)), 4) if churn else 0.0,
        "candidates": len(bars_by_symbol),
        "never_selected": sorted(set(bars_by_symbol) - set(_flatten(members_by_date))),
    }
    logger.info("Universe built", **{k: v for k, v in diagnostics.items() if k != "never_selected"})
    return Universe(members_by_date=members_by_date, diagnostics=diagnostics)


def _flatten(members_by_date: dict[date, tuple[str, ...]]) -> list[str]:
    return [s for members in members_by_date.values() for s in members]


def survivorship_report(bars_by_symbol: dict[str, list[OHLCVBar]]) -> dict[str, Any]:
    """Do names actually enter and leave this dataset, or is it a survivor list?

    The measurement that decides whether P3-1's mechanism is doing anything.
    A universe where every candidate has data from the first session to the last
    contains no delistings, no acquisitions and no collapses — every company in
    it made it to today, which is a fact about the ticker list, not about the
    selection rule applied to it. Saying so is the only honest option: the size
    of the resulting optimism cannot be estimated from inside the data.
    """
    spans: dict[str, tuple[date, date]] = {}
    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue
        stamps = sorted(_as_date(b.timestamp) for b in bars)
        spans[symbol] = (stamps[0], stamps[-1])
    if not spans:
        return {"candidates": 0, "verdict": "no data"}

    first = min(start for start, _ in spans.values())
    last = max(end for _, end in spans.values())
    # A tolerance of one rebalance quarter: a name that stops a few days early
    # because of a data gap is not a delisting.
    tolerance_days = 90
    late_starters = [s for s, (start, _) in spans.items() if (start - first).days > tolerance_days]
    early_enders = [s for s, (_, end) in spans.items() if (last - end).days > tolerance_days]

    survivors = len(spans) - len(set(late_starters) | set(early_enders))
    verdict = (
        "SURVIVOR LIST — every candidate spans the whole window, so there are no "
        "delistings in this data. Point-in-time selection cannot recover names "
        "that were never supplied; results here are optimistic by an unknown "
        "amount. Fix the ticker list, not the code."
        if not early_enders
        else f"{len(early_enders)} names stop before the end — the data contains exits."
    )
    return {
        "candidates": len(spans),
        "first_session": first.isoformat(),
        "last_session": last.isoformat(),
        "names_entering_late": len(late_starters),
        "names_ending_early": len(early_enders),
        "names_spanning_everything": survivors,
        "verdict": verdict,
    }


__all__ = ["Universe", "UniverseParams", "build_universe", "survivorship_report"]
