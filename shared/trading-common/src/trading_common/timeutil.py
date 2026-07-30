"""Timezone normalization at storage boundaries.

Every producer in this system writes UTC, but not every reader gets it back
that way: a Postgres TIMESTAMPTZ comes back aware, sqlite comes back naive
because the driver drops the zone, and a hand-posted payload can be either.
Comparing a naive datetime to an aware one raises TypeError, so code that works
on one backend crashes on the other — which is how the point-in-time
fundamentals join first hit this, and how the market-data incremental sync hit
it again when comparing the newest stored bar against "now".

Normalize at the boundary that produced the value (the repository), not at each
comparison, so a caller never has to know which backend it is talking to.
"""

from datetime import UTC, datetime

__all__ = ["as_utc"]


def as_utc(moment: datetime) -> datetime:
    """Treat a naive timestamp as UTC; leave an aware one alone."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
