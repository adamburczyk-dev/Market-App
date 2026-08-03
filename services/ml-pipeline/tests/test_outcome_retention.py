"""The ML-3 loop had two ways to die silently, and neither logged an error.

Both are bounds that encoded the 10-session horizon without saying so:

- ``OUTCOME_DROP_AFTER_DAYS = 42`` — "3 x horizon, in calendar days" written as
  a literal. At h=63 a vote needs ~91 calendar days to mature, so `too_old`
  fires while `triple_barrier_label` is still returning None (window not full),
  and the vote is marked resolved with `label=None`. Forever.
- ``INFERENCE_LOG_MAXLEN = 2000`` — a `deque` that evicts the OLDEST first,
  i.e. the votes about to mature. Already too small at h=10 (414 names x 10
  sessions = 4140).

Downstream of either, `record_outcome` is never called (the aggregator's
adaptive ml weight never learns), `rolling_metrics` always returns None, and
the drift check's performance arm reports "not measured" — which looks exactly
like a cold start.
"""

from datetime import UTC, datetime, timedelta

import pytest
from trading_common.schemas import Interval, OHLCVBar

from src.config import settings
from src.core.inference_log import InferenceLog, InferenceRecord, retention_for
from src.core.labels import LABEL_HORIZON, LabelParams
from src.core.outcomes import OutcomeResolver

MODEL = "global_v1@v1"
START = datetime(2024, 1, 1, tzinfo=UTC)


class HistoryMarket:
    """Market-data stub serving one rising path for every symbol asked about."""

    def __init__(self, bars: list[OHLCVBar]) -> None:
        self._bars = bars
        self.calls: list[str] = []

    async def get_ohlcv(self, symbol: str, interval: Interval, limit: int) -> list[OHLCVBar]:
        self.calls.append(symbol)
        return [b.model_copy(update={"symbol": symbol}) for b in self._bars[-limit:]]


def rising(n: int, symbol: str = "AAPL") -> list[OHLCVBar]:
    """+0.6%/session: wide enough to clear an upper barrier at any horizon."""
    out = []
    price = 100.0
    for i in range(n):
        out.append(
            OHLCVBar(
                symbol=symbol,
                timestamp=START + timedelta(days=i),
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=1_000_000,
                interval="1d",
                adj_close=price,
            )
        )
        price *= 1.006
    return out


def calm(n: int, symbol: str = "AAPL") -> list[OHLCVBar]:
    """Alternating +-0.5% steps: sigma is real, but the barriers are unreachable.

    Barrier half-width is sigma*sqrt(63) ~ 8 daily steps, while an alternating
    series never drifts more than one step from its start. So the label can
    only resolve on the VERTICAL barrier — which is exactly the state a vote is
    in while it waits, and the state the 42-day cutoff used to kill.
    """
    out = []
    for i in range(n):
        price = 100.0 * (1.005 if i % 2 else 0.995)
        out.append(
            OHLCVBar(
                symbol=symbol,
                timestamp=START + timedelta(days=i),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1_000_000,
                interval="1d",
                adj_close=price,
            )
        )
    return out


def pending_vote(at: datetime, symbol: str = "AAPL") -> InferenceRecord:
    return InferenceRecord(
        symbol=symbol,
        at=at,
        features={"rsi_14": 0.7},
        probability_up=0.8,
        signal="BUY",
    )


@pytest.mark.asyncio
async def test_a_matured_vote_resolves():
    """The ordinary case: entry a full horizon back, the whole window available."""
    params = LabelParams()
    history = rising(params.sigma_window + params.horizon + 100)
    entry = history[-(params.horizon + 1)].timestamp

    log = InferenceLog()
    log.append(MODEL, pending_vote(entry))
    resolver = OutcomeResolver(HistoryMarket(history), log, params)

    resolved = await resolver.resolve_pending(MODEL, now=history[-1].timestamp)

    assert resolved, "a matured vote produced no outcome"
    assert not log.pending(MODEL)
    assert log.counts(MODEL)["resolved"] == 1, "resolved WITHOUT a label is the silent failure"


@pytest.mark.asyncio
async def test_a_still_immature_vote_waits_instead_of_being_killed_by_the_cutoff():
    """The exact shape of the silent kill, and it needs no horizon flip to bite.

    A 63-session label needs ~91 calendar days. With the cutoff pinned at 42,
    a 45-day-old vote hits `too_old` while `triple_barrier_label` is still
    returning None because only 19 sessions of its window exist — so it is
    marked resolved with `label=None` and never looked at again. The next run
    has the bars but no longer has the vote.

    Derived from the label instead, the same vote simply stays pending.
    """
    params = LabelParams(horizon=63)
    history = calm(120)
    entry_bar = history[100]  # only 19 sessions after it — window nowhere near full

    log = InferenceLog()
    vote = pending_vote(entry_bar.timestamp)
    log.append(MODEL, vote)
    resolver = OutcomeResolver(HistoryMarket(history), log, params)

    resolved = await resolver.resolve_pending(MODEL, now=entry_bar.timestamp + timedelta(days=45))

    assert resolved == []
    assert vote in log.pending(MODEL), "an immature vote was dropped and can never resolve"
    assert vote.label is None and not vote.resolved


@pytest.mark.asyncio
async def test_the_drop_cutoff_tracks_the_resolvers_own_label():
    """Constructed without an explicit cutoff, the resolver derives its own.

    A resolver replaying a 63-session label with a 42-day cutoff is the defect;
    the cutoff is not an independent policy, it is a function of the label.
    """
    long_label = LabelParams(horizon=63)
    resolver = OutcomeResolver(HistoryMarket(rising(10)), InferenceLog(), long_label)
    assert resolver._drop_after_days > 63 * 365.25 / 252


def test_the_log_retains_a_pending_vote_for_a_whole_horizon():
    """Every served name appends a record per session, HOLDs included.

    At maxlen=2000 and 414 names the log remembered 4.8 sessions, so a vote was
    evicted long before it could mature — and `pending()` kept returning a
    small, healthy-looking set.
    """
    universe = 414
    log = InferenceLog()
    vote = pending_vote(START)
    log.append(MODEL, vote)

    # A full horizon of ordinary traffic passes over it.
    for session in range(LABEL_HORIZON):
        for n in range(universe):
            log.append(
                MODEL,
                InferenceRecord(
                    symbol=f"SYM{n}",
                    at=START + timedelta(days=session + 1),
                    features={},
                    probability_up=0.5,
                    signal="HOLD",
                ),
            )

    assert vote in log.pending(MODEL), "the vote was evicted before it could mature"


def test_retention_is_derived_from_the_universe_and_the_horizon():
    assert retention_for() == settings.INFERENCE_LOG_MAXLEN
    assert retention_for(414, LABEL_HORIZON) >= 414 * LABEL_HORIZON


@pytest.mark.asyncio
async def test_one_fetch_per_symbol_not_per_pending_vote():
    """Several pending votes on one name used to re-download the same window."""
    params = LabelParams()
    history = rising(params.sigma_window + params.horizon + 80)
    market = HistoryMarket(history)
    log = InferenceLog()
    for offset in range(3):
        log.append(MODEL, pending_vote(history[params.sigma_window + offset].timestamp))

    await resolver_for(market, log, params).resolve_pending(MODEL, now=history[-1].timestamp)

    assert market.calls == ["AAPL"], f"expected one fetch per symbol, got {market.calls}"


def resolver_for(market, log, params) -> OutcomeResolver:
    return OutcomeResolver(market, log, params)
