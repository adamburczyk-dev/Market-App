"""Which (symbol, side, session) orders have already been placed.

The aggregator publishes a decision every time a component arrives, and that is
correct: a regime change or a fresh ML vote is new information and the decision
must be re-evaluated. What must NOT happen is a second *order* for a decision
that has already been acted on — the strategy BUY is sized, then the ML vote
lands, the aggregate is re-published as BUY, and risk-mgmt sizes it again,
DOUBLING the position (finding N2). The split of responsibility:

- signal-aggregator **merges** components inside a short join window, so the
  common case emits one decision;
- risk-mgmt is **idempotent**, so even a late component, a durable redelivery
  or a restart replay cannot open the same position twice.

The session key is the UTC date of the *event* timestamp (never wall clock), so
a replay of yesterday's stream is judged against yesterday's session.
"""

from datetime import UTC, datetime


def session_of(timestamp: datetime) -> str:
    """UTC session date of an event timestamp (ISO date string)."""
    if timestamp.tzinfo is None:
        return timestamp.date().isoformat()
    return timestamp.astimezone(UTC).date().isoformat()


class OrderLedger:
    """Bounded record of orders already placed, keyed by (symbol, side, session)."""

    def __init__(self, keep_sessions: int = 5) -> None:
        self._placed: dict[str, str] = {}  # key -> session
        self._keep_sessions = keep_sessions

    @staticmethod
    def _key(symbol: str, side: str, session: str) -> str:
        return f"{symbol}|{side}|{session}"

    def already_placed(self, symbol: str, side: str, session: str) -> bool:
        return self._key(symbol, side, session) in self._placed

    def record(self, symbol: str, side: str, session: str) -> None:
        self._placed[self._key(symbol, side, session)] = session
        self._prune()

    def _prune(self) -> None:
        """Keep only the most recent ``keep_sessions`` sessions — the ledger is a
        guard against same-session duplicates, not an audit trail."""
        sessions = sorted(set(self._placed.values()), reverse=True)
        if len(sessions) <= self._keep_sessions:
            return
        keep = set(sessions[: self._keep_sessions])
        self._placed = {k: v for k, v in self._placed.items() if v in keep}

    def snapshot(self) -> dict[str, str]:
        return dict(self._placed)

    def restore(self, data: dict[str, str] | None) -> None:
        if not data:
            return
        self._placed = {str(k): str(v) for k, v in data.items()}
        self._prune()
