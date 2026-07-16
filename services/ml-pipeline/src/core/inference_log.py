"""Rolling log of served inferences — the live data the daily monitor reads.

In-memory and bounded (single replica; a restart clears the window and the
monitor quietly waits for it to refill — the same cold-start trade-off as the
feature store). Records double as pending-outcome trackers: an actionable vote
matures after the label horizon and is resolved against market-data history;
resolved outcomes feed rolling accuracy/Sharpe (decay detection) and the
aggregator's adaptive weights (``record_outcome``).
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

TRADING_DAYS = 252


@dataclass
class InferenceRecord:
    symbol: str
    at: datetime
    features: dict[str, float]  # assembled input row, by feature name
    probability_up: float
    signal: str  # "BUY" | "SELL" | "HOLD" — dead-zone outputs are logged too
    resolved: bool = False
    label: int | None = None  # triple-barrier outcome (None = dropped unresolved)
    signed_return: float | None = None  # realized return signed by the vote direction
    correct: bool | None = None
    resolved_at: datetime | None = None


@dataclass
class InferenceLog:
    maxlen: int = 2000
    _records: dict[str, deque[InferenceRecord]] = field(init=False)

    def __post_init__(self) -> None:
        self._records = defaultdict(lambda: deque(maxlen=self.maxlen))

    def append(self, model_id: str, record: InferenceRecord) -> None:
        self._records[model_id].append(record)

    def counts(self, model_id: str) -> dict[str, int]:
        records = self._records.get(model_id, deque())
        return {
            "total": len(records),
            "pending": len(self.pending(model_id)),
            "resolved": sum(1 for r in records if r.resolved and r.label is not None),
        }

    def feature_window(self, model_id: str, limit: int = 500) -> dict[str, list[float]]:
        """Per-feature value lists over the most recent inferences (PSI input)."""
        records = list(self._records.get(model_id, deque()))[-limit:]
        window: dict[str, list[float]] = defaultdict(list)
        for record in records:
            for name, value in record.features.items():
                window[name].append(value)
        return dict(window)

    def prediction_window(self, model_id: str, limit: int = 500) -> list[float]:
        records = list(self._records.get(model_id, deque()))[-limit:]
        return [r.probability_up for r in records]

    def pending(self, model_id: str) -> list[InferenceRecord]:
        """Actionable (BUY/SELL) votes awaiting outcome resolution."""
        return [
            r
            for r in self._records.get(model_id, deque())
            if not r.resolved and r.signal in ("BUY", "SELL")
        ]

    def resolve(
        self,
        record: InferenceRecord,
        label: int | None,
        signed_return: float | None,
        correct: bool | None,
        resolved_at: datetime | None = None,
    ) -> None:
        record.resolved = True
        record.label = label
        record.signed_return = signed_return
        record.correct = correct
        record.resolved_at = resolved_at or datetime.now(UTC)

    def rolling_metrics(
        self,
        model_id: str,
        window_days: int,
        horizon_days: int,
        min_outcomes: int = 10,
        now: datetime | None = None,
    ) -> tuple[float, float] | None:
        """(annualized Sharpe, accuracy) over outcomes resolved in the window.

        Returns None when fewer than ``min_outcomes`` resolved — the caller
        must then use neutral inputs rather than fabricated performance.
        Outcomes are h-session trade returns, so Sharpe annualizes with
        √(252/h) — an approximation (overlapping trades), documented.
        """
        cutoff = (now or datetime.now(UTC)) - timedelta(days=window_days)
        outcomes = [
            r
            for r in self._records.get(model_id, deque())
            if r.resolved
            and r.label is not None
            and r.resolved_at is not None
            and r.resolved_at >= cutoff
        ]
        if len(outcomes) < min_outcomes:
            return None
        returns = np.array([r.signed_return for r in outcomes], dtype=float)
        std = float(returns.std(ddof=1))
        sharpe = (
            float(returns.mean() / std * math.sqrt(TRADING_DAYS / horizon_days)) if std > 0 else 0.0
        )
        accuracy = float(np.mean([1.0 if r.correct else 0.0 for r in outcomes]))
        return sharpe, accuracy
