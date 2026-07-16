"""Delayed-label outcome resolution (plan §9) — closes the learning loop.

A published ML vote matures ~``horizon`` sessions later. The resolver replays
the SAME triple-barrier rule the labels were trained with over fresh
market-data history: which barrier did the path touch first? The realized,
direction-signed return feeds the aggregator's adaptive "ml" weight and the
rolling accuracy/Sharpe that the drift check reads. A vote whose entry bar
cannot be matched or that stays unresolved past ``drop_after_days`` is dropped
(marked resolved with no label) so the pending queue cannot grow unbounded.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import structlog
from trading_common.schemas import Interval

from src.core.inference_log import InferenceLog, InferenceRecord
from src.core.labels import LabelParams, triple_barrier_label
from src.core.market_data_client import MarketDataClient

logger = structlog.get_logger()


class OutcomeResolver:
    def __init__(
        self,
        market: MarketDataClient,
        log: InferenceLog,
        label_params: LabelParams | None = None,
        interval: Interval = Interval.D1,
        drop_after_days: int = 42,  # ~3× the 10-session horizon in calendar days
    ) -> None:
        self._market = market
        self._log = log
        self._params = label_params or LabelParams()
        self._interval = interval
        self._drop_after_days = drop_after_days

    async def resolve_pending(self, model_id: str, now: datetime | None = None) -> list[float]:
        """Resolve matured votes; returns the signed returns resolved this run."""
        now = now or datetime.now(UTC)
        resolved: list[float] = []
        for record in self._log.pending(model_id):
            outcome_return = await self._resolve_one(record, now)
            if outcome_return is not None:
                resolved.append(outcome_return)
        if resolved:
            logger.info(
                "Outcomes resolved",
                model_id=model_id,
                count=len(resolved),
                mean_return=round(float(np.mean(resolved)), 5),
            )
        return resolved

    async def _resolve_one(self, record: InferenceRecord, now: datetime) -> float | None:
        limit = self._params.sigma_window + self._params.horizon + 40
        bars = await self._market.get_ohlcv(record.symbol, self._interval, limit=limit)
        entry_index: int | None = None
        for i, bar in enumerate(bars):
            if bar.timestamp.date() <= record.at.date():
                entry_index = i
            else:
                break
        too_old = now - record.at > timedelta(days=self._drop_after_days)

        if entry_index is None or entry_index < self._params.sigma_window:
            if too_old:  # entry bar unrecoverable — drop instead of retrying forever
                self._log.resolve(record, None, None, None, now)
                logger.warning(
                    "Vote dropped — entry bar unmatched", symbol=record.symbol, at=str(record.at)
                )
            return None

        closes = np.array([b.close for b in bars], dtype=float)
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        outcome = triple_barrier_label(closes, highs, lows, entry_index, self._params)
        if outcome is None:
            if too_old:
                self._log.resolve(record, None, None, None, now)
                logger.warning(
                    "Vote dropped — unresolved past cutoff",
                    symbol=record.symbol,
                    at=str(record.at),
                )
            return None  # immature (window not full yet) — retry next run

        raw_return = float(closes[outcome.touch_index] / closes[entry_index] - 1.0)
        signed = raw_return if record.signal == "BUY" else -raw_return
        correct = (outcome.label == 1) == (record.signal == "BUY")
        self._log.resolve(record, outcome.label, signed, correct, now)
        return signed
