"""Delayed-label outcome resolution (plan §9) — closes the learning loop.

A published ML vote matures ~``horizon`` sessions later. The resolver replays
the SAME rule the labels were trained with over fresh market-data history:
which barrier did the path touch first? A vote whose entry bar cannot be
matched or that stays unresolved past ``drop_after_days`` is dropped (marked
resolved with no label) so the pending queue cannot grow unbounded.

Two quantities come out of one resolution and they are deliberately measured on
DIFFERENT bases:

- ``signed_return`` feeds the aggregator's adaptive "ml" weight. It is money,
  and money is absolute — a name that fell 3% while the market fell 5% did not
  pay anyone 2%.
- ``label`` / ``correct`` feed rolling accuracy and the drift check's
  performance arm, which is scored against ``ACCURACY_MIN``. They must use the
  rule the model was TRAINED on, or the monitor compares the model against a
  question it was never asked.

Under an excess label that means the benchmark leg has to exist at serving
time too — assembled once per run from the pending set with the same
``market_path`` module ``build_dataset`` uses. A benchmark ETF is not a
substitute: training uses the universe median, and swapping it here is exactly
the train/serve divergence this module exists to detect.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import structlog
from trading_common.prices import adjusted_ohlc
from trading_common.schemas import Interval, OHLCVBar

from src.core.inference_log import InferenceLog, InferenceRecord
from src.core.labels import (
    LabelParams,
    excess_barrier_label,
    outcome_drop_after_days,
    triple_barrier_label,
)
from src.core.market_data_client import MarketDataClient
from src.core.market_path import market_levels, project_levels

logger = structlog.get_logger()

# Slack on top of what a label strictly needs (sigma window + horizon). Enough
# that a monitor run delayed by a holiday week still finds the entry bar with
# its full trailing window intact.
READ_MARGIN = 60


class OutcomeResolver:
    def __init__(
        self,
        market: MarketDataClient,
        log: InferenceLog,
        label_params: LabelParams | None = None,
        interval: Interval = Interval.D1,
        drop_after_days: int | None = None,
    ) -> None:
        self._market = market
        self._log = log
        self._params = label_params or LabelParams()
        self._interval = interval
        # Derived from the label the resolver is actually replaying, so a
        # horizon change cannot leave the cutoff behind. A cutoff shorter than
        # the label window makes EVERY vote resolve as label=None: the adaptive
        # weight never learns, rolling metrics stay empty, and the drift check
        # reports "not measured" forever — indistinguishable from a cold start,
        # and nothing logs an error.
        self._drop_after_days = (
            outcome_drop_after_days(self._params.horizon)
            if drop_after_days is None
            else drop_after_days
        )

    async def resolve_pending(self, model_id: str, now: datetime | None = None) -> list[float]:
        """Resolve matured votes; returns the signed returns resolved this run."""
        now = now or datetime.now(UTC)
        pending = list(self._log.pending(model_id))
        if not pending:
            return []

        # One fetch per SYMBOL, not per record: several pending votes on one
        # name used to re-download the same window each time.
        limit = self._params.sigma_window + self._params.horizon + READ_MARGIN
        bars_by_symbol: dict[str, list[OHLCVBar]] = {}
        for symbol in sorted({record.symbol for record in pending}):
            bars_by_symbol[symbol] = await self._market.get_ohlcv(
                symbol, self._interval, limit=limit
            )

        market_by_symbol: dict[str, np.ndarray] = {}
        if self._params.excess:
            levels = market_levels(bars_by_symbol)
            market_by_symbol = {
                symbol: project_levels(levels, bars) for symbol, bars in bars_by_symbol.items()
            }

        resolved: list[float] = []
        for record in pending:
            outcome_return = self._resolve_one(
                record,
                now,
                bars_by_symbol.get(record.symbol, []),
                market_by_symbol.get(record.symbol),
            )
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

    def _resolve_one(
        self,
        record: InferenceRecord,
        now: datetime,
        bars: list[OHLCVBar],
        market: np.ndarray | None,
    ) -> float | None:
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

        # ADJUSTED, like build_dataset: barriers are compared against highs and
        # lows, so the whole bar must live on one scale. Reading raw closes here
        # while training reads adjusted ones made a split inside the label window
        # resolve a barrier that the trained rule would not have touched — a
        # train/serve divergence that only shows up on corporate actions.
        _, highs, lows, closes = adjusted_ohlc(bars)
        if self._params.excess:
            if market is None:
                return None
            outcome = excess_barrier_label(closes, market, entry_index, self._params)
        else:
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

        # Absolute on purpose, even under an excess label — see the module
        # docstring. This number is what the position actually paid.
        raw_return = float(closes[outcome.touch_index] / closes[entry_index] - 1.0)
        signed = raw_return if record.signal == "BUY" else -raw_return
        correct = (outcome.label == 1) == (record.signal == "BUY")
        self._log.resolve(record, outcome.label, signed, correct, now)
        return signed
