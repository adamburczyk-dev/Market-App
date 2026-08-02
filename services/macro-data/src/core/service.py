"""MacroDataService — fetch macro indicators, classify the regime, publish events."""

from datetime import UTC, date, datetime, timedelta

import structlog
from trading_common.events import MacroUpdatedEvent, RegimeChangedEvent
from trading_common.schemas import MacroObservation, MacroRegime, MacroSnapshot

from src.core.fred_client import MacroFetcher
from src.core.regime import RegimeThresholds, classify_regime
from src.core.repository import MacroStore, NullMacroStore
from src.events.publisher import Publisher

logger = structlog.get_logger()

# Indicators that feed the snapshot; FRED provides a subset, the rest arrive via overrides.
_INDICATOR_KEYS = (
    "yield_curve_10y_2y",
    "credit_spread_baa_10y",
    "pmi",
    "cpi_yoy",
    "unemployment_rate",
    "fed_funds_rate",
)


class MacroDataService:
    def __init__(
        self,
        fetcher: MacroFetcher,
        publisher: Publisher,
        thresholds: RegimeThresholds | None = None,
        store: MacroStore | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._publisher = publisher
        self._thresholds = thresholds or RegimeThresholds()
        self._store: MacroStore = store or NullMacroStore()
        self._snapshot: MacroSnapshot | None = None
        self._regime: MacroRegime | None = None

    # --- history (P2-4) ---------------------------------------------------

    async def backfill(
        self,
        series_by_indicator: dict[str, str],
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, int]:
        """Pull every VINTAGE of each series and store it. Rows written per series.

        Deliberately one series at a time and sequential: ALFRED returns one row
        per (period, revision), so a 20-year daily series is tens of thousands of
        rows, and firing six of those concurrently at a rate-limited public API
        is how a backfill turns into a ban.
        """
        written: dict[str, int] = {}
        for indicator, series_id in series_by_indicator.items():
            observations = await self._fetcher.fetch_vintage_history(series_id, start, end)
            written[indicator] = await self._store.save(observations)
            logger.info(
                "Backfilled series",
                indicator=indicator,
                series_id=series_id,
                rows=written[indicator],
            )
        return written

    async def regime_on(self, day: date) -> MacroRegime | None:
        """The regime as it would have been classified ON `day`.

        Derived at read time from stored observations rather than stored as a
        label. Persisting the classification would freeze one version of
        `classify_regime` into the data: change a threshold and history would
        keep asserting the old verdict, with nothing to say it was computed
        under different rules.
        """
        indicators = await self._store.as_of(day)
        return classify_regime(
            yield_curve_10y_2y=indicators.get("T10Y2Y"),
            credit_spread_baa_10y=indicators.get("BAA10Y"),
            pmi=indicators.get("PMI"),
            thresholds=self._thresholds,
        )

    async def regime_history(self, start: date, end: date) -> dict[str, str]:
        """Date → regime for every day in the range that can be classified.

        ONE pass over the panel, not one query per day: a 20-year request is
        7300 days, and asking the database each time would be 7300 round trips
        to recompute something that fits in memory many times over.

        A day that cannot be classified is simply ABSENT from the result rather
        than filled with a default. `_regime_one_hot` in ml-pipeline already
        turns a missing regime into all-zeros, and an invented "expansion" is a
        made-up fact where the honest answer is a missing one.
        """
        panel = await self._store.panel()
        # A row becomes usable on the later of its two dates: it must describe a
        # period that has happened AND already have been published.
        pending = sorted(
            panel, key=lambda o: (max(o.observation_date, o.realtime_start or _NEVER), o.series)
        )
        cursor = 0
        # series -> (observation_date, realtime_start, value) currently in force
        current: dict[str, tuple[date, date, float]] = {}

        out: dict[str, str] = {}
        day = start
        while day <= end:
            while cursor < len(pending):
                obs = pending[cursor]
                visible_from = max(obs.observation_date, obs.realtime_start or _NEVER)
                if visible_from > day:
                    break
                cursor += 1
                if obs.realtime_start is None:
                    continue  # undated: never usable point-in-time
                held = current.get(obs.series)
                candidate = (obs.observation_date, obs.realtime_start, obs.value)
                # Newest PERIOD wins; within one period, the newest VINTAGE wins.
                if held is None or (candidate[0], candidate[1]) > (held[0], held[1]):
                    current[obs.series] = candidate
            regime = classify_regime(
                yield_curve_10y_2y=_value(current, "T10Y2Y"),
                credit_spread_baa_10y=_value(current, "BAA10Y"),
                pmi=_value(current, "PMI"),
                thresholds=self._thresholds,
            )
            if regime is not None:
                out[day.isoformat()] = regime.value
            day = day + timedelta(days=1)
        return out

    async def coverage(self) -> dict[str, dict[str, str | int]]:
        return await self._store.coverage()

    async def series_history(
        self, series: str, start: date | None = None, end: date | None = None
    ) -> list[MacroObservation]:
        return await self._store.series_history(series, start, end)

    @property
    def snapshot(self) -> MacroSnapshot | None:
        return self._snapshot

    @property
    def regime(self) -> MacroRegime | None:
        return self._regime

    async def refresh(self, overrides: dict[str, float | None] | None = None) -> MacroSnapshot:
        """Fetch indicators (FRED + manual overrides), classify regime, publish events.

        ``overrides`` (e.g. PMI/CPI that FRED doesn't serve here, or manual inputs)
        take precedence over fetched values.
        """
        indicators: dict[str, float | None] = dict.fromkeys(_INDICATOR_KEYS, None)
        indicators.update(await self._fetcher.fetch_indicators())
        if overrides:
            # Only non-None overrides win; a None means "no manual value, defer to FRED".
            indicators.update(
                {k: v for k, v in overrides.items() if k in _INDICATOR_KEYS and v is not None}
            )

        regime = classify_regime(
            yield_curve_10y_2y=indicators["yield_curve_10y_2y"],
            credit_spread_baa_10y=indicators["credit_spread_baa_10y"],
            pmi=indicators["pmi"],
            thresholds=self._thresholds,
        )

        snapshot = MacroSnapshot(
            timestamp=datetime.now(UTC),
            regime=regime,
            yield_curve_10y_2y=indicators["yield_curve_10y_2y"],
            credit_spread_baa_10y=indicators["credit_spread_baa_10y"],
            pmi=indicators["pmi"],
            cpi_yoy=indicators["cpi_yoy"],
            unemployment_rate=indicators["unemployment_rate"],
            fed_funds_rate=indicators["fed_funds_rate"],
        )

        previous = self._regime
        self._snapshot = snapshot
        self._regime = regime

        await self._publisher.publish(
            MacroUpdatedEvent(regime=regime.value if regime is not None else None)
        )
        if regime is not None and previous is not None and previous != regime:
            await self._publisher.publish(
                RegimeChangedEvent(old_regime=previous.value, new_regime=regime.value)
            )
            logger.info("Regime changed", old=previous.value, new=regime.value)

        logger.info(
            "Macro refreshed",
            regime=regime.value if regime is not None else None,
            yield_curve=indicators["yield_curve_10y_2y"],
            credit_spread=indicators["credit_spread_baa_10y"],
            pmi=indicators["pmi"],
        )
        return snapshot


#: Sentinel for an observation whose vintage is unknown. Far in the future so
#: it can never become visible to a point-in-time walk.
_NEVER = date(9999, 12, 31)


def _value(current: dict[str, tuple[date, date, float]], series: str) -> float | None:
    entry = current.get(series)
    return entry[2] if entry is not None else None
