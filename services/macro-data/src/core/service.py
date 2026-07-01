"""MacroDataService — fetch macro indicators, classify the regime, publish events."""

from datetime import UTC, datetime

import structlog
from trading_common.events import MacroUpdatedEvent, RegimeChangedEvent
from trading_common.schemas import MacroRegime, MacroSnapshot

from src.core.fred_client import MacroFetcher
from src.core.regime import RegimeThresholds, classify_regime
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
    ) -> None:
        self._fetcher = fetcher
        self._publisher = publisher
        self._thresholds = thresholds or RegimeThresholds()
        self._snapshot: MacroSnapshot | None = None
        self._regime: MacroRegime | None = None

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
