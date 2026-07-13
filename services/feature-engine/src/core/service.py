"""FeatureEngineService — compute features on demand or from market-data events.

Besides the Tier-1 technical path (market_data.updated → compute → publish
features.ready), the service consumes ``fundamentals.updated`` (querying the
payload back from fundamental-data over HTTP) and ``company.classified``
(style straight from the event) into a per-symbol attribute store. Attributes
are merged into every vector at *read* time, so ``/features`` and ``/ranked``
expose them — including cross-sectional ranks of e.g. ``f_score``. Attribute
updates deliberately do NOT publish ``features.ready``: the momentum strategy
consumes technical features only, and a fundamentals refresh must not trigger
signal re-evaluation (the ML tier will read the merged vectors directly).
"""

import structlog
from trading_common.events import (
    CompanyClassifiedEvent,
    FeaturesReadyEvent,
    FundamentalsUpdatedEvent,
    MarketDataUpdatedEvent,
)
from trading_common.features import compute_feature_vector
from trading_common.ranking import cross_sectional_rank
from trading_common.schemas import FeatureVector, Interval

from src.core.attributes import SymbolAttributeStore
from src.core.enrichment import fundamental_features, style_features
from src.core.fundamental_client import FundamentalsClient
from src.core.market_data_client import MarketDataClient
from src.core.store import FeatureStore
from src.events.publisher import Publisher

logger = structlog.get_logger()


class FeatureEngineService:
    def __init__(
        self,
        client: MarketDataClient,
        store: FeatureStore,
        publisher: Publisher,
        lookback: int = 250,
        min_bars: int = 20,
        attributes: SymbolAttributeStore | None = None,
        fundamentals_client: FundamentalsClient | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._publisher = publisher
        self._lookback = lookback
        self._min_bars = min_bars
        self._attributes = attributes
        self._fundamentals = fundamentals_client

    async def compute_for_symbol(self, symbol: str, interval: Interval) -> FeatureVector | None:
        """Fetch OHLCV, compute features, store them, publish FeaturesReadyEvent."""
        bars = await self._client.get_ohlcv(symbol, interval, self._lookback)
        if len(bars) < self._min_bars:
            logger.warning("Not enough bars to compute features", symbol=symbol, bars=len(bars))
            return None

        fv = compute_feature_vector(bars)
        await self._store.put(fv)
        await self._publisher.publish(
            FeaturesReadyEvent(
                symbol=symbol,
                interval=interval.value,
                features_count=len(fv.features),
                tier=fv.tier,
            )
        )
        logger.info(
            "Computed features", symbol=symbol, interval=interval.value, count=len(fv.features)
        )
        return fv

    async def _merged(self, fv: FeatureVector) -> FeatureVector:
        """Overlay the symbol's Tier-2 attributes onto a stored technical vector."""
        if self._attributes is None:
            return fv
        attrs = await self._attributes.get(fv.symbol)
        if not attrs:
            return fv
        return fv.model_copy(update={"features": {**fv.features, **attrs}})

    async def get_features(self, symbol: str, interval: Interval) -> FeatureVector | None:
        fv = await self._store.get(symbol, interval)
        return await self._merged(fv) if fv is not None else None

    async def ranked_universe(self, interval: Interval) -> list[FeatureVector]:
        """Cross-sectional percentile-ranked features for the whole universe."""
        vectors = [await self._merged(fv) for fv in await self._store.all_for_interval(interval)]
        return cross_sectional_rank(vectors)

    async def get_ranked(self, symbol: str, interval: Interval) -> FeatureVector | None:
        """The given symbol's rank-transformed vector within the current universe."""
        for fv in await self.ranked_universe(interval):
            if fv.symbol == symbol:
                return fv
        return None

    async def list_symbols(self) -> list[str]:
        return await self._store.symbols()

    async def handle_market_data_event(self, data: bytes) -> None:
        """NATS handler: parse the event and (re)compute features for its symbol."""
        event = MarketDataUpdatedEvent.model_validate_json(data)
        await self.compute_for_symbol(event.symbol, Interval(event.interval))

    async def handle_fundamentals_event(self, data: bytes) -> None:
        """NATS handler: fresh fundamentals → query them back → store attributes."""
        event = FundamentalsUpdatedEvent.model_validate_json(data)
        if self._attributes is None or self._fundamentals is None:
            return
        payload = await self._fundamentals.get_fundamentals(event.symbol)
        if payload is None:  # symbol not stored (404) — nothing to enrich
            return
        features = fundamental_features(payload)
        if not features:
            logger.warning("Fundamentals yielded no features", symbol=event.symbol)
            return
        await self._attributes.put(event.symbol, features)
        logger.info("Stored fundamental attributes", symbol=event.symbol, keys=sorted(features))

    async def handle_company_classified_event(self, data: bytes) -> None:
        """NATS handler: company style → numeric encoding → store attributes."""
        event = CompanyClassifiedEvent.model_validate_json(data)
        if self._attributes is None:
            return
        features = style_features(event.style)
        if not features:
            logger.warning("Unknown style — no attributes", symbol=event.symbol, style=event.style)
            return
        await self._attributes.put(event.symbol, features)
        logger.info("Stored style attributes", symbol=event.symbol, style=event.style)
