"""FeatureEngineService — compute features on demand or from market-data events."""

import structlog
from trading_common.events import FeaturesReadyEvent, MarketDataUpdatedEvent
from trading_common.schemas import FeatureVector, Interval

from src.core.features import compute_feature_vector
from src.core.market_data_client import MarketDataClient
from src.core.ranking import cross_sectional_rank
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
    ) -> None:
        self._client = client
        self._store = store
        self._publisher = publisher
        self._lookback = lookback
        self._min_bars = min_bars

    async def compute_for_symbol(self, symbol: str, interval: Interval) -> FeatureVector | None:
        """Fetch OHLCV, compute features, store them, publish FeaturesReadyEvent."""
        bars = await self._client.get_ohlcv(symbol, interval, self._lookback)
        if len(bars) < self._min_bars:
            logger.warning("Not enough bars to compute features", symbol=symbol, bars=len(bars))
            return None

        fv = compute_feature_vector(bars)
        self._store.put(fv)
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

    def get_features(self, symbol: str, interval: Interval) -> FeatureVector | None:
        return self._store.get(symbol, interval)

    def ranked_universe(self, interval: Interval) -> list[FeatureVector]:
        """Cross-sectional percentile-ranked features for the whole universe."""
        return cross_sectional_rank(self._store.all_for_interval(interval))

    def get_ranked(self, symbol: str, interval: Interval) -> FeatureVector | None:
        """The given symbol's rank-transformed vector within the current universe."""
        for fv in self.ranked_universe(interval):
            if fv.symbol == symbol:
                return fv
        return None

    def list_symbols(self) -> list[str]:
        return self._store.symbols()

    async def handle_market_data_event(self, data: bytes) -> None:
        """NATS handler: parse the event and (re)compute features for its symbol."""
        event = MarketDataUpdatedEvent.model_validate_json(data)
        await self.compute_for_symbol(event.symbol, Interval(event.interval))
