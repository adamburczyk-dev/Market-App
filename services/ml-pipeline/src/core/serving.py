"""ServingEngine — features.ready → infer → publish MlSignalGeneratedEvent (plan §8).

Serving runs ONLY a production-aliased model (a failed gate never gets
promoted, so serving stays silent). The metadata's feature list is
load-bearing: the input row is assembled in exactly that order — Tier-2
attributes a symbol lacks fill with the neutral rank 0.5 (same as training),
but when the served vector is missing the MAJORITY of expected features the
engine refuses to infer (schema drift, not sparsity). The dead zone
(sell_threshold < p < buy_threshold) is silent, mirroring the strategy's
silence on HOLD — a stale ML vote then simply TTL-expires in the aggregator.
"""

import numpy as np
import structlog
from trading_common.events import FeaturesReadyEvent, MlSignalGeneratedEvent
from trading_common.schemas import Interval

from src.core.dataset import REGIMES
from src.core.feature_client import FeatureClient
from src.core.macro_client import MacroClient
from src.core.model import TrainedModel
from src.core.model_store import MlflowModelStore
from src.events.publisher import Publisher

logger = structlog.get_logger()


class ServingEngine:
    def __init__(
        self,
        publisher: Publisher,
        feature_client: FeatureClient,
        macro_client: MacroClient,
        store: MlflowModelStore | None,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        serve_interval: str = "1d",
        horizon_days: int = 10,
        max_missing_fraction: float = 0.5,
    ) -> None:
        self._publisher = publisher
        self._features = feature_client
        self._macro = macro_client
        self._store = store
        self._buy = buy_threshold
        self._sell = sell_threshold
        self._interval = serve_interval
        self._horizon_days = horizon_days
        self._max_missing = max_missing_fraction
        self._model: TrainedModel | None = None
        self._model_id: str | None = None

    @property
    def active(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def reload(self) -> str | None:
        """(Re)load the production alias from the registry; None deactivates."""
        if self._store is None:
            return None
        loaded = self._store.load_production()
        if loaded is None:
            self._model = None
            self._model_id = None
            logger.info("No production model — serving inactive")
            return None
        model, _metadata = loaded
        version = self._store.production_version()
        self._model = model
        self._model_id = f"{self._store.model_name}@v{version}"
        logger.info(
            "Serving model loaded", model_id=self._model_id, features=len(model.feature_names)
        )
        return self._model_id

    async def handle_features_ready(self, data: bytes) -> None:
        event = FeaturesReadyEvent.model_validate_json(data)
        if event.interval != self._interval:
            return
        await self.infer_symbol(event.symbol, Interval(event.interval))

    def _assemble(self, features: dict[str, float], regime: str | None) -> np.ndarray | None:
        """Input row in the metadata's exact feature order, or None (refusal)."""
        assert self._model is not None
        names = self._model.feature_names
        macro = {f"macro_{name}": 1.0 if regime == name else 0.0 for name in REGIMES}
        expected = [n for n in names if not n.startswith("macro_")]
        missing = [n for n in expected if n not in features]
        if expected and len(missing) / len(expected) > self._max_missing:
            logger.error(
                "Feature contract mismatch — inference refused",
                model_id=self._model_id,
                missing=sorted(missing),
                served=sorted(features),
            )
            return None
        return np.array([macro.get(name, features.get(name, 0.5)) for name in names], dtype=float)

    async def infer_symbol(self, symbol: str, interval: Interval) -> MlSignalGeneratedEvent | None:
        """Infer one symbol; publishes (and returns) only actionable BUY/SELL votes."""
        if self._model is None or self._model_id is None or self._store is None:
            return None  # nothing promoted — serving is silent
        ranked = await self._features.get_ranked(symbol, interval)
        if ranked is None:
            return None
        regime = await self._macro.get_regime()
        row = self._assemble(ranked.features, regime)
        if row is None:
            return None

        probability_up = float(self._model.predict_proba(row.reshape(1, -1))[0])
        if probability_up >= self._buy:
            signal = "BUY"
        elif probability_up <= self._sell:
            signal = "SELL"
        else:
            logger.debug("Dead zone — no ML vote", symbol=symbol, p=round(probability_up, 4))
            return None

        event = MlSignalGeneratedEvent(
            symbol=symbol,
            model_id=self._model_id,
            model_stack=self._store.model_name,
            signal=signal,
            confidence=min(1.0, 2.0 * abs(probability_up - 0.5)),
            probability_up=probability_up,
            horizon_days=self._horizon_days,
        )
        await self._publisher.publish(event)
        logger.info(
            "ML signal published",
            symbol=symbol,
            signal=signal,
            p_up=round(probability_up, 4),
            model_id=self._model_id,
        )
        return event
