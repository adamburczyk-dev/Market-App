"""In-memory store of the latest FeatureVector per (symbol, interval).

Kept simple for the first functional cut; a Redis-backed store is a natural
follow-up for cross-replica sharing and restart durability.
"""

from trading_common.schemas import FeatureVector, Interval


def _key(symbol: str, interval: Interval) -> str:
    return f"{symbol}:{interval.value}"


class FeatureStore:
    def __init__(self) -> None:
        self._store: dict[str, FeatureVector] = {}

    def put(self, fv: FeatureVector) -> None:
        self._store[_key(fv.symbol, fv.interval)] = fv

    def get(self, symbol: str, interval: Interval) -> FeatureVector | None:
        return self._store.get(_key(symbol, interval))

    def symbols(self) -> list[str]:
        return sorted({key.split(":", 1)[0] for key in self._store})
