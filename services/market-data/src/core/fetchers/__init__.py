"""OHLCV data-source fetchers."""

from src.config import Settings

from .alpha_vantage import AlphaVantageFetcher
from .base import FallbackFetcher, Fetcher, FetchError
from .yahoo import YahooFetcher

__all__ = [
    "AlphaVantageFetcher",
    "FallbackFetcher",
    "FetchError",
    "Fetcher",
    "YahooFetcher",
    "build_default_fetcher",
]


def build_default_fetcher(settings: Settings) -> Fetcher:
    """Yahoo as primary; Alpha Vantage as fallback when an API key is configured."""
    fetchers: list[Fetcher] = [YahooFetcher()]
    if settings.ALPHA_VANTAGE_API_KEY:
        fetchers.append(AlphaVantageFetcher(settings.ALPHA_VANTAGE_API_KEY))
    return FallbackFetcher(fetchers)
