"""Pomocnicze funkcje współdzielone przez serwisy."""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Zwraca aktualny czas UTC (aware datetime)."""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Konwertuje datetime do UTC, dodaje tzinfo jeśli brak."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def symbol_to_topic(symbol: str) -> str:
    """Konwertuje symbol do formatu NATS topic. AAPL -> aapl"""
    return symbol.lower().replace("/", "_").replace("-", "_")
