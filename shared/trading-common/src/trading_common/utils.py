"""Pomocnicze funkcje współdzielone przez serwisy."""

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Zwraca aktualny czas UTC (aware datetime)."""
    return datetime.now(UTC)


def to_utc(dt: datetime) -> datetime:
    """Konwertuje datetime do UTC, dodaje tzinfo jeśli brak."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def symbol_to_topic(symbol: str) -> str:
    """Konwertuje symbol do formatu NATS topic. AAPL -> aapl"""
    return symbol.lower().replace("/", "_").replace("-", "_")
