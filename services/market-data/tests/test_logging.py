"""The traceback has to survive the JSON renderer.

It did not. `logger.exception(...)` put `exc_info=True` into the event dict and
JSONRenderer serialized it as the boolean `true`, so a production log line read

    {"symbol": "AAPL", "exc_info": true, "event": "Fetch-and-store failed", ...}

and the traceback was simply gone. The route that raised it already returns a
named `detail` over HTTP, but the log is where an operator looks first, and for
every one of the 13 services it had been answering "something failed" with no
way to ask what. `app.debug` is False under compose, so this was the production
path and only the production path.
"""

import io
import re

import pytest
import structlog
from fastapi import FastAPI

from src.core.observability import setup_observability


def capture(debug: bool) -> str:
    app = FastAPI(debug=debug)
    setup_observability(app, "market-data")
    stream = io.StringIO()
    # Keep the configured processor chain; only redirect where it writes.
    config = structlog.get_config()
    structlog.configure(
        processors=config["processors"],
        wrapper_class=config["wrapper_class"],
        logger_factory=structlog.PrintLoggerFactory(stream),
        cache_logger_on_first_use=False,
    )
    logger = structlog.get_logger()
    try:
        raise RuntimeError('relation "ohlcv" does not exist')
    except RuntimeError:
        logger.exception("Fetch-and-store failed", symbol="AAPL")
    return stream.getvalue()


@pytest.fixture(autouse=True)
def _restore_structlog():
    yield
    structlog.reset_defaults()


def test_json_logs_carry_the_traceback_not_just_exc_info_true():
    line = capture(debug=False)
    assert "Fetch-and-store failed" in line
    assert '"exc_info": true' not in line, "the traceback was dropped again"
    assert "Traceback (most recent call last)" in line
    assert 'relation \\"ohlcv\\" does not exist' in line


def test_console_logs_still_render_the_exception_in_dev():
    """The dev path keeps ConsoleRenderer's own (prettier, coloured) rendering —
    the fix must not trade a readable local traceback for a JSON one."""
    line = re.sub(r"\x1b\[[0-9;]*m", "", capture(debug=True))  # drop ANSI colours
    assert "Fetch-and-store failed" in line
    assert 'RuntimeError: relation "ohlcv" does not exist' in line
    assert "Traceback" in line
