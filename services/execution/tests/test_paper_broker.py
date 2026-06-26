"""Testy paper brokera."""

from src.core.paper_broker import PaperBroker


def test_buy_reduces_cash_and_opens_position():
    broker = PaperBroker(initial_cash=100_000.0)
    fill = broker.fill("o1", "AAPL", "BUY", 50, 100.0)
    assert fill.price == 100.0
    assert broker.cash == 95_000.0
    assert broker.positions()["AAPL"]["quantity"] == 50
    assert broker.equity == 100_000.0  # marked at fill price


def test_sell_increases_cash():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.fill("o1", "AAPL", "BUY", 100, 100.0)
    broker.fill("o2", "AAPL", "SELL", 100, 90.0)  # realize a loss
    assert broker.cash == 99_000.0  # 100k - 10k + 9k
    assert broker.positions() == {}  # flat
    assert broker.equity == 99_000.0


def test_metrics_track_drawdown_and_daily_loss():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.fill("o1", "AAPL", "BUY", 100, 100.0)
    broker.fill("o2", "AAPL", "SELL", 100, 90.0)  # -1000 → equity 99000
    m = broker.metrics()
    assert m["value"] == 99_000.0
    assert round(m["drawdown_pct"], 4) == 0.01
    assert round(m["daily_loss_pct"], 4) == 0.01
    assert m["exposure_pct"] == 0.0  # flat


def test_exposure_reflects_open_position():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.fill("o1", "AAPL", "BUY", 100, 100.0)  # 10k position, equity 100k
    assert round(broker.metrics()["exposure_pct"], 2) == 0.10


def test_slippage_moves_fill_price():
    broker = PaperBroker(initial_cash=100_000.0, slippage_bps=10.0)  # 0.1%
    buy = broker.fill("o1", "AAPL", "BUY", 1, 100.0)
    sell = broker.fill("o2", "MSFT", "SELL", 1, 100.0)
    assert buy.price == 100.1  # BUY pays up
    assert sell.price == 99.9  # SELL receives less


def test_mark_updates_unrealized_value():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.fill("o1", "AAPL", "BUY", 50, 100.0)  # equity 100k
    broker.mark("AAPL", 90.0)  # price drops -> unrealized loss
    assert broker.positions()["AAPL"]["last_price"] == 90.0
    assert broker.equity == 99_500.0  # 95k cash + 50*90
    assert round(broker.metrics()["drawdown_pct"], 4) == 0.005


def test_mark_ignores_unheld_symbol():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.mark("AAPL", 50.0)
    assert broker.positions() == {}
    assert broker.equity == 100_000.0
