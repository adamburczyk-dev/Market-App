"""Testy reguły momentum-on-ranks."""

from trading_common.schemas import Signal

from src.core.momentum import MomentumParams, generate_signal

P = MomentumParams()


def test_buy_on_top_momentum():
    sig, conf = generate_signal(0.9, 50.0, P)
    assert sig == Signal.BUY
    assert conf == 0.9


def test_sell_on_bottom_momentum():
    sig, conf = generate_signal(0.1, 50.0, P)
    assert sig == Signal.SELL
    assert round(conf, 2) == 0.9


def test_hold_in_the_middle():
    sig, conf = generate_signal(0.5, 50.0, P)
    assert sig == Signal.HOLD
    assert conf == 0.5


def test_no_buy_when_overbought():
    sig, _ = generate_signal(0.9, 75.0, P)  # rsi > 70
    assert sig == Signal.HOLD


def test_no_sell_when_oversold():
    sig, _ = generate_signal(0.1, 25.0, P)  # rsi < 30
    assert sig == Signal.HOLD
