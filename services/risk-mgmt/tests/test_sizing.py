"""Testy PositionSizer (adaptive_sizing + regime_allocator)."""

from src.core.portfolio import PortfolioState
from src.core.sizing import PositionSizer

SIZER = PositionSizer()


def test_sizes_down_to_position_cap():
    # Tight 5% stop would risk-budget 400 shares, but the 5% position cap -> 50.
    shares, reason = SIZER.size(price=100.0, stop_loss=95.0, portfolio=PortfolioState())
    assert reason == "sized"
    assert shares == 50  # 100k * 5% / 100


def test_drawdown_scales_size_down():
    # Wide 50% stop so the risk budget (not the position cap) binds.
    full = SIZER.size(100.0, 50.0, PortfolioState(drawdown_pct=0.0))[0]
    half = SIZER.size(100.0, 50.0, PortfolioState(drawdown_pct=0.10))[0]
    assert full == 40  # 100k * 2% / 50
    assert 19 <= half <= 20  # risk budget ~halved at the midpoint of the 5–15% band


def test_zero_size_at_max_drawdown():
    shares, reason = SIZER.size(100.0, 95.0, PortfolioState(drawdown_pct=0.15))
    assert shares == 0
    assert "drawdown" in reason


def test_regime_exposure_cap_blocks():
    pf = PortfolioState(exposure_pct=0.95, regime="expansion")  # cap 90%
    shares, reason = SIZER.size(100.0, 95.0, pf)
    assert shares == 0
    assert "exposure_cap" in reason


def test_crisis_regime_low_cap_blocks():
    pf = PortfolioState(exposure_pct=0.20, regime="crisis")  # cap 15%
    shares, reason = SIZER.size(100.0, 95.0, pf)
    assert shares == 0
    assert "crisis" in reason
