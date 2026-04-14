"""Tests for RiskEnvelope — lightweight pre-trade risk checks."""

from datetime import UTC, datetime

import pytest

from trading_common.risk_envelope import RiskEnvelope, RiskLimits
from trading_common.schemas import Signal, TradingSignal


def make_signal(
    confidence: float = 0.80,
    price: float = 150.0,
    stop_loss: float | None = None,
) -> TradingSignal:
    return TradingSignal(
        symbol="AAPL",
        strategy="test",
        signal=Signal.BUY,
        confidence=confidence,
        price=price,
        timestamp=datetime.now(UTC),
        stop_loss=stop_loss,
    )


def make_envelope(**kwargs: object) -> RiskEnvelope:
    return RiskEnvelope(RiskLimits(**kwargs)) if kwargs else RiskEnvelope()


# Safe defaults for check_signal kwargs
SAFE_CONTEXT = {
    "portfolio_value": 100_000.0,
    "current_exposure_pct": 0.50,
    "current_drawdown_pct": 0.0,
    "daily_loss_pct": 0.0,
    "sector_positions": {},
}


class TestRiskLimitsDefaults:
    def test_max_position_pct(self):
        assert RiskLimits().max_position_pct == 0.05

    def test_max_portfolio_exposure_pct(self):
        assert RiskLimits().max_portfolio_exposure_pct == 0.80

    def test_max_single_loss_pct(self):
        assert RiskLimits().max_single_loss_pct == 0.02

    def test_max_daily_loss_pct(self):
        assert RiskLimits().max_daily_loss_pct == 0.05

    def test_max_drawdown_pct(self):
        assert RiskLimits().max_drawdown_pct == 0.15

    def test_max_correlated_positions(self):
        assert RiskLimits().max_correlated_positions == 3

    def test_min_confidence(self):
        assert RiskLimits().min_confidence == 0.55

    def test_custom_values(self):
        limits = RiskLimits(max_drawdown_pct=0.10, min_confidence=0.70)
        assert limits.max_drawdown_pct == 0.10
        assert limits.min_confidence == 0.70


class TestRiskLimitsValidation:
    def test_zero_max_position_pct_raises(self):
        with pytest.raises(ValueError, match="max_position_pct"):
            RiskLimits(max_position_pct=0.0)

    def test_negative_max_drawdown_pct_raises(self):
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            RiskLimits(max_drawdown_pct=-0.1)

    def test_above_one_min_confidence_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            RiskLimits(min_confidence=1.5)

    def test_zero_correlated_positions_raises(self):
        with pytest.raises(ValueError, match="max_correlated_positions"):
            RiskLimits(max_correlated_positions=0)

    def test_boundary_one_is_valid(self):
        limits = RiskLimits(max_position_pct=1.0, max_drawdown_pct=1.0)
        assert limits.max_position_pct == 1.0

    def test_valid_defaults_pass(self):
        limits = RiskLimits()
        assert limits.max_position_pct == 0.05


class TestRiskEnvelopeSectorCorrelation:
    def test_sector_full_rejects(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "sector_positions": {"Technology": 3}},
            signal_sector="Technology",
        )
        assert approved is False
        assert "sector" in reason

    def test_sector_below_limit_approves(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "sector_positions": {"Technology": 2}},
            signal_sector="Technology",
        )
        assert approved is True

    def test_no_sector_skips_check(self):
        envelope = make_envelope()
        approved, _ = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "sector_positions": {"Technology": 10}},
        )
        assert approved is True

    def test_unknown_sector_zero_count(self):
        envelope = make_envelope()
        approved, _ = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "sector_positions": {"Technology": 3}},
            signal_sector="Healthcare",
        )
        assert approved is True


class TestRiskEnvelopeApproved:
    def test_happy_path_approved(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(make_signal(), **SAFE_CONTEXT)
        assert approved is True
        assert reason == "approved"

    def test_approved_without_stop_loss(self):
        """Risk-per-trade check is skipped when stop_loss is None."""
        envelope = make_envelope()
        approved, reason = envelope.check_signal(make_signal(stop_loss=None), **SAFE_CONTEXT)
        assert approved is True

    def test_approved_at_confidence_exactly_at_threshold(self):
        """min_confidence=0.55, signal.confidence=0.55 → approved (< not <=)."""
        envelope = make_envelope()
        approved, _ = envelope.check_signal(make_signal(confidence=0.55), **SAFE_CONTEXT)
        assert approved is True


class TestRiskEnvelopeRejectDrawdown:
    def test_reject_drawdown_exceeded(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "current_drawdown_pct": 0.16},
        )
        assert approved is False
        assert "drawdown" in reason

    def test_reject_drawdown_exactly_at_limit(self):
        """Uses >= so exactly at 15% should reject."""
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "current_drawdown_pct": 0.15},
        )
        assert approved is False
        assert "drawdown" in reason

    def test_reject_negative_drawdown_uses_abs(self):
        """Drawdown passed as negative value should still trigger."""
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "current_drawdown_pct": -0.16},
        )
        assert approved is False
        assert "drawdown" in reason


class TestRiskEnvelopeRejectDailyLoss:
    def test_reject_daily_loss_exceeded(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "daily_loss_pct": 0.06},
        )
        assert approved is False
        assert "daily_loss" in reason

    def test_reject_daily_loss_exactly_at_limit(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "daily_loss_pct": 0.05},
        )
        assert approved is False
        assert "daily_loss" in reason

    def test_reject_negative_daily_loss_uses_abs(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "daily_loss_pct": -0.06},
        )
        assert approved is False
        assert "daily_loss" in reason


class TestRiskEnvelopeRejectConfidence:
    def test_reject_low_confidence(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(make_signal(confidence=0.40), **SAFE_CONTEXT)
        assert approved is False
        assert "confidence" in reason

    def test_reject_confidence_just_below_threshold(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(make_signal(confidence=0.54), **SAFE_CONTEXT)
        assert approved is False
        assert "confidence" in reason


class TestRiskEnvelopeRejectExposure:
    def test_reject_exposure_exceeded(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "current_exposure_pct": 0.85},
        )
        assert approved is False
        assert "exposure" in reason

    def test_reject_exposure_exactly_at_limit(self):
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "current_exposure_pct": 0.80},
        )
        assert approved is False
        assert "exposure" in reason


class TestRiskEnvelopeRejectRiskPerTrade:
    def test_reject_position_size_after_risk_sizing(self):
        """
        With very tight stop ($1 below entry on $150 stock),
        risk sizing allows large position that exceeds 5% limit.
        max_risk = 100k * 0.02 = $2000
        max_shares = 2000 / 1.0 = 2000
        position_value = 2000 * 150 = $300,000 → 300% of portfolio → reject
        """
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(price=150.0, stop_loss=149.0), **SAFE_CONTEXT
        )
        assert approved is False
        assert "position_size" in reason

    def test_approve_position_size_within_limits(self):
        """
        With wide stop ($30 below entry on $150 stock),
        max_shares = 2000 / 30 = 66
        position_value = 66 * 150 = $9,900 → 9.9% of portfolio
        But wait, 9.9% > 5% → reject. Let's use even wider stop.
        stop at $100 → risk_per_share = 50
        max_shares = 2000 / 50 = 40
        position_value = 40 * 150 = $6,000 → 6% → still > 5% → reject

        Actually let's check: the condition is:
        position_value / portfolio_value > max_position_pct
        6000 / 100000 = 0.06 > 0.05 → reject

        With stop at $50: risk = 100, shares = 20, value = 3000, 3% → approve
        """
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(price=150.0, stop_loss=50.0), **SAFE_CONTEXT
        )
        assert approved is True
        assert reason == "approved"

    def test_zero_risk_per_share_skips_check(self):
        """entry_price == stop_loss → risk_per_share = 0 → skip."""
        envelope = make_envelope()
        approved, reason = envelope.check_signal(
            make_signal(price=150.0, stop_loss=150.0), **SAFE_CONTEXT
        )
        assert approved is True


class TestRiskEnvelopeOrdering:
    def test_drawdown_checked_before_daily_loss(self):
        """Both violated, drawdown should be reported."""
        envelope = make_envelope()
        _, reason = envelope.check_signal(
            make_signal(),
            **{
                **SAFE_CONTEXT,
                "current_drawdown_pct": 0.20,
                "daily_loss_pct": 0.10,
            },
        )
        assert "drawdown" in reason

    def test_daily_loss_checked_before_confidence(self):
        """Both violated, daily_loss should be reported."""
        envelope = make_envelope()
        _, reason = envelope.check_signal(
            make_signal(confidence=0.30),
            **{**SAFE_CONTEXT, "daily_loss_pct": 0.10},
        )
        assert "daily_loss" in reason

    def test_confidence_checked_before_exposure(self):
        """Both violated, confidence should be reported."""
        envelope = make_envelope()
        _, reason = envelope.check_signal(
            make_signal(confidence=0.30),
            **{**SAFE_CONTEXT, "current_exposure_pct": 0.90},
        )
        assert "confidence" in reason


class TestRiskEnvelopeCustomLimits:
    def test_custom_drawdown_limit(self):
        envelope = make_envelope(max_drawdown_pct=0.10)
        approved, _ = envelope.check_signal(
            make_signal(),
            **{**SAFE_CONTEXT, "current_drawdown_pct": 0.12},
        )
        assert approved is False

    def test_custom_confidence_threshold(self):
        envelope = make_envelope(min_confidence=0.70)
        approved, _ = envelope.check_signal(make_signal(confidence=0.65), **SAFE_CONTEXT)
        assert approved is False

    def test_custom_limits_more_permissive(self):
        envelope = make_envelope(max_drawdown_pct=0.30, max_daily_loss_pct=0.10)
        approved, _ = envelope.check_signal(
            make_signal(),
            **{
                **SAFE_CONTEXT,
                "current_drawdown_pct": 0.20,
                "daily_loss_pct": 0.08,
            },
        )
        assert approved is True
