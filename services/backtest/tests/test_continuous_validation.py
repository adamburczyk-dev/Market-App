"""Tests for ContinuousWalkForward — weekly strategy revalidation."""

import pytest

from src.core.continuous_validation import ContinuousWalkForward, WalkForwardResult


class MockWalkForward(ContinuousWalkForward):
    """Test subclass that returns a configurable Sharpe."""

    def __init__(self, mock_sharpe: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.mock_sharpe = mock_sharpe

    async def _run_backtest(self, strategy_name, strategy_params, ohlcv_data):
        return self.mock_sharpe


class TestRevalidateActive:
    @pytest.mark.asyncio
    async def test_no_degradation(self):
        wf = MockWalkForward(mock_sharpe=1.0)
        result = await wf.revalidate("strat_v1", 1.0, [], {})
        assert result.recommended_status == "active"
        assert result.degradation_pct == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_small_degradation_still_active(self):
        """20% degradation < 40% threshold → active."""
        wf = MockWalkForward(mock_sharpe=0.8)
        result = await wf.revalidate("strat_v1", 1.0, [], {})
        assert result.recommended_status == "active"
        assert result.degradation_pct == pytest.approx(0.2)


class TestRevalidateProbation:
    @pytest.mark.asyncio
    async def test_degradation_at_threshold(self):
        """Exactly 40% degradation → probation (>= threshold)."""
        wf = MockWalkForward(mock_sharpe=0.6)
        result = await wf.revalidate("strat_v1", 1.0, [], {})
        assert result.recommended_status == "probation"
        assert result.degradation_pct == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_degradation_above_threshold(self):
        wf = MockWalkForward(mock_sharpe=0.3)
        result = await wf.revalidate("strat_v1", 1.0, [], {})
        assert result.recommended_status == "probation"


class TestRevalidateDeactivate:
    @pytest.mark.asyncio
    async def test_negative_sharpe_deactivates(self):
        wf = MockWalkForward(mock_sharpe=-0.5)
        result = await wf.revalidate("strat_v1", 1.0, [], {})
        assert result.recommended_status == "deactivate"

    @pytest.mark.asyncio
    async def test_negative_sharpe_overrides_degradation(self):
        """Even if degradation is small, negative Sharpe → deactivate."""
        wf = MockWalkForward(mock_sharpe=-0.1)
        result = await wf.revalidate("strat_v1", -0.05, [], {})
        assert result.recommended_status == "deactivate"


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_zero_original_sharpe(self):
        wf = MockWalkForward(mock_sharpe=0.5)
        result = await wf.revalidate("strat_v1", 0.0, [], {})
        assert result.degradation_pct == 0.0
        assert result.recommended_status == "active"

    @pytest.mark.asyncio
    async def test_zero_original_sharpe_negative_current(self):
        """Zero baseline + negative current → deactivate."""
        wf = MockWalkForward(mock_sharpe=-0.3)
        result = await wf.revalidate("strat_v1", 0.0, [], {})
        assert result.degradation_pct == 0.0
        assert result.recommended_status == "deactivate"

    @pytest.mark.asyncio
    async def test_negative_baseline_worsening(self):
        """Negative baseline, more negative current → positive degradation."""
        wf = MockWalkForward(mock_sharpe=-0.8)
        result = await wf.revalidate("strat_v1", -0.5, [], {})
        # degradation = -((-0.8) - (-0.5)) / 0.5 = -(−0.3)/0.5 = 0.6
        assert result.degradation_pct == pytest.approx(0.6)
        assert result.recommended_status == "deactivate"  # negative sharpe

    @pytest.mark.asyncio
    async def test_negative_baseline_improving(self):
        """Negative baseline, less negative current → negative degradation."""
        wf = MockWalkForward(mock_sharpe=-0.2)
        result = await wf.revalidate("strat_v1", -0.5, [], {})
        # degradation = -((-0.2) - (-0.5)) / 0.5 = -(0.3)/0.5 = -0.6
        assert result.degradation_pct == pytest.approx(-0.6)
        assert result.recommended_status == "deactivate"  # still negative sharpe

    @pytest.mark.asyncio
    async def test_custom_params(self):
        wf = MockWalkForward(
            mock_sharpe=0.5,
            oos_window_days=63,
            is_window_days=126,
            degradation_threshold=0.30,
        )
        result = await wf.revalidate("strat_v1", 1.0, [], {})
        # 50% degradation > 30% threshold → probation
        assert result.recommended_status == "probation"
        assert result.oos_window_days == 63
        assert result.is_window_days == 126

    @pytest.mark.asyncio
    async def test_result_fields(self):
        wf = MockWalkForward(mock_sharpe=0.8)
        result = await wf.revalidate("my_strat", 1.0, [], {})
        assert isinstance(result, WalkForwardResult)
        assert result.strategy_name == "my_strat"
        assert result.original_oos_sharpe == 1.0
        assert result.current_oos_sharpe == 0.8
