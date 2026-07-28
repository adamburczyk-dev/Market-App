"""The coverage check must tell a real crash from a data defect.

The first version could not: it flagged NFLX's genuine -35% session of
2022-04-20 (Q1 subscriber loss) as a suspected corporate action, because a raw
price move alone carries no information about which of the two it is. With
adj_close stored, the discriminator exists — an adjustment moves raw and
adjusted apart, a market move moves them together.
"""

import importlib.util
import pathlib
from datetime import UTC, datetime, timedelta

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bootstrap", pathlib.Path(__file__).resolve().parents[1] / "bootstrap-universe.py"
)
boot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boot)

START = datetime(2022, 1, 3, tzinfo=UTC)


def series(moves: dict[int, tuple[float, float]], n: int = 1000) -> list[dict]:
    """n flat sessions (enough to clear the training-length check) with
    (raw_factor, adj_factor) applied on the given indices."""
    bars, raw, adj = [], 100.0, 100.0
    for i in range(n):
        if i in moves:
            raw_f, adj_f = moves[i]
            raw *= raw_f
            adj *= adj_f
        bars.append(
            {
                "timestamp": (START + timedelta(days=i)).isoformat(),
                "close": raw,
                "adj_close": adj,
                "high": raw,
                "low": raw,
                "open": raw,
                "volume": 1e6,
            }
        )
    return bars


def check(bars: list[dict]) -> dict:
    boot._request = lambda *a, **k: (200, bars)  # type: ignore[assignment]
    return boot.validate_coverage("http://x", ["SYM"], START.date())["SYM"]


def test_real_earnings_crash_is_not_a_defect():
    # NFLX 2022-04-20 in miniature: raw and adjusted fall together by 35%.
    result = check(series({30: (0.65, 0.65)}))
    assert "SUSPECT" not in result["note"]
    assert result["notable_moves"], "a -35% session should still be surfaced"
    assert result["notable_moves"][0]["return"] == pytest.approx(-0.35, abs=0.01)
    assert not result["corporate_actions"]


def test_unadjusted_split_in_the_measured_series_is_a_defect():
    # A 4:1 split that the ADJUSTED series failed to absorb: -75% where returns
    # are measured. This is the case that would poison labels.
    result = check(series({30: (0.25, 0.25)}))
    assert "SUSPECT" in result["note"]
    assert result["ok"] is False


def test_adjustment_doing_its_job_is_reported_not_flagged():
    # Raw drops 75% on a split, adjusted does not move: exactly what adj_close
    # is for. Its ABSENCE would be the warning.
    result = check(series({30: (0.25, 1.0)}))
    assert "SUSPECT" not in result["note"]
    assert result["ok"] is True
    assert result["corporate_actions"][0]["raw"] == pytest.approx(-0.75, abs=0.01)
    assert result["corporate_actions"][0]["adjusted"] == pytest.approx(0.0, abs=0.01)


def test_missing_adjusted_close_is_reported():
    bars = series({})
    for b in bars:
        b["adj_close"] = None
    result = check(bars)
    assert "without adj_close" in result["note"]
    assert result["adj_close_coverage"] == 0.0


def test_full_adjusted_coverage_is_reported_positively():
    result = check(series({}))
    assert result["adj_close_coverage"] == 1.0
    assert result["ok"] is True
