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
from trading_common.sectors import normalize_sector

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


# --- P2-2: the sector map and the verdict it produces ---------------------


def test_default_universe_and_sector_map_stay_in_sync():
    """The sectors were a comment before P2-2 turned them into data. A comment
    can drift from the list beneath it silently; this cannot."""
    assert len(set(boot.DEFAULT_UNIVERSE)) == len(boot.DEFAULT_UNIVERSE)
    assert set(boot.SECTOR_BY_SYMBOL) == set(boot.DEFAULT_UNIVERSE)
    for sector in boot.DEFAULT_UNIVERSE_BY_SECTOR:
        assert normalize_sector(sector) == sector, f"{sector} is not a GICS name"


def test_verdict_refuses_to_judge_a_universe_it_could_not_neutralize():
    # The real 34-name case: ~3 names per sector, so almost nothing was demeaned
    # against peers and the comparison is noise either way.
    text = boot.sector_verdict(0.18, {"strong_plain": 4, "strong_neutral": 0})
    assert "not measurable" in text


def test_a_collapse_in_evidence_is_reported_as_a_sector_bet():
    """The reading that must not regress: mean |t| falling is the transform
    doing its job, not failing. Calling it 'no gain, keep ranking globally'
    would invert the conclusion."""
    text = boot.sector_verdict(1.0, {"strong_plain": 9, "strong_neutral": 0})
    assert "was the sector" in text
    assert "keep ranking globally" not in text


def test_surviving_evidence_recommends_adoption():
    text = boot.sector_verdict(1.0, {"strong_plain": 3, "strong_neutral": 5})
    assert "stock-specific" in text
    mixed = boot.sector_verdict(1.0, {"strong_plain": 6, "strong_neutral": 2})
    assert "mixed" in mixed


# --- P3-2: the training window must follow the backfill depth --------------


def test_train_limit_follows_the_requested_history():
    """A fixed default silently caps a deep backfill. 20 years of data trained
    on 8 is not an error anyone sees — the run just reports a smaller dataset."""
    assert boot.default_train_limit(6.0) == 6 * 252 + 253
    assert boot.default_train_limit(20.0) == 20 * 252 + 253
    # ...and it never drops below what a holdout + folds need
    assert boot.default_train_limit(0.5) == boot.MIN_SESSIONS_FOR_TRAINING
    # ...nor above what the route accepts, which would just 422
    assert boot.default_train_limit(100.0) == boot.MAX_TRAIN_LIMIT


def test_the_warmup_is_included_not_forgotten():
    """The first FULL_HISTORY bars of a window produce no rows, so a limit of
    exactly years x 252 loses a year of trainable sessions at the start."""
    assert boot.default_train_limit(10.0) > 10 * 252


# --- the universe list itself ---------------------------------------------


def test_universe_is_large_enough_for_the_measurement_it_is_for():
    """P3-1 needs 200-500 names: the IC detection threshold scales with the
    number of cross-sectional observations, and 34 put the realistic effect
    size below what could be distinguished from zero at all."""
    assert 200 <= len(boot.DEFAULT_UNIVERSE) <= 600
    assert len(set(boot.DEFAULT_UNIVERSE)) == len(boot.DEFAULT_UNIVERSE)


def test_every_sector_key_is_a_gics_name():
    for sector in boot.DEFAULT_UNIVERSE_BY_SECTOR:
        assert normalize_sector(sector) == sector, f"{sector} is not a GICS name"
    assert set(boot.SECTOR_BY_SYMBOL) == set(boot.DEFAULT_UNIVERSE)


def test_delisted_candidates_are_in_the_universe_and_flagged():
    """They must be fetched like any other symbol — that IS the check — while
    staying identifiable, so the summary can report which ones came back."""
    assert boot.DELISTED_CANDIDATES, "no removed names means no chance of an exit"
    for symbol in boot.DELISTED_CANDIDATES:
        assert symbol in boot.DEFAULT_UNIVERSE
        assert symbol in boot.SECTOR_BY_SYMBOL
    # the failures of 2023 are the whole point: nothing else here goes to zero
    for failed_bank in ("SIVB", "SBNY", "FRC"):
        assert failed_bank in boot.DELISTED_CANDIDATES


def test_every_sector_has_enough_names_to_neutralize_against():
    """P2-2 pools sectors below MIN_SECTOR_SIZE into a residual group. At 34
    names that was every sector; the point of this list is that it no longer is."""
    from trading_common.ranking import MIN_SECTOR_SIZE

    for sector, names in boot.DEFAULT_UNIVERSE_BY_SECTOR.items():
        assert len(names.split()) >= MIN_SECTOR_SIZE, f"{sector} too thin to neutralize"
