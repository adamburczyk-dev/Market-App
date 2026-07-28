"""The sector vocabulary — one canonical spelling for eleven sectors.

This exists because an unmatched sector string is not a cosmetic problem: it
fails CLOSED in risk-mgmt's regime allow-list, so a profile that says
"Technology" instead of "Information Technology" silently refuses orders in a
contraction (FLOW-8).
"""

import pytest

from trading_common.sectors import GICS_SECTORS, is_canonical, normalize_sector


def test_every_canonical_name_normalizes_to_itself():
    for sector in GICS_SECTORS:
        assert normalize_sector(sector) == sector
        assert is_canonical(sector)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Technology", "Information Technology"),
        ("tech", "Information Technology"),
        ("  INFORMATION TECHNOLOGY  ", "Information Technology"),
        ("Healthcare", "Health Care"),
        ("Consumer Cyclical", "Consumer Discretionary"),
        ("Consumer Defensive", "Consumer Staples"),
        ("Financial Services", "Financials"),
        ("Basic Materials", "Materials"),
        ("Telecom", "Communication Services"),
    ],
)
def test_wild_spellings_map_onto_gics(raw, expected):
    assert normalize_sector(raw) == expected


def test_unknown_returns_none_rather_than_guessing():
    # A guess here would let a name through a regime's allow-list on a coin
    # flip, and that gate exists precisely to be conservative.
    for raw in ("Crypto", "", None, "   ", "Miscellaneous"):
        assert normalize_sector(raw) is None
    assert not is_canonical("Technology")  # an alias is not canonical
