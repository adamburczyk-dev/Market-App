"""The project's sector vocabulary — one canonical spelling, one normalizer.

Sectors arrive as free-form strings from whatever populated `CompanyProfile`
(a data vendor, a manual POST, a classifier), and two consumers read them for
different purposes:

- risk-mgmt's `RegimeAllocator` blocks a BUY whose sector is not on the
  regime's allow-list. Matching by exact string means "Technology" from one
  source and "Information Technology" from another are different sectors, and
  the mismatch fails CLOSED — the order is silently refused in a contraction.
  That is FLOW-8, and it is a live defect, not a hypothetical one.
- cross-sectional feature construction groups the universe by sector to
  neutralize it (`trading_common.ranking.sector_neutralize`). A misspelling
  there does not block anything; it quietly puts a name in its own group,
  where "relative to peers" means "relative to itself".

So the vocabulary is GICS's 11 sectors, and `normalize_sector` maps the
spellings actually seen in the wild onto them. An unrecognized string returns
None rather than a guess: for the allocator None is a decision the caller must
make explicitly, and for ranking it is an honest "unknown peer group".
"""

GICS_SECTORS: tuple[str, ...] = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)

# Alias → canonical. Keys are compared case-insensitively with surrounding
# whitespace stripped; the canonical names map to themselves. The variants here
# are the ones Yahoo/FMP-style feeds and hand-entered profiles actually produce.
_ALIASES: dict[str, str] = {
    # Information Technology
    "technology": "Information Technology",
    "tech": "Information Technology",
    "information technology": "Information Technology",
    "infotech": "Information Technology",
    "it": "Information Technology",
    # Health Care
    "healthcare": "Health Care",
    "health care": "Health Care",
    "health": "Health Care",
    # Consumer Discretionary
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer cyclicals": "Consumer Discretionary",
    # Consumer Staples
    "consumer defensive": "Consumer Staples",
    "consumer staples": "Consumer Staples",
    "consumer non-cyclical": "Consumer Staples",
    # Financials
    "financial services": "Financials",
    "financial": "Financials",
    "financials": "Financials",
    "finance": "Financials",
    # Materials
    "basic materials": "Materials",
    "materials": "Materials",
    # Communication Services
    "communication services": "Communication Services",
    "communications": "Communication Services",
    "telecommunications": "Communication Services",
    "telecom": "Communication Services",
    # Straightforward ones
    "energy": "Energy",
    "industrials": "Industrials",
    "industrial": "Industrials",
    "real estate": "Real Estate",
    "realestate": "Real Estate",
    "utilities": "Utilities",
    "utility": "Utilities",
}


def normalize_sector(raw: str | None) -> str | None:
    """Map a free-form sector string onto a GICS sector, or None if unknown.

    None is deliberate. Guessing a sector for a name we do not recognize would
    let it through a regime's allow-list on a coin flip, and the whole point of
    that gate is that it is conservative.
    """
    if not raw:
        return None
    return _ALIASES.get(raw.strip().lower())


def is_canonical(sector: str) -> bool:
    return sector in GICS_SECTORS


__all__ = ["GICS_SECTORS", "is_canonical", "normalize_sector"]
