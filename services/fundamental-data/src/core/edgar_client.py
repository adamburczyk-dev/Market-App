"""SEC EDGAR XBRL client — pull annual (10-K) fundamentals for a ticker.

Resolves ticker → CIK via SEC's company_tickers.json, then reads the XBRL
``companyconcept`` endpoint for each needed us-gaap tag and assembles a
``FinancialStatements`` per annual period. SEC requires a descriptive
``User-Agent``; without ``SEC_USER_AGENT`` the client is disabled and returns
nothing (the service then relies on manually-posted statements).
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any, Protocol

import httpx
import structlog
from trading_common.schemas import FinancialStatements

logger = structlog.get_logger()

# FinancialStatements field → ordered candidate (us-gaap concept tag, XBRL unit)
# pairs. Filers differ in which revenue concept they report — classic ``Revenues``
# vs the ASC 606 ``RevenueFromContractWithCustomer…`` tags — so revenue carries
# fallbacks (P3 review fix). Candidates are merged per period; earlier wins.
TAG_MAP: dict[str, tuple[tuple[str, str], ...]] = {
    "revenue": (
        ("Revenues", "USD"),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("RevenueFromContractWithCustomerIncludingAssessedTax", "USD"),
        ("SalesRevenueNet", "USD"),
    ),
    "net_income": (("NetIncomeLoss", "USD"),),
    "total_assets": (("Assets", "USD"),),
    "total_liabilities": (("Liabilities", "USD"),),
    "current_assets": (("AssetsCurrent", "USD"),),
    "current_liabilities": (("LiabilitiesCurrent", "USD"),),
    "shares_outstanding": (
        ("CommonStockSharesOutstanding", "shares"),
        ("WeightedAverageNumberOfSharesOutstandingBasic", "shares"),
        ("WeightedAverageNumberOfDilutedSharesOutstanding", "shares"),
    ),
    "operating_cash_flow": (("NetCashProvidedByUsedInOperatingActivities", "USD"),),
    "eps": (("EarningsPerShareBasic", "USD/shares"),),
}


@dataclass(frozen=True)
class Observation:
    """One reported value together with when it was first filed."""

    value: float
    filed: date | None


def _parse_date(raw: object) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _earlier(candidate: date | None, current: date | None) -> bool:
    """True when `candidate` is a strictly earlier known filing date.

    An observation without a filing date never displaces one that has it: a
    value we cannot date is worse than useless for point-in-time work.
    """
    if candidate is None:
        return False
    return current is None or candidate < current


def _statement_filed_at(
    by_field: dict[str, dict[date, "Observation"]], period_end: date
) -> datetime | None:
    """When the WHOLE statement became knowable — the latest of its fields' dates.

    Deliberately the max, not the min. A statement is usable once every field it
    carries has been published; dating it by the earliest field would claim
    knowledge of numbers that were not out yet, which is look-ahead. Being a few
    days late costs a little information, being early fabricates it — and only
    one of those two errors can be found later in a backtest. If ANY populated
    field has no filing date at all, the statement gets none, and the as-of join
    must then skip it rather than guess.
    """
    dates: list[date] = []
    for values in by_field.values():
        obs = values.get(period_end)
        if obs is None:
            continue
        if obs.filed is None:
            return None
        dates.append(obs.filed)
    if not dates:
        return None
    return datetime.combine(max(dates), time.min, tzinfo=UTC)


class FundamentalsFetcher(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def latest_statements(self, symbol: str, count: int = 2) -> list[FinancialStatements]: ...

    async def aclose(self) -> None: ...


class EdgarClient:
    def __init__(
        self,
        user_agent: str | None,
        base_url: str = "https://data.sec.gov",
        tickers_url: str = "https://www.sec.gov/files/company_tickers.json",
        timeout_s: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._base = base_url.rstrip("/")
        self._tickers_url = tickers_url
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            headers={"User-Agent": user_agent} if user_agent else None,
        )
        self._cik_cache: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._user_agent)

    async def _get_json(self, url: str) -> Any | None:
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.warning("EDGAR fetch failed", url=url, error=str(exc))
            return None

    async def ticker_to_cik(self, symbol: str) -> str | None:
        """Resolve a ticker to a zero-padded 10-digit CIK."""
        if not self._cik_cache:
            data = await self._get_json(self._tickers_url)
            if data is None:
                return None
            for row in data.values():
                self._cik_cache[row["ticker"].upper()] = f"{int(row['cik_str']):010d}"
        return self._cik_cache.get(symbol.upper())

    async def _annual_by_period(self, cik: str, tag: str, unit: str) -> dict[date, Observation]:
        """period-end → (value, first filing date) from annual (10-K/FY) observations.

        The same period is reported many times — the original 10-K, any
        amendment, and as the comparative column of later filings. The one that
        matters for point-in-time work is the EARLIEST: that is when the market
        first had the number (P2-3).
        """
        url = f"{self._base}/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        data = await self._get_json(url)
        if data is None:
            return {}
        out: dict[date, Observation] = {}
        for obs in data.get("units", {}).get(unit, []):
            if obs.get("form") not in ("10-K", "10-K/A") or obs.get("fp") != "FY":
                continue
            end = obs.get("end")
            val = obs.get("val")
            if end is None or val is None:
                continue
            period = date.fromisoformat(end)
            filed = _parse_date(obs.get("filed"))
            existing = out.get(period)
            if existing is None:
                out[period] = Observation(float(val), filed)
            elif _earlier(filed, existing.filed):
                # keep the first-known value AND its date together — a later
                # restatement is not what was knowable at the time
                out[period] = Observation(float(val), filed)
        return out

    async def _field_values(
        self, cik: str, candidates: tuple[tuple[str, str], ...]
    ) -> dict[date, Observation]:
        """Merge candidate concepts per period; earlier candidates win on conflict.

        A filer may report different periods under different tags (e.g. pre- vs
        post-ASC-606 revenue), so every candidate is fetched and unioned rather
        than stopping at the first non-empty one.
        """
        merged: dict[date, Observation] = {}
        for tag, unit in candidates:
            for period, obs in (await self._annual_by_period(cik, tag, unit)).items():
                merged.setdefault(period, obs)
        return merged

    async def latest_statements(self, symbol: str, count: int = 2) -> list[FinancialStatements]:
        if not self.enabled:
            return []
        cik = await self.ticker_to_cik(symbol)
        if cik is None:
            logger.warning("Unknown ticker for EDGAR", symbol=symbol)
            return []

        by_field: dict[str, dict[date, Observation]] = {}
        for fieldname, candidates in TAG_MAP.items():
            by_field[fieldname] = await self._field_values(cik, candidates)

        # candidate annual periods = union of period-ends seen, most recent first
        periods = sorted({p for values in by_field.values() for p in values}, reverse=True)
        statements: list[FinancialStatements] = []
        for period_end in periods[:count]:

            def value(field: str, period: date = period_end) -> float | None:
                obs = by_field[field].get(period)
                return obs.value if obs is not None else None

            statements.append(
                FinancialStatements(
                    symbol=symbol.upper(),
                    period_end=period_end,
                    fiscal_period="FY",
                    revenue=value("revenue"),
                    net_income=value("net_income"),
                    total_assets=value("total_assets"),
                    total_liabilities=value("total_liabilities"),
                    current_assets=value("current_assets"),
                    current_liabilities=value("current_liabilities"),
                    shares_outstanding=value("shares_outstanding"),
                    operating_cash_flow=value("operating_cash_flow"),
                    eps=value("eps"),
                    filed_at=_statement_filed_at(by_field, period_end),
                    source="sec-edgar",
                )
            )
        return statements

    async def aclose(self) -> None:
        await self._client.aclose()
