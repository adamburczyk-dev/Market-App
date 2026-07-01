"""SEC EDGAR XBRL client — pull annual (10-K) fundamentals for a ticker.

Resolves ticker → CIK via SEC's company_tickers.json, then reads the XBRL
``companyconcept`` endpoint for each needed us-gaap tag and assembles a
``FinancialStatements`` per annual period. SEC requires a descriptive
``User-Agent``; without ``SEC_USER_AGENT`` the client is disabled and returns
nothing (the service then relies on manually-posted statements).
"""

from datetime import date
from typing import Any, Protocol

import httpx
import structlog
from trading_common.schemas import FinancialStatements

logger = structlog.get_logger()

# FinancialStatements field → (us-gaap concept tag, XBRL unit)
TAG_MAP: dict[str, tuple[str, str]] = {
    "revenue": ("Revenues", "USD"),
    "net_income": ("NetIncomeLoss", "USD"),
    "total_assets": ("Assets", "USD"),
    "total_liabilities": ("Liabilities", "USD"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities", "USD"),
    "eps": ("EarningsPerShareBasic", "USD/shares"),
}


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

    async def _annual_by_period(self, cik: str, tag: str, unit: str) -> dict[date, float]:
        """period-end → value from annual (10-K / FY) observations for a concept."""
        url = f"{self._base}/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        data = await self._get_json(url)
        if data is None:
            return {}
        out: dict[date, float] = {}
        for obs in data.get("units", {}).get(unit, []):
            if obs.get("form") not in ("10-K", "10-K/A") or obs.get("fp") != "FY":
                continue
            end = obs.get("end")
            val = obs.get("val")
            if end is None or val is None:
                continue
            out[date.fromisoformat(end)] = float(val)
        return out

    async def latest_statements(self, symbol: str, count: int = 2) -> list[FinancialStatements]:
        if not self.enabled:
            return []
        cik = await self.ticker_to_cik(symbol)
        if cik is None:
            logger.warning("Unknown ticker for EDGAR", symbol=symbol)
            return []

        by_field: dict[str, dict[date, float]] = {}
        for fieldname, (tag, unit) in TAG_MAP.items():
            by_field[fieldname] = await self._annual_by_period(cik, tag, unit)

        # candidate annual periods = union of period-ends seen, most recent first
        periods = sorted({p for values in by_field.values() for p in values}, reverse=True)
        statements: list[FinancialStatements] = []
        for period_end in periods[:count]:
            statements.append(
                FinancialStatements(
                    symbol=symbol.upper(),
                    period_end=period_end,
                    fiscal_period="FY",
                    revenue=by_field["revenue"].get(period_end),
                    net_income=by_field["net_income"].get(period_end),
                    total_assets=by_field["total_assets"].get(period_end),
                    total_liabilities=by_field["total_liabilities"].get(period_end),
                    operating_cash_flow=by_field["operating_cash_flow"].get(period_end),
                    eps=by_field["eps"].get(period_end),
                    source="sec-edgar",
                )
            )
        return statements

    async def aclose(self) -> None:
        await self._client.aclose()
