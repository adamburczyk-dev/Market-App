"""P2-4: the macro one-hot finally has a source.

`build_dataset` has always accepted `regime_by_date` and nothing ever passed
it, so all five `macro_*` columns were constant zero in every training run and
the variance filter dropped them. These tests pin the join, the date keying and
the honest behaviour when macro-data has nothing to say.
"""

from datetime import UTC, date, datetime, timedelta

import pytest
from trading_common.schemas import Interval, OHLCVBar

from src.core.dataset import REGIMES, DatasetParams, build_dataset
from src.core.macro_client import HttpMacroClient


def bars(symbol: str, n: int = 400, seed: int = 1) -> list[OHLCVBar]:
    """A plain upward drift — the labels do not matter here, the join does."""
    start = datetime(2020, 1, 1, tzinfo=UTC)
    out = []
    price = 100.0
    for i in range(n):
        price *= 1.0 + (0.004 if (i + seed) % 3 else -0.003)
        out.append(
            OHLCVBar(
                symbol=symbol,
                timestamp=start + timedelta(days=i),
                interval=Interval.D1,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1_000_000.0,
                source="test",
            )
        )
    return out


def panel(n: int = 400) -> dict[str, list[OHLCVBar]]:
    return {f"S{i:02d}": bars(f"S{i:02d}", n, seed=i) for i in range(30)}


def macro_columns(dataset) -> list[str]:
    return [name for name in dataset.feature_names if name.startswith("macro_")]


def test_without_a_regime_history_the_macro_columns_are_constant():
    """The state P2-4 found: present by name, carrying nothing."""
    ds = build_dataset(panel(), DatasetParams())
    # They are dropped by the variance filter before training, but at build time
    # they exist and are identical on every row.
    assert not macro_columns(ds) or all(
        len({row for row in ds.x[:, ds.feature_names.index(c)]}) == 1 for c in macro_columns(ds)
    )


def test_a_regime_history_makes_the_columns_vary():
    """With real history the one-hot moves, which is the whole point."""
    sessions = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(400)]
    # The switch has to land INSIDE the usable range: rows only start after
    # `min_history` (253) warms the features up, so a split at session 200 would
    # put every labelled row on one side of it and the column would be constant
    # for a reason that has nothing to do with the join.
    history = {
        s.date(): ("expansion" if i < 320 else "contraction") for i, s in enumerate(sessions)
    }
    ds = build_dataset(panel(), DatasetParams(), regime_by_date=history)

    columns = macro_columns(ds)
    assert columns, "the macro block must be present"
    varying = [c for c in columns if len({row for row in ds.x[:, ds.feature_names.index(c)]}) > 1]
    assert varying, "at least one regime column must vary once history exists"


def test_the_join_is_keyed_by_DATE_not_by_the_session_instant():
    """Sessions are tz-aware timestamps; a macro history is a calendar. Matching
    on the exact instant would never hit — and would fail SILENTLY, leaving the
    columns constant and looking exactly like no history at all."""
    sessions = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(400)]
    by_date = {s.date(): "crisis" for s in sessions}
    ds = build_dataset(panel(), DatasetParams(), regime_by_date=by_date)

    crisis = "macro_crisis"
    assert crisis in ds.feature_names
    column = ds.x[:, ds.feature_names.index(crisis)]
    assert set(column) == {1.0}, "every row falls on a session the history covers"


def test_an_unknown_day_gets_all_zeros_rather_than_a_default_regime():
    """A day macro-data could not classify must not be silently called
    'expansion' — a fabricated fact is worse than a missing one."""
    ds = build_dataset(panel(), DatasetParams(), regime_by_date={date(1990, 1, 1): "expansion"})
    for name in REGIMES:
        column = f"macro_{name}"
        if column in ds.feature_names:
            assert set(ds.x[:, ds.feature_names.index(column)]) == {0.0}


# --- the client ------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_client_parses_dates_and_skips_junk():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["start"] == "2015-01-01"
        return httpx.Response(
            200,
            json={
                "regimes": {
                    "2015-01-02": "expansion",
                    "not-a-date": "crisis",  # skipped, not crashed on
                    "2015-01-03": 7,  # not a string → skipped
                }
            },
        )

    client = HttpMacroClient("http://macro")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    history = await client.get_regime_history(date(2015, 1, 1), date(2015, 1, 31))
    assert history == {date(2015, 1, 2): "expansion"}
    await client.aclose()


@pytest.mark.asyncio
async def test_an_unreachable_macro_service_degrades_to_no_history():
    """Training must degrade, not fail: the columns simply stay unknown, which
    is what they were before this endpoint existed."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("macro-data down")

    client = HttpMacroClient("http://macro")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await client.get_regime_history(date(2015, 1, 1), date(2015, 1, 31)) == {}
    await client.aclose()
