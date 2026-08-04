"""Testy panelu vintage — dwie osie czasu i to, co się psuje przy pomyleniu ich.

Cały P2-4 stoi na jednym rozróżnieniu: co dana liczba OPISUJE (okres) versus
od kiedy BYŁA znana (rewizja). Pomylenie ich nie wywala się — daje wiarygodne
liczby, których nikt wtedy nie mógł znać.
"""

from datetime import date

import pytest

from src.models.db import UNKNOWN_VINTAGE

from .conftest import FakeFetcher, build_service, obs


@pytest.mark.asyncio
async def test_a_revision_does_not_leak_backwards(store):
    """THE test. March's figure is first published in April as 5.0 and revised
    to 4.0 a year later. Asking about May 2015 must return 5.0 — the revision
    did not exist yet, and 4.0 is a fact from the future."""
    await store.save(
        [
            obs("UNRATE", "2015-03-01", 5.0, "2015-04-03"),
            obs("UNRATE", "2015-03-01", 4.0, "2016-02-05"),
        ]
    )
    assert (await store.as_of(date(2015, 5, 1)))["UNRATE"] == 5.0
    assert (await store.as_of(date(2016, 5, 1)))["UNRATE"] == 4.0


@pytest.mark.asyncio
async def test_publication_lag_is_respected_without_a_lag_parameter(store):
    """March's number exists as an observation dated March but is not public
    until April. Between the two there is nothing to know — and `realtime_start`
    encodes that on its own, so no separate 'lag' knob is needed."""
    await store.save([obs("UNRATE", "2015-03-01", 5.0, "2015-04-03")])
    assert await store.as_of(date(2015, 3, 15)) == {}
    assert await store.as_of(date(2015, 4, 2)) == {}
    assert (await store.as_of(date(2015, 4, 3)))["UNRATE"] == 5.0


@pytest.mark.asyncio
async def test_the_newest_PERIOD_wins_not_the_newest_revision(store):
    """A revision to an old period is published after a fresh reading of a new
    one. The answer is the new period — ordering by vintage alone would return
    a year-old figure because it happened to be corrected yesterday."""
    await store.save(
        [
            obs("UNRATE", "2015-06-01", 5.5, "2015-07-02"),
            obs("UNRATE", "2015-03-01", 4.9, "2015-08-01"),  # late revision, old period
        ]
    )
    assert (await store.as_of(date(2015, 9, 1)))["UNRATE"] == 5.5


@pytest.mark.asyncio
async def test_an_undated_row_is_invisible_to_history_but_stored(store):
    """A fact we cannot date cannot be used point-in-time — the same rule as
    `filed_at` on fundamentals. It is kept (it is still the current value) but
    can never answer a historical question."""
    await store.save([obs("UNRATE", "2015-03-01", 5.0, None)])
    assert await store.as_of(date(2030, 1, 1)) == {}
    stored = await store.series_history("UNRATE")
    assert len(stored) == 1
    assert stored[0].realtime_start is None


@pytest.mark.asyncio
async def test_undated_rows_sort_as_NEWEST_not_oldest(store):
    """The sentinel is in the far future on purpose. Encoding "unknown" as an
    old date would make every undated row look like the earliest thing we ever
    knew — the exact look-ahead this table exists to prevent."""
    assert date(2100, 1, 1) < UNKNOWN_VINTAGE
    await store.save(
        [
            obs("UNRATE", "2015-03-01", 5.0, "2015-04-03"),
            obs("UNRATE", "2015-03-01", 9.9, None),
        ]
    )
    assert (await store.as_of(date(2020, 1, 1)))["UNRATE"] == 5.0


@pytest.mark.asyncio
async def test_saving_is_idempotent_and_survives_a_duplicated_batch(store):
    """Postgres refuses an ON CONFLICT whose own VALUES names a key twice, so
    one duplicated row would fail the whole write."""
    batch = [
        obs("UNRATE", "2015-03-01", 5.0, "2015-04-03"),
        obs("UNRATE", "2015-03-01", 5.0, "2015-04-03"),
    ]
    assert await store.save(batch) == 1
    await store.save(batch)
    assert len(await store.series_history("UNRATE")) == 1


@pytest.mark.asyncio
async def test_a_later_backfill_corrects_a_value_in_place(store):
    await store.save([obs("UNRATE", "2015-03-01", 5.0, "2015-04-03")])
    await store.save([obs("UNRATE", "2015-03-01", 5.1, "2015-04-03")])
    rows = await store.series_history("UNRATE")
    assert len(rows) == 1 and rows[0].value == 5.1


@pytest.mark.asyncio
async def test_coverage_reports_undated_rows_separately(store):
    await store.save(
        [
            obs("UNRATE", "2015-03-01", 5.0, "2015-04-03"),
            obs("UNRATE", "2015-04-01", 5.1, None),
        ]
    )
    summary = (await store.coverage())["UNRATE"]
    assert summary["rows"] == 2
    assert summary["undated"] == 1
    assert summary["first"] == "2015-03-01"
    assert summary["last"] == "2015-04-01"


# --- the regime walk -------------------------------------------------------


@pytest.mark.asyncio
async def test_regime_history_walks_forward_with_only_what_was_known(store):
    """A regime series is only honest if each day is classified from the
    vintages available on that day."""
    await store.save(
        [
            # Healthy curve and spread, published early in the year.
            obs("T10Y2Y", "2015-01-01", 1.5, "2015-01-02"),
            obs("BAA10Y", "2015-01-01", 2.0, "2015-01-02"),
            # An inversion published in July.
            obs("T10Y2Y", "2015-07-01", -0.5, "2015-07-02"),
        ]
    )
    service = build_service(store=store)
    history = await service.regime_history(date(2015, 1, 1), date(2015, 12, 31))

    assert history["2015-03-01"] != history["2015-09-01"], (
        "the July inversion must change the regime only from July onward"
    )
    assert "2015-01-01" not in history, "nothing was published yet on the first day"
    assert history["2015-01-02"] == history["2015-06-30"]


@pytest.mark.asyncio
async def test_a_day_that_cannot_be_classified_is_ABSENT_not_defaulted(store):
    """An invented 'expansion' would be a fabricated fact where a missing one is
    the truth. ml-pipeline already turns a missing regime into all-zeros."""
    service = build_service(store=store)
    history = await service.regime_history(date(2015, 1, 1), date(2015, 1, 10))
    assert history == {}


@pytest.mark.asyncio
async def test_regime_history_reads_the_panel_once(store):
    """One query, not one per day: a 20-year request is 7300 days."""
    await store.save(
        [
            obs("T10Y2Y", "2015-01-01", 1.5, "2015-01-02"),
            obs("BAA10Y", "2015-01-01", 2.0, "2015-01-02"),
        ]
    )

    calls = {"panel": 0, "as_of": 0}
    original_panel = store.panel
    original_as_of = store.as_of

    async def counting_panel():
        calls["panel"] += 1
        return await original_panel()

    async def counting_as_of(day):
        calls["as_of"] += 1
        return await original_as_of(day)

    store.panel = counting_panel  # type: ignore[method-assign]
    store.as_of = counting_as_of  # type: ignore[method-assign]

    service = build_service(store=store)
    await service.regime_history(date(2015, 1, 1), date(2016, 1, 1))
    assert calls["panel"] == 1
    assert calls["as_of"] == 0


@pytest.mark.asyncio
async def test_regime_on_a_single_day_matches_the_walk(store):
    """Two code paths answer the same question; they must not disagree."""
    await store.save(
        [
            obs("T10Y2Y", "2015-01-01", 1.5, "2015-01-02"),
            obs("BAA10Y", "2015-01-01", 2.0, "2015-01-02"),
        ]
    )
    service = build_service(store=store)
    walked = await service.regime_history(date(2015, 6, 1), date(2015, 6, 1))
    single = await service.regime_on(date(2015, 6, 1))
    assert single is not None
    assert walked["2015-06-01"] == single.value


# --- backfill --------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_stores_every_vintage_the_fetcher_returns(store):
    fetcher = FakeFetcher(
        vintage={
            "UNRATE": [
                obs("UNRATE", "2015-03-01", 5.0, "2015-04-03"),
                obs("UNRATE", "2015-03-01", 4.0, "2016-02-05"),
            ]
        }
    )
    service = build_service(fetcher=fetcher, store=store)
    written = await service.backfill({"unemployment_rate": "UNRATE"})
    assert written == {"unemployment_rate": 2}
    assert len(await store.series_history("UNRATE")) == 2


# --- FRED's 2000-vintage cap, found on the first real backfill -------------


def test_the_realtime_window_is_sliced_below_freds_vintage_cap():
    """A daily series has more vintages than FRED will serve in one response.

    T10Y2Y over 20 years has ~3100 vintage dates; FRED refuses anything past
    2000 with a 400 that names the count. Both series the regime classifier
    actually reads are daily, so before slicing a 20-year backfill returned
    HTTP 200 having stored nothing the classifier could use.
    """
    from src.core.fred_client import (
        MAX_VINTAGES_PER_REQUEST,
        VINTAGE_SLICE_YEARS,
        _vintage_slices,
    )

    # ~250 business days a year is the vintage rate of a daily, daily-revised
    # series — the worst case this has to survive.
    assert VINTAGE_SLICE_YEARS * 260 < MAX_VINTAGES_PER_REQUEST

    slices = _vintage_slices()
    assert slices, "no realtime windows produced"
    for start, end in slices:
        assert start <= end, f"inverted window {start}..{end}"
    # Contiguous: a gap between windows is a vintage nobody fetches, and a
    # missing vintage reads as "never revised" rather than as missing data.
    for (_, prev_end), (next_start, _) in zip(slices, slices[1:], strict=False):
        assert next_start > prev_end
        assert next_start[:4] <= str(int(prev_end[:4]) + 1)
    # The last window must stay open, or a vintage published tomorrow is lost.
    assert slices[-1][1] == "9999-12-31"


def test_the_api_key_never_reaches_a_log_line():
    """httpx puts the full URL in the exception message, and the key is a query
    parameter — so the first upstream failure wrote it to the container log in
    plaintext, where log shipping keeps it."""
    from src.core.fred_client import _redact

    key = "76d91b061ddabde772d83654ddf74d48"
    message = f"Client error '400 Bad Request' for url '...&api_key={key}&file_type=json'"
    redacted = _redact(message, key)
    assert key not in redacted
    assert "***" in redacted
    assert _redact("no secret configured", None) == "no secret configured"
