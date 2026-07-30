"""What a failed study tells the caller.

These routes run for minutes and their output is the review artifact — a run
that dies has to say why IN THE REPORT, not only in a container log. It did
not: only RuntimeError and ValueError were mapped, everything else escaped as
Starlette's plain-text 500 with an EMPTY body, and a whole campaign (five
studies plus training) came back as six identical `HTTP 500: {}` lines while
the real cause sat in the log — market-data refusing a 5293-bar request.
"""

import httpx
import pytest
from httpx import AsyncClient
from trading_common.constants import MAX_OHLCV_LIMIT

from src.core.data_contract import TrainingDataContractError
from src.core.service import MLPipelineService

STUDY_URL = "/api/v1/ml-pipeline/models/target-study"


def body(**extra: object) -> dict:
    return {"symbols": ["AAPL", "MSFT"], "interval": "1d", **extra}


def raiser(exc: BaseException):  # type: ignore[no-untyped-def]
    async def _raise(*args, **kwargs):
        raise exc

    return _raise


def upstream_error(status: int) -> httpx.HTTPStatusError:
    """The exact shape httpx raises out of HttpMarketDataClient.get_ohlcv."""
    request = httpx.Request(
        "GET", "http://market-data:8000/api/v1/market-data/ohlcv/AMZN?interval=1d&limit=5293"
    )
    return httpx.HTTPStatusError(
        f"Client error '{status}'",
        request=request,
        response=httpx.Response(status, request=request),
    )


@pytest.mark.asyncio
async def test_an_upstream_rejection_is_reported_as_such(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """The incident itself: market-data answered 422, and the caller was told
    "500" with nothing else. 502 + the upstream status and url is a diagnosis."""
    client, service = wired
    service.target_study = raiser(upstream_error(422))  # type: ignore[method-assign]

    resp = await client.post(STUDY_URL, json=body())
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "422" in detail
    assert "market-data" in detail
    assert "limit=5293" in detail


@pytest.mark.asyncio
async def test_an_unreachable_upstream_is_distinguished_from_a_rejection(
    wired: tuple[AsyncClient, MLPipelineService],
):
    client, service = wired
    service.target_study = raiser(httpx.ConnectError("connection refused"))  # type: ignore[method-assign]

    resp = await client.post(STUDY_URL, json=body())
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "ConnectError" in detail
    assert "connection refused" in detail


@pytest.mark.asyncio
async def test_an_unexpected_failure_names_its_cause_instead_of_an_empty_500(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """The catch-all matters more than the specific cases: the failure nobody
    anticipated is precisely the one worth naming."""
    client, service = wired
    service.target_study = raiser(KeyError("adj_close"))  # type: ignore[method-assign]

    resp = await client.post(STUDY_URL, json=body())
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "KeyError" in detail
    assert "adj_close" in detail


@pytest.mark.asyncio
async def test_the_data_contract_still_answers_422_with_its_report(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """TrainingDataContractError is a RuntimeError subclass, so a mapping that
    checked RuntimeError first would swallow the report and answer a bare 503.
    The report IS the answer here — it says which assertion the data failed."""
    client, service = wired
    service.train = raiser(  # type: ignore[method-assign]
        TrainingDataContractError(["too few sessions"], {"sessions": 183, "min_sessions": 1000})
    )

    resp = await client.post("/api/v1/ml-pipeline/models/train", json=body())
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "training data contract violated"
    assert detail["sessions"] == 183


@pytest.mark.asyncio
async def test_missing_upstream_configuration_is_still_503(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """No market-data client wired: the service is not ready, not broken. The
    fixture's service has none, so this exercises the real path."""
    client, _ = wired
    resp = await client.post(STUDY_URL, json=body())
    assert resp.status_code == 503
    assert "market-data client not configured" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_a_dataset_too_small_is_still_400(wired: tuple[AsyncClient, MLPipelineService]):
    client, service = wired
    service.target_study = raiser(ValueError("need >= 945 sessions"))  # type: ignore[method-assign]

    resp = await client.post(STUDY_URL, json=body())
    assert resp.status_code == 400
    assert "945" in resp.json()["detail"]


# --- the limit ceiling ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_request_ceiling_is_the_shared_one(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """Accepting a limit market-data will not serve is what broke the campaign:
    the request passed validation here and died upstream 455 symbols later."""
    client, service = wired
    seen: list[int] = []

    async def record(symbols, interval, limit=1500, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(limit)
        return {}

    service.target_study = record  # type: ignore[method-assign]

    resp = await client.post(STUDY_URL, json=body(limit=MAX_OHLCV_LIMIT))
    assert resp.status_code == 200
    assert seen == [MAX_OHLCV_LIMIT]

    resp = await client.post(STUDY_URL, json=body(limit=MAX_OHLCV_LIMIT + 1))
    assert resp.status_code == 422  # pydantic validation, before any work


@pytest.mark.asyncio
async def test_a_twenty_year_request_is_accepted(wired: tuple[AsyncClient, MLPipelineService]):
    """5040 sessions + 253 warm-up bars — the number the campaign actually
    sent, and the one that used to fall between the two ceilings."""
    client, service = wired
    seen: list[int] = []

    async def record(symbols, interval, limit=1500, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(limit)
        return {}

    service.target_study = record  # type: ignore[method-assign]

    resp = await client.post(STUDY_URL, json=body(limit=20 * 252 + 253))
    assert resp.status_code == 200
    assert seen == [5293]
