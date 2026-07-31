"""A client timeout must not destroy hours of finished work.

Training on the full universe runs for hours and its report lived only in the
HTTP response, so a read timeout on the caller's side lost it completely — even
though the server had finished: uvicorn does not cancel an endpoint when the
caller disconnects, so everything after the await still runs. The result is
recorded on the way out and can be collected afterwards.
"""

import pytest
from httpx import AsyncClient

from src.core.service import MLPipelineService

STUDY_URL = "/api/v1/ml-pipeline/models/target-study"


def body(**extra: object) -> dict:
    return {"symbols": ["AAPL", "MSFT"], "interval": "1d", **extra}


@pytest.mark.asyncio
async def test_a_completed_run_can_be_collected_after_the_fact(
    wired: tuple[AsyncClient, MLPipelineService],
):
    client, service = wired

    async def study(*args, **kwargs):
        return {"recommended_mult": 1.0}

    service.target_study = study  # type: ignore[method-assign]

    # The caller never sees this response — pretend it timed out.
    assert (await client.post(STUDY_URL, json=body())).status_code == 200

    listing = await client.get("/api/v1/ml-pipeline/runs")
    assert [r["operation"] for r in listing.json()["runs"]] == ["target-study"]
    assert listing.json()["runs"][0]["completed_at"]

    recovered = await client.get("/api/v1/ml-pipeline/runs/target-study")
    assert recovered.status_code == 200
    assert recovered.json()["result"] == {"recommended_mult": 1.0}


@pytest.mark.asyncio
async def test_an_unfinished_run_is_404_not_an_empty_result(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """404 has to mean "not finished yet", never "finished and empty" — the
    caller polls this endpoint and must not stop on a fabricated answer."""
    client, _ = wired
    resp = await client.get("/api/v1/ml-pipeline/runs/train")
    assert resp.status_code == 404
    assert "train" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_a_failed_run_is_not_recorded(wired: tuple[AsyncClient, MLPipelineService]):
    """Only completed work is retrievable. Recording a failure would let a
    polling client mistake an error for the result it was waiting for."""
    client, service = wired

    async def boom(*args, **kwargs):
        raise ValueError("dataset too small")

    service.target_study = boom  # type: ignore[method-assign]

    assert (await client.post(STUDY_URL, json=body())).status_code == 400
    assert (await client.get("/api/v1/ml-pipeline/runs/target-study")).status_code == 404


@pytest.mark.asyncio
async def test_a_rerun_replaces_the_previous_report(
    wired: tuple[AsyncClient, MLPipelineService],
):
    client, service = wired
    calls = {"n": 0}

    async def study(*args, **kwargs):
        calls["n"] += 1
        return {"run": calls["n"]}

    service.target_study = study  # type: ignore[method-assign]

    await client.post(STUDY_URL, json=body())
    await client.post(STUDY_URL, json=body())
    recovered = await client.get("/api/v1/ml-pipeline/runs/target-study")
    assert recovered.json()["result"] == {"run": 2}
