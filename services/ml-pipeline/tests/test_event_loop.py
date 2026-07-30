"""Heavy routes must not block /health.

The symptom on the real machine: `--capacity-probe` made ml-pipeline go
`unhealthy` every time. Not a resource problem — the compose healthcheck is
interval 10s / timeout 5s / retries 3, so the container is condemned after ~30
seconds, and the probe's fits ran synchronously inside an `async def`. The
event loop was held for minutes, so /health could not be answered at all.

These pin the property that fixes it: while a long compute runs, the loop still
serves other requests. Written against the loop rather than against the
healthcheck, because that is the actual invariant.
"""

import asyncio
import time

import pytest
from httpx import AsyncClient

from src.core.service import MLPipelineService


@pytest.mark.asyncio
async def test_a_long_computation_leaves_the_event_loop_free(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """A synchronous fit dispatched with asyncio.to_thread keeps the loop
    responsive; called directly it would starve every other request."""
    client, _ = wired

    def blocking() -> str:
        time.sleep(0.6)  # stands in for a model fit
        return "done"

    started = time.perf_counter()
    compute = asyncio.create_task(asyncio.to_thread(blocking))
    await asyncio.sleep(0)  # let the thread start

    # /health must answer while the computation is still running
    health = await client.get("/health")
    answered_at = time.perf_counter() - started

    assert health.status_code == 200
    assert not compute.done(), "the computation finished too early to prove anything"
    assert answered_at < 0.5, f"/health waited {answered_at:.2f}s behind the compute"
    assert await compute == "done"


@pytest.mark.asyncio
async def test_every_heavy_service_method_dispatches_off_the_loop():
    """A structural check: the expensive entry points must hand their compute to
    a thread. Adding a new one that calls run_*(...) directly re-introduces the
    exact failure — and it only shows up on a real machine under a healthcheck,
    never in a unit test of the function itself.
    """
    import inspect

    import src.core.service as service_module

    source = inspect.getsource(service_module)
    heavy = [
        "run_training",
        "run_capacity_probe",
        "run_cpcv",
        "run_sweep",
        "run_alpha_decay",
        "run_sector_study",
        "run_cost_study",
        "score_targets",
        "calibrate_barriers",
    ]
    for name in heavy:
        calls = [
            line.strip() for line in source.splitlines() if f"{name}," in line or f"{name}(" in line
        ]
        # the import line and the to_thread dispatch are fine; a bare call is not
        offenders = [
            line
            for line in calls
            if line.startswith(("return ", "report", "probe", "model,", '"'))
            and "to_thread" not in line
            and f"{name}(" in line
        ]
        assert not offenders, f"{name} is called on the event loop: {offenders}"
