"""Durability of long-run reports, and of a run that never finishes.

Every test here pins a way a finished-or-partial run used to be lost. The
motivating incident: an h=63 training pass completed, was served once from
memory, and was destroyed by stopping the container — no file, no trace.
"""

import json
import os

import pytest

from src.core.dataset import DatasetParams
from src.core.labels import LabelParams
from src.core.monitoring.drift_detector import DriftDetector
from src.core.registry import ModelRegistry
from src.core.run_store import FileRunStore, NullRunStore, build_run_store
from src.core.service import MLPipelineService
from src.core.training import run_training
from src.events.publisher import NullPublisher

from .test_training import SMALL, synthetic_dataset


def service_with(store):
    return MLPipelineService(DriftDetector(), ModelRegistry(), NullPublisher(), run_store=store)


# --- the report outlives the process -------------------------------------


def test_a_recorded_run_is_readable_by_a_process_that_never_saw_it(tmp_path):
    """The whole point: a cold container still has the report.

    Two services over one directory stand in for restart — the second has an
    empty `_runs` dict, exactly like a fresh process.
    """
    first = service_with(FileRunStore(tmp_path))
    first.record_run("train", {"gate": {"passed": False}, "version": "7"})

    second = service_with(FileRunStore(tmp_path))
    entry = second.last_run("train")

    assert entry is not None, "a restarted process lost the report"
    assert entry["result"]["version"] == "7"
    assert [r["operation"] for r in second.runs()] == ["train"]


def test_the_report_lands_on_disk_without_anyone_reading_the_response(tmp_path):
    """Durability must not depend on the HTTP caller still being connected."""
    service = service_with(FileRunStore(tmp_path))
    service.record_run("target-study", {"winner": "h63-excess"})

    written = json.loads((tmp_path / "target-study.json").read_text(encoding="utf-8"))
    assert written["result"]["winner"] == "h63-excess"
    assert written["operation"] == "target-study"
    assert written["completed_at"]


def test_a_stale_report_does_not_impersonate_a_fresh_one(tmp_path):
    """`completed_at` must move on every save.

    The bootstrap poller accepts a report only when this timestamp differs from
    the one it saw before the call. Durability makes that guard load-bearing:
    a cold container now HAS the previous report, and returning it to someone
    waiting on a running pass would look like an answer.
    """
    store = FileRunStore(tmp_path)
    first = store.save("train", {"n": 1})
    second = store.save("train", {"n": 2})

    assert first["completed_at"] != second["completed_at"]
    assert store.load("train")["result"]["n"] == 2


# --- the write cannot corrupt what was already there ----------------------


def test_a_failed_write_leaves_the_previous_report_intact(tmp_path, monkeypatch):
    """Atomicity, stated as the property that matters.

    Writing in place would turn a crash into a truncated JSON that reads as a
    corrupt result rather than an interrupted write — the difference between
    "the run produced garbage" and "the run was interrupted".
    """
    store = FileRunStore(tmp_path)
    store.save("train", {"good": True})

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.save("train", {"good": False})

    assert store.load("train")["result"] == {"good": True}
    assert not list(tmp_path.glob("*.tmp")), "temp file left behind after a failed write"


def test_a_successful_write_leaves_no_temp_files(tmp_path):
    store = FileRunStore(tmp_path)
    store.save("train", {"ok": True})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.json"]


def test_an_unreadable_report_does_not_stop_the_service(tmp_path):
    """A directory is shared with the host; something else may write into it."""
    (tmp_path / "train.json").write_text("{not json", encoding="utf-8")
    store = FileRunStore(tmp_path)
    assert store.load("train") is None
    assert store.index() == []


# --- untrusted input ------------------------------------------------------


@pytest.mark.parametrize(
    "operation", ["../../etc/passwd", "..", "train/../../x", "TRAIN", "train.json", ""]
)
def test_an_unsafe_operation_name_is_refused_not_sanitized(tmp_path, operation):
    """`operation` comes from the URL path, so it is untrusted input.

    Refused rather than rewritten: quietly turning "../../etc/passwd" into
    something harmless hides that a caller asked for it.
    """
    store = FileRunStore(tmp_path)
    with pytest.raises(ValueError):
        store.load(operation)


# --- an interrupted run still says something ------------------------------


def test_an_interrupted_run_leaves_the_folds_it_had_finished(tmp_path):
    """The pass dies after some folds; the checkpoint holds what they showed."""
    ds = synthetic_dataset()
    store = FileRunStore(tmp_path)
    seen = []

    def checkpoint(progress):
        seen.append(progress)
        store.save_progress("train", progress)

    run_training(ds, SMALL, on_progress=checkpoint)

    assert len(seen) > 2, "no checkpoints published during the run"
    saved = store.load_progress("train")
    assert saved is not None
    assert saved["stage"] in {"folds", "holdout", "final_model"}
    assert saved["folds_total"] >= 1
    # pred_std is the field that separates "we lost a learning model" from
    # "it had already collapsed" — the checkpoint is worthless without it.
    assert all("pred_std" in fold for fold in saved["folds"])


def test_the_checkpoint_grows_as_folds_complete(tmp_path):
    ds = synthetic_dataset()
    counts = []
    run_training(ds, SMALL, on_progress=lambda p: counts.append(len(p["folds"])))
    assert counts[0] == 0
    assert counts[-1] >= counts[0]
    assert max(counts) >= 1, "no fold ever reached the checkpoint"


def test_a_throwing_checkpoint_does_not_fail_the_run():
    """A diagnostic must never be able to abort the work it describes."""

    def broken(_progress):
        raise OSError("read-only file system")

    model, report = run_training(synthetic_dataset(), SMALL, on_progress=broken)
    assert model is not None
    assert report.holdout is not None


def test_the_run_clears_its_own_checkpoint_not_only_the_http_route(tmp_path):
    """Cleanup belongs to the producer.

    It used to live only in `record_run`, which the HTTP route calls — so a
    script or a scheduled job calling `train()` directly left a checkpoint that
    said "still running" forever. Caught by the end-to-end test below, not by
    review.
    """
    store = FileRunStore(tmp_path)
    service = service_with(store)
    store.save_progress("train", {"stage": "folds", "folds_done": 3})

    service.clear_progress("train")

    assert service.run_progress("train") is None


def test_the_checkpoint_is_cleared_when_the_real_report_lands(tmp_path):
    """A leftover checkpoint next to a finished report reads as 'still going'."""
    store = FileRunStore(tmp_path)
    service = service_with(store)
    store.save_progress("train", {"stage": "folds", "folds_done": 3})
    assert service.run_progress("train") is not None

    service.record_run("train", {"gate": {"passed": True}})

    assert service.run_progress("train") is None
    assert store.load("train") is not None


def test_the_index_ignores_checkpoints(tmp_path):
    store = FileRunStore(tmp_path)
    store.save("train", {"a": 1})
    store.save_progress("train", {"stage": "folds"})
    assert [e["operation"] for e in store.index()] == ["train"]


# --- the default stays in-memory -----------------------------------------


def test_no_directory_configured_means_no_files(tmp_path):
    """Importing the service in a test or a script must not create directories."""
    store = build_run_store("")
    assert isinstance(store, NullRunStore)
    service = service_with(store)
    service.record_run("train", {"x": 1})
    assert service.last_run("train")["result"]["x"] == 1  # memory still works
    assert not list(tmp_path.iterdir())


def test_an_unusable_directory_degrades_instead_of_refusing_to_start(tmp_path):
    """Losing durability must not take the service down with it."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    store = build_run_store(str(blocker / "reports"))
    assert isinstance(store, NullRunStore)


# --- the service-level path ----------------------------------------------


@pytest.mark.asyncio
async def test_training_checkpoints_land_in_the_configured_directory(tmp_path, monkeypatch):
    """End to end: a training pass writes checkpoints where compose mounts them."""
    from .test_dataset import make_bars, trending
    from .test_train_service import TOY_CONTRACT, FakeMarketDataClient

    universe = {
        "UP": make_bars("UP", trending(220, 0.004)),
        "DOWN": make_bars("DOWN", trending(220, -0.004)),
        "FLATISH": make_bars("FLATISH", trending(220, 0.0005)),
    }
    service = MLPipelineService(
        DriftDetector(),
        ModelRegistry(),
        NullPublisher(),
        market_client=FakeMarketDataClient(universe),
        data_contract=TOY_CONTRACT,
        dataset_params=DatasetParams(label=LabelParams(horizon=10), min_history=60, min_universe=2),
        run_store=FileRunStore(tmp_path),
    )
    from trading_common.schemas import Interval

    result = await service.train(list(universe), Interval.D1, limit=220, params=SMALL)
    # Exactly what POST /models/train does with the return value.
    service.record_run("train", result)

    # The checkpoint is gone (the run finished) and the report is on disk.
    assert not (tmp_path / "train.progress.json").exists()
    written = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    assert written["result"]["gate"]["passed"] == result["gate"]["passed"]
    assert written["result"]["samples"] == result["samples"]
