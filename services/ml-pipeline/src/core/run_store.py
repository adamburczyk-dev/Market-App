"""Durable storage for long-run reports — memory is not where they belong.

A full training pass costs hours of real compute and produces exactly one
artifact: the report. Three times now that artifact has been destroyed by
something that had nothing to do with the work succeeding.

  1. A client read timeout threw away a finished run (2026-07-31). Fixed by
     remembering the result in `MLPipelineService._runs` and serving it from
     `GET /runs/{operation}` — which is what `record_run` still does.
  2. That memory is per-process. Stopping the container erased the h=63 run
     (2026-08-07) with no trace left anywhere: the report lived only in a dict
     on a service instance that no longer exists.
  3. Neither mechanism survives an interruption DURING the run. Hours of
     completed folds vanish because the pass never reached its return
     statement.

This module closes all three. Reports are written to a directory that outlives
the process, and a run publishes per-fold progress as it goes, so an
interrupted pass still leaves evidence of what it had learned.

Two properties matter more than the feature itself:

**The write is atomic.** A report is serialized to a temp file in the SAME
directory, fsynced, then `os.replace`d over the target. `os.replace` is atomic
on POSIX and on Windows, so a crash mid-write leaves either the old complete
file or the new one — never a truncated JSON that parses as garbage and looks
like a corrupt result rather than an interrupted write. Writing in place would
turn a power cut into a fabricated report.

**A stale report never impersonates a fresh one.** `completed_at` is stamped at
save time and the poller in `scripts/bootstrap-universe.py` accepts a report
only when that timestamp CHANGED. Durability makes this guard load-bearing:
before, a cold container had no report at all; now it has the previous one, and
returning that to a caller waiting on a running pass would be worse than a
timeout — it would look like an answer.
"""

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import structlog

logger = structlog.get_logger()

# `operation` arrives from the URL path (`GET /runs/{operation}`), so it is
# untrusted input that gets turned into a filename. Anything outside this shape
# is refused rather than sanitized: quietly rewriting "../../etc/passwd" into
# something harmless hides the fact that a caller asked for it.
_OPERATION = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class RunStorage(Protocol):
    """What the service needs from a run store (real or null)."""

    def save(self, operation: str, result: dict[str, Any]) -> dict[str, Any]: ...
    def load(self, operation: str) -> dict[str, Any] | None: ...
    def index(self) -> list[dict[str, Any]]: ...
    def save_progress(self, operation: str, progress: dict[str, Any]) -> None: ...
    def load_progress(self, operation: str) -> dict[str, Any] | None: ...
    def clear_progress(self, operation: str) -> None: ...


class NullRunStore:
    """No-op store — same shape, no persistence.

    The default, so importing the service in a test or a script does not create
    directories. Matches the Null*Repository fallback used for Redis.
    """

    def save(self, operation: str, result: dict[str, Any]) -> dict[str, Any]:
        return _entry(operation, result)

    def load(self, operation: str) -> dict[str, Any] | None:
        return None

    def index(self) -> list[dict[str, Any]]:
        return []

    def save_progress(self, operation: str, progress: dict[str, Any]) -> None:
        return None

    def load_progress(self, operation: str) -> dict[str, Any] | None:
        return None

    def clear_progress(self, operation: str) -> None:
        return None


class FileRunStore:
    """Reports as JSON files under ``directory``, one per operation."""

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    # --- paths -----------------------------------------------------------

    def _path(self, operation: str, suffix: str = "") -> Path:
        if not _OPERATION.fullmatch(operation):
            raise ValueError(f"refusing unsafe operation name: {operation!r}")
        return self._dir / f"{operation}{suffix}.json"

    # --- writes ----------------------------------------------------------

    def _write_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        """Serialize fully, fsync, then rename over the target.

        Serialization happens BEFORE the file is touched: a report containing a
        non-serializable value must fail without destroying the previous one.
        """
        body = json.dumps(payload, indent=1, default=str)
        fd, tmp_name = tempfile.mkstemp(dir=self._dir, prefix=path.stem, suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def save(self, operation: str, result: dict[str, Any]) -> dict[str, Any]:
        entry = _entry(operation, result)
        path = self._path(operation)
        self._write_atomic(path, entry)
        logger.info("Run report persisted", operation=operation, path=str(path))
        return entry

    def save_progress(self, operation: str, progress: dict[str, Any]) -> None:
        """Checkpoint a run in flight. Best-effort: never fails the run.

        A progress write that raises would abort a training pass to protect a
        diagnostic, which inverts the priority — the run is the valuable thing.
        """
        try:
            self._write_atomic(self._path(operation, ".progress"), progress)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Progress checkpoint failed", operation=operation, error=str(exc))

    def clear_progress(self, operation: str) -> None:
        """Drop the checkpoint once the real report exists.

        A leftover progress file sitting next to a complete report is the same
        trap as a stale run: it describes a pass that is over, in a file whose
        whole meaning is "this one is still going".
        """
        try:
            self._path(operation, ".progress").unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Progress cleanup failed", operation=operation, error=str(exc))

    # --- reads -----------------------------------------------------------

    def _read(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Atomic writes make this unreachable for files we wrote, so a
            # failure here means something else touched the directory. Log the
            # path and return nothing — a service that cannot read one stale
            # report must still start.
            logger.warning("Unreadable run report", path=str(path), error=str(exc))
            return None
        return loaded if isinstance(loaded, dict) else None

    def load(self, operation: str) -> dict[str, Any] | None:
        return self._read(self._path(operation))

    def load_progress(self, operation: str) -> dict[str, Any] | None:
        return self._read(self._path(operation, ".progress"))

    def index(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self._dir.glob("*.json")):
            if path.stem.endswith(".progress"):
                continue
            entry = self._read(path)
            if entry is not None and "operation" in entry:
                out.append(
                    {"operation": entry["operation"], "completed_at": entry.get("completed_at")}
                )
        return out


def _entry(operation: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": operation,
        "completed_at": datetime.now(UTC).isoformat(),
        "result": result,
    }


def build_run_store(directory: str | None) -> RunStorage:
    """A file store when a directory is configured, otherwise the null store.

    Falls back to null (rather than raising) when the directory cannot be
    created: losing report durability must not stop the service from serving.
    The log line is the signal that it happened.
    """
    if not directory:
        return NullRunStore()
    try:
        return FileRunStore(directory)
    except OSError as exc:
        logger.warning(
            "Run report directory unusable — reports stay in memory only",
            directory=directory,
            error=str(exc),
        )
        return NullRunStore()
