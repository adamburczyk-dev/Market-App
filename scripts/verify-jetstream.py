#!/usr/bin/env python3
"""End-to-end check for NATS JetStream publishing via market-data's NatsPublisher.

Verifies: stream creation, publish, Nats-Msg-Id deduplication, and consume —
using the real `NatsPublisher` + `ensure_stream` from the market-data service.

Modes:
  python scripts/verify-jetstream.py
      Spawn an isolated `nats-server -js` on a temp port, run a deterministic
      round-trip, tear it down. Non-destructive and repeatable. Requires
      `nats-server` on PATH or in ~/go/bin
      (install: GOSUMDB=off go install github.com/nats-io/nats-server/v2@v2.10.22).

  python scripts/verify-jetstream.py --url nats://127.0.0.1:4222
      Run against an already-running NATS (e.g. `make up`). Non-destructive:
      only the dedup delta is asserted (the stream may already hold data).

Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Import the REAL publisher code from the market-data service.
_MARKET_DATA = Path(__file__).resolve().parent.parent / "services" / "market-data"
sys.path.insert(0, str(_MARKET_DATA))

import nats  # noqa: E402
from trading_common.events import MarketDataUpdatedEvent  # noqa: E402

from src.events.publisher import NatsPublisher, ensure_stream  # noqa: E402

STREAM = "MARKET_DATA"
SUBJECTS = ["market_data.>"]
SUBJECT = "market_data.updated"


def _port_open(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket()) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _find_nats_server() -> str | None:
    exe = shutil.which("nats-server")
    if exe:
        return exe
    candidate = Path.home() / "go" / "bin" / "nats-server"
    return str(candidate) if candidate.exists() else None


async def _stream_count(js) -> int:  # type: ignore[no-untyped-def]
    from nats.js.errors import NotFoundError

    try:
        info = await js.stream_info(STREAM)
        return int(info.state.messages)
    except NotFoundError:
        return 0


async def round_trip(url: str, *, strict: bool) -> None:
    nc = await nats.connect(url)
    try:
        js = nc.jetstream()
        await ensure_stream(js, STREAM, SUBJECTS)
        publisher = NatsPublisher(js)

        before = await _stream_count(js)

        ev = MarketDataUpdatedEvent(symbol="AAPL", interval="1d", rows_count=42)
        await publisher.publish(ev)
        await publisher.publish(ev)  # same event_id -> JetStream must dedup
        ev2 = MarketDataUpdatedEvent(symbol="MSFT", interval="1d", rows_count=7)
        await publisher.publish(ev2)

        delta = await _stream_count(js) - before
        assert delta == 2, f"dedup FAILED: stream grew by {delta} (expected 2)"
        print(f"  [dedup] published 3 (1 duplicate) -> stream grew by {delta} ✓")

        if strict:
            sub = await js.pull_subscribe(SUBJECT, durable="verify_jetstream")
            msgs = await sub.fetch(10, timeout=2)
            assert len(msgs) == 2, f"consume FAILED: got {len(msgs)} (expected 2)"
            for m in msgs:
                await m.ack()
            print(f"  [consume] pulled {len(msgs)} messages {[m.subject for m in msgs]} ✓")
    finally:
        await nc.drain()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="connect to an already-running NATS instead of spawning one")
    args = parser.parse_args()

    os.environ.setdefault("DB_PASSWORD", "x")
    os.environ.setdefault("REDIS_PASSWORD", "x")

    proc: subprocess.Popen | None = None
    tmpdir: str | None = None
    try:
        if args.url:
            print(f"Verifying JetStream against running NATS at {args.url} ...")
            asyncio.run(round_trip(args.url, strict=False))
        else:
            exe = _find_nats_server()
            if not exe:
                print(
                    "ERROR: nats-server not found. Either run `make up` and use "
                    "`--url nats://127.0.0.1:4222`, or install it:\n"
                    "  GOSUMDB=off go install github.com/nats-io/nats-server/v2@v2.10.22",
                    file=sys.stderr,
                )
                return 2
            host, port = "127.0.0.1", 14222
            tmpdir = tempfile.mkdtemp(prefix="jsverify-")
            print(f"Spawning isolated {Path(exe).name} -js on {host}:{port} ...")
            proc = subprocess.Popen(
                [exe, "-js", "-sd", tmpdir, "-a", host, "-p", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(40):
                if _port_open(host, port):
                    break
                time.sleep(0.25)
            else:
                print("ERROR: nats-server did not become ready", file=sys.stderr)
                return 3
            asyncio.run(round_trip(f"nats://{host}:{port}", strict=True))

        print("JetStream round-trip OK ✓")
        return 0
    except Exception as exc:  # noqa: BLE001 - report and fail
        print(f"JetStream round-trip FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if proc is not None:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
