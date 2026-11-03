"""Structured run events: one JSON object per line in runs/<run_id>/events.jsonl.

Every actor appends from its own process; each event is a single O_APPEND write, so lines never interleave.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def emit(run: str | Path, actor: str, event: str, round: int | None = None, **data: object) -> dict:
    run = Path(run)
    rec = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "run_id": run.name,
        "round": round,
        "actor": actor,
        "event": event,
        "data": data,
    }
    line = (json.dumps(rec, default=str) + "\n").encode()
    fd = os.open(run / "events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return rec


def read_events(run: str | Path) -> list[dict]:
    p = Path(run) / "events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
