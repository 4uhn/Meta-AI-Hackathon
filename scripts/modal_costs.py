"""Measured Modal cost and time per config → results/modal_costs.json (shown before a GPU launch).

Cost is what Modal billed for the run's app (`modal billing report`, grouped by app id). Each run was
one ephemeral app, so its id identifies its cost exactly. Wall time comes from the run's own logs.
Needs the Modal account that made the runs.

    python scripts/modal_costs.py
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "modal_costs.json"

# config → experiment → (app id, billing window start, end (exclusive), run directory or None)
RUNS = {
    "5g-nidd-tiny-modal.yaml": {
        "E3 (pipeline test, 3 rounds, --max-steps 2)": (
            "ap-IWg8hyMh5kNoV2gRD3RYLs",
            "2026-11-03",
            "2026-11-04",
            "runs/hzsmoke.nosync/modal_tiny_E3_smoke",
        ),
    },
    "5g-nidd-llama.yaml": {
        "E3": ("ap-A3I6NhJqtH1OJUziFmg7SX", "2026-11-03", "2026-11-04", "runs/hzsmoke.nosync/modal_llama_E3"),
        "weights download (once)": ("ap-c1KGaOo2OiN9DTxnnvvwTC", "2026-11-03", "2026-11-04", None),
    },
    "5g-nidd-glimmer.yaml": {
        "E3": ("ap-05eqBYetjXnHP3q8hwacvH", "2026-10-29", "2026-10-30", "runs/modal_glimmer_E3"),
        "E0, E1a-c, E4 (in parallel)": ("ap-GE62zGXKLgsTaJsMIof8I8", "2026-10-29", "2026-10-30", None),
    },
    "asrs-glimmer.yaml": {
        "E3": ("ap-6nYRSs8cYyOWCHgOrFWvuL", "2026-10-30", "2026-10-31", "runs/modal_glimmer_asrs_E3"),
    },
    "asrs4-glimmer.yaml": {
        "E3": ("ap-9VLCUihVnWMomdcO7IxbVT", "2026-10-30", "2026-10-31", "runs/modal_glimmer_asrs4_E3"),
    },
}
# wall time of runs without an event log: the longest of their parallel containers (logs/modal_usage.jsonl)
USAGE_DIRS = {"ap-GE62zGXKLgsTaJsMIof8I8": "runs/modal_glimmer_baselines"}
WALL_MIN = {"ap-c1KGaOo2OiN9DTxnnvvwTC": 0.6}  # from `modal app list` (created → stopped)


def billed(app: str, start: str, end: str) -> float:
    out = subprocess.run(
        [sys.executable, "-m", "modal", "billing", "report", "--start", start, "--end", end, "--resolution", "h", "--json"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return sum(float(r["cost"]) for r in json.loads(out) if r["object_id"] == app)


def minutes(app: str, run: str | None) -> float:
    if app in WALL_MIN:
        return WALL_MIN[app]
    if run:
        ts = [json.loads(line)["ts"] for line in open(ROOT / run / "events.jsonl")]
        t0, t1 = (dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in (ts[0], ts[-1]))
        return (t1 - t0).total_seconds() / 60
    rows = [json.loads(line) for line in open(ROOT / USAGE_DIRS[app] / "logs" / "modal_usage.jsonl")]
    return max(r["wall_s"] for r in rows) / 60


def main() -> None:
    configs = {}
    for cfg, exps in RUNS.items():
        configs[cfg] = {
            exp: {
                "usd": round(billed(app, s, e), 2),
                "minutes": round(minutes(app, run), 1),
                "app": app,
                "date": s,
            }
            for exp, (app, s, e, run) in exps.items()
        }
    OUT.write_text(
        json.dumps({"source": "modal billing report, by app id", "configs": configs}, indent=1) + "\n"
    )
    print(json.dumps(configs, indent=1))


if __name__ == "__main__":
    main()
