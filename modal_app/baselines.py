"""Baselines on Modal: E0 base, E1a, E1b, … local-only, E4 centralised, one container each, in
parallel. Each runs `horizon.fl.baseline` unchanged; E1x mounts only its own operator's shard.

    python -m modal_app.baselines --config configs/5g-nidd-llama.yaml --exp E1a
    python -m modal_app.baselines --config configs/5g-nidd-llama.yaml --exp E0 E1a E1b E1c E4

Results → --out, else the config's results_dir (smoke runs → <run>/smoke/); adapters, logs and
per-container cost → --run-root/<config>_baselines/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from modal_app.backend import ModalBackend, session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", nargs="+", required=True, choices=["E0", *(f"E1{c}" for c in "abcdefg"), "E4"])
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--out", help="results directory (default: the config's results_dir)")
    ap.add_argument("--run-root", default="runs")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    smoke = bool(a.max_steps or a.limit_test)
    run = Path(a.run_root) / f"{Path(a.config).stem}_baselines"
    # baseline.py sends smoke results to runs/smoke/ itself; they land under this run dir
    out = run / "smoke" if smoke else Path(a.out or cfg.get("results_dir", "runs/results"))
    run.mkdir(parents=True, exist_ok=True)
    dbg = (["--max-steps", str(a.max_steps)] if a.max_steps else []) + (
        ["--limit-test", str(a.limit_test)] if a.limit_test else []
    )
    steps = [
        (f"baseline_{e}", ["horizon.fl.baseline", "--exp", e, "--config", a.config, "--out", "results", *dbg])
        for e in a.exp
    ]
    be = ModalBackend(a.config)
    with session():
        be.run_steps(run, steps, out_dir=out)
    for e in a.exp:
        r = json.loads((out / f"{e}.json").read_text())
        m = r["metrics"]
        print(e, json.dumps({"acc": m["accuracy"], "macro_f1": m["macro_f1"], "train": r.get("train")}))
    print(f"Modal est. ${be.spent:.3f} → {out}")


if __name__ == "__main__":
    main()
