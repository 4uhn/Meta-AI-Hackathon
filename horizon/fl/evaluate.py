"""Evaluator: score a published global adapter on the shared test set.

    python -m horizon.fl.evaluate --run DIR --round R --config CFG [--limit-test N]

Scores bus/round_R/global/adapter (R = 1 is the initial adapter with B = 0; R = rounds + 1 is the final
model) and writes eval/round_R.json and eval/round_R_scores.npy. `--compare-fast` also scores with the
first-token fast path and records its agreement with the primary scores.

    python -m horizon.fl.evaluate --run DIR --round R --config CFG --split val --op op_b

Federated validation for tuning, so the test set is never used: runs as operator `op` on its own val split
and writes only a confusion matrix (no per-row predictions). The scheduler sums the operators' matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from horizon.events import emit
from horizon.fl.rundir import adapter_dir, eval_dir, write_json
from horizon.tasks import activate


def evaluate_val(run: Path, round_: int, cfg: dict, op: str) -> dict:
    from horizon.data.prepare import load_shard
    from horizon.data.serialize import LABELS
    from horizon.model.lora import load_adapter
    from horizon.model.train import confusion, evaluate, load_base

    rows = load_shard(cfg["data_dir"], op, "val", actor=op)
    base, tok, _ = load_base(cfg)
    model = load_adapter(base, adapter_dir(run, round_), trainable=False)
    m, scores = evaluate(model, tok, rows, cfg)
    c = confusion([r["label"] for r in rows], [LABELS[i] for i in scores.argmax(1)])
    out = {"round": round_, "split": "val", "operator": op, "n": len(rows), "confusion": c.tolist()}
    out["eval_ms"] = m["eval_ms"]
    d = eval_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / f"round_{round_}_val_{op}.json", out)
    emit(run, op, "val_eval_done", round_, n=len(rows))
    return out


def evaluate_round(
    run: Path,
    round_: int,
    cfg: dict,
    limit_test: int = 0,
    scoring: str | None = None,
    compare_fast: bool = False,
) -> dict:
    from horizon.data.prepare import load_test
    from horizon.model.lora import load_adapter
    from horizon.model.train import evaluate, load_base, mps_mem_gb

    if scoring:
        cfg = cfg | {"eval": cfg["eval"] | {"scoring": scoring}}
    test = load_test(cfg["data_dir"])
    if limit_test:
        test = test[:limit_test]
    base, tok, _ = load_base(cfg)
    model = load_adapter(base, adapter_dir(run, round_), trainable=False)
    m, scores = evaluate(model, tok, test, cfg)
    m["mps_peak_gb"] = round(mps_mem_gb(cfg["model"]["device"]), 2)
    pred = scores.argmax(1)
    out = {
        "round": round_,
        "n_test": len(test),
        "scoring": cfg["eval"].get("scoring", "full"),
        "pred_sha256": hashlib.sha256(pred.astype(np.int8).tobytes()).hexdigest(),
        "metrics": m,
    }
    d = eval_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"round_{round_}_scores.npy", scores)
    if compare_fast:
        mf, fast = evaluate(model, tok, test, cfg | {"eval": cfg["eval"] | {"scoring": "first_token"}})
        np.save(d / f"round_{round_}_scores_fast.npy", fast)
        out["fast_path"] = {
            "agreement_with_primary": float((fast.argmax(1) == pred).mean()),
            "pred_sha256": hashlib.sha256(fast.argmax(1).astype(np.int8).tobytes()).hexdigest(),
            "metrics": mf,
        }
    write_json(d / f"round_{round_}.json", out)
    emit(
        run,
        "evaluator",
        "eval_done",
        round_,
        metrics_ref=f"eval/round_{round_}.json",
        macro_f1=m["macro_f1"],
        accuracy=m["accuracy"],
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--round", required=True, type=int)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--op", help="--split val: the operator whose own val split this process scores")
    ap.add_argument("--scoring", choices=["full", "first_token"], help="override eval.scoring")
    ap.add_argument("--compare-fast", action="store_true", help="also score with the fast path")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    activate(cfg)
    if a.split == "val":
        out = evaluate_val(a.run, a.round, cfg, a.op)
        print(json.dumps({k: out[k] for k in ("round", "operator", "n")}))
        return
    out = evaluate_round(a.run, a.round, cfg, a.limit_test, a.scoring, a.compare_fast)
    print(
        json.dumps({k: out[k] for k in ("round", "pred_sha256")} | {"macro_f1": out["metrics"]["macro_f1"]})
    )


if __name__ == "__main__":
    main()
