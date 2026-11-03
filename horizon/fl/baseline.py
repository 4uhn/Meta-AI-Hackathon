"""Non-federated baselines: E0 base model, E1a, E1b, … local-only (one per operator), E4 centralised.

    python -m horizon.fl.baseline --exp E1a --config configs/5g-nidd-tiny.yaml

Plaintext by design (not the secure path). E1x reads only operator op_x's shard; E4 pools the shards of
every operator in the config (`fl.operators`) as the upper bound. Training runs rounds × local_epochs
epochs to match the federated compute. Writes <out>/<exp>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from horizon.data.prepare import load_shard, load_test
from horizon.fl.rundir import OPERATORS, configure
from horizon.model.lora import a_hash, create_initial_adapter, flatten_b, layout_hash, load_adapter
from horizon.model.train import evaluate, load_base, mps_mem_gb, train_local
from horizon.tasks import activate

LOCAL = {f"E1{c}": f"op_{c}" for c in "abcdefg"}  # E1x = operator op_x alone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["E0", *LOCAL, "E4"])
    ap.add_argument("--config", default="configs/5g-nidd-tiny.yaml")
    ap.add_argument("--out", help="results directory (default: the config's results_dir)")
    ap.add_argument("--limit-test", type=int, default=0, help="debug: evaluate on the first N test rows")
    ap.add_argument("--max-steps", type=int, default=0, help="debug (smoke): stop training after N steps")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    activate(cfg)
    configure(cfg)
    if args.exp in LOCAL and LOCAL[args.exp] not in OPERATORS:
        raise SystemExit(f"{args.exp}: {LOCAL[args.exp]} is not an operator of this config ({OPERATORS})")
    tc = cfg["train"]
    epochs = tc["rounds"] * tc["local_epochs"]
    test = load_test(cfg["data_dir"])
    if args.limit_test:
        test = test[: args.limit_test]

    args.out = args.out or cfg.get("results_dir", "runs/results")
    smoke = bool(args.max_steps or args.limit_test)
    if smoke:  # partial runs never land in results/
        args.out = "runs/smoke"
    base, tok, rev = load_base(cfg)
    run_dir = Path("runs") / (f"dev_{args.exp}" + ("_smoke" if smoke else ""))
    init = run_dir / "initial_adapter"
    model = create_initial_adapter(base, cfg["lora"], tc["adapter_seed"], init)
    model = load_adapter(model.unload(), init, trainable=True)
    out = {
        "exp": args.exp,
        "model": cfg["model"]["id"],
        "revision": rev,
        "epochs": epochs,
        "layout_hash": layout_hash(model),
        "a_hash": a_hash(model),
        "lr": tc["lr"],
        "dtype": cfg["model"]["dtype"],
        "smoke": smoke,
    }

    if args.exp != "E0":
        if args.exp == "E4":
            rows = []
            for op in OPERATORS:
                rows += load_shard(cfg["data_dir"], op, "train", actor=op)
        else:
            op = LOCAL[args.exp]
            rows = load_shard(cfg["data_dir"], op, "train", actor=op)
        print(f"{args.exp}: training on {len(rows)} rows for {epochs} epochs")
        out["train"] = train_local(
            model, tok, rows, cfg, epochs=epochs, seed=tc["seed"], max_steps=args.max_steps
        )
        b = flatten_b(model)
        out["b_max_abs"] = float(np.abs(b).max())
        model.save_pretrained(run_dir / "adapter")

    m, scores = evaluate(model, tok, test, cfg)
    m["mps_peak_gb"] = round(mps_mem_gb(cfg["model"]["device"]), 2)
    pred = scores.argmax(1)
    out["metrics"] = m
    out["n_test"] = len(test)
    out["pred_sha256"] = hashlib.sha256(pred.astype(np.int8).tobytes()).hexdigest()
    np.save(run_dir / "scores.npy", scores)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.exp}.json").write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "exp": args.exp,
                "acc": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "recall": {k: round(v["recall"], 3) for k, v in m["per_class"].items()},
                "eval_ms": m["eval_ms"],
                "train": out.get("train"),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"total {time.time() - t:.0f}s")
