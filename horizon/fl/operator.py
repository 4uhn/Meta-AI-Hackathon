"""One operator, one step, one process. Each operator runs this on its own.

    python -m horizon.fl.operator train     --run DIR --op op_a --round 2 --config CFG [--insecure-reference]
    python -m horizon.fl.operator dec-share --run DIR --op op_a --round 2

`train` checks the global adapter's frozen A and B layout against round.json, trains on the operator's own
shard only, and builds the weighted payload [w·B, w]. On the secure path the payload is encrypted under the
collective key and only ciphertext + manifest reach the bus. `--insecure-reference` writes plaintext instead.

`dec-share` re-verifies the aggregate and releases this operator's noise-flooded decryption share.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from horizon.events import emit
from horizon.fl.rundir import (
    OPERATORS,
    adapter_dir,
    configure,
    insecure_update,
    round_info,
    submission_dir,
)
from horizon.tasks import activate


def train(run: Path, op: str, round_: int, cfg: dict, insecure: bool, max_steps: int = 0) -> dict:
    # Heavy imports are local so `dec-share` stays a light process.
    from horizon.data.prepare import load_shard
    from horizon.he_bridge import HorizonHE, write_f64
    from horizon.model.lora import a_hash, flatten_b, layout_hash, load_adapter, weighted_payload
    from horizon.model.train import load_base, train_local

    info = round_info(run, round_)
    base, tok, rev = load_base(cfg)
    model = load_adapter(base, adapter_dir(run, round_), trainable=True)
    checks = {
        "lora_A_hash": (a_hash(model), info["lora_A_hash"]),
        "layout_hash": (layout_hash(model), info["layout_hash"]),
        "base_revision": (rev, info["base_revision"]),
    }
    for k, (mine, announced) in checks.items():
        if mine != announced:
            emit(run, op, "submission_withheld", round_, reason=f"{k} mismatch")
            raise RuntimeError(f"{op}: {k} {mine[:12]} ≠ announced {announced[:12]}; not training")

    rows = load_shard(cfg["data_dir"], op, "train", actor=op)
    emit(run, op, "train_started", round_, n=len(rows))
    seed = cfg["train"]["seed"] + round_ + OPERATORS.index(op)  # distinct per (round, operator)
    tr = train_local(
        model, tok, rows, cfg, epochs=cfg["train"]["local_epochs"], seed=seed, max_steps=max_steps
    )
    emit(run, op, "train_done", round_, loss=tr["loss_last"], n=tr["n"], ms=tr["ms"])

    payload, w = weighted_payload(flatten_b(model), tr["n"])
    manifest = {
        "run_id": run.name,
        "round": round_,
        "operator": op,
        "layout_hash": checks["layout_hash"][0],
        "lora_A_hash": checks["lora_A_hash"][0],
        "base_revision": rev,
        "n_values": int(payload.size),
        "train_examples": tr["n"],
        "train_loss_final": tr["loss_last"],
        "train_ms": tr["ms"],
        "train_steps": tr["steps"],
        "mps_peak_gb": tr["mps_peak_gb"],
        "cuda_peak_gb": tr.get("cuda_peak_gb", 0.0),
    }
    if insecure:
        p = insecure_update(run, round_, op)
        write_f64(p, payload)
        manifest |= {"mode": "insecure-reference", "path": str(p.relative_to(run))}
        emit(run, op, "insecure_reference_written", round_, warning="PLAINTEXT update (reference run)")
        print(f"WARNING {op}: --insecure-reference wrote its plaintext update to {p}")
    else:
        he = HorizonHE(run)
        res = he.encrypt(op, round_, payload)  # plaintext staged in parties/<op>/state/, then deleted
        manifest |= {
            "mode": "encrypted",
            "n_ciphertexts": res["n_ciphertexts"],
            "ct_sha256": res["ct_sha256"],
            "bytes": res["bytes"],
            "expansion_vs_fp16": res["expansion_vs_fp16"],
            "encrypt_ms": res["encrypt_ms"],
        }
        emit(run, op, "encrypted", round_, bytes=res["bytes"], ms=res["encrypt_ms"], ct_sha=res["ct_sha256"])
    manifest["created_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    d = submission_dir(run, round_, op)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    emit(run, op, "submitted", round_, mode=manifest["mode"])
    return manifest


def dec_share(run: Path, op: str, round_: int) -> dict:
    from horizon.he_bridge import HorizonHE

    t = time.time()
    res = HorizonHE(run).dec_share(op, round_)
    emit(run, op, "dec_share", round_, party=op, ms=int(1000 * (time.time() - t)))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "dec-share"])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--op", required=True, help="op_a, op_b, … (dec-share: checked against parties.json)")
    ap.add_argument("--round", required=True, type=int)
    ap.add_argument("--config")
    ap.add_argument("--insecure-reference", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: cap local training steps")
    a = ap.parse_args()
    if a.step == "train":
        cfg = yaml.safe_load(open(a.config))
        activate(cfg)
        if a.op not in configure(cfg):
            raise SystemExit(f"--op {a.op}: not an operator of this run ({OPERATORS})")
        out = train(a.run, a.op, a.round, cfg, a.insecure_reference, a.max_steps)
    else:
        out = dec_share(a.run, a.op, a.round)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
