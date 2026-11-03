"""Per-class gradient directions for the update-leakage attack (server side, auxiliary data).

For each global adapter the attacker computes, per class, the gradient of the plain label-token loss
(full softmax: it does not know any operator's label set) with respect to every lora_B value, on k rows of
that class from labelled data it holds. Here that is the evaluation set, which is string-disjoint from every
operator shard. The gradients are later matched against each operator's update (update_leakage_probe.py).

    python -m horizon.eval.probe_gradients --config CFG --adapter DIR --lora-a-hash HASH \
        --b B_r1.npy B_r2.npy … --test test.jsonl --k 32 --seed 7 --out DIR

Writes grads_<name>.npy (n_labels × n_B, float32, rows in LABELS order) per B vector and selection.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from horizon.data.serialize import LABELS
from horizon.model.labels import collate, encode_example, label_logits
from horizon.model.lora import a_hash, b_params, load_adapter, n_b_values, unflatten_b
from horizon.model.train import cuda_peak_gb, load_base, seed_all
from horizon.tasks import activate


def select_rows(rows: list[dict], k: int, seed: int) -> dict[str, list[int]]:
    """k row indices per class, seeded, sorted (the attacker's auxiliary sample)."""
    rng = np.random.default_rng(seed)
    out = {}
    for lab in LABELS:
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        if len(idx) < k:
            raise ValueError(f"{lab}: {len(idx)} rows < k = {k}")
        out[lab] = sorted(int(i) for i in rng.choice(idx, size=k, replace=False))
    return out


def class_gradient(model, tok, rows: list[dict], micro: int, device: str) -> np.ndarray:
    """∂(mean label-token cross-entropy over `rows`)/∂B, flattened in the payload order."""
    params = [p for _, p in b_params(model)]
    for p in params:
        p.grad = None
    data = [encode_example(tok, r["text"], r["label"]) for r in rows]
    n_tok = sum(sum(t != -100 for t in y[1:]) for _, y in data)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(data), micro):
        b = {k: v.to(device) for k, v in collate(data[s : s + micro], pad).items()}
        logits, tgt = label_logits(model, b["input_ids"], b["attention_mask"], b["labels"])
        (F.cross_entropy(logits, tgt, reduction="sum") / n_tok).backward()
    return np.concatenate([p.grad.detach().float().reshape(-1).cpu().numpy() for p in params])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, type=Path, help="any global adapter of the run (A is frozen)")
    ap.add_argument("--lora-a-hash", required=True, help="announced frozen-A hash; checked before use")
    ap.add_argument("--b", nargs="+", required=True, type=Path, help="global B vectors (.npy, float64)")
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open(a.config))
    activate(cfg)
    seed_all(cfg["train"]["seed"])
    base, tok, rev = load_base(cfg)
    model = load_adapter(base, a.adapter, trainable=True)
    if a_hash(model) != a.lora_a_hash:
        raise RuntimeError("frozen A differs from the announced hash")
    rows = [json.loads(line) for line in a.test.open()]
    sel = select_rows(rows, a.k, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "selection.json").write_text(json.dumps(sel))
    model.train()  # same mode as local training (gradient checkpointing is active only in train mode)
    micro, device = cfg["train"].get("micro_batch", 16), cfg["model"]["device"]
    shas, per = {}, {}
    for bp in a.b:
        t = time.time()
        unflatten_b(model, np.load(bp))
        g = np.stack(
            [class_gradient(model, tok, [rows[i] for i in sel[lab]], micro, device) for lab in LABELS]
        )
        if not np.isfinite(g).all():
            raise FloatingPointError(f"non-finite gradient for {bp.name}")
        dst = a.out / f"grads_{bp.stem}.npy"
        np.save(dst, g.astype(np.float32))
        shas[dst.name] = hashlib.sha256(dst.read_bytes()).hexdigest()
        per[bp.stem] = round(time.time() - t, 1)
        print(
            f"{bp.stem}: {per[bp.stem]} s, |g| per class {np.round(np.linalg.norm(g, axis=1), 4).tolist()}",
            flush=True,
        )
    summary = {
        "base_revision": rev,
        "n_b_values": n_b_values(model),
        "k": a.k,
        "seed": a.seed,
        "selection_sha256": hashlib.sha256((a.out / "selection.json").read_bytes()).hexdigest(),
        "grads_sha256": shas,
        "seconds_per_adapter": per,
        "seconds": round(time.time() - t0, 1),
        "cuda_peak_gb": cuda_peak_gb(),
        "loss": "mean label-token cross-entropy, full softmax (no restriction, no logit adjustment)",
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
