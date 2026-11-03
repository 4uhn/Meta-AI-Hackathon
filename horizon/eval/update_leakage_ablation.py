"""Does the leak depend on the restricted softmax? Attack round-1 updates trained with and without it.

Both runs start from the same initial global adapter (B = 0, same frozen A), shards and seeds; they differ
only in `train.restrict_missing_labels`. With it, a class the operator lacks gets no gradient at the
decision token; without it, that class is pushed down. Both leave a class-specific trace, so which one is
easier to read is an empirical question. Each run's three round-1 updates are scored with the logit lens
(update_leakage.py) and, when gradients are given, with gradient matching (update_leakage_probe.py).

    python -m horizon.eval.update_leakage_ablation --partition configs/partitions/5g-nidd.yaml \
        --run with_restriction=RUN_A --run without=RUN_B --grads GRADS_DIR \
        --out results/5g-nidd-glimmer/update_leakage_ablation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from horizon.data.serialize import LABELS
from horizon.eval.update_leakage import (
    auc,
    exact_topk,
    fetch_readout_weights,
    label_first_tokens,
    layer_scores,
    o_proj_layout,
    permutation_p,
    pooled_auc,
    readout_directions,
    total_score,
    zscore_rows,
)
from horizon.eval.update_leakage_probe import scores as probe_scores
from horizon.model.lora import read_b_file


def round1_updates(run: Path, ops: list[str]) -> tuple[dict, dict]:
    g = read_b_file(run / "bus/round_1/global/adapter")
    if np.any(g != 0):
        raise AssertionError(f"{run}: round-1 global B is not zero")
    out, sha = {}, {}
    for op in ops:
        p = run / f"insecure/round_1/{op}.f64"
        v = np.fromfile(p)
        out[op] = v[:-1] / v[-1]
        sha[op] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out, sha


def metrics(s9: np.ndarray, truth9: np.ndarray, attack: list[int], perm: bool = False) -> dict:
    s, t = s9[:, attack], truth9[:, attack]
    z = zscore_rows(s)
    extra = {}
    if perm:  # exact: relabel the attack classes by every permutation (same for every update)
        extra["presence_auc_perm_p"], extra["n_permutations"] = permutation_p(lambda tt: pooled_auc(z, tt), t)
    return extra | {
        "presence_auc": pooled_auc(z, t),
        "mean_within_update_auc": float(np.mean([auc(r[m], r[~m]) for r, m in zip(z, t, strict=True)])),
        "top3_exact": int(exact_topk(z, t).sum()),
        "held_minus_absent_mean_z": float(z[t].mean() - z[~t].mean()),
        "z": np.round(z, 3).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--partition", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, help="name=run_dir (round 1 is attacked)")
    ap.add_argument("--grads", type=Path, help="probe gradients dir (uses grads_B_r1.npy)")
    ap.add_argument("--cache", type=Path, default=Path("runs/cache/glimmer_readout_weights.npz"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--perm", action="store_true", help="add the exact permutation p-value of each pooled AUC"
    )
    a = ap.parse_args()
    part = yaml.safe_load(open(a.partition))["operators"]
    ops = sorted(part)
    attack = [i for i, lab in enumerate(LABELS) if lab != "benign"]
    truth9 = np.array([[lab in part[op] for lab in LABELS] for op in ops])
    runs = dict(x.split("=", 1) for x in a.run)
    g1 = None
    if a.grads:
        summ = json.loads((a.grads / "summary.json").read_text())
        p = a.grads / "grads_B_r1.npy"
        if hashlib.sha256(p.read_bytes()).hexdigest() != summ["grads_sha256"][p.name]:
            raise AssertionError(f"{p} does not match the container's sha256")
        g1 = np.load(p).astype(np.float64)
    out = {"labels": LABELS, "attack_classes": [LABELS[i] for i in attack], "operators": ops, "runs": {}}
    a_hashes = set()
    for name, rd in runs.items():
        run = Path(rd)
        cfg = yaml.safe_load(open(run / "config.yaml")) if (run / "config.yaml").is_file() else None
        info = json.loads((run / "bus/round_1/global/round.json").read_text())
        a_hashes.add(info["lora_A_hash"])
        deltas, sha = round1_updates(run, ops)
        blocks, a_mats, _ = o_proj_layout(run / "bus/round_1/global/adapter")
        mcfg = (cfg or yaml.safe_load(open("configs/5g-nidd-glimmer.yaml")))["model"]
        w = fetch_readout_weights(
            mcfg["id"], mcfg["revision"], label_first_tokens({"model": mcfg}), len(blocks), a.cache
        )
        res = {"run": rd, "inputs_sha256": sha, "lora_A_hash": info["lora_A_hash"], "logit_lens": {}}
        for var, ro in [
            ("primary", readout_directions(w)),
            ("uncentred", readout_directions(w, centre=False)),
        ]:
            s9 = np.stack([total_score(layer_scores(deltas[op], blocks, a_mats, ro)) for op in ops])
            res["logit_lens"][var] = metrics(s9, truth9, attack, a.perm)
        if g1 is not None:
            per = [probe_scores(deltas[op], g1) for op in ops]
            res["gradient_matching"] = {
                v: metrics(np.stack([p[v] for p in per]), truth9, attack, a.perm) for v in per[0]
            }
        out["runs"][name] = res
    if len(a_hashes) != 1:
        raise AssertionError(f"runs do not share the frozen A: {a_hashes}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    for name, r in out["runs"].items():
        print(
            name,
            {
                k: {v: (m["presence_auc"], m["top3_exact"]) for v, m in r[k].items()}
                for k in ("logit_lens", "gradient_matching")
                if k in r
            },
        )


if __name__ == "__main__":
    main()
