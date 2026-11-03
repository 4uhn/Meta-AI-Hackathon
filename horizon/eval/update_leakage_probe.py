"""Update-leakage attack with auxiliary data: match each update against per-class gradient directions.

The server computes, at the global adapter of round r, the gradient g_c of the plain label loss on a few
labelled rows of each class c (probe_gradients.py). Local training moves B against a mixture of the
operator's class gradients, so −Δ should align with the g_c of the classes it holds. Scores:
- primary: cosine(−Δ, g_c − mean_c' g_c'), the class-specific part of each gradient;
- raw cosine(−Δ, g_c);
- non-negative least squares −Δ ≈ Σ_c α_c g_c, score α_c.
Metrics and controls are those of the logit-lens attack (update_leakage.py).

    python -m horizon.eval.update_leakage_probe --run RUN --partition configs/partitions/5g-nidd.yaml \
        --grads GRADS_DIR --out results/5g-nidd-glimmer/update_leakage_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from horizon.data.serialize import LABELS
from horizon.eval.update_leakage import exact_topk, load_updates, permutation_p, pooled_auc, zscore_rows


def nnls(k: np.ndarray, b: np.ndarray, iters: int = 20000, tol: float = 1e-12) -> np.ndarray:
    """argmin_{α ≥ 0} ‖y − Gᵀα‖² from K = G Gᵀ and b = G y, by projected coordinate descent."""
    a = np.zeros(len(b))
    for _ in range(iters):
        prev = a.copy()
        for i in range(len(b)):
            if k[i, i] > 0:
                a[i] = max(0.0, a[i] + (b[i] - k[i] @ a) / k[i, i])
        if np.abs(a - prev).max() <= tol * max(1.0, np.abs(a).max()):
            break
    return a


def scores(delta: np.ndarray, g: np.ndarray) -> dict[str, np.ndarray]:
    y = -delta
    gc = g - g.mean(0, keepdims=True)
    cos = lambda m: (m @ y) / (np.linalg.norm(m, axis=1) * np.linalg.norm(y))  # noqa: E731
    return {"centred_cosine": cos(gc), "cosine": cos(g), "nnls": nnls(g @ g.T, g @ y)}


def run_probe(run: Path, partition: Path, grads: Path, tag: str = "ref") -> dict:
    part = yaml.safe_load(open(partition))["operators"]
    ops = sorted(part)
    rounds = sorted(int(p.name.split("_")[1]) for p in (run / "insecure").glob("round_*"))
    deltas, _ = load_updates(run, rounds, ops)
    summ = json.loads((grads / "summary.json").read_text())
    g = {}
    for r in rounds:
        p = grads / f"grads_B_r{r}.npy"
        if hashlib.sha256(p.read_bytes()).hexdigest() != summ["grads_sha256"][p.name]:
            raise AssertionError(f"{p} does not match the container's sha256")
        g[r] = np.load(p).astype(np.float64)
    return evaluate(deltas, g, part, ops, rounds) | {
        "run": str(run),
        "grads": str(grads),
        "aux": {k: summ[k] for k in ("k", "seed", "selection_sha256", "loss")},
        "aux_source": "evaluation set held by the server-side evaluator (string-disjoint from every shard)",
        "tag": tag,
    }


def evaluate(deltas: dict, g: dict, part: dict, ops: list[str], rounds: list[int]) -> dict:
    keys = [(r, op) for r in rounds for op in ops]
    attack = [i for i, lab in enumerate(LABELS) if lab != "benign"]
    truth = np.array([[LABELS[i] in part[op] for i in attack] for _, op in keys])
    rnd = np.array([r for r, _ in keys])
    per = [scores(deltas[k], g[k[0]]) for k in keys]
    out = {"labels": LABELS, "attack_classes": [LABELS[i] for i in attack], "rounds": rounds, "variants": {}}
    for v in per[0]:
        s = np.stack([p[v] for p in per])[:, attack]
        z = zscore_rows(s)
        ex = exact_topk(z, truth)
        r1 = rnd == rounds[0]
        out["variants"][v] = {
            "presence_auc": pooled_auc(z, truth),
            "presence_auc_round1": pooled_auc(z[r1], truth[r1]) if r1.sum() else None,
            "top3_exact": int(ex.sum()),
            "top3_exact_round1": int(ex[r1].sum()),
            "top3_exact_by_update": {f"r{r}/{op}": bool(e) for (r, op), e in zip(keys, ex, strict=True)},
            "scores": {f"r{r}/{op}": np.round(row, 6).tolist() for (r, op), row in zip(keys, s, strict=True)},
        }
    zp = zscore_rows(np.stack([p["centred_cosine"] for p in per])[:, attack])
    p_auc, n = permutation_p(lambda t: pooled_auc(zp, t), truth)
    out["primary"] = "centred_cosine"
    out["permutation_test"] = {"n_permutations": n, "p_presence_auc": p_auc}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--partition", type=Path, required=True)
    ap.add_argument("--grads", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = run_probe(a.run, a.partition, a.grads)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    for v, m in res["variants"].items():
        print(v, {k: x for k, x in m.items() if k not in ("scores", "top3_exact_by_update")})
    print(res["permutation_test"])


if __name__ == "__main__":
    main()
