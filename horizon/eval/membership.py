"""Membership inference on the shared final model (an empirical leakage measurement, not a guarantee).

Encryption hides each update, but the aggregate model can still leak about its training rows. Two
black-box attacks on the label log-probs:

    loss         s(x) = log p_target(y | x)                     (higher → "member")
    reference    s(x) = log p_target(y | x) − log p_base(y | x)  (calibrated by the base model)

Metrics: ROC AUC and TPR at 1% FPR (Carlini et al. 2022) with bootstrap 95% intervals. Members are
operator train rows, non-members are test rows, matched so membership is the only systematic difference.
Experimenter-side: one container reads rows from every operator's shard.

    python -m horizon.eval.membership score --config CFG --run RUN

    RUN/selection.json   rows to score: [{set, op, split, line, sha256, label}, ...]
    RUN/adapters/<name>/ adapters to score besides the base (e.g. E3, E4)
    → RUN/scores_<name>.npy (n_rows × n_labels, summed label log-probs), RUN/scores_base.npy,
      RUN/score_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from horizon.tasks import activate

FPR_POINTS = (0.01,)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_rows(data_dir: str, selection: list[dict]) -> list[dict]:
    """The selected rows, each checked against the text hash recorded at selection time."""
    from horizon.data.prepare import load_shard, load_test

    cache: dict[tuple[str, str], list[dict]] = {}
    out = []
    for s in selection:
        key = (s["op"], s["split"])
        if key not in cache:
            # experimenter side: reads every operator's shard
            cache[key] = (
                load_test(data_dir)
                if s["split"] == "test"
                else load_shard(data_dir, s["op"], s["split"], actor=s["op"])
            )
        row = cache[key][s["line"]]
        if sha(row["text"]) != s["sha256"] or row["label"] != s["label"]:
            raise ValueError(f"row {s} does not match the mounted data")
        out.append(row)
    return out


def score(run: Path, cfg: dict) -> dict:
    import torch

    from horizon.model.labels import score_labels
    from horizon.model.lora import load_adapter
    from horizon.model.train import load_base

    sel = json.loads((run / "selection.json").read_text())
    rows = load_rows(cfg["data_dir"], sel)
    texts = [r["text"] for r in rows]
    names = sorted(p.name for p in (run / "adapters").iterdir() if p.is_dir())
    base, tok, rev = load_base(cfg)
    dev, bf = cfg["model"]["device"], cfg["eval"]["batch_flows"]
    model = load_adapter(base, run / "adapters" / names[0], trainable=False)
    for n in names[1:]:
        model.load_adapter(str(run / "adapters" / n), adapter_name=n)
    summary = {"n_rows": len(rows), "base_revision": rev, "adapters": names, "seconds": {}, "sha256": {}}

    def save(name: str, s: np.ndarray, t0: float) -> None:
        np.save(run / f"scores_{name}.npy", s)
        summary["seconds"][name] = round(time.time() - t0, 1)
        summary["sha256"][name] = hashlib.sha256(s.tobytes()).hexdigest()
        print(f"  {name}: {len(rows)} rows in {summary['seconds'][name]}s", flush=True)

    t0 = time.time()
    with model.disable_adapter():  # base model (equivalent to the initial adapter, B = 0)
        save("base", score_labels(model, tok, texts, dev, bf), t0)
    for i, n in enumerate(names):
        model.set_adapter("default" if i == 0 else n)
        t0 = time.time()
        save(n, score_labels(model, tok, texts, dev, bf), t0)
    if torch.cuda.is_available():
        summary["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (run / "score_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def roc_auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    """P(score of a random member > score of a random non-member), ties counted half (Mann-Whitney)."""
    _, inv, cnt = np.unique(np.concatenate([member, nonmember]), return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]  # 1-based ranks, ties get their average rank
    m, n = len(member), len(nonmember)
    return float((ranks[:m].sum() - m * (m + 1) / 2) / (m * n))


def tpr_at_fpr(member: np.ndarray, nonmember: np.ndarray, fpr: float) -> float:
    """Highest TPR over thresholds t (predict member iff s > t) whose FPR ≤ fpr."""
    allowed = int(np.floor(fpr * len(nonmember)))  # non-members allowed above the threshold
    t = np.sort(nonmember)[::-1][allowed] if allowed < len(nonmember) else -np.inf
    return float((member > t).mean())


def attack_metrics(member: np.ndarray, nonmember: np.ndarray, seed: int, n_boot: int = 2000) -> dict:
    out = {"auc": roc_auc(member, nonmember)}
    for f in FPR_POINTS:
        out[f"tpr_at_{f:g}_fpr"] = tpr_at_fpr(member, nonmember, f)
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in out}
    for _ in range(n_boot):
        m = member[rng.integers(0, len(member), len(member))]
        n = nonmember[rng.integers(0, len(nonmember), len(nonmember))]
        boot["auc"].append(roc_auc(m, n))
        for f in FPR_POINTS:
            boot[f"tpr_at_{f:g}_fpr"].append(tpr_at_fpr(m, n, f))
    out["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
    return out


def analyze(selection: list[dict], scores: dict[str, np.ndarray], labels: list[str], seed: int) -> dict:
    """Attack metrics per target adapter (every key of `scores` except 'base')."""
    y = np.array([labels.index(s["label"]) for s in selection])
    is_m = np.array([s["set"] == "member" for s in selection])
    idx = np.arange(len(y))
    true = {k: v[idx, y] for k, v in scores.items()}
    out = {}
    for name in sorted(k for k in scores if k != "base"):
        sig = {"loss": true[name], "reference": true[name] - true["base"]}
        out[name] = {a: attack_metrics(s[is_m], s[~is_m], seed) for a, s in sig.items()}
        out[name]["mean_true_label_logprob"] = {
            "member": float(true[name][is_m].mean()),
            "nonmember": float(true[name][~is_m].mean()),
        }
    out["base_only"] = {"loss": attack_metrics(true["base"][is_m], true["base"][~is_m], seed)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, type=Path)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    activate(cfg)
    s = score(a.run, cfg)
    print(json.dumps({k: s[k] for k in ("n_rows", "adapters", "seconds")}))


if __name__ == "__main__":
    main()
