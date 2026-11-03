"""Re-score saved predictions on the test flows that have no near-duplicate in any train/val shard.

    python -m scripts.leakage_subset_eval [--suite 5g-nidd-llama|5g-nidd-glimmer] [--e3-run DIR] [--baselines-run DIR]

Writes results/<suite>/leakage_subset.json. Uses the saved score arrays of each experiment (no reruns),
checks each against its result JSON's pred_sha256, then recomputes accuracy, macro-F1 and blind-spot
recall (as defined in horizon/eval/report.py) on: all test rows, rows with no train/val flow equal in
all but 1 field (k1-clean), and in all but 2 fields (k2-clean). See scripts/leakage_check.py for the
near-duplicate definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from horizon.data.serialize import LABELS
from horizon.eval.report import blind_spots
from horizon.model.train import metrics
from scripts.leakage_check import OPS, fields, keys, read

SUITES = {  # where each suite's experiments saved their test scores
    "5g-nidd-llama": {
        "E0": "runs/modal_llama_baselines/dev_E0/scores.npy",
        "E1a": "runs/modal_llama_baselines/dev_E1a/scores.npy",
        "E1b": "runs/modal_llama_baselines/dev_E1b/scores.npy",
        "E1c": "runs/modal_llama_baselines/dev_E1c/scores.npy",
        "E4": "runs/modal_llama_baselines/dev_E4/scores.npy",
        "E3": "runs/modal_llama_E3/eval/round_7_scores.npy",
        "E2": "runs/modal_llama_E2/eval/round_7_scores.npy",
    },
    "5g-nidd-glimmer": {
        "E0": "runs/modal_glimmer_baselines/dev_E0/scores.npy",
        "E1a": "runs/modal_glimmer_baselines/dev_E1a/scores.npy",
        "E1b": "runs/modal_glimmer_baselines/dev_E1b/scores.npy",
        "E1c": "runs/modal_glimmer_baselines/dev_E1c/scores.npy",
        "E4": "runs/modal_glimmer_baselines/dev_E4/scores.npy",
        "E3": "runs/modal_glimmer_E3/eval/round_7_scores.npy",
    },
}
LOCAL = {"E1a": "op_a", "E1b": "op_b", "E1c": "op_c"}


def clean_mask(test: list[dict], pool: list[dict], k: int) -> np.ndarray:
    index = {key for r in pool for _, key in keys(fields(r["text"]), k)}
    return np.array([not any(key in index for _, key in keys(fields(r["text"]), k)) for r in test])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed", type=Path)
    ap.add_argument("--suite", default="5g-nidd-llama", choices=sorted(SUITES))
    ap.add_argument("--e3-run", help="your encrypted (E3) run directory (default: the recorded one)")
    ap.add_argument("--baselines-run", help="your baselines run directory, with dev_E0/ … dev_E4/")
    a = ap.parse_args()
    s = SUITES[a.suite]
    for exp, path in s.items():
        run, rest = path.split("/", 2)[1:]
        mine = a.e3_run if exp == "E3" else a.baselines_run if run.endswith("_baselines") else None
        if mine:
            s[exp] = f"{mine.rstrip('/')}/{rest}"
    a.results = Path("results") / a.suite
    a.out = a.results / "leakage_subset.json"
    SCORES = SUITES[a.suite]
    test = read(a.data / "test.jsonl")
    pool = [r for op in OPS for s in ("train", "val") for r in read(a.data / op / f"{s}.jsonl")]
    y = [r["label"] for r in test]
    subsets = {
        "all": np.ones(len(test), bool),
        "k1_clean": clean_mask(test, pool, 1),
        "k2_clean": clean_mask(test, pool, 2),
    }
    spots = blind_spots(yaml.safe_load(open("configs/partitions/5g-nidd.yaml")))
    out: dict = {
        "n": {s: int(m.sum()) for s, m in subsets.items()},
        "support": {},
        "verified_pred_sha256": {},
    }
    for s, m in subsets.items():
        sup = defaultdict(int)
        for lab, keep in zip(y, m, strict=True):
            sup[lab] += int(keep)
        out["support"][s] = dict(sup)
    res: dict = defaultdict(dict)
    for e, path in SCORES.items():
        sc = np.load(path)
        pred = sc.argmax(1)
        ok = (
            hashlib.sha256(pred.astype(np.int8).tobytes()).hexdigest()
            == json.loads((a.results / f"{e}.json").read_text())["pred_sha256"]
        )
        out["verified_pred_sha256"][e] = ok
        if not ok:
            raise SystemExit(f"{e}: {path} does not match {e}.json's pred_sha256")
        for s, m in subsets.items():
            mt = metrics([y[i] for i in np.flatnonzero(m)], [LABELS[p] for p in pred[m]])
            rec = {lab: v["recall"] for lab, v in mt["per_class"].items()}
            res[s][e] = {"accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"], "recall": rec}
    for s in subsets:
        bs = {}
        for e, op in LOCAL.items():
            cls = [c for c in spots[op] if not np.isnan(res[s]["E3"]["recall"][c])]
            bs[op] = {
                k: float(np.mean([res[s][x]["recall"][c] for c in cls]))
                for k, x in (("local", e), ("horizon_E3", "E3"), ("centralised_E4", "E4"))
            }
        bs["mean"] = {
            k: float(np.mean([v[k] for v in bs.values()])) for k in ("local", "horizon_E3", "centralised_E4")
        }
        res[s]["blind_spot_recall"] = bs
        res[s]["E3_share_of_E4_macro_f1"] = res[s]["E3"]["macro_f1"] / res[s]["E4"]["macro_f1"]
    out["results"] = res
    a.out.write_text(json.dumps(out, indent=1))
    for s in subsets:
        r = res[s]
        print(
            s,
            out["n"][s],
            {e: round(r[e]["macro_f1"], 3) for e in SCORES},
            "E3/E4",
            round(r["E3_share_of_E4_macro_f1"], 3),
            "blind",
            {k: round(v, 3) for k, v in r["blind_spot_recall"]["mean"].items()},
        )
    print("support k1_clean", out["support"]["k1_clean"])


if __name__ == "__main__":
    main()
