"""Membership inference on Modal: select rows locally, score base / E3 / E4 in one GPU container, analyse.

    python -m modal_app.membership --suite 5g-nidd-glimmer --smoke  # → runs/5g-nidd-glimmer_mia_smoke/
    python -m modal_app.membership --suite 5g-nidd-glimmer  # → results/<suite>/membership_inference.json
    python -m modal_app.membership --suite asrs-glimmer

See horizon/eval/membership.py for the attacks and metrics. Row selection (`select`) makes membership
the only systematic difference between the two sets:
- class-matched: equal counts per cell (5G-NIDD: class; ASRS: class × reporter organisation, since
  members come only from airline / ATC / GA reports and writing style differs by organisation; test
  reports from other organisations are not used);
- isolation on both sides: a non-member has no near-duplicate in any train/val shard, and a member
  has no near-duplicate among the *other* train/val rows (5G-NIDD: no row equal in all but one field,
  the k1 test of scripts/leakage_check.py; ASRS: 5-gram Jaccard < 0.2, the level every test report
  already meets). Without the member side, members would be the repeated, typical rows and
  non-members the unusual ones, which inflates AUC without any memorisation.
The base model (never trained on any of these rows) is attacked too, as a control: AUC ≈ 0.5 there
means the matching left no non-membership signal.

The container is an experimenter-side evaluation (it mounts all three shards, like E4) and runs
`horizon.eval.membership score` through the generic `run_job` entry point; backend.plan() is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from horizon.eval.membership import analyze, sha
from horizon.tasks import activate
from modal_app.app import M_MODELS, actor_glimmer, data_vols, eval_vol, models_vol
from modal_app.backend import _vol, est_usd, last_json, session

OPS = ("op_a", "op_b", "op_c")
SUITES = {  # the saved test scores double as a cross-check of the container's scores on non-members
    "5g-nidd-glimmer": {
        "config": "configs/5g-nidd-glimmer.yaml",
        "data": "data/processed",
        "adapters": {
            "E3": "runs/modal_glimmer_E3/bus/round_7/global/adapter",
            "E4": "runs/modal_glimmer_baselines/dev_E4/adapter",
        },
        "test_scores": {
            "E3": "runs/modal_glimmer_E3/eval/round_7_scores.npy",
            "E4": "runs/modal_glimmer_baselines/dev_E4/scores.npy",
            "base": "runs/modal_glimmer_baselines/dev_E0/scores.npy",
        },
    },
    "asrs-glimmer": {
        "config": "configs/asrs-glimmer.yaml",
        "data": "data/asrs",
        "partition": "configs/partitions/asrs.yaml",
        "adapters": {
            "E3": "runs/modal_glimmer_asrs_E3/bus/round_7/global/adapter",
            "E4": "runs/modal_glimmer_asrs_baselines/dev_E4/adapter",
        },
        "test_scores": {
            "E3": "runs/modal_glimmer_asrs_E3/eval/round_7_scores.npy",
            "E4": "runs/modal_glimmer_asrs_baselines/dev_E4/scores.npy",
            "base": "runs/modal_glimmer_asrs_baselines/dev_E0/scores.npy",
        },
    },
}


def with_runs(s: dict, e3: str | None, baselines: str | None) -> dict:
    """The suite with its recorded run directories replaced by yours (same layout)."""
    old = {s["adapters"]["E3"].split("/bus/")[0]: e3, s["adapters"]["E4"].rsplit("/dev_E4/", 1)[0]: baselines}

    def move(p: str) -> str:
        for a, b in old.items():
            if b and p.startswith(a + "/"):
                return b.rstrip("/") + p[len(a) :]
        return p

    return {**s, **{k: {n: move(v) for n, v in s[k].items()} for k in ("adapters", "test_scores")}}


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def item(set_: str, op: str, split: str, line: int, row: dict) -> dict:
    return {
        "set": set_,
        "op": op,
        "split": split,
        "line": line,
        "sha256": sha(row["text"]),
        "label": row["label"],
    }


def candidates_nidd(data: Path) -> tuple[dict, dict]:
    """Cell = class. Non-member: test row k1-clean vs all train/val; member: train row with no other
    train/val row equal in all but one field."""
    from scripts.leakage_check import fields, keys

    shards = {(op, s): read(data / op / f"{s}.jsonl") for op in OPS for s in ("train", "val")}
    count = Counter(k for rows in shards.values() for r in rows for _, k in keys(fields(r["text"]), 1))
    mem, non = defaultdict(list), defaultdict(list)
    for op in OPS:
        for i, r in enumerate(shards[(op, "train")]):
            if all(count[k] == 1 for _, k in keys(fields(r["text"]), 1)):
                mem[r["label"]].append(item("member", op, "train", i, r))
    for i, r in enumerate(read(data / "test.jsonl")):
        if all(count[k] == 0 for _, k in keys(fields(r["text"]), 1)):
            non[r["label"]].append(item("nonmember", "test", "test", i, r))
    return mem, non


def candidates_asrs(data: Path, partition: str) -> tuple[dict, dict]:
    """Cell = (class, operator organisation). Test reports' organisation is recovered read-only from
    the pinned raw export (exact text match; texts are unique after prep)."""
    from horizon.data.asrs import load_reports, shingles

    pcfg = yaml.safe_load(open(partition))
    rep, _ = load_reports(Path(os.path.expanduser(pcfg["raw_csv"])), pcfg)
    org = dict(zip(rep["text"], rep["org"], strict=True))
    group = {o: op for op, v in pcfg["operators"].items() for o in v["reporter_orgs"]}
    shard = [
        (op, s, i, r)
        for op in OPS
        for s in ("train", "val")
        for i, r in enumerate(read(data / op / f"{s}.jsonl"))
    ]
    sh = [shingles(r["text"], pcfg["near_dup_shingle"]) for *_, r in shard]
    vocab: dict[str, int] = {}
    post: list[list[int]] = []
    for j, s in enumerate(sh):
        for g in s:
            k = vocab.setdefault(g, len(vocab))
            if k == len(post):
                post.append([])
            post[k].append(j)
    post_np = [np.asarray(p) for p in post]
    size = np.array([len(s) for s in sh])
    mem, non = defaultdict(list), defaultdict(list)
    for j, (op, split, i, r) in enumerate(shard):
        if split != "train":
            continue
        inter = np.bincount(np.concatenate([post_np[vocab[g]] for g in sh[j]]), minlength=len(sh))
        jac = inter / (len(sh[j]) + size - inter)
        jac[j] = 0.0
        if jac.max() < 0.2:
            mem[(r["label"], op)].append(item("member", op, "train", i, r))
    for i, r in enumerate(read(data / "test.jsonl")):
        op = group.get(org[r["text"]])
        if op:  # test reports from other organisations have no member counterpart
            non[(r["label"], op)].append(item("nonmember", "test", "test", i, r))
    return mem, non


def select(suite: str, n: int, seed: int) -> tuple[list[dict], dict]:
    s = SUITES[suite]
    data = Path(s["data"])
    mem, non = candidates_asrs(data, s["partition"]) if "partition" in s else candidates_nidd(data)
    cap = {c: min(len(mem[c]), len(non[c])) for c in sorted(set(mem) | set(non), key=str)}
    cap = {c: v for c, v in cap.items() if v}
    total = sum(cap.values())
    if n > total:
        raise ValueError(f"{n} per side > {total} matched rows available")
    raw = {c: n * v / total for c, v in cap.items()}  # proportional, largest remainder
    alloc = {c: int(np.floor(x)) for c, x in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - alloc[c], reverse=True)[: n - sum(alloc.values())]:
        alloc[c] += 1
    rng = np.random.default_rng(seed)
    sel = []
    for c, k in alloc.items():
        for pool in (mem[c], non[c]):
            sel += [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
    info = {
        "cells": {
            str(c): {"members_available": len(mem[c]), "nonmembers_available": len(non[c]), "used": alloc[c]}
            for c in cap
        },
        "n_per_side": n,
        "seed": seed,
    }
    return sel, info


def tree(base: Path, prefix: str) -> dict[str, bytes]:
    return {f"{prefix}/{p.relative_to(base)}": p.read_bytes() for p in sorted(base.rglob("*")) if p.is_file()}


def cross_check(sel: list[dict], scores: dict[str, np.ndarray], suite: str) -> dict:
    """Non-members are test rows: the container's scores on them should reproduce the saved test scores
    of the same model (bf16/nf4 kernels and different batch neighbours → small differences)."""
    out = {}
    lines = np.array([s["line"] for s in sel if s["set"] == "nonmember"])
    nm = np.array([s["set"] == "nonmember" for s in sel])
    for name, p in SUITES[suite]["test_scores"].items():
        saved = np.load(p)[lines]
        mine = scores[name][nm]
        out[name] = {
            "n": int(len(lines)),
            "argmax_agreement": float((saved.argmax(1) == mine.argmax(1)).mean()),
            "max_abs_diff": float(np.abs(saved - mine).max()),
            "median_abs_diff": float(np.median(np.abs(saved - mine))),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--n", type=int, default=300, help="rows per side (members = non-members)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--smoke", action="store_true", help="10 rows per side, results under runs/ only")
    ap.add_argument("--e3-run", help="your encrypted (E3) run directory (default: the recorded one)")
    ap.add_argument("--baselines-run", help="your baselines run directory, with dev_E0/ and dev_E4/")
    a = ap.parse_args()
    s = SUITES[a.suite] = with_runs(SUITES[a.suite], a.e3_run, a.baselines_run)
    n = 10 if a.smoke else a.n
    cfg = yaml.safe_load(open(s["config"]))
    task = activate(cfg)
    from horizon.data.serialize import LABELS

    labels = list(LABELS)
    run = Path("runs") / f"{a.suite}_mia{'_smoke' if a.smoke else ''}"
    if run.exists():
        raise SystemExit(f"{run} exists; move it away first")
    sel, info = select(a.suite, n, a.seed)
    (run / "logs").mkdir(parents=True)
    sel_bytes = json.dumps(sel, indent=1).encode()
    (run / "selection.json").write_bytes(sel_bytes)
    print(f"selected {n} members + {n} non-members over {len(info['cells'])} cells → {run}/selection.json")

    files = {"selection.json": sel_bytes}
    for name, d in s["adapters"].items():
        files |= tree(Path(d), f"adapters/{name}")
    dv, ev = data_vols(task.name), eval_vol(task.name)
    mounts = {f"/vol/{dv[op]}": dv[op] for op in OPS} | {f"/vol/{ev}": ev}
    job = {
        "run_name": run.name,
        "module": "horizon.eval.membership",
        "args": ["score", "--config", "{CFG}", "--run", "{RUN}"],
        "files": files,
        "data_links": {op: f"/vol/{dv[op]}" for op in OPS} | {"test.jsonl": f"/vol/{ev}/test.jsonl"},
        "config_text": Path(s["config"]).read_text(),
        "signer": None,
        "require_device": None,
    }
    mc = cfg["modal"]
    fn = actor_glimmer.with_options(
        gpu=mc["gpu"],
        cpu=float(mc["cpu"]),
        memory=int(mc["memory_mb"]),
        timeout=int(mc["timeout_s"]),
        volumes={M_MODELS: models_vol} | {p: _vol(v) for p, v in mounts.items()},
        retries=0,
        scaledown_window=2,
    )
    with session():
        call = fn.spawn(job)
        with open(run / "logs" / "modal_calls.jsonl", "a") as f:
            f.write(json.dumps({"step": "mia_score", "call_id": call.object_id, "gpu": mc["gpu"]}) + "\n")
        print(f"  mia_score: → Modal ({mc['gpu']}) {call.object_id}", flush=True)
        res = call.get()
    usd = est_usd(mc["gpu"], float(mc["cpu"]), int(mc["memory_mb"]), res["wall_s"])
    with open(run / "logs" / "modal_usage.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "step": "mia_score",
                    "device": res["device"],
                    "gpu": mc["gpu"],
                    "wall_s": round(res["wall_s"], 1),
                    "est_usd": round(usd, 4),
                    "returncode": res["returncode"],
                }
            )
            + "\n"
        )
    (run / "logs" / "mia_score.log").write_text(res["log"])
    if res["returncode"] != 0:
        raise SystemExit(f"mia_score exit {res['returncode']}:\n{res['log'][-3000:]}")
    prefix = f"runs/{run.name}/"
    for rel, data in res["files"].items():
        if rel.startswith(prefix) and not rel[len(prefix) :].startswith(("adapters/", "selection.json")):
            (run / rel[len(prefix) :]).write_bytes(data)
    print(f"  mia_score: {res['wall_s']:.0f}s on {res['device']} (≈ ${usd:.3f})  {last_json(res['log'])}")

    summ = json.loads((run / "score_summary.json").read_text())
    scores = {k: np.load(run / f"scores_{k}.npy") for k in ["base", *s["adapters"]]}
    for k, v in scores.items():
        assert hashlib.sha256(v.tobytes()).hexdigest() == summ["sha256"][k], k
    out = {
        "suite": a.suite,
        "smoke": a.smoke,
        "framing": "empirical membership-inference measurement on the shared final model, not a privacy "
        "guarantee; experimenter-side (one container reads all three shards, like E4); loss on label "
        "tokens only, so text extraction does not apply",
        "attacks": {
            "loss": "true-label full-label log-prob under the target model",
            "reference": "same minus the base model's (E0, no adapter) score on the same row",
        },
        "selection": info | {"selection_sha256": hashlib.sha256(sel_bytes).hexdigest()},
        "targets": {"E3": s["adapters"]["E3"], "E4": s["adapters"]["E4"]},
        "results": analyze(sel, scores, labels, a.seed),
        "cross_check_vs_saved_test_scores": cross_check(sel, scores, a.suite),
        "score_sha256": summ["sha256"],
        "seconds": summ["seconds"],
        "cuda_peak_gb": summ.get("cuda_peak_gb"),
        "modal": {"device": res["device"], "wall_s": round(res["wall_s"], 1), "est_usd": round(usd, 4)},
    }
    dst = (
        run / "membership_inference.json"
        if a.smoke
        else Path(cfg["results_dir"]) / "membership_inference.json"
    )
    dst.write_text(json.dumps(out, indent=2))
    r = out["results"]
    for t in ("E3", "E4", "base_only"):
        print(
            t,
            {
                k: {"auc": round(v["auc"], 3), "tpr@1%": round(v["tpr_at_0.01_fpr"], 3)}
                for k, v in r[t].items()
                if isinstance(v, dict) and "auc" in v
            },
        )
    print("cross-check", json.dumps(out["cross_check_vs_saved_test_scores"]))
    print(f"→ {dst}")


if __name__ == "__main__":
    main()
