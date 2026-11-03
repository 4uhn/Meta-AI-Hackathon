"""Federated run orchestrator: scheduler + aggregator for E2 (plaintext), E3 (encrypted) and E5 (async).

    python -m horizon.fl.run --config configs/5g-nidd-tiny.yaml --mode encrypted            # E3
    python -m horizon.fl.run --config configs/5g-nidd-tiny.yaml --mode plaintext            # E2 (reference)
    python -m horizon.fl.run --config configs/5g-nidd-tiny.yaml --mode encrypted --scenario e5

This process never loads a model: every model step is its own subprocess, one at a time. Operators are
separate processes that read only their own shard and key directory. `--backend modal` runs model steps
in isolated Modal containers instead; crypto, aggregation and decryption shares stay local.

E5: op_c skips round 2, so the 2-contributor aggregate is refused and the global adapter carries over;
op_b is offline when round 3 is decrypted, so {op_a, op_c} decrypt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from horizon.events import emit
from horizon.fl.aggregator import ACTOR, close_round_encrypted, close_round_plaintext, submissions
from horizon.fl.rundir import (
    OPERATORS,
    configure,
    default_active,
    eval_dir,
    round_info,
    threshold,
    write_json,
)
from horizon.he_bridge import HorizonHE
from horizon.tasks import activate

SCENARIOS = {
    "none": {"skip": {}, "offline_at_decrypt": {}},
    "e5": {"skip": {2: ["op_c"]}, "offline_at_decrypt": {3: ["op_b"]}},
}


def step(run: Path, name: str, args: list[str]) -> dict:
    """Run one actor step as its own Python process; its output goes to logs/<name>.log."""
    log = run / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, "w") as f:
        p = subprocess.run([sys.executable, "-m", *args], stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {p.returncode}):\n{log.read_text()[-2000:]}")
    lines = [x for x in log.read_text().splitlines() if x.startswith("{")]
    print(f"  {name}: {time.time() - t:.0f}s", flush=True)
    return json.loads(lines[-1]) if lines else {}


class LocalBackend:
    """Model steps as local processes, one at a time (one model in memory)."""

    spent = 0.0

    def run_steps(self, run: Path, steps: list[tuple[str, list[str]]]) -> list[dict]:
        return [step(run, name, args) for name, args in steps]


def evaluate_global(be, run: Path, round_: int, cfg_path: str, cfg: dict, evl: list[str]) -> dict:
    """Score the global adapter of `round_` on the test set or, with eval.split = val, by federated
    evaluation (each operator on its own val split, confusion counts summed here)."""
    base = ["horizon.fl.evaluate", "--run", str(run), "--round", str(round_), "--config", cfg_path]
    if cfg["eval"].get("split", "test") == "test":
        (res,) = be.run_steps(run, [(f"eval_round_{round_}", [*base, *evl])])
        return res
    from horizon.model.train import metrics_from_confusion

    be.run_steps(
        run, [(f"val_round_{round_}_{op}", [*base, "--split", "val", "--op", op]) for op in OPERATORS]
    )
    parts = [json.loads((eval_dir(run) / f"round_{round_}_val_{op}.json").read_text()) for op in OPERATORS]
    conf = sum(np.array(p["confusion"]) for p in parts)
    m = metrics_from_confusion(conf)
    out = {"round": round_, "split": "val", "n_test": int(conf.sum()), "pred_sha256": None, "metrics": m}
    out["confusion"] = conf.tolist()
    write_json(eval_dir(run) / f"round_{round_}.json", out)
    return {"round": round_, "split": "val", "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


def ceremony(run: Path) -> dict:
    """Key ceremony: one horizon-he process per (party, step); no trusted dealer."""
    he = HorizonHE(run)
    t = time.time()
    he.init(list(OPERATORS), threshold())
    for p in OPERATORS:
        he.keygen(p)
    emit(run, ACTOR, "keygen_done", parties=list(OPERATORS))
    for p in OPERATORS:
        he.pk_share(p)
    for p in OPERATORS:
        he.shamir_deal(p)
    for p in OPERATORS:
        he.shamir_agg(p)
    emit(run, ACTOR, "shamir_done", threshold=threshold())
    he.pk_combine()
    emit(run, ACTOR, "pk_published", pk_bytes=(run / "public" / "pk.bin").stat().st_size)
    return {"ms": int(1000 * (time.time() - t))}


def run_experiment(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(open(args.config))
    activate(cfg)
    configure(cfg)
    tc = cfg["train"]
    rounds = args.rounds or tc["rounds"]
    sc = SCENARIOS[args.scenario]
    if args.scenario == "e5" and (args.mode != "encrypted" or rounds < 3):
        raise SystemExit("--scenario e5 needs --mode encrypted and ≥ 3 rounds")
    exp = args.exp or {"plaintext": "E2", "encrypted": "E3"}[args.mode]
    if args.scenario == "e5" and not args.exp:
        exp = "E5"
    smoke = bool(args.max_steps or args.limit_test)
    out_dir = Path(args.out or ("runs/smoke" if smoke else cfg.get("results_dir", "runs/results")))

    run = Path(args.run_root) / f"{Path(args.config).stem}_{exp}{'_smoke' if smoke else ''}"
    if run.exists():
        if not args.fresh:
            raise SystemExit(f"{run} exists; pass --fresh to replace it")
        shutil.rmtree(run)
    run.mkdir(parents=True)
    shutil.copy(args.config, run / "config.yaml")
    t0 = time.time()
    emit(run, ACTOR, "run_started", exp=exp, mode=args.mode, scenario=args.scenario, rounds=rounds)
    if args.mode == "plaintext":
        emit(run, ACTOR, "insecure_reference_run", warning="operators emit PLAINTEXT updates (E2 only)")

    backend = getattr(args, "backend", "local")
    if backend == "modal":
        from modal_app.backend import ModalBackend

        be = ModalBackend(args.config)
    else:
        be = LocalBackend()
    result: dict = {
        "exp": exp,
        "mode": args.mode,
        "scenario": args.scenario,
        "run": str(run),
        "backend": backend,
    }
    if args.mode == "encrypted":
        result["ceremony"] = ceremony(run)
    (init,) = be.run_steps(
        run,
        [("aggregator_init", ["horizon.fl.aggregator", "init", "--run", str(run), "--config", args.config])],
    )
    result |= {k: init[k] for k in ("model", "base_revision", "layout_hash", "lora_A_hash", "n_values")}

    dbg = ["--max-steps", str(args.max_steps)] if args.max_steps else []
    evl = ["--limit-test", str(args.limit_test)] if args.limit_test else []
    every = cfg["eval"].get("every_round", True)
    every_n = int(cfg["eval"].get("every_n_rounds", 1))  # with every_round: evaluate rounds n, 2n, …
    result["rounds"] = []
    for r in range(1, rounds + 1):
        print(f"round {r}/{rounds}", flush=True)
        emit(run, ACTOR, "round_started", r, min_submissions=round_info(run, r)["min_submissions"])
        skip = sc["skip"].get(r, [])
        train_steps = []
        for op in OPERATORS:
            if op in skip:
                emit(run, op, "offline", r, phase="training")
                continue
            flags = ["--insecure-reference"] if args.mode == "plaintext" else []
            train_steps.append(
                (
                    f"r{r}_{op}_train",
                    [
                        "horizon.fl.operator",
                        "train",
                        "--run",
                        str(run),
                        "--op",
                        op,
                        "--round",
                        str(r),
                        "--config",
                        args.config,
                        *flags,
                        *dbg,
                    ],
                )
            )
        be.run_steps(run, train_steps)
        if args.mode == "encrypted":
            offline = sc["offline_at_decrypt"].get(r, [])
            for op in offline:
                emit(run, op, "offline", r, phase="decryption")
            online = [op for op in OPERATORS if op not in offline]
            active = online[: threshold()] if offline else default_active()
            rec = close_round_encrypted(run, r, active)
        else:
            rec = close_round_plaintext(run, r)
        rec["skipped_training"] = skip
        rec["manifests"] = submissions(run, r)
        if (every and r % every_n == 0) or r == rounds:
            # eval.round_scoring scores intermediate rounds with the fast path; eval.compare_fast_final
            # also scores the final model with it, to record agreement.
            ec, extra = cfg["eval"], []
            if r < rounds and ec.get("round_scoring"):
                extra = ["--scoring", ec["round_scoring"]]
            elif r == rounds and ec.get("compare_fast_final"):
                extra = ["--compare-fast"]
            rec["eval"] = evaluate_global(be, run, r + 1, args.config, cfg, [*evl, *extra])
        result["rounds"].append(rec)

    final = json.loads((eval_dir(run) / f"round_{rounds + 1}.json").read_text())
    result["final"] = final
    result["metrics"] = final["metrics"]
    result["pred_sha256"] = final["pred_sha256"]
    result["crypto"] = crypto_summary(result["rounds"]) if args.mode == "encrypted" else None
    result["total_s"] = round(time.time() - t0, 1)
    result["smoke"] = smoke
    if backend == "modal":
        result["modal_est_usd"] = round(be.spent, 4)
        result["modal_gpu"] = be.gpu
        result["modal_device"] = be.require_device
        result["device_mismatches"] = be.device_mismatches
    emit(run, ACTOR, "run_finished", exp=exp, total_s=result["total_s"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{exp}.json", result)
    return result


def crypto_summary(rounds: list[dict]) -> dict:
    """Per-round crypto cost vs local training time."""
    rows = []
    for r in rounds:
        ms = r["manifests"]
        enc = {op: m["encrypt_ms"] for op, m in ms.items() if "encrypt_ms" in m}
        train = {op: m["train_ms"] for op, m in ms.items()}
        row = {"round": r["round"], "skipped": r.get("skipped", False), "encrypt_ms": enc, "train_ms": train}
        if not r.get("skipped"):
            crypto_ms = (
                sum(enc.values()) + r["aggregate_ms"] + sum(r["dec_share_ms"].values()) + r["combine_ms"]
            )
            row |= {k: r[k] for k in ("aggregate_ms", "dec_share_ms", "combine_ms", "solo_rel_err", "active")}
            row["crypto_ms"] = crypto_ms
            row["crypto_over_train"] = crypto_ms / max(sum(train.values()), 1)
        rows.append(row)
    m0 = next(iter(rounds[0]["manifests"].values()))
    return {
        "bytes_per_submission": m0["bytes"],
        "expansion_vs_fp16": m0["expansion_vs_fp16"],
        "n_values": m0["n_values"],
        "n_ciphertexts": m0["n_ciphertexts"],
        "per_round": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/5g-nidd-tiny.yaml")
    ap.add_argument("--mode", choices=["encrypted", "plaintext"], default="encrypted")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="none")
    ap.add_argument("--exp", help="result name (default E3 / E2 / E5)")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--out", help="results directory (default: the config's results_dir, else runs/results)")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--fresh", action="store_true", help="replace an existing run directory")
    ap.add_argument("--max-steps", type=int, default=0, help="debug (smoke): cap each local training")
    ap.add_argument("--limit-test", type=int, default=0, help="debug (smoke): evaluate on N test rows")
    ap.add_argument("--backend", choices=["local", "modal"], default="local")
    args = ap.parse_args()
    res = None
    if args.backend == "modal":
        from modal_app.backend import session

        with session():
            res = run_experiment(args)
    else:
        res = run_experiment(args)
    if res is None:  # interrupted inside the Modal session (it stops the app and its containers)
        raise SystemExit("interrupted")
    m = res["metrics"]
    print(
        json.dumps(
            {
                "exp": res["exp"],
                "acc": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "recall": {k: round(v["recall"], 3) for k, v in m["per_class"].items()},
                "total_s": res["total_s"],
                "modal_est_usd": res.get("modal_est_usd"),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
