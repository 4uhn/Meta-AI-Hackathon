"""Equivalence of E2 (plaintext FedAvg reference) and E3 (threshold-encrypted FedAvg).

    python -m horizon.eval.equivalence --e2 RESULTS/E2.json --e3 RESULTS/E3.json
    python -m horizon.eval.equivalence --e2 … --e3 … --replay --config CFG [--backend modal]

The gate uses the replay: E2's plaintext round updates go through the real threshold-CKKS path (fresh
ceremony → per-operator encrypt → aggregate → t-of-N decrypt → FedAvg) and are compared with E2's
plaintext FedAvg round by round, which isolates the encryption error. The final encrypted adapter is
then evaluated and its predictions compared with E2's. Two GPU runs of the same training can differ by
kernel nondeterminism across hosts, so an end-to-end comparison alone would measure GPU reproducibility
rather than the crypto.

The end-to-end comparison is still reported: in round 1 the only difference is CKKS error
(`agg_max_abs_err`), and FedAvg gives ΔB ≈ agg error / Σw; later rounds also include how training
propagates that ΔB. Writes equivalence.json next to --e3.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from horizon.fl.rundir import (
    OPERATORS,
    adapter_dir,
    configure,
    default_active,
    eval_dir,
    insecure_update,
    write_json,
)
from horizon.he_bridge import HorizonHE, read_f64
from horizon.model.lora import fedavg_from_sum, read_b_file, server_update, write_b_file

GATE_MAX_ABS_DB = 1e-5
GATE_AGREEMENT = 0.995


def compare(e2: dict, e3: dict) -> dict:
    r2, r3 = Path(e2["run"]), Path(e3["run"])
    if (e2["layout_hash"], e2["lora_A_hash"]) != (e3["layout_hash"], e3["lora_A_hash"]):
        raise ValueError("E2 and E3 used different adapters/layouts; not comparable")
    n = len(e3["rounds"])
    per_round = []
    for r in range(1, n + 1):
        b2, b3 = read_b_file(adapter_dir(r2, r + 1)), read_b_file(adapter_dir(r3, r + 1))
        d = np.abs(b2 - b3)
        row = {
            "round": r,
            "max_abs_dB": float(d.max()),
            "mean_abs_dB": float(d.mean()),
            "max_abs_B": float(np.abs(b2).max()),
        }
        dec = r3 / "bus" / f"round_{r}" / "decrypt" / "agg.f64"
        ins = [insecure_update(r2, r, op) for op in OPERATORS]
        if dec.is_file() and all(p.is_file() for p in ins):
            plain, enc = sum(read_f64(p) for p in ins), read_f64(dec)
            row |= {"agg_max_abs_err": float(np.abs(plain - enc).max()), "weight_sum": float(plain[-1])}
        per_round.append(row)
    p2 = np.load(eval_dir(r2) / f"round_{n + 1}_scores.npy").argmax(1)
    p3 = np.load(eval_dir(r3) / f"round_{n + 1}_scores.npy").argmax(1)
    agree = float((p2 == p3).mean())
    worst = max(x["max_abs_dB"] for x in per_round)
    same_training = {  # round 1 starts from the same adapter in both runs
        op: e2["rounds"][0]["manifests"][op]["train_loss_final"]
        == e3["rounds"][0]["manifests"][op]["train_loss_final"]
        for op in OPERATORS
    }
    return {
        "e2_run": str(r2),
        "e3_run": str(r3),
        "rounds": n,
        "per_round": per_round,
        "max_abs_dB": worst,
        "prediction_agreement": agree,
        "n_test": int(p2.size),
        "macro_f1": {"E2": e2["metrics"]["macro_f1"], "E3": e3["metrics"]["macro_f1"]},
        "round1_identical_training": same_training,
        "gate": {
            "max_abs_dB_le_1e-5": worst <= GATE_MAX_ABS_DB,
            "agreement_ge_99.5": agree >= GATE_AGREEMENT,
        },
    }


def replay(e2: dict, out_run: Path) -> dict:
    """E2's plaintext updates through the real threshold-CKKS path; per-round encryption error."""
    from horizon.fl.run import ceremony

    r2, n = Path(e2["run"]), len(e2["rounds"])
    if out_run.exists():
        shutil.rmtree(out_run)
    out_run.mkdir(parents=True)
    cfg = yaml.safe_load((r2 / "config.yaml").read_text())
    configure(cfg)
    active = default_active()
    cer = ceremony(out_run)
    he = HorizonHE(out_run)
    lr = float((cfg.get("fl") or {}).get("server_lr", 1.0))
    rows = []
    for r in range(1, n + 1):
        ups = {op: read_f64(insecure_update(r2, r, op)) for op in OPERATORS}
        for op, v in ups.items():  # each operator encrypts (and signs) its own update
            he.encrypt(op, r, v)
        he.aggregate(r)
        he.request(r, active)
        for op in active:
            he.dec_share(op, r)
        he.dec_combine(r)
        enc, plain = he.aggregate_values(r), sum(ups.values())
        b_prev = read_b_file(adapter_dir(r2, r))  # the public global adapter both paths start from
        b_enc = server_update(b_prev, fedavg_from_sum(enc), lr)
        b_plain = server_update(b_prev, fedavg_from_sum(plain), lr)
        d = np.abs(b_enc - b_plain)
        rows.append(
            {
                "round": r,
                "agg_max_abs_err": float(np.abs(enc - plain).max()),
                "weight_sum": float(plain[-1]),
                "max_abs_dB": float(d.max()),
                "mean_abs_dB": float(d.mean()),
                "max_abs_B": float(np.abs(b_plain).max()),
                # sanity: recomputed plaintext FedAvg vs the adapter E2 published (fp32 storage)
                "e2_published_vs_fedavg": float(np.abs(read_b_file(adapter_dir(r2, r + 1)) - b_plain).max()),
            }
        )
    write_b_file(adapter_dir(r2, n + 1), adapter_dir(out_run, n + 1), b_enc)
    return {"run": str(out_run), "rounds": n, "server_lr": lr, "ceremony_ms": cer["ms"], "per_round": rows}


def _eval_adapters(
    adapters: dict[str, Path], work: Path, cfg_path: str, backend: str
) -> dict[str, np.ndarray]:
    """Evaluate each adapter on the test set (same evaluator config and GPU model); return scores."""
    from horizon.fl.run import LocalBackend

    out = {}
    for tag, src in adapters.items():
        run = work / tag
        if run.exists():
            shutil.rmtree(run)
        shutil.copytree(src, adapter_dir(run, 1))
    steps = [
        (
            f"eval_{tag}",
            ["horizon.fl.evaluate", "--run", str(work / tag), "--round", "1", "--config", cfg_path],
        )
        for tag in adapters
    ]
    if backend == "modal":  # one backend, one step at a time: both adapters on the same GPU model
        from modal_app.backend import ModalBackend, session

        be = ModalBackend(cfg_path)
        with session():
            for st in steps:
                be.run_steps(work, [st])
    else:
        LocalBackend().run_steps(work, steps)
    for tag in adapters:
        out[tag] = np.load(eval_dir(work / tag) / "round_1_scores.npy")
    return out


def replay_agreement(e2: dict, rep: dict, cfg_path: str, backend: str) -> dict:
    """Final encrypted vs plaintext adapter, both evaluated fresh with `cfg_path`: prediction agreement
    and max |Δscore|."""
    r2, run, n = Path(e2["run"]), Path(rep["run"]), rep["rounds"]
    work = run.parent / f"{run.name}_eval_{Path(cfg_path).stem}"
    sc = _eval_adapters(
        {"plain": adapter_dir(r2, n + 1), "encrypted": adapter_dir(run, n + 1)}, work, cfg_path, backend
    )
    a, b = sc["plain"], sc["encrypted"]
    return {
        "config": cfg_path,
        "prediction_agreement": float((a.argmax(1) == b.argmax(1)).mean()),
        "max_abs_dscore": float(np.abs(a - b).max()),
        "n_test": int(a.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", default="results/5g-nidd-llama/E2.json")
    ap.add_argument("--e3", default="results/5g-nidd-llama/E3.json")
    ap.add_argument("--out", default=None, help="default: equivalence.json next to --e3")
    ap.add_argument(
        "--replay", action="store_true", help="replay E2's updates through the real threshold path (see top)"
    )
    ap.add_argument("--config", help="--replay: run config (bf16 agreement)")
    ap.add_argument("--agreement-config", help="--replay: gate agreement with this evaluator config (fp32)")
    ap.add_argument("--backend", choices=["local", "modal"], default="local")
    a = ap.parse_args()
    if a.replay and not a.config:
        ap.error("--replay needs --config")
    e2 = json.loads(Path(a.e2).read_text())
    res = {"end_to_end": compare(e2, json.loads(Path(a.e3).read_text()))}
    if a.replay:
        rep = replay(e2, Path(e2["run"] + "_replay"))
        rep["agreement"] = {Path(a.config).stem: replay_agreement(e2, rep, a.config, a.backend)}
        gate_cfg = a.agreement_config or a.config
        if a.agreement_config:
            rep["agreement"][Path(gate_cfg).stem] = replay_agreement(e2, rep, gate_cfg, a.backend)
        worst = max(x["max_abs_dB"] for x in rep["per_round"])
        rep["max_abs_dB"] = worst
        rep["prediction_agreement"] = rep["agreement"][Path(gate_cfg).stem]["prediction_agreement"]
        rep["gate_agreement_config"] = gate_cfg
        rep["gate"] = {
            "max_abs_dB_le_1e-5": worst <= GATE_MAX_ABS_DB,
            "agreement_ge_99.5": rep["prediction_agreement"] >= GATE_AGREEMENT,
        }
        res["replay"] = rep
        res["gate_G1b"] = rep["gate"]
    write_json(Path(a.out) if a.out else Path(a.e3).with_name("equivalence.json"), res)
    show = {
        k: res["end_to_end"][k] for k in ("max_abs_dB", "prediction_agreement", "round1_identical_training")
    }
    print(
        json.dumps(
            {"end_to_end": show}
            | (
                {"replay": {k: res["replay"][k] for k in ("max_abs_dB", "prediction_agreement", "gate")}}
                if a.replay
                else {}
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
