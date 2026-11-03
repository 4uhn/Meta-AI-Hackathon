"""The aggregator. It never opens a parties/*/secret directory and never sees an individual update:
on the secure path it only adds ciphertexts and asks t operators for decryption shares.

    python -m horizon.fl.aggregator init --run DIR --config CFG     # once: seeded initial adapter

Only `init` loads a model (seeded shared lora_A, B = 0). Closing a round works on files: the decrypted
aggregate [Σ w_i B_i, Σ w_i] becomes the new global B, written into a copy of the adapter file.
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
from horizon.fl.rundir import (
    OPERATORS,
    adapter_dir,
    configure,
    global_dir,
    insecure_update,
    min_contributors,
    round_info,
    submission_dir,
    write_json,
)
from horizon.he_bridge import HEError, HorizonHE, read_f64
from horizon.model.lora import fedavg_from_sum, file_layout_hash, read_b_file, server_update, write_b_file

ACTOR = "aggregator"


def init(run: Path, cfg: dict) -> dict:
    from horizon.model.lora import a_hash, create_initial_adapter, layout_hash, n_b_values
    from horizon.model.train import load_base

    base, _, rev = load_base(cfg)
    model = create_initial_adapter(base, cfg["lora"], cfg["train"]["adapter_seed"], adapter_dir(run, 1))
    info = {
        "round": 1,
        "model": cfg["model"]["id"],
        "base_revision": rev,
        "layout_hash": layout_hash(model),
        "lora_A_hash": a_hash(model),
        "n_values": n_b_values(model) + 1,  # + the weight slot
        "min_submissions": min_contributors(),
    }
    write_json(global_dir(run, 1) / "round.json", info)
    emit(run, ACTOR, "global_published", 1, initial=True, layout_hash=info["layout_hash"])
    return info


def publish(run: Path, round_: int, b_new: np.ndarray | None) -> dict:
    """Publish round_+1's global adapter: the new B, or (b_new None) round_'s adapter carried over."""
    info = round_info(run, round_)
    src, dst = adapter_dir(run, round_), adapter_dir(run, round_ + 1)
    if b_new is None:
        shutil.copytree(src, dst)
    else:
        if b_new.size != info["n_values"] - 1:
            raise ValueError(f"aggregate has {b_new.size} B values, layout needs {info['n_values'] - 1}")
        write_b_file(src, dst, b_new)
    if file_layout_hash(dst) != info["layout_hash"]:
        raise AssertionError("published adapter's layout differs from the announced one")
    nxt = info | {"round": round_ + 1, "carried_over": b_new is None}
    write_json(global_dir(run, round_ + 1) / "round.json", nxt)
    emit(run, ACTOR, "global_published", round_ + 1, carried_over=b_new is None)
    return nxt


def next_global_b(run: Path, round_: int, agg: np.ndarray) -> np.ndarray:
    """FedAvg of the decrypted aggregate, then the server step η_s from the run's config (fl.server_lr)."""
    cfg = (
        yaml.safe_load((Path(run) / "config.yaml").read_text())
        if (Path(run) / "config.yaml").is_file()
        else {}
    )
    lr = float((cfg.get("fl") or {}).get("server_lr", 1.0))
    return server_update(read_b_file(adapter_dir(run, round_)), fedavg_from_sum(agg), lr)


def _operator_step(run: Path, op: str, round_: int) -> None:
    """Ask operator `op` for its decryption share (its own process, its own key)."""
    cmd = [sys.executable, "-m", "horizon.fl.operator", "dec-share", "--run", str(run), "--op", op]
    p = subprocess.run([*cmd, "--round", str(round_)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{op} dec-share failed: {p.stderr.strip()[-800:]}")


def close_round_encrypted(run: Path, round_: int, active: list[str]) -> dict:
    """Aggregate → request → shares → combine → FedAvg → publish.

    If the crypto layer refuses the aggregate (e.g. fewer than t + 1 contributors) the round is
    skipped and the adapter carried over."""
    he = HorizonHE(run)
    out: dict = {"round": round_, "active": active}
    t = time.time()
    try:
        agg = he.aggregate(round_)
    except HEError as e:
        out |= {"skipped": True, "reason": str(e)}
        emit(run, ACTOR, "round_skipped", round_, reason=str(e))
        publish(run, round_, None)
        return out
    out |= {"contributors": agg["contributors"], "rejected": agg["rejected"], "aggregate_ms": agg["sum_ms"]}
    out["agg_bytes"] = agg["bytes"]
    for op, why in agg["rejected"].items():
        emit(run, ACTOR, "submission_rejected", round_, party=op, reason=why)
    emit(run, ACTOR, "aggregated", round_, n_subs=len(agg["contributors"]), ms=agg["sum_ms"])

    he.request(round_, active)
    emit(run, ACTOR, "decrypt_requested", round_, active=active)
    out["dec_share_ms"] = {}
    for op in active:
        t1 = time.time()
        _operator_step(run, op, round_)
        out["dec_share_ms"][op] = int(1000 * (time.time() - t1))  # incl. process start and checks
    comb = he.dec_combine(round_)
    out["combine_ms"] = comb["combine_ms"]
    emit(run, ACTOR, "decrypted", round_, ms=comb["combine_ms"], weight_slot=comb["weight_slot"])
    solo = he.dec_combine(round_, solo=active[0])  # demonstrates that one share alone yields noise
    out["solo_rel_err"] = solo["rel_err_vs_true"]
    emit(run, ACTOR, "solo_decrypt_attempt", round_, party=active[0], rel_err=solo["rel_err_vs_true"])
    if len(active) > 2:  # and so do t − 1 shares
        part = active[: len(active) - 1]
        pr = he.dec_combine(round_, solo=",".join(part))
        out["partial_rel_err"] = pr["rel_err_vs_true"]
        emit(run, ACTOR, "partial_decrypt_attempt", round_, parties=part, rel_err=pr["rel_err_vs_true"])

    agg_vals = he.aggregate_values(round_)
    out["weight_sum"] = float(agg_vals[-1])
    publish(run, round_, next_global_b(run, round_, agg_vals))
    out["close_ms"] = int(1000 * (time.time() - t))
    return out


def close_round_plaintext(run: Path, round_: int) -> dict:
    """Plaintext reference: the same FedAvg on plaintext updates (`--insecure-reference` runs only).

    Applies the same contributor rule so it skips the same rounds as the encrypted path."""
    present = [op for op in OPERATORS if insecure_update(run, round_, op).is_file()]
    out: dict = {"round": round_, "contributors": present, "mode": "insecure-reference"}
    if len(present) < (need := min_contributors()):
        out |= {"skipped": True, "reason": f"{len(present)} submissions, need {need}"}
        emit(run, ACTOR, "round_skipped", round_, reason=out["reason"])
        publish(run, round_, None)
        return out
    agg = sum(read_f64(insecure_update(run, round_, op)) for op in present)
    out["weight_sum"] = float(agg[-1])
    emit(run, ACTOR, "aggregated", round_, n_subs=len(present), mode="insecure-reference")
    publish(run, round_, next_global_b(run, round_, agg))
    return out


def submissions(run: Path, round_: int) -> dict[str, dict]:
    """Manifests on the bus for this round (public metadata only)."""
    out = {}
    for op in OPERATORS:
        m = submission_dir(run, round_, op) / "manifest.json"
        if m.is_file():
            out[op] = json.loads(m.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["init"])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    configure(cfg)
    print(json.dumps(init(a.run, cfg)))


if __name__ == "__main__":
    main()
