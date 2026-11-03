"""Modal jobs for the update-leakage study. Outputs go to runs/leakage/.

    python -m modal_app.leakage ablation [--smoke]   # one round of local training WITHOUT the restricted
                                                     # softmax, from the reference run's initial adapter
    python -m modal_app.leakage probe [--smoke]      # per-class gradient directions at each global adapter

ablation: the three operator steps run exactly as in an FL round (`horizon.fl.operator train
--insecure-reference`, each operator on its own shard volume) with
configs/5g-nidd-glimmer-no-restriction.yaml, which differs from the reference recipe in one flag.
The initial global adapter is copied from the reference run, so frozen A, seeds and data are
identical and the updates are directly comparable.

probe: the attacker's gradients (horizon/eval/probe_gradients.py) in one Glimmer container, on rows of the
evaluation set. Inputs and the large outputs (9 × 4.7M floats per adapter) travel through a dedicated
volume rather than the call payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import modal
import numpy as np
import yaml

from horizon.model.lora import read_b_file
from horizon.tasks import task_name
from modal_app.app import M_MODELS, PREFIX, actor_glimmer, eval_vol, models_vol
from modal_app.backend import ModalBackend, _vol, est_usd, last_json, session

REF_RUN = Path("runs/modal_glimmer_E2")
ROOT = Path("runs/leakage")
ABLATION_CFG = "configs/5g-nidd-glimmer-no-restriction.yaml"
PROBE_CFG = "configs/5g-nidd-glimmer.yaml"
LEAK_VOL = f"{PREFIX}-leakage"
M_LEAK = "/vol/leak"
OPS = ("op_a", "op_b", "op_c")


def ablation(smoke: bool) -> None:
    run = ROOT / f"glimmer_nofedrs_r1{'_smoke' if smoke else ''}"
    if run.exists():
        raise SystemExit(f"{run} exists; move it away first")
    src = REF_RUN / "bus" / "round_1" / "global"
    shutil.copytree(src, run / "bus" / "round_1" / "global")
    (run / "logs").mkdir(parents=True)
    shutil.copy2(ABLATION_CFG, run / "config.yaml")  # provenance: the exact recipe these updates used
    ops = ["op_b"] if smoke else list(OPS)
    dbg = ["--max-steps", "2"] if smoke else []
    steps = [
        (
            f"r1_{op}_train",
            ["horizon.fl.operator", "train", "--run", str(run), "--op", op, "--round", "1"]
            + ["--config", ABLATION_CFG, "--insecure-reference", *dbg],
        )
        for op in ops
    ]
    be = ModalBackend(ABLATION_CFG)
    with session():
        out = be.run_steps(run, steps)
    print(json.dumps({"run": str(run), "steps": out, "est_usd": round(be.spent, 4)}, indent=1))


def _upload(vol: modal.Volume, files: dict[str, Path]) -> None:
    with vol.batch_upload(force=True) as b:
        for remote, local in files.items():
            b.put_file(str(local), remote)


def _download(vol: modal.Volume, remote_dir: str, dst: Path, grads: list[str], wait_s: int = 300) -> dict:
    """Fetch what the container wrote and check it against its own summary (names + sha256).

    The volume commits on container shutdown, so a file can be listed before its bytes are final:
    re-fetch until the summary parses and every gradient file matches the hash the container recorded."""
    dst.mkdir(parents=True, exist_ok=True)
    t, err = time.time(), ""
    while time.time() - t < wait_s:
        try:
            files = {}
            for name in ["summary.json", "selection.json", *grads]:
                files[name] = b"".join(vol.read_file(f"{remote_dir}/{name}"))
            summ = json.loads(files["summary.json"])
            bad = [g for g in grads if hashlib.sha256(files[g]).hexdigest() != summ["grads_sha256"].get(g)]
            if hashlib.sha256(files["selection.json"]).hexdigest() != summ["selection_sha256"]:
                bad.append("selection.json")
            if not bad:
                for name, data in files.items():
                    (dst / name).write_bytes(data)
                return summ
            err = f"hash mismatch {bad}"
        except Exception as e:  # not committed yet (missing or partial file)
            err = f"{type(e).__name__}: {e}"
        time.sleep(10)
    raise TimeoutError(f"outputs not final on the volume after {wait_s}s ({err})")


def probe(smoke: bool, k: int, seed: int) -> None:
    tag = f"probe{'_smoke' if smoke else ''}_k{k}_s{seed}"
    out = ROOT / tag
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first")
    rounds = (
        [1] if smoke else sorted(int(p.name.split("_")[1]) for p in (REF_RUN / "insecure").glob("round_*"))
    )
    staging = ROOT / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    files = {}
    for r in rounds:  # the public global adapters the server itself published
        p = staging / f"B_r{r}.npy"
        np.save(p, read_b_file(REF_RUN / f"bus/round_{r}/global/adapter"))
        files[f"in/B_r{r}.npy"] = p
    adapter = REF_RUN / "bus/round_1/global/adapter"
    for f in adapter.iterdir():
        files[f"in/adapter/{f.name}"] = f
    info = json.loads((REF_RUN / "bus/round_1/global/round.json").read_text())

    vol = modal.Volume.from_name(LEAK_VOL, create_if_missing=True, version=2)
    _upload(vol, files)
    cfg = yaml.safe_load(open(PROBE_CFG))
    mc = cfg["modal"]
    ev = eval_vol(task_name(cfg))
    job = {
        "run_name": tag,
        "module": "horizon.eval.probe_gradients",
        "args": [
            "--config",
            "{CFG}",
            "--adapter",
            f"{M_LEAK}/in/adapter",
            "--lora-a-hash",
            info["lora_A_hash"],
        ]
        + ["--b", *[f"{M_LEAK}/in/B_r{r}.npy" for r in rounds]]
        + [
            "--test",
            f"/vol/{ev}/test.jsonl",
            "--k",
            str(k),
            "--seed",
            str(seed),
            "--out",
            f"{M_LEAK}/out/{tag}",
        ],
        "files": {},
        "data_links": {},
        "config_text": Path(PROBE_CFG).read_text(),
        "signer": None,
        "require_device": None,
    }
    fn = actor_glimmer.with_options(
        gpu=mc["gpu"],
        cpu=float(mc["cpu"]),
        memory=int(mc["memory_mb"]),
        timeout=int(mc["timeout_s"]),
        volumes={M_MODELS: models_vol, M_LEAK: vol, f"/vol/{ev}": _vol(ev)},
        retries=0,
        scaledown_window=2,
    )
    out.mkdir(parents=True)
    with session():
        call = fn.spawn(job)
        (out / "modal_calls.jsonl").write_text(
            json.dumps({"call_id": call.object_id, "gpu": mc["gpu"]}) + "\n"
        )
        print(f"probe: → Modal ({mc['gpu']}) {call.object_id}", flush=True)
        res = call.get()
    usd = est_usd(mc["gpu"], float(mc["cpu"]), int(mc["memory_mb"]), res["wall_s"])
    (out / "probe.log").write_text(res["log"])
    usage = {
        "device": res["device"],
        "gpu": mc["gpu"],
        "wall_s": round(res["wall_s"], 1),
        "est_usd": round(usd, 4),
    }
    (out / "modal_usage.json").write_text(json.dumps(usage | {"returncode": res["returncode"]}))
    if res["returncode"] != 0:
        raise SystemExit(f"probe exit {res['returncode']}:\n{res['log'][-3000:]}")
    summ = _download(vol, f"out/{tag}", out, [f"grads_B_r{r}.npy" for r in rounds])
    print(f"probe: verified {len(summ['grads_sha256'])} gradient file(s) against the container's sha256")
    print(f"probe: {res['wall_s']:.0f}s on {res['device']} (≈ ${usd:.3f}) {last_json(res['log'])}")
    print(f"→ {out}")


def main() -> None:
    global REF_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["ablation", "probe"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=32, help="probe: auxiliary rows per class")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--ref-run",
        type=Path,
        default=REF_RUN,
        help=f"plaintext-reference run to attack (default: {REF_RUN})",
    )
    a = ap.parse_args()
    REF_RUN = a.ref_run
    if a.job == "ablation":
        ablation(a.smoke)
    else:
        probe(a.smoke, 2 if a.smoke else a.k, a.seed)


if __name__ == "__main__":
    main()
