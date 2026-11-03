"""Local side of the Modal backend: turn a pipeline step into a remote container call.

`run.py --backend modal` hands the same (name, module args) steps it would run as local processes to
`ModalBackend.run_steps`. For each step this module decides:
- which files the container gets (public keys + the round's global adapter; never a secret/ dir),
- what it mounts (operator X: only `<prefix>-data-op_x`; evaluator: only `<prefix>-eval`; + weights),
- for an encrypting operator, its Ed25519 signing key as a per-call Modal Secret,
then runs the steps in parallel, writes the returned files into the local run dir and logs the
GPU time and an estimated cost per container (`logs/modal_usage.jsonl`).

`dec-share` never goes through here: threshold key shares never leave the local machine.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import modal
import yaml

from horizon.tasks import task_name
from modal_app.app import M_MODELS, actor, actor_glimmer, app, data_vols, eval_vol, models_vol

# GPU $/h, Modal list prices (modal.com/pricing, Oct 2026); CPU and RAM are billed on top
GPU_USD_H = {
    "T4": 0.59,
    "L4": 0.80,
    "A10": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}
CPU_CORE_USD_H, MEM_GIB_USD_H = 0.0473, 0.008


# Measured cost per config (from `modal billing report`), shown before a GPU launch.
MEASURED_COSTS = Path(__file__).resolve().parent.parent / "results" / "modal_costs.json"


def confirm_gpu(cfg_path: str, gpu: str) -> None:
    """Ask before starting paid GPU containers. HORIZON_YES=1 answers yes (for scripts and detached runs)."""
    if os.environ.get("HORIZON_YES") == "1":
        return
    costs = json.loads(MEASURED_COSTS.read_text()) if MEASURED_COSTS.is_file() else {}
    lines = [
        f"This starts {gpu} GPU containers on your Modal account "
        f"(list price ${GPU_USD_H.get(gpu, '?')}/h per container, plus CPU and memory)."
    ]
    for exp, m in costs.get("configs", {}).get(Path(cfg_path).name, {}).items():
        lines.append(f"  measured for this config: {exp} ${m['usd']:.2f}, {m['minutes']} min")
    print("\n".join(lines), flush=True)
    if not sys.stdin.isatty():
        raise SystemExit("not confirmed: no terminal to ask. Set HORIZON_YES=1 to confirm.")
    if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SystemExit("cancelled")


def est_usd(gpu: str | None, cpu: float, memory_mb: int, seconds: float) -> float:
    per_h = GPU_USD_H.get(gpu or "", 0.0) + cpu * CPU_CORE_USD_H + memory_mb / 1024 * MEM_GIB_USD_H
    return per_h * seconds / 3600


def _vol(name: str) -> modal.Volume:
    return modal.Volume.from_name(name, create_if_missing=False, version=2)


def _flag(args: list[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args else None


def _tree(base: Path, rel_to: Path) -> dict[str, bytes]:
    return {str(p.relative_to(rel_to)): p.read_bytes() for p in sorted(base.rglob("*")) if p.is_file()}


def plan(run: Path, args: list[str], cfg_path: str) -> dict:
    """Describe the container for one step: job dict (minus config) + mounts + signer.

    Pure (reads only public run files and, for an encrypting operator, that operator's signing key),
    so it is unit-tested without Modal.
    """
    module, rest = args[0], [a for a in args[1:]]
    cfg = yaml.safe_load(open(cfg_path)) if Path(cfg_path).is_file() else {}
    task, ops = task_name(cfg), (cfg.get("fl") or {}).get("operators")
    DATA_VOL, EVAL_VOL = data_vols(task, ops), eval_vol(task)  # noqa: N806 (this run's volume names)
    rest = ["{RUN}" if a == str(run) else "{CFG}" if a == cfg_path else a for a in rest]
    job = {"run_name": run.name, "module": module, "args": rest, "files": {}, "data_links": {}}
    mounts: dict[str, str] = {}
    signer = None
    if module == "horizon.fl.aggregator":
        pass  # init: weights only
    elif module == "horizon.fl.operator" and args[1] == "train":
        op, r = _flag(args, "--op"), int(_flag(args, "--round"))
        g = run / "bus" / f"round_{r}" / "global"
        job["files"] |= _tree(g, run)
        if "--insecure-reference" not in args:  # secure path: public keys + this operator's signing key
            job["files"] |= _tree(run / "public", run)
            signer = op
        mounts[f"/vol/{DATA_VOL[op]}"] = DATA_VOL[op]
        job["data_links"][op] = f"/vol/{DATA_VOL[op]}"
    elif module == "horizon.fl.evaluate":
        r = int(_flag(args, "--round"))
        job["files"] |= _tree(run / "bus" / f"round_{r}" / "global", run)
        op = _flag(args, "--op")
        if op:  # federated validation: runs as that operator, on its own shard volume only
            mounts[f"/vol/{DATA_VOL[op]}"] = DATA_VOL[op]
            job["data_links"][op] = f"/vol/{DATA_VOL[op]}"
        else:
            mounts[f"/vol/{EVAL_VOL}"] = EVAL_VOL
            job["data_links"]["test.jsonl"] = f"/vol/{EVAL_VOL}/test.jsonl"
    elif module == "horizon.fl.baseline":
        exp = _flag(args, "--exp")
        local = {f"E1{op[-1]}": [op] for op in DATA_VOL}  # E1x: operator op_x alone
        ops = (local | {"E4": list(DATA_VOL), "E0": []})[exp]
        for op in ops:  # a local baseline sees only its own shard; E4 (the pooled ceiling) sees all
            mounts[f"/vol/{DATA_VOL[op]}"] = DATA_VOL[op]
            job["data_links"][op] = f"/vol/{DATA_VOL[op]}"
        mounts[f"/vol/{EVAL_VOL}"] = EVAL_VOL
        job["data_links"]["test.jsonl"] = f"/vol/{EVAL_VOL}/test.jsonl"
    else:
        raise ValueError(f"no remote plan for {args[:2]}")
    if any("/parties/" in f"/{k}" for k in job["files"]):
        raise AssertionError("refusing to ship an operator's private directory")
    return {"job": job, "mounts": mounts, "signer": signer}


def signing_secret(run: Path, op: str) -> modal.Secret:
    """Acting as operator `op`: hand its own signing key to its own GPU container (per call)."""
    key = (run / "parties" / op / "secret" / "ed25519.key").read_bytes()
    return modal.Secret.from_dict({"HORIZON_SIGNING_KEY": base64.b64encode(key).decode()})


def write_outputs(run: Path, files: dict[str, bytes], out_dir: Path, extra_root: Path) -> list[str]:
    """Place returned files: runs/<run>/X → run/X (events appended), results/X → out_dir/X,
    other runs/... (baseline artefacts) → extra_root/...; never under a parties/ dir."""
    written = []
    prefix = f"runs/{run.name}/"
    for rel, data in sorted(files.items()):
        if "/parties/" in f"/{rel}" or ".." in Path(rel).parts:
            raise AssertionError(f"refusing returned path {rel}")
        if rel.startswith(prefix):
            dst = run / rel[len(prefix) :]
            if dst.name == "events.jsonl":
                with open(dst, "ab") as f:
                    f.write(data)
                continue
            if dst.exists():
                raise FileExistsError(f"remote step would overwrite {dst}")
        elif rel.startswith("results/"):
            dst = out_dir / rel[len("results/") :]
        elif rel.startswith("runs/"):
            dst = extra_root / rel[len("runs/") :]
        else:
            continue  # e.g. the container's data/ scratch
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        written.append(str(dst))
    return written


class ModalBackend:
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        self.cfg = yaml.safe_load(open(cfg_path))
        mc = self.cfg.get("modal", {})
        # HORIZON_MODAL_GPU overrides the config for one run; the GPU used is recorded per container
        # in logs/modal_usage.jsonl and in the run's result (modal_gpu).
        self.gpu = os.environ.get("HORIZON_MODAL_GPU") or mc.get("gpu")
        self.cpu = float(mc.get("cpu", 2))
        self.memory = int(mc.get("memory_mb", 8192))
        self.timeout = int(mc.get("timeout_s", 1800))
        # exact nvidia-smi name; "auto" = the first device this backend gets (one model per run); None = any
        self.require_device = mc.get("require_device")
        self.actor = {"default": actor, "glimmer": actor_glimmer}[mc.get("image", "default")]
        self.max_device_retries = 2  # A10 pool sometimes serves only A10G for a while: don't stall
        self.device_mismatches: list[dict] = []  # steps that ran on another model after the retries
        self.spent = 0.0
        if self.gpu:
            confirm_gpu(cfg_path, self.gpu)

    def run_steps(
        self, run: Path, steps: list[tuple[str, list[str]]], out_dir: Path | None = None
    ) -> list[dict]:
        """Run steps concurrently, one container each; return each step's last JSON stdout line.

        Every call's outputs are written before any failure is raised, and each call id is logged at
        spawn (`logs/modal_calls.jsonl`; Modal keeps spawned results 7 days), so a local error
        never throws away paid work: `recover(call_id)` fetches it again.
        """
        (run / "logs").mkdir(parents=True, exist_ok=True)
        calls = []
        for name, args in steps:
            srun = Path(_flag(args, "--run")) if "--run" in args else run  # the step's own run dir
            p = plan(srun, args, self.cfg_path)
            job = p["job"] | {"config_text": Path(self.cfg_path).read_text(), "signer": p["signer"]}
            want = self.require_device if self.gpu and self.require_device != "auto" else None
            job["require_device"] = want
            vols = {M_MODELS: models_vol} | {path: _vol(v) for path, v in p["mounts"].items()}
            secrets = [signing_secret(srun, p["signer"])] if p["signer"] else []
            fn = self.actor.with_options(
                gpu=self.gpu,
                cpu=self.cpu,
                memory=self.memory,
                timeout=self.timeout,
                volumes=vols,
                secrets=secrets,
                retries=0,  # preemption restarts the same input anyway; never pay 3x for a real bug
                scaledown_window=2,  # minimum: idle containers are billed (default 60 s)
            )
            call = self._spawn(run, name, fn, job)
            calls.append((name, srun, fn, job, call))
        results, failures = [], []
        for name, srun, fn, job, call in calls:
            try:
                res = call.get()
                for _ in range(self.max_device_retries):  # wrong GPU model: ask again (seconds each)
                    if not res.get("wrong_device"):
                        break
                    self._log_usage(run, name + " (wrong device)", res)
                    print(
                        f"  {name}: got {res['device']}, want {job['require_device']}; retrying", flush=True
                    )
                    res = self._spawn(run, name, fn, job).get()
                if res.get("wrong_device"):  # fail soft: run on what Modal has, and record it
                    miss = {"step": name, "wanted": job["require_device"], "got": res["device"]}
                    self.device_mismatches.append(miss)
                    print(f"  {name}: WARNING no {miss['wanted']} after retries; accepting {miss['got']}")
                    res = self._spawn(run, name, fn, job | {"require_device": None}).get()
            except Exception as e:  # remote crash/timeout: keep collecting the others first
                failures.append(f"{name}: {type(e).__name__}: {e}")
                results.append({})
                continue
            usd = self._log_usage(run, name, res)
            if self.require_device == "auto" and res.get("device", "cpu") != "cpu":
                self.require_device = res["device"]  # the rest of this run must match
            (run / "logs" / f"{name}.log").write_text(res["log"])
            if res["returncode"] != 0:
                failures.append(f"{name} exit {res['returncode']}:\n{res['log'][-2000:]}")
                results.append({})
                continue
            write_outputs(srun, res["files"], out_dir or run / "results", run)
            print(f"  {name}: {res['wall_s']:.0f}s on {res['device']} (≈ ${usd:.3f})", flush=True)
            results.append(last_json(res["log"]))
        if failures:
            raise RuntimeError("Modal step(s) failed:\n" + "\n".join(failures))
        return results

    def _spawn(self, run: Path, name: str, fn, job: dict):
        call = fn.spawn(job)
        with open(run / "logs" / "modal_calls.jsonl", "a") as f:
            f.write(json.dumps({"step": name, "call_id": call.object_id, "gpu": self.gpu}) + "\n")
        print(f"  {name}: → Modal ({self.gpu or 'cpu'}) {call.object_id}", flush=True)
        return call

    def _log_usage(self, run: Path, name: str, res: dict) -> float:
        usd = est_usd(self.gpu, self.cpu, self.memory, res["wall_s"])
        self.spent += usd
        usage = {"step": name, "device": res["device"], "gpu": self.gpu, "wall_s": round(res["wall_s"], 1)}
        usage |= {"est_usd": round(usd, 4), "returncode": res["returncode"]}
        with open(run / "logs" / "modal_usage.jsonl", "a") as f:
            f.write(json.dumps(usage) + "\n")
        return usd


def last_json(log: str) -> dict:
    """The step's one-line JSON summary (last stdout line that parses); {} if none does."""
    for line in reversed(log.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def recover(call_id: str) -> dict:
    """Fetch a spawned call's result again (kept 7 days by Modal)."""
    return modal.FunctionCall.from_id(call_id).get()


def session():
    """Context manager: an ephemeral Modal app for the duration of one experiment (logs streamed)."""
    modal.enable_output()
    return app.run()
