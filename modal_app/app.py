"""Modal app: image, volumes and the container side of every remote step.

A remote step is a pure function: files in → run one of our modules → new files out. The container
rebuilds a minimal run directory (same basename as locally, so the crypto run id matches), runs
the same module the local backend runs (`horizon.fl.operator train`, `horizon.fl.evaluate`, …) and
returns only the files that step created. Nothing persists between calls, so a preempted call that
Modal restarts on the same input is harmless, and the caller writes outputs only after a call returns.

What each container can see is decided by the caller (`modal_app/backend.py`) through
`with_options(volumes=…, secrets=…)`: operator X mounts only its own data volume.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal
import modal.experimental

# Every Modal name (app and volumes) starts with this prefix. Set HORIZON_MODAL_PREFIX to keep your own
# set of volumes, e.g. several projects on one account; the containers receive the same value.
DEFAULT_PREFIX = "horizon"
PREFIX = os.environ.get("HORIZON_MODAL_PREFIX") or DEFAULT_PREFIX
if not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", PREFIX):
    raise ValueError(f"HORIZON_MODAL_PREFIX={PREFIX!r}: use 1-20 lowercase letters, digits or '-'")

APP_NAME = PREFIX
MODELS_VOL = f"{PREFIX}-models"  # HF cache + tiny test model (public weights)
EVAL_VOL = f"{PREFIX}-eval"  # test.jsonl (evaluator and baselines only)
DATA_VOL = {op: f"{PREFIX}-data-{op}" for op in ("op_a", "op_b", "op_c")}  # one private shard each


def data_vols(task: str | None = None, ops: list[str] | None = None) -> dict[str, str]:
    """Per-operator shard volumes for a task (5G-NIDD uses unsuffixed names), so tasks never share a
    volume. `ops` defaults to op_a, op_b, op_c; a larger consortium adds one volume per new operator."""
    ops = list(ops or DATA_VOL)
    if task in (None, "nidd5g"):
        return {op: f"{PREFIX}-data-{op}" for op in ops}
    return {op: f"{PREFIX}-data-{task}-{op}" for op in ops}


def eval_vol(task: str | None = None) -> str:
    return EVAL_VOL if task in (None, "nidd5g") else f"{EVAL_VOL}-{task}"


M_MODELS = "/vol/models"
HE_BIN = "/opt/horizon/horizon-he"
ROOT = Path("/tmp/hz")  # per-call scratch: runs/<run>, data/ (symlinks), results/, config.yaml

app = modal.App(APP_NAME)
models_vol = modal.Volume.from_name(MODELS_VOL, create_if_missing=True, version=2)

BASE_PIP = (  # versions tested locally (.venv); torch's PyPI wheel ships CUDA
    "torch==2.14.1",
    "transformers==5.21.0",
    "peft==0.21.2",
    "safetensors==0.8.0",
    "huggingface_hub==2.3.0",
    "numpy==2.5.4",
    "pandas==3.1.0",
    "pyyaml==6.0.3",
)
# Glimmer only: the nf4 checkpoint needs bitsandbytes, device_map placement needs accelerate.
# A separate image keeps the Llama image, and so its results, unchanged.
GLIMMER_PIP = ("bitsandbytes==0.50.2", "accelerate==1.15.0")


def _prefix_env() -> dict[str, str]:
    # Only a non-default prefix is added, so the default image definition is unchanged.
    return {} if PREFIX == DEFAULT_PREFIX else {"HORIZON_MODAL_PREFIX": PREFIX}


def _image(*extra: str) -> modal.Image:
    img = modal.Image.debian_slim(python_version="3.12").pip_install(*BASE_PIP)
    if extra:
        img = img.pip_install(*extra)
    return (
        img.env({"HF_HOME": f"{M_MODELS}/hf", "PYTHONUNBUFFERED": "1"} | _prefix_env())
        .add_local_file("crypto/bin/horizon-he-linux-amd64", HE_BIN, copy=True)
        .run_commands(f"chmod 755 {HE_BIN}")
        .add_local_python_source("horizon", "modal_app")
    )


image = _image()
glimmer_image = _image(*GLIMMER_PIP)


def _hashes(root: Path) -> dict[str, str]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@app.function(image=image, volumes={M_MODELS: models_vol}, timeout=3600)
def actor(job: dict) -> dict:
    """Run one step (Llama / tiny model image). See `run_job`."""
    return run_job(job)


@app.function(image=glimmer_image, volumes={M_MODELS: models_vol}, timeout=3600)
def actor_glimmer(job: dict) -> dict:
    """Run one step on the Glimmer image (config `modal.image: glimmer`). See `run_job`."""
    return run_job(job)


def run_job(job: dict) -> dict:
    """Run one step. job = {run_name, module, args, files, config_text, data_links, signer}.

    - args: module arguments with {RUN} / {CFG} placeholders for the container's paths;
    - files: {path relative to the run dir: bytes} (public keys, the round's global adapter);
    - data_links: {name under data_dir: mounted path} (an operator's shard dir, test.jsonl);
    - signer: operator id whose Ed25519 key is in env HORIZON_SIGNING_KEY (per-call Modal Secret).
    Returns {returncode, log, files: {path relative to ROOT: bytes}, wall_s, device}.
    """
    import yaml

    t0 = time.time()
    want = job.get("require_device")
    if want:  # e.g. gpu="A10" lands on an NVIDIA A10 or A10G; kernels differ → not bit-reproducible
        name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
        if name != want:
            modal.experimental.stop_fetching_inputs()  # don't let this warm container take the retry
            return {
                "returncode": 75,
                "wrong_device": True,
                "device": name,
                "log": f"wrong device {name}",
                "files": {},
                "wall_s": time.time() - t0,
            }
    shutil.rmtree(ROOT, ignore_errors=True)
    run = ROOT / "runs" / job["run_name"]
    for rel, data in job["files"].items():
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True)
    for name, target in job.get("data_links", {}).items():
        (data_dir / name).symlink_to(target)
    cfg = yaml.safe_load(job["config_text"])
    cfg["data_dir"] = str(data_dir)
    (ROOT / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if job.get("signer"):  # this container acts as that operator: its key, in its own secret/ dir
        sec = run / "parties" / job["signer"] / "secret"
        sec.mkdir(parents=True, mode=0o700, exist_ok=True)
        kf = sec / "ed25519.key"
        kf.write_bytes(base64.b64decode(os.environ["HORIZON_SIGNING_KEY"]))
        kf.chmod(0o600)
    before = _hashes(ROOT)

    args = [a.replace("{RUN}", str(run)).replace("{CFG}", str(ROOT / "config.yaml")) for a in job["args"]]
    env = os.environ | {"PYTHONPATH": "/root", "HORIZON_HE": HE_BIN, "HF_HUB_OFFLINE": "1"}
    env.pop("HORIZON_SIGNING_KEY", None)  # the key is in the operator's secret/ file only
    proc = subprocess.Popen(
        [sys.executable, "-m", job["module"], *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    for line in proc.stdout:  # stream to the Modal log and keep a copy for the caller's logs/
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()

    # One call per container by design: exit right after this input instead of idling for the
    # scaledown window, because idle containers are billed.
    modal.experimental.stop_fetching_inputs()
    files = {}
    for rel, h in _hashes(ROOT).items():
        if before.get(rel) == h or rel == "config.yaml" or "/parties/" in f"/{rel}":
            continue  # unchanged inputs, and never anything from an operator's private dir
        files[rel] = (ROOT / rel).read_bytes()
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "returncode": rc,
        "log": "".join(lines),
        "files": files,
        "wall_s": time.time() - t0,
        "device": device,
    }


@app.function(
    image=image,
    volumes={M_MODELS: models_vol},
    timeout=3600,
    cpu=2,
    memory=4096,
)
def download_weights(repo_id: str, revision: str = "main") -> dict:
    """Download a base model once into the models volume. A gated model needs the caller to attach a
    Hugging Face secret (`with_options(secrets=…)`); this is the only place a token is used."""
    from huggingface_hub import snapshot_download

    t = time.time()
    path = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.jinja"],
    )
    models_vol.commit()
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return {
        "repo_id": repo_id,
        "commit": Path(path).name,
        "gb": round(size / 1e9, 2),
        "s": round(time.time() - t),
    }
