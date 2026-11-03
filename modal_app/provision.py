"""One-time Modal provisioning: data volumes, base weights and a tiny test model.

    python -m modal_app.provision --config configs/5g-nidd-glimmer.yaml   # everything this config needs
    python -m modal_app.provision data --config CFG [--ops op_d] [--verify]
    python -m modal_app.provision weights --config CFG                  # or: --repo ID --revision REV
    python -m modal_app.provision tiny                                  # tiny random model (CPU tests)

Volume names start with $HORIZON_MODAL_PREFIX (default "horizon"): `<prefix>-data-<op>` holds one
operator's shard, `<prefix>-eval` the test set, `<prefix>-models` the weights. `data` and `tiny` upload
from this machine. `weights` runs a small CPU container; a gated model (e.g. Llama) also needs a Modal
secret holding HF_TOKEN, named by $HORIZON_HF_SECRET (default "huggingface"). Muse Glimmer is not gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import modal
import yaml

from horizon.tasks import task_name
from modal_app.app import MODELS_VOL, PREFIX, data_vols, eval_vol

DATA_DIRS = {"nidd5g": Path("data/processed"), "asrs": Path("data/asrs")}  # each task's prepared split
TINY_DIR = "/tiny_llama"
TINY_ID = "/vol/models/tiny_llama"  # model.id of a config that uses the tiny model
HF_SECRET = os.environ.get("HORIZON_HF_SECRET", "huggingface")


def upload_data(
    task: str = "nidd5g",
    data_dir: str | None = None,
    ops: list[str] | None = None,
    with_test: bool | None = None,
) -> dict:
    """Upload the shards of `ops` (default op_a, op_b, op_c) and the test set. By default an explicit `ops`
    list means "add these members" and skips the shared test set (already provisioned)."""
    DATA = Path(data_dir) if data_dir else DATA_DIRS[task]  # noqa: N806
    if not (DATA / "stats.json").is_file():
        raise SystemExit(f"{DATA}/ is not prepared yet: run `make data` (5G-NIDD) or `make data-asrs` (ASRS)")
    out = {}
    for op, vname in data_vols(task, ops).items():
        vol = modal.Volume.from_name(vname, create_if_missing=True, version=2)
        files = [DATA / op / f"{s}.jsonl" for s in ("train", "val")]
        with vol.batch_upload(force=True) as b:
            for f in files:
                b.put_file(f, f"/{f.name}")
        out[vname] = {f.name: sum(1 for _ in open(f)) for f in files}
    if not (with_test if with_test is not None else not ops):
        return out
    vol = modal.Volume.from_name(eval_vol(task), create_if_missing=True, version=2)
    with vol.batch_upload(force=True) as b:
        b.put_file(DATA / "test.jsonl", "/test.jsonl")
    out[eval_vol(task)] = {"test.jsonl": sum(1 for _ in open(DATA / "test.jsonl"))}
    return out


def verify_data(task: str, data_dir: str | None = None) -> dict:
    """Compare every operator shard under the data dir and test.jsonl with the files on their volumes
    (line count + sha256). Raises if any differ."""
    DATA = Path(data_dir) if data_dir else DATA_DIRS[task]  # noqa: N806
    ops = sorted(p.name for p in DATA.iterdir() if p.is_dir() and p.name.startswith("op_"))
    pairs = [
        (vn, DATA / op / f"{s}.jsonl", f"/{s}.jsonl")
        for op, vn in data_vols(task, ops).items()
        for s in ("train", "val")
    ]
    pairs.append((eval_vol(task), DATA / "test.jsonl", "/test.jsonl"))
    out, bad = {}, []
    for vname, local, remote in pairs:
        data = b"".join(modal.Volume.from_name(vname, version=2).read_file(remote))
        same = hashlib.sha256(data).hexdigest() == hashlib.sha256(local.read_bytes()).hexdigest()
        out[f"{vname}{remote}"] = {
            "lines": data.count(b"\n"),
            "sha256": hashlib.sha256(data).hexdigest()[:16],
            "identical": same,
        }
        if not same:
            bad.append(f"{vname}{remote}")
    if bad:
        raise SystemExit(f"volume files differ from {DATA}: {bad}")
    return out


def upload_tiny(tokenizer: str | None = None) -> dict:
    """The tiny random model of horizon.model.tiny (public tokenizer by default), for CPU pipeline tests."""
    from horizon.model import tiny

    with tempfile.TemporaryDirectory() as d:
        if tokenizer:
            tiny.build(d, tokenizer, revision=None)
        else:
            tiny.build(d)
        vol = modal.Volume.from_name(MODELS_VOL, create_if_missing=True, version=2)
        with vol.batch_upload(force=True) as b:
            b.put_directory(d, TINY_DIR)
        return {"uploaded": sorted(p.name for p in Path(d).iterdir()), "to": f"{MODELS_VOL}:{TINY_DIR}"}


def hf_gated(repo: str) -> bool:
    """Whether the Hub requires accepting a licence (and a token) to download `repo`."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
        return bool(json.load(r).get("gated"))


def hf_secret() -> modal.Secret | None:
    try:
        s = modal.Secret.from_name(HF_SECRET)
        s.hydrate()
        return s
    except modal.exception.NotFoundError:
        return None


def download_weights(repos: list[tuple[str, str]]) -> list[dict]:
    """Download (repo, revision) pairs into the models volume, attaching the HF secret only if needed."""
    from modal_app.app import app
    from modal_app.app import download_weights as fn

    secret = hf_secret()
    for repo, _ in repos:
        if hf_gated(repo) and secret is None:
            raise SystemExit(
                f"{repo} is gated on Hugging Face. Accept its licence on huggingface.co, then store a "
                f"read token as a Modal secret:\n  modal secret create {HF_SECRET} HF_TOKEN=hf_...\n"
                "(Muse Glimmer is not gated and needs no token.)"
            )
    call = fn.with_options(secrets=[secret]) if secret else fn
    modal.enable_output()
    with app.run():
        return [call.remote(repo, rev) for repo, rev in repos]


def config_repos(cfg: dict) -> list[tuple[str, str]]:
    """The Hub repos a config's model needs: weights (+ a separate tokenizer repo, e.g. Glimmer's)."""
    m = cfg["model"]
    repos = [(m["id"], m.get("revision", "main"))]
    if m.get("tokenizer_id"):
        repos.append((m["tokenizer_id"], m.get("tokenizer_revision", "main")))
    return repos


def main() -> None:
    ap = argparse.ArgumentParser(description="Create and fill this project's Modal volumes.")
    ap.add_argument("what", nargs="?", default="all", choices=["all", "data", "weights", "tiny"])
    ap.add_argument("--config", help="run config: task, data_dir, operators and model are read from it")
    ap.add_argument("--repo", help="weights: a Hub repo instead of the config's model")
    ap.add_argument("--revision", default="main", help="weights: with --repo")
    ap.add_argument("--tokenizer", help="tiny: tokenizer to build it with (default: public Muse Glimmer)")
    ap.add_argument("--task", choices=sorted(DATA_DIRS), help="data without --config: a built-in split")
    ap.add_argument("--data-dir", help="data: prepared split to upload from (default: the config's)")
    ap.add_argument("--ops", help="data: comma list; upload only these operators' shards (not the test set)")
    ap.add_argument("--verify", action="store_true", help="data: check all volumes against the data dir")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)) if a.config else {}
    if a.what == "all" and not cfg:
        ap.error("give --config (or one of: data, weights, tiny)")
    task = task_name(cfg) if cfg else (a.task or "nidd5g")
    data_dir = a.data_dir or cfg.get("data_dir")
    res: dict = {"prefix": PREFIX}
    if a.what in ("all", "data"):
        if a.ops:  # new members only
            res["data"] = upload_data(task, data_dir, a.ops.split(","))
        else:  # every operator in the config, and the test set
            ops = (cfg.get("fl") or {}).get("operators")
            res["data"] = upload_data(task, data_dir, ops, with_test=True)
        if a.verify:
            res["verified"] = verify_data(task, data_dir)
    if a.what == "tiny" or (a.what == "all" and cfg["model"]["id"] == TINY_ID):
        res["tiny"] = upload_tiny(a.tokenizer)
    elif a.what in ("all", "weights"):
        if not (a.repo or cfg):
            ap.error("weights: give --config or --repo")
        res["weights"] = download_weights([(a.repo, a.revision)] if a.repo else config_repos(cfg))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
