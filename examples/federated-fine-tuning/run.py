"""Encrypted federated fine-tuning end to end, exactly as our published runs were made.

    python examples/federated-fine-tuning/run.py                              # laptop CPU, tiny model, free
    python examples/federated-fine-tuning/run.py --on modal --model tiny      # your Modal account, CPU, cents
    python examples/federated-fine-tuning/run.py --on modal --model llama     # Llama-3.2-1B on an A10
    python examples/federated-fine-tuning/run.py --on modal --model glimmer   # Muse Glimmer-30B on an L40S

    --dataset 5g-nidd (default) | asrs (aviation, 3 organisations) | asrs4 (aviation, 4 organisations, 3-of-4)
    --baselines             also train the comparison rows: base model, each organisation alone, pooled data
    --plaintext-reference   also run the unencrypted reference and check it matches the encrypted run
    --yes                   don't ask before starting GPU containers

Steps: download and split the data, upload it to your Modal volumes (Modal only), run the encrypted
federated training, write the per-class report and compare with our published result. Each step prints
the command it runs, so you can run them by hand. Outputs go to runs/examples/<dataset>-<model>/, never
to results/. The tiny model checks the pipeline only: its weights are random and its scores mean nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {  # name: (download name, prepare module, partition config, prepared data dir)
    "5g-nidd": ("nidd5g", "horizon.data.prepare", "configs/partitions/5g-nidd.yaml", "data/processed"),
    "asrs": ("asrs", "horizon.data.asrs", "configs/partitions/asrs.yaml", "data/asrs"),
    "asrs4": ("asrs", "horizon.data.asrs", "configs/partitions/asrs4.yaml", "data/asrs4"),
}
# (dataset, model) → (config, our published result to compare with)
RUNS = {
    ("5g-nidd", "tiny-local"): ("configs/5g-nidd-tiny.yaml", None),
    ("5g-nidd", "tiny"): ("configs/5g-nidd-tiny-modal.yaml", None),
    ("5g-nidd", "llama"): ("configs/5g-nidd-llama.yaml", "results/5g-nidd-llama"),
    ("5g-nidd", "glimmer"): ("configs/5g-nidd-glimmer.yaml", "results/5g-nidd-glimmer"),
    ("asrs", "glimmer"): ("configs/asrs-glimmer.yaml", "results/asrs-glimmer"),
    ("asrs4", "glimmer"): ("configs/asrs4-glimmer.yaml", "results/asrs4-glimmer"),
}
TASK = {"5g-nidd": "nidd5g", "asrs": "asrs", "asrs4": "asrs"}
# The replay checks prediction agreement in fp32 for Llama (bf16 rounding flips near-ties), as published.
AGREEMENT = {"configs/5g-nidd-llama.yaml": "configs/5g-nidd-llama-fp32-eval.yaml"}
SMOKE = ["--max-steps", "4", "--limit-test", "60"]  # tiny model: minutes of CPU


def py(module: str, *args: str, env: dict | None = None) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("$ python -m " + " ".join(shlex.quote(c) for c in cmd[2:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def config_for(dataset: str, model: str, out: Path) -> tuple[str, str | None]:
    """Our config, unchanged, for every combination we published; otherwise ours retargeted."""
    if (dataset, model) in RUNS:
        return RUNS[(dataset, model)]
    cfg = yaml.safe_load(open(ROOT / RUNS[("5g-nidd", model)][0]))
    cfg["task"], cfg["data_dir"] = TASK[dataset], DATASETS[dataset][3]
    if dataset == "asrs4":
        cfg["fl"] = cfg.get("fl", {}) | {"operators": ["op_a", "op_b", "op_c", "op_d"], "threshold": 3}
    (ROOT / out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out / "config.yaml"), None


def prepare_data(dataset: str) -> None:
    name, prepare, partition, data_dir = DATASETS[dataset]
    if (ROOT / data_dir / "stats.json").is_file():
        print(f"{data_dir}/ is already prepared")
        return
    py("horizon.data.download", name)
    if dataset == "asrs4" and not (ROOT / "data/asrs/stats.json").is_file():
        py(prepare, "--config", DATASETS["asrs"][2])  # the 4-organisation split reuses its test set
    py(prepare, "--config", partition)


def pipeline_summary(result: dict) -> None:
    for r in result["rounds"]:
        print(
            f"  round {r['round']}: {len(r['contributors'])} encrypted updates summed, decrypted by "
            f"{' + '.join(r['active'])}; one share alone gave noise (relative error {r['solo_rel_err']:.0e})"
        )
    print("tiny model: the pipeline works end to end; its scores mean nothing (random weights)")


def compare(result: dict, published: str | None) -> None:
    m = result["final"]["metrics"]
    print(f"encrypted federated model (E3): macro-F1 {m['macro_f1']:.3f}, accuracy {m['accuracy']:.3f}")
    if published:
        ref = json.loads((ROOT / published / "E3.json").read_text())["final"]
        same = result["final"]["pred_sha256"] == ref["pred_sha256"]
        print(
            f"published {published}/E3.json: macro-F1 {ref['metrics']['macro_f1']:.3f}; "
            + ("identical predictions on every test row" if same else "predictions differ (see README)")
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--on", choices=["local", "modal"], default="local")
    ap.add_argument("--model", choices=["tiny", "llama", "glimmer"], default="tiny")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="5g-nidd")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--plaintext-reference", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.on == "local" and a.model != "tiny":
        ap.error("--on local runs the tiny model only; larger models need a GPU (--on modal)")
    if a.baselines and a.model == "tiny":
        ap.error("--baselines needs a real model (--on modal --model llama|glimmer)")
    model = "tiny-local" if a.on == "local" else a.model
    out = Path("runs") / "examples" / f"{a.dataset}-{model}"
    res = str(out / "results")
    (ROOT / out).mkdir(parents=True, exist_ok=True)
    env = os.environ | ({"HORIZON_YES": "1"} if a.yes else {})
    cfg, published = config_for(a.dataset, model, out)
    smoke = SMOKE if a.model == "tiny" else []

    print(f"== 1. data: {a.dataset}")
    prepare_data(a.dataset)
    print("== 2. model and volumes")
    if a.on == "local":
        if not (ROOT / "models/tiny/config.json").is_file():
            py("horizon.model.tiny", "--out", "models/tiny")
    else:
        py("modal_app.provision", "--config", cfg, env=env)
    print("== 3. encrypted federated training (E3)")
    run = [
        "--config",
        cfg,
        "--backend",
        a.on,
        "--fresh",
        "--run-root",
        str(out / "runs"),
        "--out",
        res,
        *smoke,
    ]
    py("horizon.fl.run", "--mode", "encrypted", *run, env=env)
    if a.plaintext_reference:
        print("== 3b. plaintext reference (E2), replayed through the real threshold path")
        py("horizon.fl.run", "--mode", "plaintext", *run, env=env)
        agree = AGREEMENT.get(cfg, cfg)
        eq = ["--e2", f"{res}/E2.json", "--e3", f"{res}/E3.json", "--replay", "--config", cfg]
        py("horizon.eval.equivalence", *eq, "--agreement-config", agree, "--backend", a.on, env=env)
    if a.baselines:
        ops = (yaml.safe_load(open(ROOT / cfg)).get("fl") or {}).get("operators", ["op_a", "op_b", "op_c"])
        exps = ["E0", *(f"E1{op[-1]}" for op in ops), "E4"]
        print("== 3c. baselines: " + ", ".join(exps))
        py(
            "modal_app.baselines",
            "--config",
            cfg,
            "--exp",
            *exps,
            "--out",
            res,
            "--run-root",
            str(out / "runs"),
            env=env,
        )
    print("== 4. results")
    e3 = json.loads((ROOT / res / "E3.json").read_text())
    if smoke:
        pipeline_summary(e3)
    else:
        title = f"Per-class recall: {a.dataset}, {model}"
        py("horizon.eval.report", "--results", res, "--partition", DATASETS[a.dataset][2], "--title", title)
        compare(e3, published)
    print(f"outputs: {res}/")


if __name__ == "__main__":
    main()
