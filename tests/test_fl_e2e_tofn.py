"""End-to-end federated loop at 3-of-4 on CPU: a tiny random Llama, real ASRS rows from data/asrs4
(four organisations), real horizon-he, every actor its own process. Two encrypted rounds, the async
scenario at 3-of-4 (a skipped round is refused, one operator offline at decryption), and the E1d
baseline.

Slow (~2 min, CPU only): `pytest -m slow tests/test_fl_e2e_tofn.py`. Needs data/asrs4 and `make he`;
the tokenizer (public Muse Glimmer by default, or HORIZON_TOKENIZER) is downloaded once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from horizon.fl.run import run_experiment
from horizon.he_bridge import DEFAULT_BINARY, read_f64
from horizon.model.lora import fedavg_from_sum, read_b_file

ASRS4 = Path("data/asrs4")
FOUR = ["op_a", "op_b", "op_c", "op_d"]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DEFAULT_BINARY.is_file(), reason="run `make he`"),
    pytest.mark.skipif(not (ASRS4 / "stats.json").is_file(), reason="prepare data/asrs4 first"),
]
TOKENIZER = os.environ.get("HORIZON_TOKENIZER", "meta-models/Muse-Glimmer-30B")  # public


def _take(path: Path, per_class: int) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for line in open(path):
        lab = json.loads(line)["label"]
        if seen.get(lab, 0) < per_class:
            seen[lab] = seen.get(lab, 0) + 1
            out.append(line)
    return out


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from horizon.model import tiny

    root = tmp_path_factory.mktemp("e2e4")
    model_dir = root / "tiny_llama"
    try:
        tiny.build(model_dir, TOKENIZER, tiny.TOKENIZER_REVISION if TOKENIZER == tiny.TOKENIZER else None)
    except OSError:
        pytest.skip(f"{TOKENIZER} tokenizer not available (offline, or a gated model without a token)")

    data = root / "data"
    for op in FOUR:  # one real report per held class, each organisation from its own shard
        (data / op).mkdir(parents=True)
        (data / op / "train.jsonl").write_text("".join(_take(ASRS4 / op / "train.jsonl", 1)))
    (data / "test.jsonl").write_text("".join(_take(ASRS4 / "test.jsonl", 1)))

    conf = {
        "task": "asrs",
        "model": {"id": str(model_dir), "revision": "main", "dtype": "float32", "device": "cpu"},
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "train": {
            "lr": 5e-4,
            "batch_size": 4,
            "micro_batch": 2,
            "local_epochs": 1,
            "rounds": 2,
            "seed": 1234,
            "adapter_seed": 4321,
            "restrict_missing_labels": True,
            "logit_adjust_tau": 1.0,
        },
        "fl": {"server_lr": 1.0, "operators": FOUR, "threshold": 3},
        "data_dir": str(data),
        "eval": {"batch_flows": 3, "every_round": True},
    }
    cfg_path = root / "tiny4.yaml"
    cfg_path.write_text(yaml.safe_dump(conf))
    return root, cfg_path


def go(env, scenario="none", rounds=0):
    root, cfg = env
    ns = argparse.Namespace(
        config=str(cfg),
        mode="encrypted",
        scenario=scenario,
        exp=None,
        rounds=rounds,
        out=str(root / "results"),
        run_root=str(root / "runs"),
        fresh=True,
        max_steps=0,
        limit_test=0,
    )
    return run_experiment(ns)


def test_three_of_four_encrypted_loop(env):
    root, _ = env
    res = go(env)
    run = root / "runs" / "tiny4_E3"
    parties = json.loads((run / "public" / "parties.json").read_text())
    assert parties["threshold"] == 3 and [p["name"] for p in parties["parties"]] == FOUR
    assert json.loads((run / "public" / "params.json").read_text())["flood_log_width"] == 86
    assert json.loads((run / "bus" / "round_1" / "global" / "round.json").read_text())["min_submissions"] == 4
    for r in res["rounds"]:
        assert r["contributors"] == FOUR and r["active"] == ["op_a", "op_b", "op_c"]
        assert r["solo_rel_err"] > 1e3 and r["partial_rel_err"] > 1e3  # 1 and 2 shares are noise
        agg = read_f64(run / "bus" / f"round_{r['round']}" / "decrypt" / "agg.f64")
        b_new = read_b_file(run / "bus" / f"round_{r['round'] + 1}" / "global" / "adapter")
        assert np.max(np.abs(b_new - fedavg_from_sum(agg))) < 1e-6  # published B = decrypted FedAvg
    events = [json.loads(x) for x in open(run / "events.jsonl")]
    for r in (1, 2):
        assert sorted(e["actor"] for e in events if e["event"] == "submitted" and e["round"] == r) == FOUR
    assert not any("insecure" in e["event"] for e in events) and not list(run.rglob("insecure"))
    assert all(p.parent.name == "decrypt" for p in run.rglob("*.f64"))  # plaintext only once decrypted


def test_three_of_four_async(env):
    """op_c skips round 2 → 3 contributors < t + 1 = 4 → refused; op_b offline at round 3 → {a, c, d}."""
    root, _ = env
    res = go(env, scenario="e5", rounds=3)
    r1, r2, r3 = res["rounds"]
    assert r1["contributors"] == FOUR
    assert r2["skipped"] and r2["skipped_training"] == ["op_c"] and "need ≥ 4" in r2["reason"]
    assert r3["active"] == ["op_a", "op_c", "op_d"] and r3["contributors"] == FOUR


def test_local_baseline_for_the_new_member(env):
    root, cfg = env
    out = root / "results_b"
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "horizon.fl.baseline",
            "--exp",
            "E1d",
            "--config",
            str(cfg),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=root,  # its run dir (runs/dev_E1d) lands in the temp dir, not the repo
        env=os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads((out / "E1d.json").read_text())
    assert r["n_test"] == 9 and "training on 7 rows" in p.stdout  # op_d holds 7 classes
