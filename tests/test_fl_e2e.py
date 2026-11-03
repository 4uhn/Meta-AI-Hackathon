"""End-to-end federated loop on CPU with a tiny random Llama (real tokenizer, real horizon-he, every
actor its own process): E3 encrypted ≡ E2 plaintext, and the E5 async scenario.

Slow (~2 min, CPU only): `pytest -m slow tests/test_fl_e2e.py`. Needs `make he`; the tokenizer (public
Muse Glimmer by default, or HORIZON_TOKENIZER) is downloaded once.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pytest
import yaml

from horizon.data.serialize import LABELS
from horizon.eval.equivalence import compare
from horizon.events import read_events
from horizon.fl.run import run_experiment
from horizon.fl.rundir import adapter_dir
from horizon.he_bridge import DEFAULT_BINARY
from horizon.model.lora import read_b_file

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not DEFAULT_BINARY.is_file(), reason="run `make he`")]
# HF id, or a directory holding just the tokenizer files
TOKENIZER = os.environ.get("HORIZON_TOKENIZER", "meta-models/Muse-Glimmer-30B")  # public
SHARDS = {"op_a": LABELS[:4], "op_b": LABELS[3:7], "op_c": LABELS[5:]}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from horizon.model import tiny

    root = tmp_path_factory.mktemp("e2e")
    model_dir = root / "tiny_llama"
    try:
        tiny.build(model_dir, TOKENIZER, tiny.TOKENIZER_REVISION if TOKENIZER == tiny.TOKENIZER else None)
    except OSError:
        pytest.skip(f"{TOKENIZER} tokenizer not available (offline, or a gated model without a token)")

    data = root / "data"
    for op, labs in SHARDS.items():
        (data / op).mkdir(parents=True)
        rows = [{"text": f"proto={lab} n={i}", "label": lab} for i in range(3) for lab in labs[:2]]
        (data / op / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    test = [{"text": f"proto={lab} n=9", "label": lab} for lab in LABELS]
    (data / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test))

    conf = {
        "model": {"id": str(model_dir), "revision": "main", "dtype": "float32", "device": "cpu"},
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "train": {
            "lr": 1e-3,
            "batch_size": 4,
            "micro_batch": 2,
            "local_epochs": 1,
            "rounds": 2,
            "seed": 1234,
            "adapter_seed": 4321,
        },
        "data_dir": str(data),
        "eval": {"batch_flows": 3, "every_round": True},
    }
    cfg_path = root / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(conf))
    return root, cfg_path


def go(env, mode, scenario="none", rounds=0):
    root, cfg = env
    ns = argparse.Namespace(
        config=str(cfg),
        mode=mode,
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


@pytest.fixture(scope="module")
def e3(env):
    return go(env, "encrypted")


@pytest.fixture(scope="module")
def e2(env):
    return go(env, "plaintext")


def test_encrypted_loop_completes(e3):
    assert [r["contributors"] for r in e3["rounds"]] == [["op_a", "op_b", "op_c"]] * 2
    assert all(r["solo_rel_err"] > 1e3 for r in e3["rounds"])  # one share alone = noise
    ev = [e["event"] for e in read_events(e3["run"])]
    for want in (
        "pk_published",
        "encrypted",
        "aggregated",
        "decrypted",
        "solo_decrypt_attempt",
        "run_finished",
    ):
        assert want in ev
    assert "insecure_reference_written" not in ev
    # an encrypted run never writes a plaintext update
    from pathlib import Path

    run = Path(e3["run"])
    assert not (run / "insecure").exists()
    assert not list(run.glob("parties/*/state/*.f64"))
    c = e3["crypto"]
    assert c["n_values"] == e3["n_values"] and c["bytes_per_submission"] > 0
    assert np.any(read_b_file(adapter_dir(run, 3)) != 0)


def test_e2_matches_e3(e2, e3):
    """Toy shards (n = 6) give Σw = 0.0018, so the 1e-5 ΔB tolerance used for real runs (Σw ≈ 0.34)
    does not apply; check its ingredients instead: identical training, CKKS-sized aggregate error,
    ΔB ≈ err / Σw."""
    assert e2["layout_hash"] == e3["layout_hash"]
    for op in ("op_a", "op_b", "op_c"):  # round 1: bit-identical local training in both runs
        assert (
            e2["rounds"][0]["manifests"][op]["train_loss_final"]
            == e3["rounds"][0]["manifests"][op]["train_loss_final"]
        )
    eq = compare(e2, e3)
    r1 = eq["per_round"][0]
    assert r1["agg_max_abs_err"] <= 1e-5
    assert r1["max_abs_dB"] <= 2 * r1["agg_max_abs_err"] / r1["weight_sum"]
    assert eq["prediction_agreement"] == 1.0
    assert "insecure_reference_written" in [e["event"] for e in read_events(e2["run"])]


def test_e5_skip_refused_and_offline_decrypt(env):
    e5 = go(env, "encrypted", scenario="e5", rounds=3)
    r1, r2, r3 = e5["rounds"]
    assert r1["active"] == ["op_a", "op_b"] and not r1.get("skipped")
    assert r2["skipped"] and r2["skipped_training"] == ["op_c"] and "need" in r2["reason"]
    assert r3["active"] == ["op_a", "op_c"] and sorted(r3["contributors"]) == ["op_a", "op_b", "op_c"]
    run = e5["run"]
    assert np.array_equal(read_b_file(adapter_dir(run, 3)), read_b_file(adapter_dir(run, 2)))  # carried over
    assert not np.array_equal(read_b_file(adapter_dir(run, 4)), read_b_file(adapter_dir(run, 3)))
    ev = read_events(run)
    assert any(e["event"] == "round_skipped" and e["round"] == 2 for e in ev)
    assert any(e["event"] == "offline" and e["actor"] == "op_b" and e["round"] == 3 for e in ev)
