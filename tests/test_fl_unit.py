"""Aggregator-side pieces that work on files only: adapter B read/write without a model, layout hash
from the file, publish/carry-over, and the event log."""

from __future__ import annotations

import json

import numpy as np
import pytest

from horizon.events import emit, read_events
from horizon.fl.aggregator import publish
from horizon.fl.rundir import adapter_dir, global_dir, write_json
from horizon.model.lora import (
    a_hash,
    create_initial_adapter,
    file_layout_hash,
    flatten_b,
    layout_hash,
    load_adapter,
    n_b_values,
    read_b_file,
    write_b_file,
)
from tests.test_flatten import LORA, tiny


@pytest.fixture
def run(tmp_path):
    run = tmp_path / "run_x"
    m = create_initial_adapter(tiny(), LORA, seed=7, out_dir=adapter_dir(run, 1))
    write_json(
        global_dir(run, 1) / "round.json",
        {"round": 1, "layout_hash": layout_hash(m), "lora_A_hash": a_hash(m), "n_values": n_b_values(m) + 1},
    )
    return run


def test_file_layout_and_b_match_live_model(run, tmp_path):
    m = load_adapter(tiny(), adapter_dir(run, 1), trainable=True)
    assert file_layout_hash(adapter_dir(run, 1)) == layout_hash(m)
    v = np.random.default_rng(0).normal(0, 0.01, n_b_values(m)).astype(np.float32).astype(np.float64)
    write_b_file(adapter_dir(run, 1), tmp_path / "new", v)
    assert np.array_equal(read_b_file(tmp_path / "new"), v)
    m2 = load_adapter(tiny(), tmp_path / "new", trainable=True)  # PEFT loads what we wrote
    assert np.array_equal(flatten_b(m2), v)
    assert a_hash(m2) == a_hash(m), "writing B must leave A untouched"
    with pytest.raises(ValueError):
        write_b_file(adapter_dir(run, 1), tmp_path / "bad", v[:-1])


def test_publish_new_and_carry_over(run):
    n = json.loads((global_dir(run, 1) / "round.json").read_text())["n_values"] - 1
    v = np.full(n, 0.25)
    info = publish(run, 1, v)
    assert info["round"] == 2 and not info["carried_over"]
    assert np.array_equal(read_b_file(adapter_dir(run, 2)), v)
    info = publish(run, 2, None)  # skipped round: adapter carries over unchanged
    assert info["carried_over"] and np.array_equal(read_b_file(adapter_dir(run, 3)), v)
    with pytest.raises(ValueError):
        publish(run, 3, v[:-1])
    ev = [e["event"] for e in read_events(run)]
    assert ev == ["global_published", "global_published"]


def test_events_schema(tmp_path):
    emit(tmp_path, "op_b", "train_done", 2, loss=0.1, n=5, ms=7)
    (e,) = read_events(tmp_path)
    assert set(e) == {"ts", "run_id", "round", "actor", "event", "data"}
    assert (e["actor"], e["round"], e["data"]["n"]) == ("op_b", 2, 5) and e["ts"].endswith("Z")
