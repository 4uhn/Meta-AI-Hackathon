"""Second domain (ASRS): task switching and the data split's leakage guards.

The invariants on the real prepared split run only when data/asrs/ exists (`python -m horizon.data.asrs`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from horizon import tasks
from horizon.data import asrs, serialize
from horizon.model import labels as L

CFG = yaml.safe_load(open("configs/partitions/asrs.yaml"))
OUT = Path(CFG["out_dir"])
OPS = ["op_a", "op_b", "op_c"]


@pytest.fixture
def asrs_active():
    tasks.activate("asrs")
    yield
    tasks.activate("nidd5g")


def _toks():
    from transformers import AutoTokenizer

    g = yaml.safe_load(open("configs/5g-nidd-glimmer.yaml"))["model"]
    try:
        return {
            "glimmer": AutoTokenizer.from_pretrained(
                g["tokenizer_id"], revision=g["tokenizer_revision"], local_files_only=True
            ),
            "llama": AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", local_files_only=True),
        }
    except OSError:
        pytest.skip("tokenizers not in the HF cache")


def test_activate_rewrites_labels_in_place_and_restores_nidd():
    nidd = (list(serialize.LABELS), dict(L.VERBALIZER), L.SYSTEM)
    ref = serialize.LABELS  # what `from horizon.data.serialize import LABELS` holds elsewhere
    try:
        t = tasks.activate({"task": "asrs"})
        assert ref is serialize.LABELS and list(ref) == list(tasks.ASRS_LABELS)
        assert L.SYSTEM == t.system and "aviation safety" in L.SYSTEM
        assert set(L.VERBALIZER) == set(tasks.ASRS_LABELS) and tasks.active().name == "asrs"
    finally:
        tasks.activate({})  # no `task:` key = 5G-NIDD, the default
    assert (list(serialize.LABELS), dict(L.VERBALIZER), L.SYSTEM) == nidd
    assert tasks.active().name == "nidd5g"
    with pytest.raises(ValueError):
        tasks.activate("nope")


def test_asrs_first_tokens_unique_under_both_tokenizers(asrs_active):
    for name, tok in _toks().items():
        assert L.first_tokens_unique(tok), name
        p = L.prompt_text(tok, "Encountered moderate chop at FL350.")
        assert "aviation safety analyst" in p and "proto=" not in p


def test_event_class_needs_exactly_one_configured_class():
    c = CFG["classes"]
    assert asrs.event_class("Aircraft Equipment Problem Critical", c) == "equipment_failure"
    # generic items are ignored; the one configured class decides
    assert (
        asrs.event_class("Deviation / Discrepancy - Procedural FAR; Conflict NMAC", c) == "traffic_conflict"
    )
    assert asrs.event_class("Aircraft Equipment Problem Critical; Conflict NMAC", c) is None  # two classes
    assert asrs.event_class("Deviation - Speed All Types", c) is None  # none
    assert asrs.event_class("Flight Deck / Cabin / Aircraft Event Illness / Injury", c) is None  # excluded


def test_near_duplicates_is_exact():
    pool = ["a b c d e f g h", "x y z w v u t s", "a b c d e f g q"]
    got = asrs.near_duplicates(["a b c d e f g h"], pool, 3, 0.3)
    assert got.keys() == {0, 2} and got[0] == 1.0 and abs(got[2] - 5 / 7) < 1e-12


def _synthetic(n_per=40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k, lab in enumerate(tasks.ASRS_LABELS):
        for i in range(n_per):
            org = ["Air Carrier", "Government", "Personal"][i % 3]
            words = " ".join(f"w{rng.integers(1_000_000)}" for _ in range(30))
            rows.append(
                {
                    "acn": f"{k}-{i}",
                    "label": lab,
                    "org": org,
                    "text": f"{lab} {words}",
                    "event": f"acn:{k}-{i}",
                }
            )
    return pd.DataFrame(rows)


def test_partition_guards_on_synthetic_reports():
    cfg = dict(CFG, test_per_class=5, cap_per_class=8, min_per_class=3)
    rep = _synthetic()
    # plant a twin of every report: same event key for one, near-identical text for the other
    twins = rep.copy()
    twins["acn"] = "t" + twins["acn"]
    twins["text"] = twins["text"] + " extra"
    rep = pd.concat([rep, twins], ignore_index=True)
    shards, test, notes = asrs.partition(rep, cfg)
    assert test["label"].value_counts().to_dict() == {lab: 5 for lab in tasks.ASRS_LABELS}
    shard_text = set().union(*(set(d["text"]) for d in shards.values()))
    assert not shard_text & set(test["text"])
    # no shard text is a near-duplicate (Jaccard ≥ thr) of a test text
    assert not asrs.near_duplicates(test["text"].tolist(), sorted(shard_text), 5, CFG["near_dup_jaccard"])
    assert notes["near_duplicate_quarantined"] > 0
    for op, d in shards.items():
        assert set(d["label"]) == set(notes["held"][op])
        assert (d["label"].value_counts() <= cfg["cap_per_class"]).all()
    a, b, c = (set(shards[o]["acn"]) for o in OPS)
    assert not (a & b or a & c or b & c)


def test_deidentified_locations_are_not_one_event():
    df = pd.DataFrame(
        {
            "Report_1_Narrative": ["one two three"] * 3,
            "Events_Anomaly": ["Conflict NMAC"] * 3,
            "Person_1_Reporter_Organization": ["Government"] * 3,
            "Time_Date": ["202401"] * 3,
            "Place_Locale_Reference": ["ZZZ.Airport", "ZZZ.Airport", "ORD.Airport"],
            "acn": ["1", "2", "3"],
        }
    )
    df["Report_1_Narrative"] = [f"report number {i} words" for i in range(3)]
    p = Path(os.environ.get("TMPDIR", "/tmp")) / "asrs_test_raw.csv"
    df.to_csv(p, index=False)
    rep, _ = asrs.load_reports(p, CFG)
    ev = dict(zip(rep["acn"], rep["event"], strict=True))
    assert ev["1"] != ev["2"] and ev["3"] == "202401|ORD.Airport|traffic_conflict"


@pytest.mark.skipif(not (OUT / "stats.json").is_file(), reason="run python -m horizon.data.asrs first")
def test_real_split_invariants():
    read = lambda p: [json.loads(x) for x in open(p)]  # noqa: E731
    stats = json.loads((OUT / "stats.json").read_text())
    test = read(OUT / "test.jsonl")
    assert len(test) == 9 * CFG["test_per_class"] and {r["label"] for r in test} == set(tasks.ASRS_LABELS)
    tt = {r["text"] for r in test}
    seen: set[str] = set()
    for op in OPS:
        assert (os.stat(OUT / op).st_mode & 0o777) == 0o700
        rows = read(OUT / op / "train.jsonl") + read(OUT / op / "val.jsonl")
        texts = {r["text"] for r in rows}
        assert len(texts) == len(rows)  # no duplicate inside a shard
        assert not texts & tt and not texts & seen  # disjoint from test and other shards
        seen |= texts
        assert {r["label"] for r in rows} == set(stats["notes"]["held"][op])
        assert all(len(r["text"].split()) <= CFG["max_words"] for r in rows)
    assert not asrs.near_duplicates(
        sorted(tt), sorted(seen), CFG["near_dup_shingle"], CFG["near_dup_jaccard"]
    )


def test_modal_plan_mounts_only_the_tasks_volumes(tmp_path):
    from modal_app.backend import plan

    run = tmp_path / "r"
    (run / "bus" / "round_1" / "global").mkdir(parents=True)
    for cfg, vol, ev in (
        ("configs/asrs-glimmer.yaml", "horizon-data-asrs-op_b", "horizon-eval-asrs"),
        ("configs/5g-nidd-glimmer.yaml", "horizon-data-op_b", "horizon-eval"),  # 5G-NIDD: unsuffixed names
    ):
        e1b = plan(run, ["horizon.fl.baseline", "--exp", "E1b", "--config", cfg], cfg)
        assert set(e1b["mounts"].values()) == {vol, ev}
        val = plan(
            run,
            [
                "horizon.fl.evaluate",
                "--run",
                str(run),
                "--round",
                "1",
                "--config",
                cfg,
                "--split",
                "val",
                "--op",
                "op_b",
            ],
            cfg,
        )
        assert set(val["mounts"].values()) == {vol}
