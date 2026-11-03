"""Data pipeline: serialiser determinism, no test-string overlap, disjoint shards, silo guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from horizon.data.prepare import load_shard, load_test, partition, read_jsonl
from horizon.data.serialize import FEATURE_COLUMNS, LABELS, fmt_num, serialize_frame

CFG = yaml.safe_load(open(Path(__file__).parent.parent / "configs" / "partitions/5g-nidd.yaml"))
PROCESSED = Path(__file__).parent.parent / CFG["out_dir"]


def test_fmt_num():
    assert fmt_num(float("nan")) == "na"
    assert fmt_num(0.0) == "0"
    assert fmt_num(64.0) == "64"
    assert fmt_num(1412.7) == "1410"
    assert fmt_num(0.0021345) == "0.00213"
    assert fmt_num(4.99802) == "5"
    assert fmt_num(3737600.0) == "3740000"
    assert fmt_num(1.234e-7) == "1.23e-07"
    assert fmt_num("tcp") == "tcp"


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = {c: rng.integers(0, 50, n).astype(float) for c in FEATURE_COLUMNS}
    d["Proto"] = rng.choice(["tcp", "udp", "icmp"], n)
    d["State"] = rng.choice(["CON", "REQ"], n)
    d["Cause"] = "Start"
    d["dTtl"] = np.where(rng.random(n) < 0.5, np.nan, d["dTtl"])
    return pd.DataFrame(d)


def test_serializer_deterministic_and_na():
    df = _synthetic()
    a, b = serialize_frame(df), serialize_frame(df.copy())
    assert a.equals(b)
    assert any("ttl=" in s and "/na" in s for s in a)
    assert a.iloc[0].startswith("proto=")


def _uniq(n_per=300, seed=0):
    rows = [(f"{lab} flow {i}", lab) for lab in LABELS for i in range(n_per)]
    return pd.DataFrame(rows, columns=["text", "label"]).sample(frac=1, random_state=seed)


def test_partition_disjoint_and_deterministic():
    s1, t1, _ = partition(_uniq(seed=0), CFG)
    s2, t2, _ = partition(_uniq(seed=1), CFG)  # input order must not matter
    assert t1.equals(t2)
    for op in s1:
        assert s1[op].equals(s2[op])
    test = set(t1["text"])
    seen: set[str] = set()
    for op, d in s1.items():
        texts = set(d["text"])
        assert not texts & test, f"{op} shard overlaps test"
        assert not texts & seen, f"{op} shard overlaps another shard"
        seen |= texts
        assert set(d["label"]) == set(CFG["operators"][op])


@pytest.mark.skipif(not (PROCESSED / "test.jsonl").exists(), reason="run `make data` first")
def test_processed_files_no_overlap():
    test = {r["text"] for r in load_test(PROCESSED)}
    union: set[str] = set()
    for op in CFG["operators"]:
        rows = load_shard(PROCESSED, op, "train", actor=op) + load_shard(PROCESSED, op, "val", actor=op)
        texts = {r["text"] for r in rows}
        assert len(texts) == len(rows), "duplicate string inside a shard"
        assert not texts & test
        assert not texts & union
        union |= texts
        assert {r["label"] for r in rows} == set(CFG["operators"][op])
    assert {r["label"] for r in read_jsonl(PROCESSED / "test.jsonl")} == set(LABELS)


def test_silo_guard():
    with pytest.raises(PermissionError):
        load_shard(PROCESSED, "op_b", "train", actor="op_a")
    with pytest.raises(PermissionError):
        load_shard(PROCESSED, "op_b", "train", actor="aggregator")
