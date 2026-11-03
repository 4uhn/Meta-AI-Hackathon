"""Report on synthetic results: blind-spot math, smoke runs excluded, figure written."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from horizon.data.serialize import LABELS
from horizon.eval.report import blind_spots, build, plot_matrix

PART = yaml.safe_load(open("configs/partitions/5g-nidd.yaml"))


def fake(tmp_path, exp, recall, smoke=False):
    per = {lab: {"recall": recall(lab)} for lab in LABELS}
    m = {"accuracy": 0.5, "macro_f1": 0.4, "per_class": per}
    (tmp_path / f"{exp}.json").write_text(json.dumps({"metrics": m, "smoke": smoke}))


def test_blind_spots_are_unseen_attacks():
    s = blind_spots(PART)
    assert len(s["op_a"]) == 5 and "udp_flood" not in s["op_a"] and "benign" not in s["op_a"]


def test_build_and_plot(tmp_path):
    spots = blind_spots(PART)
    for e, op in (("E1a", "op_a"), ("E1b", "op_b"), ("E1c", "op_c")):
        fake(tmp_path, e, lambda lab, op=op: 0.0 if lab in spots[op] else 1.0)
    fake(tmp_path, "E3", lambda lab: 0.8)
    fake(tmp_path, "E4", lambda lab: 0.9, smoke=True)  # smoke runs never enter the report
    rep = build(tmp_path, PART)
    assert set(rep["recall"]) == {"E1a", "E1b", "E1c", "E3"}
    assert rep["blind_spot_recall"]["mean"] == pytest.approx({"local": 0.0, "horizon_E3": 0.8})
    rep["recall"]["E3"]["udp_scan"] = float("nan")  # e.g. a class with no support
    plot_matrix(rep, tmp_path / "m.png", "t")
    assert (tmp_path / "m.png").stat().st_size > 10_000
    assert np.isnan(rep["recall"]["E3"]["udp_scan"])


@pytest.mark.skipif(
    not Path("data/asrs/stats.json").is_file(), reason="prepare data/asrs first (make data-asrs)"
)
def test_operator_without_blind_spots_is_skipped(tmp_path):
    """ASRS: the airline holds all 9 classes; the mean runs over operators that have blind spots."""
    from horizon import tasks

    tasks.activate("asrs")
    try:
        part = yaml.safe_load(open("configs/partitions/asrs.yaml"))
        spots = blind_spots(part)
        assert spots["op_a"] == [] and spots["op_b"] and spots["op_c"]
        labs = list(tasks.ASRS_LABELS)
        for e, op in (("E1a", "op_a"), ("E1b", "op_b"), ("E1c", "op_c")):
            per = {lab: {"recall": 0.0 if lab in spots[op] else 1.0} for lab in labs}
            (tmp_path / f"{e}.json").write_text(
                json.dumps({"metrics": {"accuracy": 0.5, "macro_f1": 0.4, "per_class": per}})
            )
        per = {lab: {"recall": 0.6 if lab in spots["op_b"] else 0.9} for lab in labs}
        (tmp_path / "E3.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.5, "macro_f1": 0.4, "per_class": per}})
        )
        bs = build(tmp_path, part)["blind_spot_recall"]
        assert "op_a" not in bs and set(bs) == {"op_b", "op_c", "mean"}
        exp_c = np.mean([0.6 if c in spots["op_b"] else 0.9 for c in spots["op_c"]])
        assert bs["mean"]["horizon_E3"] == pytest.approx((0.6 + exp_c) / 2)
        assert bs["mean"]["local"] == 0.0
    finally:
        tasks.activate("nidd5g")
