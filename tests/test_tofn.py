"""t-of-N consortia in the Python pipeline: config, CLI bridge at 3-of-4, Modal plans, report rows,
and the four-organisation ASRS split (op_a/b/c + test byte-identical to the three-organisation one)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from horizon.fl import rundir
from horizon.he_bridge import DEFAULT_BINARY, HorizonHE

FOUR = ["op_a", "op_b", "op_c", "op_d"]


def test_default_consortium_is_two_of_three():
    assert rundir.OPERATORS == ["op_a", "op_b", "op_c"]
    assert (rundir.threshold(), rundir.min_contributors(), rundir.default_active()) == (
        2,
        3,
        ["op_a", "op_b"],
    )
    rundir.configure({"fl": {"server_lr": 1.0}})  # an fl block without consortium keys
    assert rundir.OPERATORS == ["op_a", "op_b", "op_c"] and rundir.threshold() == 2


def test_configure_three_of_four_in_place():
    from horizon.fl.rundir import OPERATORS  # the name other modules import

    rundir.configure({"fl": {"operators": FOUR, "threshold": 3}})
    assert OPERATORS == FOUR and rundir.threshold() == 3
    assert rundir.min_contributors() == 4 and rundir.default_active() == ["op_a", "op_b", "op_c"]


@pytest.mark.parametrize(
    "ops,t",
    [
        (["op_a", "op_b"], 2),  # N < 3
        (FOUR, 2),  # 2t ≤ N: two disjoint decrypting sets could difference two aggregates
        (FOUR, 1),
        (FOUR, 5),
        (["op_a", "op_c", "op_b"], 2),  # not in point order
        (["op_a", "op_b", "op_x"], 2),
        ([f"op_{c}" for c in "abcdefgh"], 5),  # N > 7
        (["op_a", "op_b", "op_c"], 3),  # t = N: an aggregate would need N + 1 contributors
        ([f"op_{c}" for c in "abcdef"], 5),  # N = 6 without fl.unverified
    ],
)
def test_configure_rejects(ops, t):
    with pytest.raises(ValueError):
        rundir.configure({"fl": {"operators": ops, "threshold": t}})


def test_bridge_passes_party_flags_only_when_not_default(monkeypatch, tmp_path):
    if not DEFAULT_BINARY.is_file():
        pytest.skip("run `make he`")
    calls = []
    he = HorizonHE(tmp_path / "r")
    monkeypatch.setattr(he, "call", lambda cmd, **kw: calls.append((cmd, kw)) or {})
    he.init()
    he.init(["op_a", "op_b", "op_c"], 2)
    he.init(FOUR, 3)
    assert calls == [("init", {}), ("init", {}), ("init", {"parties": "op_a,op_b,op_c,op_d", "threshold": 3})]


@pytest.mark.skipif(not DEFAULT_BINARY.is_file(), reason="run `make he`")
def test_bridge_three_of_four_every_subset(tmp_path):
    """Real CLI: 4 operators submit; every 3-subset decrypts its own round; 1 or 2 shares are noise;
    3 contributors (one skipped) are refused."""
    he = HorizonHE(tmp_path / "run_tofn")
    assert all(s["ok"] for s in he.ceremony_stepwise(FOUR, 3))
    parties = json.loads((he.run / "public" / "parties.json").read_text())
    params = json.loads((he.run / "public" / "params.json").read_text())
    assert parties["threshold"] == 3 and [p["point"] for p in parties["parties"]] == [1, 2, 3, 4]
    assert params["flood_log_width"] == 86 and params["params_sha256"].startswith("ed03de312c948e0d")
    n = 20_000
    sets = [
        ["op_a", "op_b", "op_c"],
        ["op_a", "op_b", "op_d"],
        ["op_a", "op_c", "op_d"],
        ["op_b", "op_c", "op_d"],
    ]
    for r, active in enumerate(sets, start=1):
        rng = np.random.default_rng(r)
        ups = {p: rng.uniform(-0.05, 0.05, n) for p in FOUR}
        for p in FOUR:
            he.encrypt(p, r, ups[p])
        assert he.aggregate(r)["contributors"] == FOUR
        he.request(r, active)
        for p in active:
            he.dec_share(p, r)
        he.dec_combine(r)
        assert np.max(np.abs(he.aggregate_values(r) - sum(ups.values()))) < 1e-5
        assert he.dec_combine(r, solo=active[0])["rel_err_vs_true"] > 1e3
        assert he.dec_combine(r, solo=",".join(active[:2]))["rel_err_vs_true"] > 1e3
        assert (he.run / "bus" / f"round_{r}" / "decrypt" / f"solo_{active[0]}+{active[1]}.f64").is_file()
    r = len(sets) + 1
    for p in FOUR[:3]:  # op_d skips: two colluders among three contributors would read the third
        he.encrypt(p, r, np.zeros(10))
    with pytest.raises(Exception, match="need ≥ 4"):
        he.aggregate(r)


def test_modal_plan_three_of_four(tmp_path):
    from modal_app.backend import plan

    cfg = "configs/asrs4-glimmer.yaml"
    run = tmp_path / "r"
    (run / "bus" / "round_1" / "global").mkdir(parents=True)
    (run / "public").mkdir()
    (run / "parties" / "op_d" / "secret").mkdir(parents=True)
    (run / "parties" / "op_d" / "secret" / "ed25519.key").write_bytes(b"k")
    tr = plan(
        run,
        ["horizon.fl.operator", "train", "--run", str(run), "--op", "op_d", "--round", "1", "--config", cfg],
        cfg,
    )
    assert set(tr["mounts"].values()) == {"horizon-data-asrs-op_d"} and tr["signer"] == "op_d"
    e1d = plan(run, ["horizon.fl.baseline", "--exp", "E1d", "--config", cfg], cfg)
    assert set(e1d["mounts"].values()) == {"horizon-data-asrs-op_d", "horizon-eval-asrs"}
    e4 = plan(run, ["horizon.fl.baseline", "--exp", "E4", "--config", cfg], cfg)
    assert set(e4["mounts"].values()) == {f"horizon-data-asrs-{op}" for op in FOUR} | {"horizon-eval-asrs"}
    # the three-organisation config is unchanged: E4 pools exactly three shards
    e4_3 = plan(
        run,
        ["horizon.fl.baseline", "--exp", "E4", "--config", "configs/asrs-glimmer.yaml"],
        "configs/asrs-glimmer.yaml",
    )
    assert len(e4_3["mounts"]) == 4


def test_report_rows_follow_the_operators():
    from horizon.eval.report import LOCAL, ROWS, local_map, rows_for

    assert rows_for(None) == ROWS and local_map(None) == LOCAL
    assert rows_for({"op_a": {}, "op_b": {}, "op_c": {}}) == ROWS
    four = rows_for(FOUR)
    assert [e for e, _ in four] == ["E0", "E1a", "E1b", "E1c", "E1d", "E3", "E4"]


def test_asrs4_config_differs_from_asrs_only_where_intended():
    a = yaml.safe_load(open("configs/asrs-glimmer.yaml"))
    b = yaml.safe_load(open("configs/asrs4-glimmer.yaml"))
    assert b["fl"] == a["fl"] | {"operators": FOUR, "threshold": 3}
    assert (b["data_dir"], b["results_dir"]) == ("data/asrs4", "results/asrs4-glimmer")
    assert {k: v for k, v in a.items() if k not in ("fl", "data_dir", "results_dir")} == {
        k: v for k, v in b.items() if k not in ("fl", "data_dir", "results_dir")
    }
    pa = yaml.safe_load(open("configs/partitions/asrs.yaml"))
    pb = yaml.safe_load(open("configs/partitions/asrs4.yaml"))
    assert list(pb["operators"])[:3] == list(pa["operators"]) and list(pb["operators"])[3] == "op_d"
    assert {k: v for k, v in pa.items() if k not in ("operators", "out_dir")} == {
        k: v for k, v in pb.items() if k not in ("operators", "out_dir", "test_from")
    }


ASRS, ASRS4 = Path("data/asrs"), Path("data/asrs4")


@pytest.mark.skipif(not (ASRS4 / "stats.json").is_file(), reason="prepare data/asrs4 first")
def test_asrs4_shares_the_three_organisation_split_byte_for_byte():
    """What lets E0 and E1a–c be reused: same shards, same test set, same bytes."""
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    files = [f"{op}/{s}.jsonl" for op in ("op_a", "op_b", "op_c") for s in ("train", "val")] + ["test.jsonl"]
    for f in files:
        assert sha(ASRS / f) == sha(ASRS4 / f), f
    stats = json.loads((ASRS4 / "stats.json").read_text())
    assert stats["notes"]["held"]["op_d"] == [
        "equipment_failure",
        "traffic_conflict",
        "ground_conflict",
        "turbulence",
        "terrain_warning",
        "airspace_violation",
        "loss_of_control",
    ]  # blind spots: hazmat, passenger_misconduct
    rows = [json.loads(x) for s in ("train", "val") for x in open(ASRS4 / "op_d" / f"{s}.jsonl")]
    texts = {r["text"] for r in rows}
    test = {json.loads(x)["text"] for x in open(ASRS4 / "test.jsonl")}
    others = {json.loads(x)["text"] for f in files[:-1] for x in open(ASRS4 / f)}
    assert len(texts) == len(rows) == 1110 and not texts & test and not texts & others


def test_unmeasured_consortium_needs_opt_in():
    six = [f"op_{c}" for c in "abcdef"]
    assert rundir.configure({"fl": {"operators": six, "threshold": 5, "unverified": True}}) == six
