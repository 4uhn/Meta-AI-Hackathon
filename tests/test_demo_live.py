"""`make demo` engine: live re-decryption of a recorded round on a small real-CLI run."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from horizon.demo import __main__ as demo_cli
from horizon.demo.live import Live, stage, summary
from horizon.he_bridge import DEFAULT_BINARY, HorizonHE

PARTIES = ["op_a", "op_b", "op_c"]
pytestmark = pytest.mark.skipif(not DEFAULT_BINARY.is_file(), reason="run `make he` first")


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A finished 1-round encrypted run, decrypted by op_a + op_b (like the real one)."""
    he = HorizonHE(tmp_path_factory.mktemp("rec") / "run_rec")
    he.ceremony_stepwise(PARTIES)
    rng = np.random.default_rng(7)
    ups = {p: rng.uniform(-0.05, 0.05, 20_000) for p in PARTIES}
    for p in PARTIES:
        he.encrypt(p, 1, ups[p])
    he.aggregate(1)
    he.request(1, ["op_a", "op_b"])
    for p in ("op_a", "op_b"):
        he.dec_share(p, 1)
    he.dec_combine(1)
    return he.run, sum(ups.values())


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_stage_rewinds_only_that_round_and_leaves_offline_key_out(recorded, tmp_path):
    run, _ = recorded
    ws = stage(run, 1, tmp_path, offline=["op_b"])
    assert ws.name == run.name  # signed statements bind the run id
    assert not (ws / "parties" / "op_b" / "secret").exists()
    for op in ("op_a", "op_c"):
        assert (ws / "parties" / op / "secret" / "tsk.bin").is_file()
    a_src = (run / "parties" / "op_a" / "state" / "decrypt_log.jsonl").read_text()
    assert '"round":1' in a_src.replace(" ", "")
    assert (ws / "parties" / "op_a" / "state" / "decrypt_log.jsonl").read_text() == ""
    assert json.loads((ws / "staged.json").read_text())["rewound_log_entries"] == {"op_a": 1}


def test_live_round_matches_recording_and_refuses_every_attack(recorded, tmp_path):
    run, true_sum = recorded
    before = snapshot(run)
    live = Live(run, 1, tmp_path / "live", ["op_a", "op_c"], ["op_b"])
    live.run()
    res = summary(live, DEFAULT_BINARY)

    assert snapshot(run) == before, "the recorded run must never be written"
    assert res["aggregate"]["identical"]
    assert res["key_shares_present"] == ["op_a", "op_c"]
    assert res["decrypted"]["max_abs_diff_vs_recorded"] < 1e-5
    ws = tmp_path / "live" / run.name
    got = np.fromfile(ws / "bus" / "round_1" / "decrypt" / "agg.f64", dtype="<f8")
    assert np.max(np.abs(got - true_sum)) < 1e-5
    assert res["decrypted"]["solo_rel_err"] > 1e3  # one key share alone is noise
    assert res["published"] is None  # no adapter in this run
    reasons = {a["attack"]: a["refusal"] for a in res["attacks"]}
    assert "recomputed aggregate" in reasons["fake_sum"]
    assert "contributors, need" in reasons["two_party_sum"]
    assert "one-shot" in reasons["second_decryption"]
    assert res["all_attacks_refused"]
    # The honest decryption is logged once, by the live quorum only.
    log = (ws / "parties" / "op_a" / "state" / "decrypt_log.jsonl").read_text().splitlines()
    assert [json.loads(x)["active"] for x in log] == [["op_a", "op_c"]]


def test_cli_narrates_and_writes_results(recorded, tmp_path, capsys):
    run, _ = recorded
    rc = demo_cli.main(
        [
            "--run",
            str(run),
            "--pace",
            "0",
            "--root",
            str(tmp_path / "ws"),
            "--out",
            str(tmp_path / "out"),
            "--no-color",
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "3 cheating attempts, 3 refused" in text
    assert "Operator B's key isn't even on this machine" in text
    assert not any((tmp_path / "ws").iterdir()), "workspace removed afterwards"
    (out,) = (tmp_path / "out").glob("live_*.json")
    assert json.loads(out.read_text())["all_attacks_refused"]
