"""Bridge tests: a real stepwise key ceremony and encrypted rounds through the horizon-he CLI."""

from __future__ import annotations

import numpy as np
import pytest

from horizon.he_bridge import DEFAULT_BINARY, HEError, HorizonHE, read_f64, write_f64

PARTIES = ["op_a", "op_b", "op_c"]
pytestmark = pytest.mark.skipif(not DEFAULT_BINARY.is_file(), reason="run `make he` first")


@pytest.fixture(scope="module")
def he(tmp_path_factory):
    h = HorizonHE(tmp_path_factory.mktemp("run") / "run_bridge")
    steps = h.ceremony_stepwise(PARTIES)
    assert all(s["ok"] for s in steps)
    assert (h.run / "public" / "pk.bin").is_file()
    # sk_i erased after Shamir aggregation; threshold share kept.
    for p in PARTIES:
        assert not (h.run / "parties" / p / "secret" / "sk.bin").exists()
        assert (h.run / "parties" / p / "secret" / "tsk.bin").is_file()
    return h


def test_f64_roundtrip(tmp_path):
    v = np.array([0.0, -1.5, 1e-9, np.pi], dtype=np.float64)
    write_f64(tmp_path / "x.f64", v)
    assert np.array_equal(read_f64(tmp_path / "x.f64"), v)
    assert (tmp_path / "x.f64").stat().st_size == 32


PAIRS = [["op_a", "op_b"], ["op_a", "op_c"], ["op_b", "op_c"]]  # the offline party is the third


@pytest.mark.parametrize("round_,active", [(i + 1, pair) for i, pair in enumerate(PAIRS)])
def test_three_contributors_every_decrypting_pair(he, round_, active):
    """All three submit; each 2-of-3 pair decrypts its own round (one decryption per round)."""
    rng = np.random.default_rng(round_)
    n = 20_000  # > one ciphertext of slots (2^14), exercises the multi-ct path
    ups = {p: rng.uniform(-0.05, 0.05, n) for p in PARTIES}
    for p in PARTIES:
        r = he.encrypt(p, round_, ups[p])
        assert r["n_values"] == n and r["n_ciphertexts"] == 2
        assert not he.update_path(p, round_).exists(), "plaintext update must be deleted after encrypt"
    assert sorted(he.aggregate(round_)["contributors"]) == PARTIES
    he.request(round_, active)
    for p in active:
        he.dec_share(p, round_)
    he.dec_combine(round_)
    got = he.aggregate_values(round_)
    assert got.shape == (n,)
    assert np.max(np.abs(got - sum(ups.values()))) < 1e-5
    solo = he.dec_combine(round_, solo=active[0])  # one share alone decrypts to noise
    assert solo["rel_err_vs_true"] > 1e3


def test_one_shot_and_value_guard(he):
    # Round 1 was decrypted above: neither a second share nor a second request is allowed.
    with pytest.raises(HEError, match="one-shot"):
        he.dec_share("op_a", 1)
    with pytest.raises(HEError):
        he.request(1, ["op_b", "op_c"])
    # Values outside the |v| ≤ 4 guard are rejected by encrypt; the plaintext file still goes.
    with pytest.raises(HEError):
        he.encrypt("op_a", 10, np.array([5.0]))
    assert not he.update_path("op_a", 10).exists()


def test_skipped_round_is_refused_then_next_round_recovers(he):
    """A round needs all three submissions: with only two, each submitter could subtract its own
    update from the aggregate and read the other's."""
    rng = np.random.default_rng(4)
    for p in ["op_a", "op_b"]:  # op_c skips round 4
        he.encrypt(p, 4, rng.uniform(-0.05, 0.05, 1000))
    with pytest.raises(HEError, match="need ≥ 3"):
        he.aggregate(4)
    # Round 5: everyone submits fresh encryptions and the protocol carries on.
    ups = {p: rng.uniform(-0.05, 0.05, 1000) for p in PARTIES}
    for p, v in ups.items():
        he.encrypt(p, 5, v)
    he.aggregate(5)
    he.request(5, ["op_a", "op_c"])
    he.dec_share("op_a", 5)
    he.dec_share("op_c", 5)
    he.dec_combine(5)
    assert np.max(np.abs(he.aggregate_values(5) - sum(ups.values()))) < 1e-5


def test_single_contributor_refused(he):
    he.encrypt("op_a", 6, np.zeros(10))
    with pytest.raises(HEError, match="need ≥ 3"):
        he.aggregate(6)
