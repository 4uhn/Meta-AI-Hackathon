"""Modal backend, local side, without network: what each container receives (isolation) and where
its returned files land."""

from __future__ import annotations

import pytest

pytest.importorskip("modal")

from modal_app.backend import plan, write_outputs  # noqa: E402

CFG = "configs/5g-nidd-tiny-modal.yaml"


@pytest.fixture
def run(tmp_path):
    run = tmp_path / "runs" / "modal_tiny_E3"
    for rel in [
        "public/pk.bin",
        "public/params.json",
        "bus/round_2/global/round.json",
        "bus/round_2/global/adapter/adapter_model.safetensors",
        "bus/round_1/global/round.json",
        "parties/op_b/secret/ed25519.key",
        "parties/op_b/secret/tsk.bin",
    ]:
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_bytes(rel.encode())
    return run


def train_args(run, op, extra=()):
    return [
        "horizon.fl.operator",
        "train",
        "--run",
        str(run),
        "--op",
        op,
        "--round",
        "2",
        "--config",
        CFG,
        *extra,
    ]


def test_operator_gets_only_its_shard_round_and_public_files(run):
    p = plan(run, train_args(run, "op_b"), CFG)
    assert p["mounts"] == {"/vol/horizon-data-op_b": "horizon-data-op_b"}
    assert p["signer"] == "op_b"
    assert sorted(p["job"]["files"]) == [
        "bus/round_2/global/adapter/adapter_model.safetensors",
        "bus/round_2/global/round.json",
        "public/params.json",
        "public/pk.bin",
    ]  # no other round, and nothing from parties/ (tsk.bin stays local)
    assert p["job"]["args"] == [
        "train",
        "--run",
        "{RUN}",
        "--op",
        "op_b",
        "--round",
        "2",
        "--config",
        "{CFG}",
    ]
    assert p["job"]["data_links"] == {"op_b": "/vol/horizon-data-op_b"}


def test_plaintext_reference_gets_no_key_material(run):
    p = plan(run, train_args(run, "op_a", ["--insecure-reference"]), CFG)
    assert p["signer"] is None and not any(k.startswith("public/") for k in p["job"]["files"])
    assert p["mounts"] == {"/vol/horizon-data-op_a": "horizon-data-op_a"}


def test_evaluator_and_baselines_mounts(run):
    ev = plan(run, ["horizon.fl.evaluate", "--run", str(run), "--round", "2", "--config", CFG], CFG)
    assert ev["mounts"] == {"/vol/horizon-eval": "horizon-eval"} and ev["signer"] is None
    e1b = plan(run, ["horizon.fl.baseline", "--exp", "E1b", "--config", CFG], CFG)
    assert set(e1b["mounts"].values()) == {"horizon-data-op_b", "horizon-eval"}
    e4 = plan(run, ["horizon.fl.baseline", "--exp", "E4", "--config", CFG], CFG)
    assert set(e4["mounts"].values()) == {
        "horizon-data-op_a",
        "horizon-data-op_b",
        "horizon-data-op_c",
        "horizon-eval",
    }
    with pytest.raises(ValueError):  # decryption shares are never remote
        plan(
            run, ["horizon.fl.operator", "dec-share", "--run", str(run), "--op", "op_a", "--round", "1"], CFG
        )


def test_write_outputs_placement_and_refusals(run, tmp_path):
    pre = f"runs/{run.name}/"
    (run / "events.jsonl").write_text('{"a":1}\n')
    out = tmp_path / "results"
    written = write_outputs(
        run,
        {
            pre + "bus/round_2/submissions/op_b/ct.bin": b"ct",
            pre + "events.jsonl": b'{"b":2}\n',
            "results/E1a.json": b"{}",
            "runs/dev_E1a/scores.npy": b"s",
            "data/ignored": b"x",
        },
        out,
        run,
    )
    assert (run / "bus/round_2/submissions/op_b/ct.bin").read_bytes() == b"ct"
    assert (run / "events.jsonl").read_text() == '{"a":1}\n{"b":2}\n'  # appended, not replaced
    assert (out / "E1a.json").is_file() and (run / "dev_E1a" / "scores.npy").is_file()
    assert len(written) == 3
    with pytest.raises(AssertionError):
        write_outputs(run, {pre + "parties/op_b/secret/ed25519.key": b"k"}, out, run)
    with pytest.raises(FileExistsError):  # a remote step never overwrites local run files
        write_outputs(run, {pre + "public/pk.bin": b"evil"}, out, run)


def test_last_json_tolerates_multiline_summaries():
    from modal_app.backend import last_json

    assert last_json('{"ok": 1}\n{\n "exp": "E1a"\n}\ntotal 3s\n') == {"ok": 1}
    assert last_json("{\n}\n") == {}


def test_federated_val_eval_runs_as_the_operator(run):
    p = plan(
        run,
        [
            "horizon.fl.evaluate",
            "--run",
            str(run),
            "--round",
            "2",
            "--config",
            CFG,
            "--split",
            "val",
            "--op",
            "op_c",
        ],
        CFG,
    )
    assert p["mounts"] == {"/vol/horizon-data-op_c": "horizon-data-op_c"}  # its own shard, no test set
    assert p["job"]["data_links"] == {"op_c": "/vol/horizon-data-op_c"} and p["signer"] is None
