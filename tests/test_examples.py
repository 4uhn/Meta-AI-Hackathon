"""The examples run as documented (so they cannot silently break)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from horizon.he_bridge import DEFAULT_BINARY

ROOT = Path(__file__).resolve().parent.parent
needs_he = pytest.mark.skipif(not DEFAULT_BINARY.is_file(), reason="run `make he`")


def run(*args: str, cwd: Path = ROOT) -> str:
    p = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p.stdout


@needs_he
@pytest.mark.parametrize("n, t", [(3, 2), (4, 3)])
def test_secure_sum(n, t):
    out = run(
        "examples/secure-aggregation/secure_sum.py",
        "--parties",
        str(n),
        "--threshold",
        str(t),
        "--size",
        "3000",
    )
    assert f"key ceremony: {t}-of-{n}" in out and "sum matches the plaintext sum" in out and "(noise)" in out


@needs_he
@pytest.mark.slow
def test_custom_dataset_runs_with_a_config_defined_task(tmp_path):
    run("examples/custom-dataset/make_toy_data.py")
    run("-m", "horizon.model.tiny", "--out", "models/tiny")
    run(
        "-m", "horizon.fl.run", "--config", "examples/custom-dataset/config.yaml", "--mode", "encrypted",
        "--fresh", "--run-root", str(tmp_path / "runs"), "--out", str(tmp_path / "results"), "--rounds", "2",
    )  # fmt: skip
    e3 = json.loads((tmp_path / "results" / "E3.json").read_text())
    assert set(e3["final"]["metrics"]["per_class"]) == {"billing", "outage", "account", "shipping"}
    assert [len(r["contributors"]) for r in e3["rounds"]] == [3, 3]
