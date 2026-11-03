"""Modal names follow HORIZON_MODAL_PREFIX (default "horizon"). GPU launches ask first."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("modal")

NAMES = (
    "import json; from modal_app import app as a; print(json.dumps([a.APP_NAME, a.MODELS_VOL, a.EVAL_VOL,"
    " a.data_vols(), a.data_vols('asrs', ['op_d']), a.eval_vol('asrs'), a._prefix_env()]))"
)


def names(prefix: str | None) -> list:
    env = {k: v for k, v in os.environ.items() if k != "HORIZON_MODAL_PREFIX"}
    if prefix is not None:
        env["HORIZON_MODAL_PREFIX"] = prefix
    out = subprocess.run([sys.executable, "-c", NAMES], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_default_prefix_keeps_the_original_names():
    assert names(None) == [
        "horizon",
        "horizon-models",
        "horizon-eval",
        {"op_a": "horizon-data-op_a", "op_b": "horizon-data-op_b", "op_c": "horizon-data-op_c"},
        {"op_d": "horizon-data-asrs-op_d"},
        "horizon-eval-asrs",
        {},  # the default image gets no extra env
    ]


def test_custom_prefix_renames_everything_and_reaches_the_containers():
    app, models, ev, nidd, asrs, ev_asrs, env = names("acme-fl")
    assert (app, models, ev, ev_asrs) == ("acme-fl", "acme-fl-models", "acme-fl-eval", "acme-fl-eval-asrs")
    assert nidd["op_b"] == "acme-fl-data-op_b" and asrs == {"op_d": "acme-fl-data-asrs-op_d"}
    assert env == {"HORIZON_MODAL_PREFIX": "acme-fl"}


@pytest.mark.parametrize("bad", ["Acme", "a b", "-x", "x" * 21])
def test_bad_prefix_is_refused(bad):
    with pytest.raises(subprocess.CalledProcessError):
        names(bad)


def test_gpu_launch_needs_confirmation(monkeypatch, tmp_path):
    from modal_app import backend

    monkeypatch.delenv("HORIZON_YES", raising=False)
    monkeypatch.setattr(sys, "stdin", open(os.devnull))  # not a terminal
    with pytest.raises(SystemExit, match="HORIZON_YES=1"):
        backend.confirm_gpu("configs/5g-nidd-llama.yaml", "A10")
    monkeypatch.setenv("HORIZON_YES", "1")
    backend.confirm_gpu("configs/5g-nidd-llama.yaml", "A10")  # no prompt, no error
