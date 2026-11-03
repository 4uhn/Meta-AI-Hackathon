"""Check this machine and (if configured) your Modal account, then print the next command to run.

    python3 -m horizon.doctor        # or: make doctor

Standard library only, so it runs before `make setup`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
BIN = ROOT / "crypto" / "bin"
GO_MIN = (1, 24)  # Lattigo v6.2


def run(cmd: list[str], timeout: float = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


class Report:
    def __init__(self) -> None:
        self.next: str | None = None

    def line(self, ok: bool | None, what: str, detail: str = "", fix: str | None = None) -> bool:
        mark = {True: "ok ", False: "!! ", None: " - "}[ok]
        print(f"  {mark} {what}" + (f"  ({detail})" if detail else ""))
        if ok is False and fix and self.next is None:
            self.next = fix
        return bool(ok)


def check_local(r: Report) -> dict:
    print("Local")
    go = shutil.which("go")
    m = re.search(r"go(\d+)\.(\d+)", run(["go", "version"])[1]) if go else None
    have = (int(m[1]), int(m[2])) if m else None
    r.line(
        bool(have and have >= GO_MIN),
        f"Go >= {GO_MIN[0]}.{GO_MIN[1]}",
        f"go {have[0]}.{have[1]}" if have else "not found",
        "install Go (https://go.dev/dl/), then: make he",
    )
    src = max(p.stat().st_mtime for p in (ROOT / "crypto").rglob("*.go"))
    for name in ("horizon-he", "horizon-he-linux-amd64"):
        b = BIN / name
        fresh = b.is_file() and b.stat().st_mtime >= src
        state = "built" if fresh else ("older than crypto/ sources" if b.is_file() else "missing")
        role = "this machine" if name == "horizon-he" else "Modal containers"
        r.line(fresh, f"crypto binary for {role}: crypto/bin/{name}", state, "make he")

    venv = VENV_PY.is_file()
    r.line(venv, "Python environment .venv", "" if venv else "missing", "make setup")
    mods = {}
    if venv:
        code = (
            "import importlib.util as u, json, sys; print(json.dumps([sys.version.split()[0]] + "
            "[u.find_spec(m) is not None for m in sys.argv[1:]]))"
        )
        names = ["numpy", "yaml", "torch", "transformers", "peft", "modal"]
        rc, out = run([str(VENV_PY), "-c", code, *names])
        found = json.loads(out.splitlines()[-1]) if rc == 0 else [None] + [False] * len(names)
        mods = dict(zip(names, found[1:], strict=True))
        train = all(mods[m] for m in ("torch", "transformers", "peft"))
        r.line(
            all(mods[m] for m in ("numpy", "yaml")), f"core packages (python {found[0]})", fix="make setup"
        )
        r.line(train, "training packages: torch, transformers, peft", fix="make setup")
        r.line(mods["modal"] or None, "Modal client", "" if mods["modal"] else "optional: make setup")

    for task, d in (("5G-NIDD", "data/processed"), ("ASRS", "data/asrs")):
        ok = (ROOT / d / "stats.json").is_file()
        hint = "make data" if task == "5G-NIDD" else "make data-asrs"
        r.line(ok or None, f"{task} shards in {d}/", "" if ok else f"optional: {hint}")
    free = shutil.disk_usage(ROOT).free / 1e9
    r.line(free >= 5, "free disk >= 5 GB", f"{free:.0f} GB")
    return mods


def check_modal(r: Report) -> None:
    prefix = os.environ.get("HORIZON_MODAL_PREFIX") or "horizon"
    secret = os.environ.get("HORIZON_HF_SECRET", "huggingface")
    print(f"Modal (prefix {prefix!r}; set HORIZON_MODAL_PREFIX to change)")
    code = (
        "import json, sys, modal\n"
        "vols = sorted(v.name for v in modal.Volume.objects.list())\n"
        "try:\n"
        "    modal.Secret.from_name(sys.argv[1]).hydrate(); hf = True\n"
        "except modal.exception.NotFoundError:\n"
        "    hf = False\n"
        "print(json.dumps({'volumes': vols, 'hf': hf}))\n"
    )
    rc, out = run([str(VENV_PY), "-c", code, secret], timeout=60)
    if rc != 0:
        r.line(None, "Modal account", "not signed in (optional, for GPU runs: modal setup)")
        return
    info = json.loads(out.splitlines()[-1])
    r.line(True, "Modal account", "signed in")
    vols = set(info["volumes"])
    sets = {
        "models (weights)": [f"{prefix}-models"],
        "5G-NIDD data": [f"{prefix}-eval"] + [f"{prefix}-data-op_{x}" for x in "abc"],
        "ASRS data": [f"{prefix}-eval-asrs"] + [f"{prefix}-data-asrs-op_{x}" for x in "abc"],
    }
    for what, need in sets.items():
        have = sum(v in vols for v in need)
        cfg = (
            "configs/5g-nidd-tiny-modal.yaml" if what.startswith("models") else "configs/5g-nidd-glimmer.yaml"
        )
        if what.startswith("ASRS"):
            cfg = "configs/asrs-glimmer.yaml"
        r.line(
            have == len(need) or None,
            f"volumes: {what}",
            f"{have}/{len(need)}"
            + ("" if have == len(need) else f"; python -m modal_app.provision --config {cfg}"),
        )
    r.line(
        info["hf"] or None,
        f"Hugging Face secret {secret!r}",
        "found" if info["hf"] else "only needed for gated models such as Llama; Muse Glimmer needs none",
    )


def main() -> None:
    r = Report()
    mods = check_local(r)
    if mods.get("modal"):
        check_modal(r)
    print()
    if r.next:
        print(f"Next: {r.next}")
    else:
        print("Next: .venv/bin/python examples/federated-fine-tuning/run.py   (local CPU, no account)")
        print("      examples/README.md lists every example; docs/run-on-modal.md what each Modal run costs")
    sys.exit(0 if r.next is None else 1)


if __name__ == "__main__":
    main()
