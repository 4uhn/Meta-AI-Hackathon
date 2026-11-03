"""Thin wrapper around the `horizon-he` Go CLI (crypto/cmd/horizon-he).

Python never touches key material: each protocol step is a separate `horizon-he` process acting as exactly
one party. Each call prints one JSON line; `ok=false` or a non-zero exit raises `HEError`.

Plaintext updates cross the boundary as little-endian float64 (`.f64`), staged in the operator's private
`parties/<op>/state/` and deleted once `encrypt` returns, so plaintext B never reaches the bus.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# The native build, or on linux/amd64 the static one that containers also use (`make he` builds both).
_LINUX_AMD64 = sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64")
DEFAULT_BINARY = REPO_ROOT / "crypto" / "bin" / ("horizon-he-linux-amd64" if _LINUX_AMD64 else "horizon-he")
AGGREGATOR = "aggregator"


class HEError(RuntimeError):
    """A horizon-he step failed. `.result` holds the parsed JSON (if any)."""

    def __init__(self, cmd: str, message: str, result: dict | None = None):
        super().__init__(f"horizon-he {cmd}: {message}")
        self.cmd = cmd
        self.result = result or {}


def write_f64(path: Path, values: np.ndarray) -> None:
    arr = np.ascontiguousarray(values, dtype="<f8").ravel()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    arr.tofile(path)


def read_f64(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


class HorizonHE:
    """One run directory, driven through the CLI. Methods name the acting party explicitly."""

    def __init__(self, run_dir: str | Path, binary: str | Path | None = None, timeout: float = 600.0):
        self.run = Path(run_dir).resolve()
        self.binary = Path(binary or os.environ.get("HORIZON_HE", DEFAULT_BINARY))
        if not self.binary.is_file():
            raise FileNotFoundError(f"{self.binary} not found; run `make he`")
        self.timeout = timeout

    def call(self, cmd: str, **flags: object) -> dict:
        argv = [str(self.binary), cmd, "--run", str(self.run)]
        for k, v in flags.items():
            if v is None or v is False:
                continue
            argv += [f"--{k.replace('_', '-')}", str(v)]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            raise HEError(
                cmd, f"no JSON on stdout (exit {proc.returncode}); stderr: {proc.stderr.strip()[-500:]}"
            )
        try:
            res = json.loads(lines[-1])
        except json.JSONDecodeError as e:
            raise HEError(cmd, f"unparseable output {lines[-1][:200]!r}") from e
        if proc.returncode != 0 or not res.get("ok", False):
            raise HEError(cmd, str(res.get("error", f"exit {proc.returncode}")), res)
        return res

    def init(self, parties: list[str] | None = None, threshold: int | None = None) -> dict:
        """Run setup; the party flags are passed only for a consortium other than 2-of-3."""
        if parties is None or (list(parties) == ["op_a", "op_b", "op_c"] and threshold in (None, 2)):
            return self.call("init")
        return self.call("init", parties=",".join(parties), threshold=threshold)

    def keygen(self, party: str) -> dict:
        return self.call("keygen", party=party)

    def pk_share(self, party: str) -> dict:
        return self.call("pk-share", party=party)

    def shamir_deal(self, party: str) -> dict:
        return self.call("shamir-deal", party=party)

    def shamir_agg(self, party: str) -> dict:
        return self.call("shamir-agg", party=party)

    def pk_combine(self) -> dict:
        return self.call("pk-combine")

    def ceremony_stepwise(self, parties: list[str], threshold: int | None = None) -> list[dict]:
        """Full setup, one process per (party, step), in protocol order. Returns every step's JSON."""
        out = [self.init(parties, threshold)]
        for step in (self.keygen, self.pk_share, self.shamir_deal, self.shamir_agg):
            out += [step(p) for p in parties]
        out.append(self.pk_combine())
        return out

    def update_path(self, party: str, round_: int) -> Path:
        """Where `party` stages its plaintext update: inside its own private area."""
        return self.run / "parties" / party / "state" / f"update_round_{round_}.f64"

    def encrypt(self, party: str, round_: int, values: np.ndarray) -> dict:
        """Operator `party` encrypts `values` under the collective key. The plaintext file is deleted."""
        p = self.update_path(party, round_)
        write_f64(p, values)
        try:
            return self.call("encrypt", party=party, round=round_, **{"in": p})
        finally:
            p.unlink(missing_ok=True)

    def aggregate(self, round_: int) -> dict:
        return self.call("aggregate", round=round_)

    def request(self, round_: int, active: list[str]) -> dict:
        return self.call("request", round=round_, active=",".join(active))

    def dec_share(self, party: str, round_: int) -> dict:
        return self.call("dec-share", party=party, round=round_)

    def dec_combine(self, round_: int, solo: str | None = None) -> dict:
        return self.call("dec-combine", round=round_, solo=solo)

    def aggregate_values(self, round_: int) -> np.ndarray:
        return read_f64(self.run / "bus" / f"round_{round_}" / "decrypt" / "agg.f64")
