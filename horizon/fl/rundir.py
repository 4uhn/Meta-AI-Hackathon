"""Run-directory paths shared by the orchestrator, operators, aggregator and evaluator.

Crypto artefacts (public/, parties/, bus/round_r/{submissions,aggregate,decrypt}) are owned by `horizon-he`.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_OPERATORS = ("op_a", "op_b", "op_c")
DEFAULT_THRESHOLD = 2
MAX_OPERATORS = 7  # the crypto layer's limit
MAX_VERIFIED = 5  # the largest consortium measured at the Glimmer payload

# The consortium of this process, set from the config by `configure` (in place, so that
# `from horizon.fl.rundir import OPERATORS` sees it). Without `fl.operators` it is 2-of-3.
OPERATORS = list(DEFAULT_OPERATORS)
_THRESHOLD = [DEFAULT_THRESHOLD]


def configure(cfg: dict | None) -> list[str]:
    """Make the config's consortium (`fl.operators`, `fl.threshold`) the active one in this process.

    Operators are op_a, op_b, … in order (they get Shamir points 1..N in that order), with
    2 ≤ t ≤ N − 1 and 2t > N, as the crypto layer requires. N ≤ 5 has been measured; 6 or 7
    needs `fl.unverified: true` (run `horizon-he check --parties N --threshold t --unverified`)."""
    fl = (cfg or {}).get("fl") or {}
    ops = list(fl.get("operators", DEFAULT_OPERATORS))
    t = int(fl.get("threshold", DEFAULT_THRESHOLD))
    n = len(ops)
    if ops != [f"op_{chr(ord('a') + i)}" for i in range(n)]:
        raise ValueError(f"fl.operators must be op_a, op_b, … in order, got {ops}")
    if not (3 <= n <= MAX_OPERATORS and 2 <= t <= n - 1 and 2 * t > n):
        raise ValueError(f"fl: {t}-of-{n} unsupported (need 3 ≤ N ≤ 7, 2 ≤ t ≤ N − 1, 2t > N)")
    if n > MAX_VERIFIED and not fl.get("unverified"):
        raise ValueError(
            f"fl: N = {n} has not been measured (N ≤ {MAX_VERIFIED} has). Run `horizon-he check --parties {n} "
            f"--threshold {t} --unverified`, then set fl.unverified: true"
        )
    OPERATORS[:] = ops
    _THRESHOLD[0] = t
    return OPERATORS


def threshold() -> int:
    """t: how many operators must take part in a decryption."""
    return _THRESHOLD[0]


def min_contributors() -> int:
    """Fewest submissions a round needs to be decrypted: t + 1 (≥ 3), so any t − 1 colluding
    contributors are left with a sum of at least two other updates."""
    return max(3, threshold() + 1)


def default_active() -> list[str]:
    """The decrypting set when nobody is offline: the first t operators."""
    return OPERATORS[: threshold()]


def global_dir(run: Path, round_: int) -> Path:
    """The global adapter operators train from in `round_` (round 1 = the seeded initial adapter)."""
    return Path(run) / "bus" / f"round_{round_}" / "global"


def adapter_dir(run: Path, round_: int) -> Path:
    return global_dir(run, round_) / "adapter"


def round_info(run: Path, round_: int) -> dict:
    return json.loads((global_dir(run, round_) / "round.json").read_text())


def submission_dir(run: Path, round_: int, op: str) -> Path:
    return Path(run) / "bus" / f"round_{round_}" / "submissions" / op


def insecure_update(run: Path, round_: int, op: str) -> Path:
    """Plaintext update path; used only by reference runs under `--insecure-reference`."""
    return Path(run) / "insecure" / f"round_{round_}" / f"{op}.f64"


def eval_dir(run: Path) -> Path:
    return Path(run) / "eval"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
