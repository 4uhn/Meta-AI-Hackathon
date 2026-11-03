"""Threshold-encrypted aggregation of plain vectors, with no model: the layer Horizon's FL loop uses.

N organisations each hold a private vector (in federated learning: a model update). The server adds up
their ciphertexts and never sees a plaintext; any t organisations together decrypt the sum, and nothing
else. Every step below runs as its own `horizon-he` process acting as exactly one party, with that
party's secrets in its own directory.

    python examples/secure-aggregation/secure_sum.py                        # 2-of-3, 100k values
    python examples/secure-aggregation/secure_sum.py --parties 5 --threshold 4 --size 4685825

`--size 4685825` is one Muse Glimmer-30B LoRA update (every B value plus its weight). Needs `make setup`
and `make he`.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from horizon.he_bridge import HorizonHE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parties", type=int, default=3, help="N organisations (3 to 5 are verified)")
    ap.add_argument("--threshold", type=int, default=2, help="t: how many must cooperate to decrypt")
    ap.add_argument("--size", type=int, default=100_000, help="values per update")
    ap.add_argument("--keep", help="keep the run directory here instead of a temp dir")
    a = ap.parse_args()
    names = [f"op_{chr(ord('a') + i)}" for i in range(a.parties)]
    run = Path(a.keep or tempfile.mkdtemp(prefix="horizon-secure-sum-"))
    he = HorizonHE(run)
    t0 = time.time()

    # 1. Setup, once: every party makes its own secret key and deals Shamir shares of it to the others;
    #    the public key is the combination of their public shares. No one ever holds the whole key.
    he.ceremony_stepwise(names, a.threshold)
    print(f"key ceremony: {a.threshold}-of-{a.parties}, no trusted dealer  ({time.time() - t0:.1f} s)")

    # 2. Each party encrypts its private vector under the collective public key. Values must lie in
    #    [-4, 4] (model updates are far smaller); the plaintext file never leaves the party's directory.
    rng = np.random.default_rng(0)
    updates = {p: rng.uniform(-0.01, 0.01, a.size) for p in names}
    for p in names:
        he.encrypt(p, 1, updates[p])
    print(f"{a.parties} parties encrypted {a.size:,} values each")

    # 3. The server checks each signed submission and adds the ciphertexts. It holds no key share.
    he.aggregate(1)

    # 4. The server names which t parties decrypt, once. Each one recomputes the sum itself, refuses an
    #    aggregate with too few contributors, and releases one decryption share per round, never more.
    active = names[: a.threshold]
    he.request(1, active)
    for p in active:
        he.dec_share(p, 1)
    he.dec_combine(1)
    total = he.aggregate_values(1)[: a.size]
    err = float(np.max(np.abs(total - sum(updates.values()))))
    print(f"decrypted by {', '.join(active)}: sum matches the plaintext sum, max abs error {err:.1e}")

    # 5. Fewer than t shares reveal nothing useful: the "decryption" is noise.
    solo = he.dec_combine(1, solo=",".join(active[:-1]))
    print(f"{a.threshold - 1} share(s) only: relative error {solo['rel_err_vs_true']:.1e} (noise)")

    if not a.keep:
        shutil.rmtree(run)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
