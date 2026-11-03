"""Live re-decryption of a recorded encrypted round, on its real ciphertexts.

The workspace is a copy-on-write clone of the recorded run at the moment that round's
signed submissions had arrived: public parameters and key, the submissions, the published global
adapter, and each online operator's key share and one-shot log *as it stood before the round was
decrypted*. The offline operator's key share is not copied at all, so decryption visibly works
without it. Every protocol step is a separate `horizon-he` process acting as exactly one party;
the "server tries to cheat" steps write files the way a malicious aggregator would and then ask
the honest operator binary to cooperate. The recorded run is only ever read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from horizon.events import emit
from horizon.he_bridge import AGGREGATOR, HEError, HorizonHE, read_f64

SAMPLE = 4  # values shown from the noise / true decryptions


def _clone(src: Path, dst: Path) -> None:
    """Copy a file or tree; on APFS `cp -c` makes instant copy-on-write clones."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        r = subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True)
        if r.returncode == 0:
            return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parties_of(run: Path) -> list[str]:
    return [p["name"] for p in _read(run / "public" / "parties.json")["parties"]]


def signed_statement(run: Path, round_: int, op: str) -> dict:
    """The operator's signed submission statement (the signature is checked by the CLI, not here)."""
    return _read(run / "bus" / f"round_{round_}" / "submissions" / op / "statement.json")["statement"]


def recorded_request(run: Path, round_: int) -> dict:
    return _read(run / "bus" / f"round_{round_}" / "decrypt" / "request.json")


def stage(src: Path, round_: int, root: Path, offline: list[str]) -> Path:
    """Build the workspace `root/<run name>` for a live re-decryption of `round_` (see module doc).

    The directory keeps the run's name because signed statements bind the run id."""
    src = Path(src)
    ws = Path(root) / src.name
    if ws.exists():
        raise FileExistsError(ws)
    rd = f"bus/round_{round_}"
    for rel in ("public", f"{rd}/submissions", f"{rd}/global"):
        if (src / rel).exists():
            _clone(src / rel, ws / rel)
    if (src / "config.yaml").is_file():
        _clone(src / "config.yaml", ws / "config.yaml")
    rewound = {}
    for op in parties_of(src):
        (ws / "parties" / op / "state").mkdir(parents=True)
        if op in offline:
            continue  # an offline operator's machine is simply not here
        _clone(src / "parties" / op / "secret", ws / "parties" / op / "secret")
        log = src / "parties" / op / "state" / "decrypt_log.jsonl"
        if log.is_file():
            lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
            keep = [ln for ln in lines if json.loads(ln)["round"] != round_]
            (ws / "parties" / op / "state" / "decrypt_log.jsonl").write_text("".join(k + "\n" for k in keep))
            rewound[op] = len(lines) - len(keep)
    info = {"source_run": str(src), "round": round_, "offline": offline, "rewound_log_entries": rewound}
    (ws / "staged.json").write_text(json.dumps(info, indent=2) + "\n")
    return ws


# Attacks: a malicious aggregator writes files directly, then asks an honest operator.
def _forge_request(ws: Path, round_: int, active: list[str], contributors: list[str], agg_sha: str) -> None:
    d = ws / "bus" / f"round_{round_}" / "decrypt"
    d.mkdir(parents=True, exist_ok=True)
    q = {
        "run_id": ws.name,
        "round": round_,
        "active": active,
        "contributors": contributors,
        "agg_sha256": agg_sha,
    }
    (d / "request.json").write_text(json.dumps(q, indent=2) + "\n")


def _refusal(he: HorizonHE, party: str, round_: int) -> str | None:
    """The operator's refusal message, or None if it (wrongly) released a share."""
    try:
        he.dec_share(party, round_)
    except HEError as e:
        return str(e.result.get("error") or e)
    return None


def attack_fake_sum(ws: Path, round_: int, active: list[str], victim: str, asked: str) -> dict:
    """Label one operator's ciphertext as 'the aggregate' and ask for its decryption."""
    sub = signed_statement(ws, round_, victim)
    agg_dir = ws / "bus" / f"round_{round_}" / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    _clone(ws / "bus" / f"round_{round_}" / "submissions" / victim / "ct.bin", agg_dir / "ct_agg.bin")
    _forge_request(ws, round_, active, parties_of(ws), sub["ct_sha256"])
    return {
        "attack": "fake_sum",
        "victim": victim,
        "asked": asked,
        "refusal": _refusal(HorizonHE(ws), asked, round_),
    }


def attack_two_party_sum(ws: Path, round_: int, active: list[str], pair: list[str], asked: str) -> dict:
    """Ask for the decryption of a sum of only two operators (each would learn the other's update)."""
    _forge_request(ws, round_, active, pair, "0" * 64)
    return {
        "attack": "two_party_sum",
        "contributors": pair,
        "asked": asked,
        "refusal": _refusal(HorizonHE(ws), asked, round_),
    }


def attack_second_decryption(ws: Path, round_: int, active: list[str], asked: str, agg_sha: str) -> dict:
    """After the round is decrypted, ask again with a different quorum (to difference two sums)."""
    _forge_request(ws, round_, active, parties_of(ws), agg_sha)
    return {
        "attack": "second_decryption",
        "active": active,
        "asked": asked,
        "refusal": _refusal(HorizonHE(ws), asked, round_),
    }


def _preload_publish() -> None:
    """Import the publish path (torch) once. Silences torch's macOS-only redirect notice."""
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    import horizon.fl.aggregator  # noqa: F401
    import horizon.model.lora  # noqa: F401


@dataclass
class Beat:
    key: str
    data: dict
    ms: float


@dataclass
class Live:
    source: Path
    round_: int
    root: Path
    active: list[str]
    offline: list[str]
    on_beat: Callable[[Beat], None] = lambda b: None
    on_start: Callable[[str, dict], None] = lambda key, ctx: None
    beats: list[Beat] = field(default_factory=list)

    def _beat(self, key: str, t0: float, **data: object) -> Beat:
        b = Beat(key, dict(data), 1000 * (time.perf_counter() - t0))
        self.beats.append(b)
        self.on_beat(b)
        return b

    def sandbox(self, ws: Path, name: str) -> Path:
        dst = self.root / f"_attack_{name}" / ws.name
        _clone(ws, dst)
        return dst

    def run(self) -> dict:
        # The publish step needs torch, which is slow to import: load it while the crypto runs.
        threading.Thread(target=_preload_publish, daemon=True).start()
        src, r = Path(self.source), self.round_
        rec = recorded_request(src, r)
        rec_agg = _read(src / "bus" / f"round_{r}" / "aggregate" / "aggregate.json")

        t = time.perf_counter()
        ws = stage(src, r, self.root, self.offline)
        subs = {}
        for op in parties_of(ws):
            st = signed_statement(ws, r, op)
            ct = ws / "bus" / f"round_{r}" / "submissions" / op / "ct.bin"
            subs[op] = {"bytes": ct.stat().st_size, "ct_sha256": st["ct_sha256"], "n_values": st["n_values"]}
        self._beat(
            "staged",
            t,
            workspace=str(ws),
            submissions=subs,
            recorded_active=rec["active"],
            offline=self.offline,
            key_shares_present=sorted(
                p.name for p in (ws / "parties").iterdir() if (p / "secret" / "tsk.bin").is_file()
            ),
        )
        # Sandboxes for the pre-decryption attacks are cloned now, before any request exists.
        sb_fake, sb_pair = self.sandbox(ws, "fake_sum"), self.sandbox(ws, "two_party")

        he = HorizonHE(ws)
        self.on_start("aggregate", {})
        t = time.perf_counter()
        agg = he.aggregate(r)
        emit(ws, AGGREGATOR, "aggregated", r, n_subs=len(agg["contributors"]), ms=agg["sum_ms"], live=True)
        self._beat(
            "aggregated",
            t,
            agg_sha256=agg["agg_sha256"],
            recorded_agg_sha256=rec_agg["agg_sha256"],
            identical=agg["agg_sha256"] == rec_agg["agg_sha256"],
            contributors=agg["contributors"],
            sum_ms=agg["sum_ms"],
            bytes=agg["bytes"],
        )

        asked = self.active[0]
        victim = next(p for p in parties_of(ws) if p not in self.active)
        self.on_start("attack", {"attack": "fake_sum", "victim": victim, "asked": asked})
        t = time.perf_counter()
        self._beat("attack", t, **attack_fake_sum(sb_fake, r, self.active, victim, asked))
        pair = sorted(set(parties_of(ws)) - {self.active[-1]})[:2]
        self.on_start("attack", {"attack": "two_party_sum", "contributors": pair, "asked": asked})
        t = time.perf_counter()
        self._beat("attack", t, **attack_two_party_sum(sb_pair, r, self.active, pair, asked))

        self.on_start("request", {"active": self.active, "offline": self.offline})
        t = time.perf_counter()
        he.request(r, self.active)
        emit(ws, AGGREGATOR, "decrypt_requested", r, active=self.active, offline=self.offline, live=True)
        self._beat("requested", t, active=self.active, offline=self.offline)

        first, rest = self.active[0], self.active[1:]
        self.on_start("share", {"party": first})
        t = time.perf_counter()
        s = he.dec_share(first, r)
        self._beat("share", t, party=first, verify_ms=s["verify_ms"], share_ms=s["share_ms"])
        self.on_start("solo", {"party": first})
        t = time.perf_counter()
        he.dec_combine(r, solo=first)
        solo = read_f64(ws / "bus" / f"round_{r}" / "decrypt" / f"solo_{first}.f64")
        self._beat("solo", t, party=first, sample=solo[:SAMPLE].tolist())
        for op in rest:
            self.on_start("share", {"party": op})
            t = time.perf_counter()
            s = he.dec_share(op, r)
            self._beat("share", t, party=op, verify_ms=s["verify_ms"], share_ms=s["share_ms"])
        self.on_start("combine", {"active": self.active})
        t = time.perf_counter()
        comb = he.dec_combine(r)
        live_agg = he.aggregate_values(r)
        rel = he.dec_combine(r, solo=first)["rel_err_vs_true"]
        emit(
            ws, AGGREGATOR, "decrypted", r, ms=comb["combine_ms"], weight_slot=comb["weight_slot"], live=True
        )
        emit(ws, AGGREGATOR, "solo_decrypt_attempt", r, party=first, rel_err=rel, live=True)
        rec_vals = src / "bus" / f"round_{r}" / "decrypt" / "agg.f64"
        diff = float(np.max(np.abs(live_agg - read_f64(rec_vals)))) if rec_vals.is_file() else None
        self._beat(
            "decrypted",
            t,
            sample=live_agg[:SAMPLE].tolist(),
            solo_rel_err=rel,
            weight_slot=comb["weight_slot"],
            n_values=int(live_agg.size),
            max_abs_diff_vs_recorded=diff,
        )

        pub = self._publish(ws, src, r, live_agg)

        again = [p for p in parties_of(ws) if p not in self.offline][:1] + self.offline[:1]
        self.on_start("attack", {"attack": "second_decryption", "active": again, "asked": asked})
        t = time.perf_counter()
        self._beat("attack", t, **attack_second_decryption(ws, r, again, asked, agg["agg_sha256"]))
        return {"workspace": ws, "published": pub}

    def _publish(self, ws: Path, src: Path, r: int, live_agg: np.ndarray) -> dict | None:
        """Publish round r+1's adapter from the live decryption and compare it with the recorded one."""
        rec_dir = src / "bus" / f"round_{r + 1}" / "global" / "adapter"
        if not (ws / "bus" / f"round_{r}" / "global" / "adapter").is_dir() or not rec_dir.is_dir():
            return None
        _preload_publish()  # normally already done by the background thread
        from horizon.fl.aggregator import next_global_b, publish
        from horizon.model.lora import read_b_file

        self.on_start("publish", {})
        t = time.perf_counter()
        b_live = next_global_b(ws, r, live_agg)
        publish(ws, r, b_live)
        b_rec = read_b_file(rec_dir)
        out = {
            "round": r + 1,
            "max_abs_diff_vs_recorded": float(np.max(np.abs(b_live - b_rec))),
            "max_abs_b": float(np.max(np.abs(b_rec))),
            "n_b_values": int(b_rec.size),
        }
        self._beat("published", t, **out)
        return out


def _display_path(p: Path) -> str:
    """The path relative to the working directory when it is inside it (no home paths in results/)."""
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return Path(p).name


def summary(live: Live, binary: Path) -> dict:
    """Everything the live round measured, for results/demo/ (numbers on screen come from here)."""
    by = {}
    for b in live.beats:
        by.setdefault(b.key, []).append(b)
    crypto_ms = sum(
        b.ms for b in live.beats if b.key in ("aggregated", "requested", "share", "solo", "decrypted")
    )
    one = lambda k: by[k][0].data if k in by else None  # noqa: E731
    return {
        "kind": "live_redecryption",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_run": live.source.name,
        "round": live.round_,
        "recorded_active": one("staged")["recorded_active"],
        "live_active": live.active,
        "offline": live.offline,
        "key_shares_present": one("staged")["key_shares_present"],
        "submissions": one("staged")["submissions"],
        "aggregate": one("aggregated"),
        "solo": one("solo"),
        "decrypted": one("decrypted"),
        "published": one("published"),
        "attacks": [b.data for b in by.get("attack", [])],
        "all_attacks_refused": all(b.data["refusal"] for b in by.get("attack", [])),
        "step_ms": [{"step": b.key, "party": b.data.get("party"), "ms": round(b.ms, 1)} for b in live.beats],
        "live_crypto_ms": round(crypto_ms, 1),
        "binary": _display_path(binary),
        "binary_sha256": _sha256(binary),
    }
