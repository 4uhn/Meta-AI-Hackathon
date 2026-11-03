"""`make demo`: a live, narrated replay of the final round of a recorded encrypted training run.

    python -m horizon.demo [--run runs/modal_glimmer_E3] [--step] [--detail] [--pace 1.8]

Plain-English by default; `--detail` adds hashes, timings and the operators' verbatim replies.
Every measured number is written to results/demo/live_<run>_r<round>_<stamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from horizon.demo.live import Beat, Live, parties_of, recorded_request, summary

REPO = Path(__file__).resolve().parents[2]

# What each attack's refusal means, in plain words, and the text the operator's reply must contain
# for that explanation to be shown (otherwise the verbatim reply is shown instead).
ATTACKS = {
    "fake_sum": (
        "Unlock just {victim}'s update",
        "Refused: {asked} re-adds the updates itself, and that isn't the combined one.",
        "recomputed aggregate",
    ),
    "two_party_sum": (
        "Unlock a mix of only two operators",
        "Refused: either of the two could work out the other's update.",
        "contributors, need",
    ),
    "second_decryption": (
        "Unlock the same round a second time",
        "Refused: each operator unlocks a round only once.",
        "one-shot",
    ),
}
DOMAIN = {"nidd5g": "mobile operators", "asrs": "aviation organisations"}


def name(op: str) -> str:
    """op_a → Operator A."""
    return "Operator " + op.split("_")[-1].upper() if op.startswith("op_") else op


class Style:
    def __init__(self, color: bool):
        c = color
        self.amber = "\033[38;2;255;183;77m" if c else ""
        self.green = "\033[38;2;110;231;160m" if c else ""
        self.red = "\033[38;2;255;107;107m" if c else ""
        self.dim = "\033[38;2;130;140;160m" if c else ""
        self.bold = "\033[1m" if c else ""
        self.off = "\033[0m" if c else ""


def num(v: float) -> str:
    s = f"{v:+.5f}" if abs(v) < 10 else f"{v:+,.1f}"
    return s.replace("-", "−")


class Narrator:
    def __init__(self, st: Style, pace: float, step: bool, detail: bool, active: list[str]):
        self.s, self.pace, self.step, self.detail = st, pace, step, detail
        self.letters = [name(o).split()[-1] for o in active]
        self.cheat_header = False

    def out(self, text: str = "") -> None:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def wait(self, k: float = 1.0) -> None:
        if self.step:
            input()
        elif self.pace > 0:
            time.sleep(self.pace * k)

    def section(self, n: int, title: str) -> None:
        self.out()
        self.out(f"  {self.s.amber}{self.s.bold}{n}  {title}{self.s.off}")

    def ok(self, text: str) -> None:
        self.out(f"     {self.s.green}✓{self.s.off}  {text}")

    def no(self, text: str) -> None:
        self.out(f"     {self.s.red}✗{self.s.off}  {text}")

    def say(self, text: str, color: str = "") -> None:
        self.out(f"        {color}{text}{self.s.off}")

    def tech(self, text: str) -> None:
        if self.detail:
            self.out(f"        {self.s.dim}{text}{self.s.off}")

    def start(self, key: str, ctx: dict) -> None:
        if key == "aggregate":
            self.section(2, "The server combines them, still encrypted")
        elif key == "attack":
            a = ctx["attack"]
            if a == "second_decryption":
                self.section(6, "The server asks again, hoping to compare two answers")
            elif not self.cheat_header:
                self.cheat_header = True
                self.section(3, "The server tries to cheat")
            self.no(ATTACKS[a][0].format(victim=name(ctx.get("victim", "")), asked=name(ctx["asked"])))
        elif key == "request":
            off = " and ".join(name(o) for o in ctx["offline"])
            on = " and ".join(name(o).split()[-1] for o in ctx["active"])
            self.section(4, f"{off} goes offline. {on} unlock the result")
        elif key == "publish":
            self.section(5, "The new shared model is ready")

    def beat(self, b: Beat) -> None:
        s, d = self.s, b.data
        if b.key == "staged":
            self.section(1, "Each operator has sent its update, encrypted")
            for op, m in d["submissions"].items():
                self.ok(f"{name(op)}  encrypted and sent")
                self.tech(f"{m['bytes'] / 1e6:.1f} MB · {m['ct_sha256'][:12]}…")
            self.say("Nobody can read these: not the server, not each other.", s.dim)
            self.wait()
        elif b.key == "aggregated":
            self.ok("combined without decrypting anything")
            if d["identical"]:
                self.ok("exactly the same result as the original run")
            else:
                self.no("differs from the original run")
            self.tech(f"Σ {d['agg_sha256'][:12]}… · {b.ms / 1000:.2f} s")
            self.wait()
        elif b.key == "attack":
            plain, needle = ATTACKS[d["attack"]][1], ATTACKS[d["attack"]][2]
            reply = d["refusal"]
            if reply and needle in reply:
                self.say(plain.format(asked=name(d["asked"])), s.red)
                self.tech(f"{d['asked']}: {reply}")
            elif reply:
                self.say(f"Refused by {name(d['asked'])}: {reply}", s.red)
            else:
                self.say(f"NOT REFUSED: {name(d['asked'])} released a share", s.red)
            self.wait()
        elif b.key == "requested":
            for op in d["offline"]:
                self.say(f"({name(op)}'s key isn't even on this machine.)", s.dim)
            self.wait(0.5)
        elif b.key == "share":
            self.tech(f"{d['party']} re-checked the sum, released its share · {b.ms / 1000:.2f} s")
        elif b.key == "solo":
            vals = "  ".join(num(v) for v in d["sample"][:3])
            self.out(
                f"     With {name(d['party']).split()[-1]}'s key alone:   {s.red}{vals} …  unreadable{s.off}"
            )
            self.wait()
        elif b.key == "decrypted":
            vals = "  ".join(num(v) for v in d["sample"][:3])
            keys = " and ".join(self.letters)
            self.out(f"     With {keys}'s keys:  {s.green}{vals} …  unlocked ✓{s.off}")
            self.tech(
                f"{d['n_values']:,} values · one key alone off by {d['solo_rel_err']:.2e}× · {b.ms / 1000:.2f} s"
            )
            self.wait()
        elif b.key == "published":
            self.ok("it matches the model from the real training run")
            self.tech(f"max |ΔB| {d['max_abs_diff_vs_recorded']:.1e} (largest weight {d['max_abs_b']:.3f})")
            self.wait()


def default_round(run: Path) -> int:
    rounds = [int(p.parent.parent.name.split("_")[1]) for p in run.glob("bus/round_*/decrypt/request.json")]
    if not rounds:
        raise SystemExit(f"{run}: no decrypted round found")
    return max(rounds)


def describe(run: Path) -> tuple[str, str, int]:
    """(who, model, number of rounds) for the opening lines, from the run's own files."""
    cfg = yaml.safe_load((run / "config.yaml").read_text()) if (run / "config.yaml").is_file() else {}
    who = DOMAIN.get((cfg or {}).get("task", "nidd5g"), "organisations")
    mid = ((cfg or {}).get("model") or {}).get("id", "")
    model = "Meta's Muse Glimmer" if "Glimmer" in mid else (mid or "a shared model")
    rounds = len(list(run.glob("bus/round_*/submissions")))
    return who, model, rounds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/modal_glimmer_E3")
    ap.add_argument("--round", type=int)
    ap.add_argument("--active", default="op_a,op_c")
    ap.add_argument("--offline", default="op_b")
    ap.add_argument("--root", default="runs/live", help="workspace parent (deleted afterwards unless --keep)")
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--pace", type=float, default=1.8, help="seconds between beats")
    ap.add_argument("--step", action="store_true", help="press Enter to advance")
    ap.add_argument("--detail", action="store_true", help="also show hashes, timings, verbatim replies")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)

    run = Path(a.run)
    if not (run / "public" / "parties.json").is_file():
        print(
            f"{run} not found. `make demo` replays a recorded run's real encrypted updates and key\n"
            "shares, which are not in the repository. Without them, run\n"
            "`python examples/secure-aggregation/secure_sum.py`: the same protocol on synthetic data."
        )
        return 2
    r = a.round or default_round(run)
    active, offline = a.active.split(","), [x for x in a.offline.split(",") if x]
    binary = Path(os.environ.get("HORIZON_HE") or REPO / "crypto" / "bin" / "horizon-he")
    thr = json.loads((run / "public" / "parties.json").read_text())["threshold"]
    n = len(parties_of(run))
    who, model, rounds = describe(run)

    st = Style(not a.no_color and sys.stdout.isatty())
    nar = Narrator(st, a.pace, a.step, a.detail, active)
    nar.out()
    nar.out(f"  {st.amber}{st.bold}HORIZON{st.off}  {st.dim}· live demo{st.off}")
    nar.out(f"  {n} {who} trained {model} together over {rounds} rounds.")
    nar.out(f"  This is round {r}, replayed live on its real encrypted updates.")
    nar.out(f"  {st.dim}Unlocking anything takes {thr} of the {n} operators.{st.off}")
    nar.tech(
        f"run {run.name}; round {r} was unlocked by {', '.join(recorded_request(run, r)['active'])} "
        f"on the day, by {', '.join(active)} now"
    )
    nar.wait()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(a.root) / stamp
    live = Live(run, r, root, active, offline, on_beat=nar.beat, on_start=nar.start)
    try:
        live.run()
        res = summary(live, binary)
    finally:
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
    out = Path(a.out) / f"live_{run.name}_r{r}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")

    refused = sum(1 for x in res["attacks"] if x["refusal"])
    nar.out()
    nar.out(
        f"  {st.bold}Done in {res['live_crypto_ms'] / 1000:.0f} seconds on this laptop.{st.off} "
        f"{len(res['attacks'])} cheating attempts, {refused} refused."
    )
    nar.out(f"  {st.dim}Every measurement: {out}{st.off}")
    nar.out()
    return 0 if res["aggregate"]["identical"] and res["all_attacks_refused"] else 1


if __name__ == "__main__":
    sys.exit(main())
