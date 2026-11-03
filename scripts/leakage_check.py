"""Near-duplicate leakage check: 5G-NIDD test flows vs every operator's train/val rows.

    python scripts/leakage_check.py [--data data/processed] [--out results/5g-nidd-split-check.json]

Exact-text overlap test ↔ train/val is 0 by construction (unique-string split). This asks the harder
question: how many test flows match some train/val flow in all but k of their serialised fields
(k = 1, 2)? For each k, every row is keyed by its fields with k positions blanked out (all C(F, k)
choices); a test row is a near-duplicate if any of its keys occurs among the train/val keys. Reported
overall, per class, and whether the matched train/val label equals the test label (same label = the
model could score by recall rather than generalisation; different label = an ambiguous flow pair).
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

OPS = ("op_a", "op_b", "op_c")


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def fields(text: str) -> tuple[str, ...]:
    return tuple(text.split(" "))


def keys(f: tuple[str, ...], k: int):
    for blank in itertools.combinations(range(len(f)), k):
        yield blank, tuple("*" if i in blank else v for i, v in enumerate(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed", type=Path)
    ap.add_argument("--out", default="results/5g-nidd-split-check.json", type=Path)
    a = ap.parse_args()
    test = read(a.data / "test.jsonl")
    pool = [
        r | {"src": f"{op}/{s}"}
        for op in OPS
        for s in ("train", "val")
        for r in read(a.data / op / f"{s}.jsonl")
    ]
    n_fields = Counter(len(fields(r["text"])) for r in test + pool)
    out: dict = {
        "n_test": len(test),
        "n_train_val": len(pool),
        "fields_per_row": dict(n_fields),
        "exact_overlap": len({r["text"] for r in test} & {r["text"] for r in pool}),
    }
    for k in (1, 2):
        index: dict[tuple, set[str]] = defaultdict(set)
        for r in pool:
            for _, key in keys(fields(r["text"]), k):
                index[key].add(r["label"])
        hit_same, hit_diff, per_class, blanked = 0, 0, Counter(), Counter()
        for r in test:
            labs, fld = set(), None
            for blank, key in keys(fields(r["text"]), k):
                if key in index:
                    labs |= index[key]
                    fld = fld or blank
            if labs:
                same = r["label"] in labs
                hit_same += same
                hit_diff += not same
                per_class[(r["label"], "same" if same else "other_label_only")] += 1
                blanked[",".join(fields(r["text"])[i].split("=")[0] for i in fld)] += 1
        out[f"k{k}"] = {
            "near_dup_same_label": hit_same,
            "near_dup_other_label_only": hit_diff,
            "share_of_test": round((hit_same + hit_diff) / len(test), 4),
            "per_class": {f"{c}|{s}": n for (c, s), n in sorted(per_class.items())},
            "top_differing_fields": dict(blanked.most_common(8)),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
