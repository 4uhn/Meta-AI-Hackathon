"""5G-NIDD Combined.csv → serialised, deduplicated, label-skew partition.

    python -m horizon.data.prepare --config configs/partitions/5g-nidd.yaml

Order matters for leakage: serialise → dedupe → split by unique string. Strings seen under two labels
are dropped. Test strings are drawn first and never appear in a shard; shards are pairwise disjoint.
Each operator's shard lives in its own 0700 directory and is read only via `load_shard(op, actor=op)`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from horizon.data.serialize import (
    DROP_COLUMNS,
    FEATURE_COLUMNS,
    FIELDS,
    LABELS,
    RAW_TO_CANONICAL,
    SERIALIZER_VERSION,
    serialize_frame,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_unique(raw_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Return one row per unambiguous unique string: columns text, label, count."""
    df = pd.read_csv(raw_csv, low_memory=False)
    missing = [c for c in [*FEATURE_COLUMNS, "Attack Type"] if c not in df.columns]
    if missing:
        raise ValueError(f"Combined.csv lacks columns {missing}")
    unknown = set(df["Attack Type"].unique()) - set(RAW_TO_CANONICAL)
    if unknown:
        raise ValueError(f"unknown Attack Type values {unknown}")
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    text = serialize_frame(df)
    label = df["Attack Type"].map(RAW_TO_CANONICAL)
    flows = pd.DataFrame({"text": text, "label": label})
    per = flows.groupby(["text", "label"]).size().rename("count").reset_index()
    n_labels = per.groupby("text")["label"].transform("nunique")
    ambiguous = per[n_labels > 1]
    uniq = per[n_labels == 1].reset_index(drop=True)
    info = {
        "rows": int(len(df)),
        "rows_per_class": {k: int(v) for k, v in label.value_counts().items()},
        "unique_strings": int(per["text"].nunique()),
        "ambiguous_strings_dropped": int(ambiguous["text"].nunique()),
        "ambiguous_rows_dropped": int(ambiguous["count"].sum()),
        "unique_per_class": {k: int(v) for k, v in uniq["label"].value_counts().items()},
    }
    return uniq, info


def partition(uniq: pd.DataFrame, cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Draw the global test set first, then disjoint operator shards from what's left."""
    rng = np.random.default_rng(cfg["seed"])
    ops: dict[str, list[str]] = cfg["operators"]
    pools: dict[str, list[str]] = {}
    test_rows, notes = [], {}
    for lab in LABELS:
        strings = np.sort(uniq.loc[uniq["label"] == lab, "text"].to_numpy())  # sort → order-independent
        strings = strings[rng.permutation(len(strings))]
        n_test = min(cfg["test_per_class"], int(len(strings) * cfg["test_max_share"]))
        if n_test < cfg["test_per_class"]:
            notes[lab] = f"only {len(strings)} unique strings; test gets {n_test}"
        test_rows += [(s, lab) for s in strings[:n_test]]
        pools[lab] = list(strings[n_test:])

    # Each class's remaining pool is split across the operators that hold it, in operator order.
    holders = {lab: [op for op in ops if lab in ops[op]] for lab in LABELS}
    shards: dict[str, list[tuple[str, str]]] = {op: [] for op in ops}
    for lab in LABELS:
        hs = holders[lab]
        if not hs:
            continue
        cap = cfg["benign_per_operator"] if lab == "benign" else cfg["attack_per_class"]
        pool = pools[lab]
        per_op = min(cap, len(pool) // len(hs))
        if per_op < cap:
            notes.setdefault(lab, "")
            notes[lab] = (notes[lab] + f"; shards get {per_op}/operator (cap {cap})").lstrip("; ")
        for i, op in enumerate(hs):
            shards[op] += [(s, lab) for s in pool[i * per_op : (i + 1) * per_op]]

    out = {}
    for op, rows in shards.items():
        d = pd.DataFrame(rows, columns=["text", "label"])
        d = d.iloc[rng.permutation(len(d))].reset_index(drop=True)
        is_val = np.zeros(len(d), dtype=bool)
        for _lab, idx in d.groupby("label").indices.items():
            k = max(1, int(round(len(idx) * cfg["val_fraction"])))
            is_val[idx[:k]] = True
        out[op] = d.assign(split=np.where(is_val, "val", "train"))
    test = pd.DataFrame(test_rows, columns=["text", "label"])
    test = test.iloc[rng.permutation(len(test))].reset_index(drop=True)
    return out, test, notes


def write_jsonl(path: Path, d: pd.DataFrame) -> None:
    with open(path, "w") as f:
        for t, lab in zip(d["text"], d["label"], strict=True):
            f.write(json.dumps({"text": t, "label": lab}) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_shard(out_dir: str | Path, op: str, split: str, actor: str) -> list[dict]:
    """Read operator `op`'s private shard. Only `op`'s own code path may do this (silo guard)."""
    if actor != op:
        raise PermissionError(f"isolation: {actor} may not read {op}'s private shard")
    return read_jsonl(Path(out_dir) / op / f"{split}.jsonl")


def load_test(out_dir: str | Path) -> list[dict]:
    return read_jsonl(Path(out_dir) / "test.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/partitions/5g-nidd.yaml")
    ap.add_argument("--skip-sha", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    raw = Path(os.path.expanduser(cfg["raw_csv"]))
    if not args.skip_sha:
        got = sha256_file(raw)
        if got != cfg["raw_sha256"]:
            raise SystemExit(f"{raw} sha256 {got} != pinned {cfg['raw_sha256']}")
    uniq, info = load_unique(raw)
    shards, test, notes = partition(uniq, cfg)

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for op, d in shards.items():
        od = out / op
        od.mkdir(exist_ok=True, mode=0o700)
        os.chmod(od, 0o700)
        for split in ("train", "val"):
            write_jsonl(od / f"{split}.jsonl", d[d["split"] == split])
    write_jsonl(out / "test.jsonl", test)

    lens = pd.concat([d["text"] for d in shards.values()] + [test["text"]]).str.len()
    stats = {
        "serializer_version": SERIALIZER_VERSION,
        "fields": [k for k, _ in FIELDS],
        "dropped_columns": DROP_COLUMNS,
        "raw_sha256": cfg["raw_sha256"],
        "seed": cfg["seed"],
        **info,
        "notes": notes,
        "shards": {
            op: {
                s: {k: int(v) for k, v in d[d["split"] == s]["label"].value_counts().items()}
                for s in ("train", "val")
            }
            for op, d in shards.items()
        },
        "test": {k: int(v) for k, v in test["label"].value_counts().items()},
        "text_chars": {"mean": float(lens.mean()), "max": int(lens.max())},
        "files_sha256": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*.jsonl"))},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(
        json.dumps(
            {k: stats[k] for k in ("unique_per_class", "ambiguous_strings_dropped", "notes", "test")},
            indent=1,
        )
    )
    print({op: {s: sum(v.values()) for s, v in x.items()} for op, x in stats["shards"].items()})


if __name__ == "__main__":
    main()
