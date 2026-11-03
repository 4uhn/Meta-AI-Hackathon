"""NASA ASRS export → single-event narratives → natural partition by reporter organisation.

    python -m horizon.data.asrs --config configs/partitions/asrs.yaml

Same output layout as horizon/data/prepare.py (`{text, label}` JSONL, one 0700 directory per operator,
a shared test.jsonl, stats.json), so the rest of the pipeline is unchanged.

Steps, in leakage order:
1. Keep reports with a narrative whose Anomaly field maps to exactly one configured class.
2. text = first `max_words` words of the first narrative, whitespace-normalised. Exact duplicates are
   dropped (all copies if their labels differ).
3. Test first: `test_per_class` reports per class from all organisations. Two guards keep test reports'
   twins out of every shard:
   - same event: shares (month, location, class) with a test report, since a pilot and a controller can
     both report one event. Only real location codes count; de-identified ones ("ZZZ") would over-match.
   - near-duplicate text: word-shingle Jaccard ≥ `near_dup_jaccard` with any test report (templated forms).
4. Each operator gets its own organisation's remaining reports, capped per class; classes with fewer
   than `min_per_class` are not held (stats.json → `held`). Other organisations' reports are test-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from horizon.data.prepare import sha256_file, write_jsonl
from horizon.tasks import ASRS_LABELS

NARRATIVE, ANOMALY, ORG = "Report_1_Narrative", "Events_Anomaly", "Person_1_Reporter_Organization"


def event_class(anomaly: str, classes: dict[str, list[str]]) -> str | None:
    """The one configured class this Anomaly field maps to, or None (zero or several)."""
    items = [x.strip() for x in str(anomaly).split(";") if x.strip()]
    hit = {c for c, prefixes in classes.items() if any(i.startswith(p) for i in items for p in prefixes)}
    return hit.pop() if len(hit) == 1 else None


def clean(text: str, max_words: int) -> str:
    return " ".join(str(text).split()[:max_words])


def shingles(text: str, n: int) -> set[str]:
    w = text.lower().split()
    return {" ".join(w[i : i + n]) for i in range(max(1, len(w) - n + 1))}


def near_duplicates(test: list[str], pool: list[str], n: int, thr: float) -> dict[int, float]:
    """Pool index → highest Jaccard with any test text, for pairs ≥ thr.

    Exact over all pairs: bincount over a test text's shingle postings gives its intersection with every
    pool text at once."""
    ps = [shingles(t, n) for t in pool]
    vocab: dict[str, int] = {}
    post: list[list[int]] = []
    for i, sh in enumerate(ps):
        for g in sh:
            k = vocab.setdefault(g, len(vocab))
            if k == len(post):
                post.append([])
            post[k].append(i)
    post_np = [np.asarray(x, dtype=np.int64) for x in post]
    size = np.array([len(x) for x in ps], dtype=np.int64)
    out: dict[int, float] = {}
    for t in test:
        ts = shingles(t, n)
        ids = [post_np[vocab[g]] for g in ts if g in vocab]
        if not ids:
            continue
        inter = np.bincount(np.concatenate(ids), minlength=len(pool))
        jac = inter / (len(ts) + size - inter)
        for i in np.flatnonzero(jac >= thr):
            out[int(i)] = max(out.get(int(i), 0.0), float(jac[i]))
    return out


def load_reports(raw: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    if list(cfg["classes"]) != list(ASRS_LABELS):
        raise ValueError("configs classes must match horizon.tasks.ASRS_LABELS (same order)")
    df = pd.read_csv(raw, dtype=str, low_memory=False)
    n_raw = len(df)
    df = df[df[NARRATIVE].notna()].copy()
    df["label"] = df[ANOMALY].map(lambda a: event_class(a, cfg["classes"]))
    single = df[df["label"].notna()].copy()
    single["text"] = single[NARRATIVE].map(lambda t: clean(t, cfg["max_words"]))
    single["org"] = single[ORG].fillna("unknown")
    place = single["Place_Locale_Reference"].fillna("")
    real = (place != "") & ~place.str.startswith("ZZZ")  # ZZZ/ZZZZ = de-identified location
    single["event"] = np.where(
        real, single["Time_Date"].fillna("") + "|" + place + "|" + single["label"], "acn:" + single["acn"]
    )
    n_labels = single.groupby("text")["label"].transform("nunique")
    ambiguous = int((n_labels > 1).sum())
    single = single[n_labels == 1].drop_duplicates("text").reset_index(drop=True)
    info = {
        "raw_reports": n_raw,
        "with_narrative": int(len(df)),
        "single_event": int(df["label"].notna().sum()),
        "ambiguous_texts_dropped": ambiguous,
        "unique_single_event": int(len(single)),
        "per_class": {k: int(v) for k, v in single["label"].value_counts().items()},
        "per_org": {k: int(v) for k, v in single["org"].value_counts().head(12).items()},
    }
    return single, info


def partition(rep: pd.DataFrame, cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    rng = np.random.default_rng(cfg["seed"])
    rep = rep.iloc[rng.permutation(len(rep))].reset_index(drop=True)
    test_idx = []
    for lab in ASRS_LABELS:
        idx = rep.index[rep["label"] == lab]
        if len(idx) < cfg["test_per_class"]:
            raise ValueError(f"{lab}: {len(idx)} reports < test_per_class")
        test_idx += list(idx[: cfg["test_per_class"]])
    test = rep.loc[test_idx]
    rest = rep[~rep.index.isin(test_idx)]
    same_event = rest["event"].isin(set(test["event"]))
    dup = near_duplicates(
        test["text"].tolist(), rest["text"].tolist(), cfg["near_dup_shingle"], cfg["near_dup_jaccard"]
    )
    near_dup = pd.Series(False, index=rest.index)
    near_dup.iloc[list(dup)] = True
    pool = rest[~same_event & ~near_dup]
    shards, held = {}, {}
    for op, o in cfg["operators"].items():
        mine = pool[pool["org"].isin(o["reporter_orgs"])]
        keep, held[op] = [], []
        for lab in ASRS_LABELS:
            rows = mine[mine["label"] == lab]
            if len(rows) >= cfg["min_per_class"]:
                keep.append(rows.iloc[: cfg["cap_per_class"]])
                held[op].append(lab)
        d = pd.concat(keep).reset_index(drop=True)
        d = d.iloc[rng.permutation(len(d))].reset_index(drop=True)
        is_val = np.zeros(len(d), dtype=bool)
        for _lab, idx in d.groupby("label").indices.items():
            is_val[idx[: max(1, int(round(len(idx) * cfg["val_fraction"])))]] = True
        shards[op] = d.assign(split=np.where(is_val, "val", "train"))
    test = test.iloc[rng.permutation(len(test))].reset_index(drop=True)
    notes = {
        "held": held,
        "same_event_quarantined": int(same_event.sum()),
        "near_duplicate_quarantined": int(near_dup.sum()),
        "near_duplicate_quarantined_by_class": {
            k: int(v) for k, v in rest[near_dup]["label"].value_counts().items()
        },
        "quarantined_both": int((same_event & near_dup).sum()),
        "test_by_org": {k: int(v) for k, v in test["org"].value_counts().items()},
        "available_per_op": {
            op: {
                lab: int(((pool["org"].isin(o["reporter_orgs"])) & (pool["label"] == lab)).sum())
                for lab in ASRS_LABELS
            }
            for op, o in cfg["operators"].items()
        },
    }
    return shards, test, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/partitions/asrs.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    raw = Path(os.path.expanduser(cfg["raw_csv"]))
    got = sha256_file(raw)
    if got != cfg["raw_sha256"]:
        raise SystemExit(f"{raw} sha256 {got} != pinned {cfg['raw_sha256']}")
    rep, info = load_reports(raw, cfg)
    shards, test, notes = partition(rep, cfg)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for op, d in shards.items():
        od = out / op
        od.mkdir(exist_ok=True, mode=0o700)
        os.chmod(od, 0o700)
        for split in ("train", "val"):
            write_jsonl(od / f"{split}.jsonl", d[d["split"] == split])
    write_jsonl(out / "test.jsonl", test)
    if cfg.get("test_from"):  # a consortium grown from an earlier split keeps that split's test file
        ref = Path(cfg["test_from"]).read_bytes()
        if sorted(ref.splitlines()) != sorted((out / "test.jsonl").read_bytes().splitlines()):
            raise SystemExit(f"test set differs from {cfg['test_from']} (not just in order)")
        (out / "test.jsonl").write_bytes(ref)
    words = pd.concat([d["text"] for d in shards.values()] + [test["text"]]).str.split().str.len()
    stats = {
        "task": "asrs",
        "raw_source": cfg["raw_source"],
        "raw_sha256": cfg["raw_sha256"],
        "seed": cfg["seed"],
        "max_words": cfg["max_words"],
        "operators": {op: o for op, o in cfg["operators"].items()},
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
        "text_words": {"mean": float(words.mean()), "max": int(words.max())},
        "files_sha256": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*.jsonl"))},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in ("per_class", "notes", "test")}, indent=1))
    print({op: {s: sum(v.values()) for s, v in x.items()} for op, x in stats["shards"].items()})


if __name__ == "__main__":
    main()
