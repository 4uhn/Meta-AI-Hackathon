"""Evaluation report: detection matrix and blind-spot recall.

    python -m horizon.eval.report --results results/5g-nidd-llama
    python -m horizon.eval.report --results results/asrs-glimmer --partition configs/partitions/asrs.yaml

Reads whichever of E0, E1a, E1b, … (one per operator), E2, E3, E4 exist in the results directory and writes:
- metrics.json: per-class recall, macro-F1, accuracy, and blind-spot recall (per operator, mean recall
  over classes absent from its shard: local model vs encrypted FL);
- detection_matrix.png: rows = models, columns = classes, cell = recall; blind spots outlined.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from horizon import tasks
from horizon.data.serialize import LABELS

ROWS = [  # (experiment, row label)
    ("E0", "Base model (E0)"),
    ("E1a", "Local only: op_a (E1a)"),
    ("E1b", "Local only: op_b (E1b)"),
    ("E1c", "Local only: op_c (E1c)"),
    ("E3", "Horizon: encrypted FL (E3)"),
    ("E4", "Centralised, pooled (E4)"),
]
LOCAL = {"E1a": "op_a", "E1b": "op_b", "E1c": "op_c"}


def local_map(ops) -> dict[str, str]:
    """E1x → op_x for the given operators (default: op_a, op_b, op_c)."""
    return {f"E1{op[-1]}": op for op in ops} if ops else dict(LOCAL)


def rows_for(ops) -> list[tuple[str, str]]:
    """ROWS with one local row per operator, in the same order (E0, E1x…, E3, E4)."""
    loc = [(e, f"Local only: {op} ({e})") for e, op in local_map(ops).items()]
    return [ROWS[0], *loc, *ROWS[-2:]]


# Sequential blue palette and light chart chrome.
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, NAN_FILL = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def blind_spots(partition_cfg: dict) -> dict[str, list[str]]:
    """Classes each operator never sees in training (excluding the shared class).

    Explicit partitions list each operator's classes; natural ones (ASRS) use stats.json `notes.held`."""
    t = tasks.TASKS[tasks.task_name(partition_cfg)]
    cands = [lab for lab in t.labels if lab != t.shared_class]
    if all(isinstance(v, list) for v in partition_cfg["operators"].values()):
        held = partition_cfg["operators"]
    else:
        stats = json.loads((Path(partition_cfg["out_dir"]) / "stats.json").read_text())
        held = stats["notes"]["held"]
    return {op: [c for c in cands if c not in held[op]] for op in partition_cfg["operators"]}


def row_names(partition_cfg: dict) -> dict[str, str]:
    """Row labels; a local row names its organisation when the partition gives one (ASRS)."""
    names = dict(rows_for(partition_cfg.get("operators")))
    for e, op in local_map(partition_cfg.get("operators")).items():
        o = partition_cfg.get("operators", {}).get(op)
        if isinstance(o, dict) and o.get("name"):
            names[e] = f"Local only: {o['name'].replace('_', ' ')} (E{e[1:]})"
    return names


def build(results: Path, partition_cfg: dict) -> dict:
    exps = {
        e: json.loads((results / f"{e}.json").read_text())
        for e, _ in rows_for(partition_cfg.get("operators")) + [("E2", "")]
        if (results / f"{e}.json").is_file()
    }
    exps = {e: r for e, r in exps.items() if not r.get("smoke")}
    rec = {e: {lab: r["metrics"]["per_class"][lab]["recall"] for lab in LABELS} for e, r in exps.items()}
    spots = blind_spots(partition_cfg)
    bs = {}
    for e, op in local_map(partition_cfg.get("operators")).items():
        if e in rec and spots[op]:  # an operator holding every class (ASRS airline) has no blind spot
            row = {"classes": spots[op], "local": float(np.mean([rec[e][c] for c in spots[op]]))}
            if "E3" in rec:
                row["horizon_E3"] = float(np.mean([rec["E3"][c] for c in spots[op]]))
            bs[op] = row
    if bs:
        bs["mean"] = {
            k: float(np.mean([v[k] for v in bs.values()]))
            for k in ("local", "horizon_E3")
            if all(k in v for v in bs.values())
        }
    return {
        "source_files": sorted(f"{e}.json" for e in exps),
        "recall": rec,
        "macro_f1": {e: r["metrics"]["macro_f1"] for e, r in exps.items()},
        "accuracy": {e: r["metrics"]["accuracy"] for e, r in exps.items()},
        "blind_spot_recall": bs,
        "blind_spot_classes": spots,
        "row_names": row_names(partition_cfg),
    }


def plot_matrix(rep: dict, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.patches import Rectangle

    spots = rep["blind_spot_classes"]
    local = local_map(list(spots))
    rows = [(e, name) for e, name in rows_for(list(spots)) if e in rep["recall"]]
    m = np.array([[rep["recall"][e][lab] for lab in LABELS] for e, _ in rows], dtype=float)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i in range(len(rows)):
        for j in range(len(LABELS)):
            v = m[i, j]
            fill = NAN_FILL if np.isnan(v) else cmap(v)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fill, edgecolor=SURFACE, linewidth=2))
            r, g, b = to_rgb(fill)
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
            ax.text(
                j + 0.5,
                i + 0.5,
                "–" if np.isnan(v) else f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if dark else INK,
            )
    for i, (e, _) in enumerate(rows):
        if e in local:
            for c in spots[local[e]]:
                j = LABELS.index(c)
                ax.add_patch(
                    Rectangle(
                        (j + 0.06, i + 0.08),
                        0.88,
                        0.84,
                        fill=False,
                        edgecolor=INK,
                        linewidth=1.2,
                        linestyle=(0, (3, 2)),
                    )
                )
    ax.set_xlim(0, len(LABELS))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(
        np.arange(len(LABELS)) + 0.5,
        [lab.replace("_", " ") for lab in LABELS],
        rotation=30,
        ha="right",
        color=INK2,
        fontsize=9,
    )
    names = rep.get("row_names", {})
    ax.set_yticks(np.arange(len(rows)) + 0.5, [names.get(e, n) for e, n in rows], color=INK2, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    fig.text(
        0.01,
        0.01,
        "Cell = per-class recall on the shared test set. Dashed outline = class "
        "absent from that operator's training shard (a blind spot).",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def partition_from_run_config(path: str) -> dict:
    """The partition a report needs, from a run config: its task and each operator's training labels."""
    cfg = yaml.safe_load(open(path))
    data = Path(cfg["data_dir"])
    ops = (cfg.get("fl") or {}).get("operators", ["op_a", "op_b", "op_c"])
    held = {
        op: sorted({json.loads(line)["label"] for line in open(data / op / "train.jsonl") if line.strip()})
        for op in ops
    }
    return {"task": cfg.get("task", tasks.DEFAULT), "operators": held, "out_dir": str(data)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/5g-nidd-llama")
    ap.add_argument("--partition", default="configs/partitions/5g-nidd.yaml")
    ap.add_argument(
        "--config", help="a run config instead of --partition: classes held are read from its shards"
    )
    ap.add_argument("--title", default="Detection matrix: per-class recall")
    a = ap.parse_args()
    results = Path(a.results)
    part = partition_from_run_config(a.config) if a.config else yaml.safe_load(open(a.partition))
    tasks.activate(part)
    rep = build(results, part)
    (results / "metrics.json").write_text(json.dumps(rep, indent=2))
    if rep["recall"]:
        plot_matrix(rep, results / "detection_matrix.png", a.title)
    print(
        json.dumps({"models": list(rep["recall"]), "blind_spot_recall": rep["blind_spot_recall"]}, indent=1)
    )


if __name__ == "__main__":
    main()
