"""What an honest-but-curious server would learn from one operator's plaintext update.

Attack: a logit lens on the weights. With FFA-LoRA the o_proj update ΔW = (α/r)·ΔB·A writes into the
residual stream; projected onto each label's readout direction (its decision token's lm_head row, through
the final norm and the layer's post-attention norm scale) it says how strongly the update can move that
label's logit. The attack ranks the classes by that strength. It needs only the public base weights and
the public frozen A, no data and no GPU.

Inputs are the plaintext updates of an `--insecure-reference` run. Horizon's secure path never exposes
them: operators submit only CKKS ciphertexts (IND-CPA), so there is nothing for this attack to read.

    python -m horizon.eval.update_leakage --run RUN --partition configs/partitions/5g-nidd.yaml \
        --out results/5g-nidd-glimmer/update_leakage.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import struct
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import load_file

from horizon.data.serialize import LABELS
from horizon.model.lora import ADAPTER_FILE, _b_keys, read_b_file

LAYER = re.compile(r"\.layers\.(\d+)\.")
HF = "https://huggingface.co/{repo}/resolve/{rev}/{file}"


# Public base weights: only the rows the attack needs, fetched by HTTP range reads.


def bf16_to_f32(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def read_tensor(
    read: Callable[[int, int], bytes], header: dict, base: int, name: str, rows=None
) -> np.ndarray:
    """One bf16 tensor (or some rows of a 2-D one) from a safetensors file via `read(start, end_incl)`."""
    meta = header[name]
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name}: expected BF16, got {meta['dtype']}")
    lo, _ = meta["data_offsets"]
    shape = meta["shape"]
    if rows is None:
        n = int(np.prod(shape))
        return bf16_to_f32(read(base + lo, base + lo + 2 * n - 1)).reshape(shape)
    width = 2 * shape[1]
    return np.stack([bf16_to_f32(read(base + lo + i * width, base + lo + (i + 1) * width - 1)) for i in rows])


def safetensors_header(read: Callable[[int, int], bytes]) -> tuple[dict, int]:
    n = struct.unpack("<Q", read(0, 7))[0]
    return json.loads(read(8, 8 + n - 1)), 8 + n


def hf_reader(repo: str, rev: str, file: str) -> Callable[[int, int], bytes]:
    url = HF.format(repo=repo, rev=rev, file=file)

    def read(a: int, b: int) -> bytes:
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                raise OSError(f"range read not honoured ({r.status}) for {url}")
            data = r.read()
        if len(data) != b - a + 1:
            raise OSError(f"short read {len(data)} ≠ {b - a + 1} from {url}")
        return data

    return read


def fetch_readout_weights(repo: str, rev: str, token_ids: list[int], n_layers: int, cache: Path) -> dict:
    """lm_head rows for `token_ids`, the final norm weight and every post-attention norm weight (fp32).

    Cached as .npz so later runs work offline; results record the arrays' sha256 (`arrays_sha256`)."""
    if cache.is_file():
        z = np.load(cache)
        if list(z["token_ids"]) == list(token_ids):
            return {k: z[k] for k in z.files}
    index = json.loads(
        urllib.request.urlopen(HF.format(repo=repo, rev=rev, file="model.safetensors.index.json")).read()
    )
    wmap = index["weight_map"]
    names = {
        "lm_head": "lm_head.weight",
        "final_norm": "model.language_model.norm.weight",
        **{
            f"post_attn_{i}": f"model.language_model.layers.{i}.post_attention_layernorm.weight"
            for i in range(n_layers)
        },
    }
    readers: dict[str, tuple] = {}
    out = {}
    for key, name in names.items():
        f = wmap[name]
        if f not in readers:
            rd = hf_reader(repo, rev, f)
            readers[f] = (rd, *safetensors_header(rd))
        rd, hdr, base = readers[f]
        out[key] = read_tensor(rd, hdr, base, name, rows=token_ids if key == "lm_head" else None)
    post = np.stack([out.pop(f"post_attn_{i}") for i in range(n_layers)])
    res = {
        "lm_head_rows": out["lm_head"],
        "final_norm": out["final_norm"],
        "post_attn_norm": post,
        "token_ids": np.array(token_ids),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **res)
    return res


def arrays_sha256(d: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def o_proj_layout(adapter_dir: Path) -> tuple[list[tuple[int, int, int]], dict[int, np.ndarray], int]:
    """[(layer, offset, size)] of each o_proj lora_B block in the flat vector, A per layer, total length."""
    t = load_file(Path(adapter_dir) / ADAPTER_FILE)
    blocks, a_mats, off = [], {}, 0
    for k in _b_keys(t):
        n = t[k].size
        if ".o_proj." in k:
            layer = int(LAYER.search(k).group(1))
            blocks.append((layer, off, n))
            a_mats[layer] = t[k.replace(".lora_B.", ".lora_A.")].astype(np.float64)
        off += n
    return sorted(blocks), a_mats, off


def readout_directions(w: dict, centre: bool = True) -> np.ndarray:
    """(n_layers, n_labels, d) readout per layer: (1 + w_post,ℓ) ⊙ g ⊙ W_U[label], centred over labels.

    Softmax ignores anything that moves every label's logit equally, so the mean over labels is removed."""
    d = w["lm_head_rows"].astype(np.float64) * w["final_norm"].astype(np.float64)
    if centre:
        d = d - d.mean(0, keepdims=True)
    scale = 1.0 + w["post_attn_norm"].astype(np.float64)  # centred RMSNorm: weight stored as an offset from 1
    return scale[:, None, :] * d[None, :, :]


def layer_scores(delta: np.ndarray, blocks, a_mats, readout: np.ndarray) -> np.ndarray:
    """(n_layers, n_labels): ‖d̃_cᵀ ΔB_ℓ A_ℓ‖ / ‖d̃_c‖, how strongly layer ℓ can move label c."""
    out = np.zeros((len(blocks), readout.shape[1]))
    for i, (layer, off, size) in enumerate(blocks):
        a = a_mats[layer]  # (r, d_in)
        db = delta[off : off + size].reshape(-1, a.shape[0])  # (d_model, r), row-major as flattened
        r = readout[layer] if readout.shape[0] > 1 else readout[0]
        m = r @ db  # (labels, r)
        out[i] = np.sqrt(np.einsum("ij,jk,ik->i", m, a @ a.T, m)) / np.linalg.norm(r, axis=1)
    return out


def total_score(per_layer: np.ndarray) -> np.ndarray:
    """All layers, no selection: energy √Σ_ℓ S_c(ℓ)²."""
    return np.sqrt((per_layer**2).sum(0))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + ½·P(tie), over all pairs (Mann–Whitney)."""
    pos, neg = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def zscore_rows(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)


def pooled_auc(z: np.ndarray, truth: np.ndarray) -> float:
    return auc(z[truth], z[~truth])


def exact_topk(s: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per row: are the k highest scores exactly the k true classes (k = row's true count)?"""
    out = []
    for row, t in zip(s, truth, strict=True):
        k = int(t.sum())
        out.append(set(np.argsort(-row)[:k]) == set(np.flatnonzero(t)))
    return np.array(out)


def permutation_p(stat: Callable[[np.ndarray], float], truth: np.ndarray) -> tuple[float, int]:
    """Exact p-value: relabel the classes by every permutation (same one for every update) and count how often
    the statistic is at least the observed one. Under 'scores carry no class identity' all are equally likely.
    """
    obs, ge, n = stat(truth), 0, 0
    for perm in itertools.permutations(range(truth.shape[1])):
        ge += stat(truth[:, list(perm)]) >= obs - 1e-12
        n += 1
    return ge / n, n


def label_first_tokens(cfg: dict) -> list[int]:
    from transformers import AutoTokenizer

    from horizon.model.labels import first_tokens_unique, target_text

    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(
        m.get("tokenizer_id", m["id"]), revision=m.get("tokenizer_revision", m["revision"])
    )
    if not first_tokens_unique(tok):
        raise ValueError("the attack reads one decision token per class; label first tokens are not unique")
    return [tok(target_text(tok, lab), add_special_tokens=False)["input_ids"][0] for lab in LABELS]


def load_updates(run: Path, rounds: list[int], ops: list[str]) -> tuple[dict, dict]:
    """Δ = B_local − B_global(r−1) per (round, op), plus the global adapters (r = 1 … R+1)."""
    glob = {r: read_b_file(run / f"bus/round_{r}/global/adapter") for r in [*rounds, rounds[-1] + 1]}
    deltas, weights = {}, {}
    for r in rounds:
        for op in ops:
            p = np.fromfile(run / f"insecure/round_{r}/{op}.f64")
            w = p[-1]
            deltas[(r, op)] = p[:-1] / w - glob[r]
            weights[(r, op)] = w
        # sanity: the next global adapter is the weighted FedAvg of these updates (η_s = 1)
        avg = sum(weights[(r, op)] * (deltas[(r, op)] + glob[r]) for op in ops) / sum(
            weights[(r, op)] for op in ops
        )
        err = float(np.abs(avg - glob[r + 1]).max())
        if err > 1e-6:
            raise AssertionError(f"round {r}: next global ≠ FedAvg of the updates (max err {err:.2e})")
    return deltas, glob


def run_attack(run: Path, partition: Path, cache: Path, null_seeds: int = 200) -> dict:
    t0 = time.time()
    cfg = yaml.safe_load(open(run / "config.yaml"))
    part = yaml.safe_load(open(partition))["operators"]
    ops = sorted(part)
    rounds = sorted(int(p.name.split("_")[1]) for p in (run / "insecure").glob("round_*"))
    deltas, glob = load_updates(run, rounds, ops)
    blocks, a_mats, total = o_proj_layout(run / f"bus/round_{rounds[0]}/global/adapter")
    if total != len(next(iter(deltas.values()))):
        raise AssertionError("adapter layout and update length disagree")

    tokens = label_first_tokens(cfg)
    n_layers = len(blocks)
    w = fetch_readout_weights(cfg["model"]["id"], cfg["model"]["revision"], tokens, n_layers, cache)
    readouts = {
        "primary": readout_directions(w),  # post-attention norm scale × final norm × lm_head row, centred
        "no_post_norm": readout_directions({**w, "post_attn_norm": np.zeros_like(w["post_attn_norm"][:1])}),
        "uncentred": readout_directions(w, centre=False),
    }

    keys = [(r, op) for r in rounds for op in ops]
    attack = [i for i, lab in enumerate(LABELS) if lab != "benign"]
    truth9 = np.array([[lab in part[op] for lab in LABELS] for _, op in keys])
    truth = truth9[:, attack]
    rnd = np.array([r for r, _ in keys])

    def evaluate(scores9: np.ndarray) -> dict:
        s = scores9[:, attack]
        z = zscore_rows(s)
        cohort = np.log(s)
        # Server-side calibration: each class against the cohort of updates it saw that round.
        for r in rounds:
            cohort[rnd == r] -= cohort[rnd == r].mean(0, keepdims=True)
        out = {}
        for name, sc in [("strict", z), ("cohort", zscore_rows(cohort))]:
            ex = exact_topk(sc, truth)
            r1 = rnd == rounds[0]
            out[name] = {
                "presence_auc": pooled_auc(sc, truth),
                "presence_auc_round1": pooled_auc(sc[r1], truth[r1]),
                "mean_within_update_auc": float(
                    np.mean([auc(row[t], row[~t]) for row, t in zip(sc, truth, strict=True)])
                ),
                "top3_exact": int(ex.sum()),
                "top3_exact_round1": int(ex[r1].sum()),
                "top3_exact_by_update": {f"r{r}/{op}": bool(e) for (r, op), e in zip(keys, ex, strict=True)},
            }
        out["strict"]["presence_auc_9class_incl_benign"] = pooled_auc(zscore_rows(scores9), truth9)
        return out

    per_layer = {
        name: np.stack([layer_scores(deltas[k], blocks, a_mats, ro) for k in keys])
        for name, ro in readouts.items()
    }
    totals = {name: np.stack([total_score(pl) for pl in v]) for name, v in per_layer.items()}
    results = {name: evaluate(s) for name, s in totals.items()}

    # exact permutation test on the primary statistics (8! relabelings of the attack classes)
    zp = zscore_rows(totals["primary"][:, attack])
    p_auc, n_perm = permutation_p(lambda t: pooled_auc(zp, t), truth)
    p_top3, _ = permutation_p(lambda t: float(exact_topk(zp, t).sum()), truth)

    # null control: shuffle each o_proj block's entries (same per-layer norm, class geometry destroyed)
    rng = np.random.default_rng(0)
    null_auc = []
    for _ in range(null_seeds):
        rows = []
        for k in keys:
            d = deltas[k].copy()
            for _, off, size in blocks:
                d[off : off + size] = rng.permutation(d[off : off + size])
            rows.append(total_score(layer_scores(d, blocks, a_mats, readouts["primary"])))
        null_auc.append(pooled_auc(zscore_rows(np.stack(rows)[:, attack]), truth))

    # aggregate control: what the decrypted FedAvg aggregate gives about each operator (same vector for all)
    agg = np.stack(
        [
            total_score(layer_scores(glob[r + 1] - glob[r], blocks, a_mats, readouts["primary"]))
            for r in rounds
        ]
    )
    agg_rows = np.stack([agg[rounds.index(r)] for r, _ in keys])
    za = zscore_rows(agg_rows[:, attack])
    agg_ctrl = {
        "presence_auc_per_operator": pooled_auc(za, truth),
        "top3_exact": int(exact_topk(za, truth).sum()),
        "union_rank_auc": "n/a: every attack class is held by some operator, so the union is all 8",
        "z_by_round": {
            f"r{r}": dict(
                zip(
                    [LABELS[i] for i in attack],
                    np.round(zscore_rows(agg[:, attack])[j], 3).tolist(),
                    strict=True,
                )
            )
            for j, r in enumerate(rounds)
        },
    }

    pl_primary = per_layer["primary"][:, :, attack]  # (updates, layers, classes)
    layer_auc = [pooled_auc(zscore_rows(pl_primary[:, i]), truth) for i in range(n_layers)]
    layer_auc_r1 = [
        pooled_auc(zscore_rows(pl_primary[rnd == rounds[0], i]), truth[rnd == rounds[0]])
        for i in range(n_layers)
    ]

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    return {
        "experiment": "update leakage: class presence from one operator's plaintext update (logit lens on weights)",
        "threat_model": "honest-but-curious server WITHOUT encryption; Horizon's secure path exposes only CKKS ciphertexts",
        "run": str(run),
        "inputs_sha256": {f"r{r}/{op}": sha(run / f"insecure/round_{r}/{op}.f64") for r, op in keys},
        "base": {
            "repo": cfg["model"]["id"],
            "revision": cfg["model"]["revision"],
            "readout_sha256": arrays_sha256(w),
        },
        "labels": LABELS,
        "attack_classes": [LABELS[i] for i in attack],
        "decision_token_ids": tokens,
        "truth": {op: sorted(part[op]) for op in ops},
        "n_updates": len(keys),
        "rounds": rounds,
        "uses": f"o_proj lora_B only ({len(blocks)} layers), frozen A, public base rows; no data, no GPU",
        "score": "S_c = sqrt(sum_l ||d_c,l^T dB_l A_l||^2 / ||d_c,l||^2), d_c,l = (1+w_post,l)*g*W_U[c] centred over labels",
        "results": results,
        "primary": results["primary"],
        "permutation_test": {"n_permutations": n_perm, "p_presence_auc": p_auc, "p_top3_exact": p_top3},
        "null_shuffled_blocks": {
            "seeds": null_seeds,
            "auc_mean": float(np.mean(null_auc)),
            "auc_p2_5": float(np.percentile(null_auc, 2.5)),
            "auc_p97_5": float(np.percentile(null_auc, 97.5)),
            "auc_max": float(np.max(null_auc)),
        },
        "aggregate_control": agg_ctrl,
        "per_layer_auc_diagnostic": {"all_rounds": layer_auc, "round1": layer_auc_r1},
        "scores_primary": {
            f"r{r}/{op}": dict(zip(LABELS, np.round(totals["primary"][i], 6).tolist(), strict=True))
            for i, (r, op) in enumerate(keys)
        },
        "update_weight_slot_n": {
            op: round(float(np.fromfile(run / f"insecure/round_{rounds[0]}/{op}.f64")[-1]) * 1e4, 3)
            for op in ops
        },
        "wall_s": round(time.time() - t0, 1),
    }


def plot(res: dict, path: Path) -> None:
    """Left: per-update class scores (within-update z), held classes outlined. Right: per-layer AUC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    blues = LinearSegmentedColormap.from_list(
        "blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    cls = res["attack_classes"]
    keys = list(res["scores_primary"])
    s = np.array([[res["scores_primary"][k][c] for c in cls] for k in keys])
    z = zscore_rows(s)
    held = np.array([[c in res["truth"][k.split("/")[1]] for c in cls] for k in keys])
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.05, 1]}, facecolor="#fcfcfb"
    )
    im = ax.imshow(z, cmap=blues, vmin=-1.5, vmax=2.5, aspect="auto")
    for i, j in zip(*np.nonzero(held), strict=True):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ink, lw=2))
    ax.set_xticks(
        range(len(cls)), [c.replace("_", " ") for c in cls], rotation=35, ha="right", fontsize=9, color=ink
    )
    ax.set_yticks(
        range(len(keys)),
        [k.replace("r", "round ", 1).replace("/op_", " · op ") for k in keys],
        fontsize=8.5,
        color=ink,
    )
    for r in range(3, len(keys), 3):
        ax.axhline(r - 0.5, color="#fcfcfb", lw=3)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    p = res["primary"]["strict"]
    ax.set_title(
        f"Class scores read from each plaintext update (outlined = classes the operator holds)\n"
        f"AUC {p['presence_auc']:.3f} · exact top-3 {p['top3_exact']}/{len(keys)} · round 1 AUC {p['presence_auc_round1']:.3f}",
        fontsize=10,
        color=ink,
        loc="left",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("score, z within update", fontsize=8.5, color=muted)
    cb.outline.set_visible(False)
    pl = res["per_layer_auc_diagnostic"]
    x = np.arange(len(pl["all_rounds"]))
    bx.axhline(0.5, color=muted, lw=1, ls=(0, (4, 3)))
    bx.text(len(pl["all_rounds"]) + 0.8, 0.5, "chance", color=muted, fontsize=8.5, va="center")
    bx.plot(x, pl["all_rounds"], color="#2a78d6", lw=2, label="all 18 updates")
    bx.plot(x, pl["round1"], color="#eb6834", lw=2, label="round 1 only (3 updates)")
    bx.text(x[-1] + 0.8, pl["all_rounds"][-1], "all rounds", color="#2a78d6", fontsize=8.5, va="center")
    bx.text(x[-1] + 0.8, pl["round1"][-1] - 0.03, "round 1", color="#eb6834", fontsize=8.5, va="center")
    bx.set_xlim(-1, len(x) + 6)
    bx.set_ylim(0.15, 1.03)
    bx.set_xlabel("decoder layer (o_proj update)", fontsize=9, color=ink)
    bx.set_ylabel("presence AUC from this layer alone", fontsize=9, color=ink)
    bx.grid(axis="y", color=grid, lw=0.8)
    bx.legend(frameon=False, fontsize=8.5, loc="lower left")
    for sp in ["top", "right"]:
        bx.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        bx.spines[sp].set_color(grid)
    bx.tick_params(colors=muted, labelsize=8.5)
    bx.set_facecolor("#fcfcfb")
    bx.set_title(
        "Where the signal sits (diagnostic, not used by the attack)", fontsize=10, color=ink, loc="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, help="an --insecure-reference run directory")
    ap.add_argument("--partition", type=Path, help="ground truth: which classes each operator holds")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plot", type=Path, help="only redraw the figure from an existing result JSON")
    ap.add_argument("--cache", type=Path, default=Path("runs/cache/readout_weights.npz"))
    ap.add_argument("--null-seeds", type=int, default=200)
    a = ap.parse_args()
    if a.plot:
        plot(json.loads(a.plot.read_text()), a.plot.with_suffix(".png"))
        return
    if not (a.run and a.partition and a.out):
        ap.error("--run, --partition and --out are required")
    res = run_attack(a.run, a.partition, a.cache, a.null_seeds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    plot(res, a.out.with_suffix(".png"))
    p = res["primary"]
    print(
        json.dumps(
            {
                "strict": {k: v for k, v in p["strict"].items() if k != "top3_exact_by_update"},
                "cohort": {k: v for k, v in p["cohort"].items() if k != "top3_exact_by_update"},
                "permutation_test": res["permutation_test"],
                "null": res["null_shuffled_blocks"],
                "aggregate_control": {k: v for k, v in res["aggregate_control"].items() if k != "z_by_round"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
