"""Base-model loading, local FFA-LoRA training and evaluation, shared by every experiment."""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

from horizon.data.serialize import LABELS
from horizon.model.labels import (
    collate,
    encode_example,
    first_tokens_unique,
    label_logits,
    missing_label_masks,
    score_first_tokens,
    score_labels,
    target_text,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# model.loader → transformers auto class. Muse Glimmer is a VLM: AutoModelForCausalLM rejects its config,
# so it loads through AutoModelForImageTextToText.
LOADERS = {"causal_lm": "AutoModelForCausalLM", "image_text_to_text": "AutoModelForImageTextToText"}


def load_base(cfg: dict):
    """Base model, tokenizer and resolved revision.

    With `model.quantized` (Glimmer nf4 checkpoint) the model is placed by device_map, since 4-bit weights
    can't be moved with `.to()`; `model.tokenizer_id` points at the canonical repo's tokenizer/template."""
    m = cfg["model"]
    if m["device"].startswith("cuda"):  # deterministic kernels: plaintext ≡ encrypted up to CKKS error
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(
        m.get("tokenizer_id", m["id"]), revision=m.get("tokenizer_revision", m["revision"])
    )
    cls = getattr(transformers, LOADERS[m.get("loader", "causal_lm")])
    kw = {"dtype": getattr(torch, m["dtype"]), "attn_implementation": m.get("attn_implementation", "sdpa")}
    if m.get("quantized"):
        dev = "cuda:0" if m["device"] == "cuda" else m["device"]
        model = cls.from_pretrained(m["id"], revision=m["revision"], device_map=dev, **kw)
    else:
        model = cls.from_pretrained(m["id"], revision=m["revision"], **kw).to(m["device"])
    model.config.use_cache = False
    if m.get("gradient_checkpointing"):  # trades compute for activation memory
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    rev = getattr(model.config, "_commit_hash", None) or m["revision"]
    return model, tok, rev


def mps_mem_gb(device: str) -> float:
    """Memory the MPS driver holds for this process (GB); 0 on any other device."""
    return torch.mps.driver_allocated_memory() / 2**30 if device == "mps" else 0.0


def cuda_peak_gb() -> float:
    """Peak CUDA memory allocated by this process since the last reset (GiB); 0 without CUDA."""
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def train_local(
    model, tok, rows: list[dict], cfg: dict, epochs: float, seed: int, log=print, max_steps: int = 0
) -> dict:
    """Local training: AdamW from fresh state each round, loss on label tokens only.

    Micro-batch losses are divided by the label-token count of the whole step, so gradient accumulation
    equals the single-batch gradient exactly. `epochs` may be fractional (a step budget).

    Options (`train.*`, all derived from this shard only):
    - restrict_missing_labels: drop tokens leading only to absent labels from the softmax (FedRS-style).
    - logit_adjust_tau = τ: add τ·log π(label) to the decision-token logits during training, π = local
      label frequencies (logit adjustment, Menon et al. ICLR'21; FedLC, Zhang et al. ICML'22), so the
      shared adapter learns prior-free scores. Needs unique first tokens; inference is unchanged.
    - class_balanced: weight examples n / (K · n_label) so each local class has equal total weight.
    """
    tc, device = cfg["train"], cfg["model"]["device"]
    bs, mb = tc["batch_size"], tc.get("micro_batch", tc["batch_size"])
    if bs % mb:
        raise ValueError(f"batch_size {bs} is not a multiple of micro_batch {mb}")
    seed_all(seed)
    data = [encode_example(tok, r["text"], r["label"]) for r in rows]
    steps_per_epoch = math.ceil(len(data) / bs)
    budget = math.ceil(epochs * steps_per_epoch)  # fractional epochs → a step budget
    max_steps = min(max_steps, budget) if max_steps else budget
    epochs = math.ceil(epochs)
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in {r["label"] for r in rows}}
    if tc.get("class_balanced"):
        ex_w = [len(rows) / (len(counts) * counts[r["label"]]) for r in rows]
    else:
        ex_w = [1.0] * len(rows)
    n_lab = [sum(t != -100 for t in y[1:]) for _, y in data]  # label positions per example
    masks = None
    if tc.get("restrict_missing_labels"):
        by_label = missing_label_masks(tok, {r["label"] for r in rows})
        masks = [by_label[r["label"]] for r in rows]
    tau = float(tc.get("logit_adjust_tau", 0.0))
    adj = None
    if tau:
        if not first_tokens_unique(tok):
            raise ValueError("logit adjustment needs one first token per class")
        first = {lab: tok(target_text(tok, lab), add_special_tokens=False)["input_ids"][0] for lab in counts}
        adj = ([first[lab] for lab in counts], [tau * math.log(counts[lab] / len(rows)) for lab in counts])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=0.0)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0, losses, step, peak = time.time(), [], 0, 0.0
    for ep in range(epochs):
        if max_steps and step >= max_steps:
            break
        order = np.random.default_rng(seed + ep).permutation(len(data))
        for s in range(0, len(order), bs):
            idx = [int(i) for i in order[s : s + bs]]
            batch = [data[i] for i in idx]
            n_tok = sum(ex_w[i] * n_lab[i] for i in idx)  # (weighted) label tokens in the whole step
            step_loss = 0.0
            for m in range(0, len(batch), mb):
                b = {k: v.to(device) for k, v in collate(batch[m : m + mb], pad).items()}
                logits, tgt = label_logits(model, b["input_ids"], b["attention_mask"], b["labels"])
                if masks is not None:
                    logits = restrict_logits(logits, [masks[i] for i in idx[m : m + mb]])
                if adj is not None:
                    logits = adjust_first_logits(logits, [n_lab[i] for i in idx[m : m + mb]], *adj)
                ce = F.cross_entropy(logits, tgt, reduction="none")
                w = torch.tensor(
                    [ex_w[i] for i in idx[m : m + mb] for _ in range(n_lab[i])],
                    device=ce.device,
                    dtype=ce.dtype,
                )
                loss = (ce * w).sum() / n_tok
                if not torch.isfinite(loss):  # fail fast: one NaN step would corrupt B permanently
                    bad = [int(i) for i in order[s + m : s + m + mb]]
                    raise FloatingPointError(f"non-finite loss at step {step} (rows {bad})")
                loss.backward()
                step_loss += loss.item()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params, float("inf"))):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(step_loss)
            step += 1
            peak = max(peak, mps_mem_gb(device))
            if step % 20 == 0 or step == max_steps:
                log(
                    f"  step {step} ep {ep} loss {np.mean(losses[-20:]):.4f} "
                    f"({time.time() - t0:.0f}s, mps {peak:.2f} GB)"
                )
            if max_steps and step >= max_steps:
                break
    return {
        "steps": step,
        "n": len(data),
        "loss_first": float(np.mean(losses[:5])),
        "loss_last": float(np.mean(losses[-10:])),
        "ms": int(1000 * (time.time() - t0)),
        "ms_per_step": int(1000 * (time.time() - t0) / max(step, 1)),
        "mps_peak_gb": round(peak, 2),
        "cuda_peak_gb": round(cuda_peak_gb(), 2),
    }


def adjust_first_logits(
    logits: torch.Tensor, n_pos: list[int], token_ids: list[int], biases: list[float]
) -> torch.Tensor:
    """Add `biases` to `token_ids` at each example's first label position (its decision token).

    Rows are label positions in (example, position) order; example e has n_pos[e] of them."""
    starts = np.cumsum([0, *n_pos[:-1]]).tolist()
    if sum(n_pos) != logits.shape[0]:
        raise AssertionError(f"{sum(n_pos)} label positions, {logits.shape[0]} logit rows")
    dev = logits.device
    rows = torch.tensor([r for r in starts for _ in token_ids], device=dev)
    cols = torch.tensor(token_ids * len(starts), device=dev)
    vals = torch.tensor(biases * len(starts), device=dev, dtype=logits.dtype)
    return logits.index_put((rows, cols), vals, accumulate=True)


def restrict_logits(logits: torch.Tensor, ex_masks: list[list[list[int]]]) -> torch.Tensor:
    """Set masked next-token logits to −inf.

    Rows are label positions in (example, position) order; `ex_masks[e][j]` = ids masked at position j."""
    rows, cols, r = [], [], 0
    for per_pos in ex_masks:
        for ids in per_pos:
            rows += [r] * len(ids)
            cols += ids
            r += 1
    if r != logits.shape[0]:
        raise AssertionError(f"{r} label positions in masks, {logits.shape[0]} in logits")
    if not rows:
        return logits
    dev = logits.device
    ninf = torch.tensor(float("-inf"), device=dev, dtype=logits.dtype)
    return logits.index_put((torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)), ninf)


def confusion(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    """Confusion counts, rows = true label, columns = predicted (LABELS order)."""
    c = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        c[LABELS.index(t), LABELS.index(p)] += 1
    return c


def metrics_from_confusion(c: np.ndarray) -> dict:
    """`metrics` from (summed) confusion counts, for federated evaluation."""
    c = np.asarray(c)
    y_true = [LABELS[i] for i in range(len(LABELS)) for _ in range(int(c[i].sum()))]
    y_pred = [LABELS[j] for i in range(len(LABELS)) for j in range(len(LABELS)) for _ in range(int(c[i, j]))]
    return metrics(y_true, y_pred)


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    per = {}
    f1s = []
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        n_t = sum(t == lab for t in y_true)
        n_p = sum(p == lab for p in y_pred)
        rec = tp / n_t if n_t else float("nan")
        prec = tp / n_p if n_p else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per[lab] = {"recall": rec, "precision": prec, "f1": f1, "support": n_t}
        f1s.append(f1)
    acc = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "per_class": per}


def evaluate(model, tok, rows: list[dict], cfg: dict) -> tuple[dict, np.ndarray]:
    t0 = time.time()
    # full = every label continuation teacher-forced (primary); first_token = fast path
    score = {"full": score_labels, "first_token": score_first_tokens}[cfg["eval"].get("scoring", "full")]
    scores = score(model, tok, [r["text"] for r in rows], cfg["model"]["device"], cfg["eval"]["batch_flows"])
    pred = [LABELS[i] for i in scores.argmax(1)]
    m = metrics([r["label"] for r in rows], pred)
    m["eval_ms"] = int(1000 * (time.time() - t0))
    return m, scores
