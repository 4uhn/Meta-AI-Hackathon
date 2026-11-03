"""FFA-LoRA adapter handling: shared frozen A, trainable B, flatten/unflatten B.

With every lora_A frozen at the same seeded value, ΔW_i = B_i·A, so FedAvg of the B's is exactly FedAvg
of the ΔW's and the server only adds vectors, which additive HE supports directly.

Flat layout: lora_B names sorted lexicographically, each tensor row-major. Parties exchange a layout hash;
a mismatch would sum slots across different weights.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.numpy import load_file, save_file

WEIGHT_CAP = 10_000.0  # public constant: w_i = n_i / WEIGHT_CAP


def lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=0.0,
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def b_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return sorted(((n, p) for n, p in model.named_parameters() if "lora_B" in n), key=lambda x: x[0])


def a_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return sorted(((n, p) for n, p in model.named_parameters() if "lora_A" in n), key=lambda x: x[0])


def freeze_a(model: torch.nn.Module) -> None:
    """FFA-LoRA: lora_A frozen, lora_B trainable, everything else frozen."""
    for n, p in model.named_parameters():
        p.requires_grad = "lora_B" in n
    if not b_params(model):
        raise ValueError("model has no lora_B parameters")


def layout(model: torch.nn.Module) -> list[tuple[str, list[int]]]:
    return [(n, list(p.shape)) for n, p in b_params(model)]


def layout_hash(model: torch.nn.Module) -> str:
    return hashlib.sha256(json.dumps(layout(model)).encode()).hexdigest()


def a_hash(model: torch.nn.Module) -> str:
    """sha256 over every lora_A tensor (name + fp32 bytes). Must be identical at every party."""
    h = hashlib.sha256()
    for n, p in a_params(model):
        h.update(n.encode())
        h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def n_b_values(model: torch.nn.Module) -> int:
    return sum(p.numel() for _, p in b_params(model))


@torch.no_grad()
def flatten_b(model: torch.nn.Module) -> np.ndarray:
    """All lora_B values as one float64 vector (fp32 → fp64 is exact)."""
    return np.concatenate(
        [
            p.detach().to("cpu", torch.float32).reshape(-1).numpy().astype(np.float64)
            for _, p in b_params(model)
        ]
    )


@torch.no_grad()
def unflatten_b(model: torch.nn.Module, vec: np.ndarray) -> None:
    """Write `vec` back into the lora_B tensors (inverse of flatten_b)."""
    params = b_params(model)
    total = sum(p.numel() for _, p in params)
    if vec.shape != (total,):
        raise ValueError(f"vector has {vec.shape} values, layout needs {total}")
    off = 0
    for _, p in params:
        k = p.numel()
        p.copy_(torch.from_numpy(np.ascontiguousarray(vec[off : off + k])).reshape(p.shape).to(p.dtype))
        off += k


def create_initial_adapter(base: torch.nn.Module, cfg: dict, seed: int, out_dir: Path) -> PeftModel:
    """Aggregator, once per run: seeded A (Kaiming-uniform, PEFT default) and B = 0."""
    torch.manual_seed(seed)
    model = get_peft_model(base, lora_config(cfg))
    if np.any(flatten_b(model) != 0):
        raise AssertionError("PEFT initial lora_B is expected to be zero")
    model.save_pretrained(out_dir)
    return model


def load_adapter(base: torch.nn.Module, adapter_dir: Path, trainable: bool) -> PeftModel:
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=trainable)
    if trainable:
        freeze_a(model)
    return model


def weighted_payload(b: np.ndarray, n_examples: int) -> tuple[np.ndarray, float]:
    """Payload [w·B, w] with w = n / WEIGHT_CAP, so the encrypted sum yields a weighted FedAvg."""
    w = n_examples / WEIGHT_CAP
    return np.concatenate([w * b, [w]]), w


def fedavg_from_sum(agg: np.ndarray) -> np.ndarray:
    """Σ w_i B_i / Σ w_i from the decrypted aggregate [Σ w_i B_i, Σ w_i]."""
    if agg[-1] <= 0:
        raise ValueError(f"aggregate weight slot {agg[-1]} ≤ 0")
    return agg[:-1] / agg[-1]


def server_update(b_prev: np.ndarray, b_avg: np.ndarray, server_lr: float = 1.0) -> np.ndarray:
    """B_{r+1} = B_r + η_s (FedAvg − B_r), a FedOpt-style server step (Reddi et al. 2021).

    Uses only the decrypted aggregate and the public global adapter. η_s = 1 is plain FedAvg.
    """
    if server_lr == 1.0:
        return b_avg
    return b_prev + server_lr * (b_avg - b_prev)


# Adapter files read without a model (aggregator side). PEFT saves `...q_proj.lora_B.weight`; in the
# live model the same tensor is `...lora_B.default.weight`. The layout (and its hash) is defined on the
# live names, so file keys are mapped onto them.
ADAPTER_FILE = "adapter_model.safetensors"


def _live_name(file_key: str) -> str:
    return file_key.replace(".lora_B.weight", ".lora_B.default.weight")


def _b_keys(tensors: dict[str, np.ndarray]) -> list[str]:
    return sorted((k for k in tensors if ".lora_B." in k), key=_live_name)


def file_layout_hash(adapter_dir: Path) -> str:
    t = load_file(Path(adapter_dir) / ADAPTER_FILE)
    lay = [(_live_name(k), list(t[k].shape)) for k in _b_keys(t)]
    return hashlib.sha256(json.dumps(lay).encode()).hexdigest()


def read_b_file(adapter_dir: Path) -> np.ndarray:
    """flatten_b, read straight from a saved adapter (same order, fp64)."""
    t = load_file(Path(adapter_dir) / ADAPTER_FILE)
    return np.concatenate([t[k].astype(np.float64).reshape(-1) for k in _b_keys(t)])


def write_b_file(src_dir: Path, dst_dir: Path, vec: np.ndarray) -> None:
    """Copy adapter `src_dir` to `dst_dir` with every lora_B replaced from `vec` (A untouched)."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    t = load_file(src_dir / ADAPTER_FILE)
    keys = _b_keys(t)
    total = sum(t[k].size for k in keys)
    if vec.shape != (total,):
        raise ValueError(f"vector has {vec.shape} values, adapter needs {total}")
    off = 0
    for k in keys:
        n = t[k].size
        t[k] = np.ascontiguousarray(vec[off : off + n].reshape(t[k].shape).astype(t[k].dtype))
        off += n
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if f.name != ADAPTER_FILE:
            shutil.copy2(f, dst_dir / f.name)
    save_file(t, str(dst_dir / ADAPTER_FILE), metadata={"format": "pt"})
