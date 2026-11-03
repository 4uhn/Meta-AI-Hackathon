"""FFA-LoRA flatten/unflatten on a tiny random Llama (no download): bit-exact, layout hash, FFA freeze."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from horizon.model.lora import (
    a_hash,
    b_params,
    create_initial_adapter,
    fedavg_from_sum,
    flatten_b,
    layout_hash,
    load_adapter,
    n_b_values,
    unflatten_b,
    weighted_payload,
)

LORA = {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}


def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


@pytest.fixture
def adapter(tmp_path):
    create_initial_adapter(tiny(), LORA, seed=7, out_dir=tmp_path / "init")
    return tmp_path / "init"


def test_roundtrip_bit_exact(adapter):
    m = load_adapter(tiny(), adapter, trainable=True)
    v = np.random.default_rng(0).normal(0, 0.01, n_b_values(m)).astype(np.float32).astype(np.float64)
    unflatten_b(m, v)
    assert np.array_equal(flatten_b(m), v)
    # q: 64→64, k/v: 64→32 (2 kv heads × 16), o: 64→64; B is out×r → (64+32+32+64)·8 per layer × 2
    assert n_b_values(m) == (64 + 32 + 32 + 64) * 8 * 2
    with pytest.raises(ValueError):
        unflatten_b(m, v[:-1])


def test_shared_a_and_layout_identical_across_parties(adapter):
    a = load_adapter(tiny(), adapter, trainable=True)
    b = load_adapter(tiny(), adapter, trainable=True)
    assert a_hash(a) == a_hash(b)
    assert layout_hash(a) == layout_hash(b)
    assert np.all(flatten_b(a) == 0)


def test_initial_adapter_seeded(tmp_path):
    m1 = create_initial_adapter(tiny(), LORA, seed=7, out_dir=tmp_path / "x")
    m2 = create_initial_adapter(tiny(), LORA, seed=7, out_dir=tmp_path / "y")
    m3 = create_initial_adapter(tiny(), LORA, seed=8, out_dir=tmp_path / "z")
    assert a_hash(m1) == a_hash(m2) != a_hash(m3)


def test_ffa_freeze_and_training_moves_only_b(adapter):
    m = load_adapter(tiny(), adapter, trainable=True)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {n for n, _ in b_params(m)}
    ah = a_hash(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(0, 128, (2, 10))
    m(input_ids=x, labels=x).loss.backward()
    opt.step()
    assert a_hash(m) == ah
    assert np.any(flatten_b(m) != 0)


def test_weighted_fedavg_math():
    rng = np.random.default_rng(0)
    bs = [rng.normal(0, 0.01, 50) for _ in range(3)]
    ns = [867, 1352, 1182]
    agg = sum(weighted_payload(b, n)[0] for b, n in zip(bs, ns, strict=True))
    want = sum(n * b for b, n in zip(bs, ns, strict=True)) / sum(ns)
    assert np.allclose(fedavg_from_sum(agg), want, rtol=0, atol=1e-15)
