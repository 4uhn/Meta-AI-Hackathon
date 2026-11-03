"""The memory-saving paths change *how* loss and scores are computed, never *what*: label-position logits
equal the full-logits loss, micro-batched gradients equal the single-batch gradient, the scorer equals
a full-logits reference, and a bf16 base keeps the adapter in fp32 (so flatten stays exact)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from horizon.model.labels import collate, label_logits
from horizon.model.lora import b_params, create_initial_adapter, load_adapter
from horizon.model.train import train_local

LORA = {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}


def tiny(dtype=torch.float32):
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
    return LlamaForCausalLM(cfg).to(dtype)


def examples(n: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """Random prompts of varying length with 2–4 label tokens each (like `label<eot>`)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        p = torch.randint(1, 128, (int(torch.randint(5, 15, (1,), generator=g)),), generator=g).tolist()
        t = torch.randint(1, 128, (int(torch.randint(2, 5, (1,), generator=g)),), generator=g).tolist()
        out.append((p + t, [-100] * len(p) + t))
    return out


def peft_tiny(tmp_path, dtype=torch.float32):
    create_initial_adapter(tiny(dtype), LORA, seed=7, out_dir=tmp_path / "init")
    m = load_adapter(tiny(dtype), tmp_path / "init", trainable=True)
    with torch.no_grad():  # non-zero B so the adapter path matters
        for _, p in b_params(m):
            p.normal_(0, 0.05)
    return m


def test_label_logits_loss_equals_full_loss(tmp_path):
    m = peft_tiny(tmp_path)
    b = collate(examples(6), pad_id=0)
    want = m(**b).loss
    logits, tgt = label_logits(m, b["input_ids"], b["attention_mask"], b["labels"])
    assert torch.allclose(F.cross_entropy(logits, tgt), want, atol=1e-6)


def _grads(m) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for _, p in b_params(m)])


def test_micro_batch_gradient_equals_full_batch(tmp_path):
    data = examples(8, seed=1)
    m = peft_tiny(tmp_path)
    b = collate(data, pad_id=0)
    m(**b).loss.backward()
    full = _grads(m).clone()
    m.zero_grad(set_to_none=True)
    n_tok = sum(sum(t != -100 for t in y[1:]) for _, y in data)
    for s in range(0, 8, 2):
        mb = collate(data[s : s + 2], pad_id=0)
        logits, tgt = label_logits(m, mb["input_ids"], mb["attention_mask"], mb["labels"])
        (F.cross_entropy(logits, tgt, reduction="sum") / n_tok).backward()
    assert torch.allclose(_grads(m), full, atol=1e-6)


def test_train_local_micro_batching_matches(tmp_path):
    """Whole training steps agree: micro_batch 2 vs 8 give the same B after 2 optimiser steps."""

    class Tok:
        pad_token_id = 0
        eos_token_id = 0

    rows = [{"text": str(i), "label": "x"} for i in range(16)]
    data = examples(16, seed=2)
    import horizon.model.train as tr

    orig = tr.encode_example
    tr.encode_example = lambda tok, text, label: data[int(text)]
    try:
        out = []
        for mb in (8, 2):
            m = peft_tiny(tmp_path / str(mb))
            cfg = {"model": {"device": "cpu"}, "train": {"lr": 1e-2, "batch_size": 8, "micro_batch": mb}}
            train_local(m, Tok(), rows, cfg, epochs=1, seed=0, log=lambda *_: None)
            out.append(torch.cat([p.detach().reshape(-1) for _, p in b_params(m)]))
    finally:
        tr.encode_example = orig
    assert torch.allclose(out[0], out[1], atol=1e-5)


def test_bf16_base_keeps_fp32_adapter(tmp_path):
    m = peft_tiny(tmp_path, dtype=torch.bfloat16)
    assert {p.dtype for _, p in b_params(m)} == {torch.float32}
    assert {p.dtype for n, p in m.named_parameters() if "lora_A" in n} == {torch.float32}
    assert m.get_base_model().model.layers[0].mlp.up_proj.weight.dtype == torch.bfloat16
    b = collate(examples(3), pad_id=0)
    logits, _ = label_logits(m, b["input_ids"], b["attention_mask"], b["labels"])
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()


class StubTok:
    """Character-level stand-in: enough of the tokenizer API for prompt_text/target_text/score_labels."""

    pad_token_id = 0
    eos_token_id = 0
    eos_token = "#"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 127 + 1 for c in text]}

    def apply_chat_template(self, msgs, **_):
        return "S" + msgs[-1]["content"] + "A"


def test_score_labels_equals_full_logits_reference(tmp_path):
    from horizon.data.serialize import LABELS
    from horizon.model.labels import prompt_text, score_labels, target_text

    m = peft_tiny(tmp_path)
    tok = StubTok()
    flows = ["ab", "hello flow", "x", "q=1 r=2"]
    got = score_labels(m, tok, flows, "cpu", batch_flows=3)  # uneven last chunk on purpose
    m.eval()
    with torch.no_grad():
        for i, f in enumerate(flows):
            p = tok(prompt_text(tok, f))["input_ids"]
            for j, lab in enumerate(LABELS):
                t = tok(target_text(tok, lab))["input_ids"]
                lp = F.log_softmax(m(input_ids=torch.tensor([p + t])).logits[0].float(), -1)
                want = sum(lp[len(p) - 1 + k, t[k]].item() for k in range(len(t)))
                assert abs(got[i, j] - want) < 1e-4
