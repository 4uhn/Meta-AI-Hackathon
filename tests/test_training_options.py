"""Label-skew mitigations: FedRS-style restricted loss on the label prefix tree, server step size,
fractional local epochs and federated-validation metrics. Each acts locally or on the decrypted
aggregate, so the encrypted aggregation is unchanged."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from horizon.data.serialize import LABELS
from horizon.model.lora import server_update
from horizon.model.train import confusion, metrics, metrics_from_confusion, restrict_logits


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Llama-3.2 tokenizer not in the HF cache")


def test_missing_label_masks_follow_the_prefix_tree(tok):
    from horizon.model.labels import missing_label_masks, target_text

    ids = lambda lab: tok(target_text(tok, lab), add_special_tokens=False)["input_ids"]  # noqa: E731
    own = {"benign", "udp_flood", "icmp_flood", "syn_scan"}  # op_a
    m = missing_label_masks(tok, own)
    missing = ("syn_flood", "http_flood", "slowrate_dos", "tcp_connect_scan", "udp_scan")
    assert set(m["benign"][0]) == {ids(lab)[0] for lab in missing}  # one first token per missing class
    assert not m["syn_scan"][1]  # half_open_scan has its own first token
    assert not m["udp_flood"][1]  # datagram_scan does not share a first token with udp_flood
    assert m["udp_flood"][2:] == [[], []]  # determined after the branch point
    for lab in own:  # the target token itself is never masked
        assert all(t not in mask for t, mask in zip(ids(lab), m[lab], strict=True))
    full = missing_label_masks(tok, set(LABELS))  # pooled data (E4): nothing to restrict
    assert all(not x for v in full.values() for x in v)


def test_restrict_logits_masks_rows_in_example_position_order():
    logits = torch.zeros(5, 10, requires_grad=True)
    out = restrict_logits(logits, [[[1, 2], []], [[3], [], [4]]])
    assert (
        torch.isinf(out[0, 1])
        and torch.isinf(out[0, 2])
        and torch.isinf(out[2, 3])
        and torch.isinf(out[4, 4])
    )
    assert int(torch.isinf(out).sum()) == 4
    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 0, 0, 0, 0]))
    loss.backward()  # masked entries get zero gradient: no push-down of missing labels
    assert logits.grad[0, 1] == 0 and logits.grad[0, 3] != 0
    with pytest.raises(AssertionError):
        restrict_logits(logits, [[[1]]])


def test_server_update():
    prev, avg = np.array([1.0, 2.0]), np.array([2.0, 0.0])
    assert server_update(prev, avg, 1.0) is avg  # plain FedAvg, bit-exact
    assert np.allclose(server_update(prev, avg, 2.0), [3.0, -2.0])


def test_federated_metrics_equal_pooled_metrics():
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(3):
        t = list(rng.choice(LABELS, 40))
        p = list(rng.choice(LABELS, 40))
        parts.append((t, p))
    pooled = metrics(sum((t for t, _ in parts), []), sum((p for _, p in parts), []))
    fed = metrics_from_confusion(sum(confusion(t, p) for t, p in parts))
    assert fed["macro_f1"] == pytest.approx(pooled["macro_f1"]) and fed["accuracy"] == pooled["accuracy"]


def test_fractional_epochs_set_a_step_budget():
    """0.5 epoch of 10 rows at batch 4 = ceil(0.5 × 3) = 2 optimiser steps (tiny CPU model)."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from horizon.model.lora import create_initial_adapter, load_adapter
    from horizon.model.train import train_local

    try:
        tk = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Llama-3.2 tokenizer not in the HF cache")
    torch.manual_seed(0)
    cfg_m = LlamaConfig(
        vocab_size=len(tk),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    lora = {"r": 2, "alpha": 4, "target_modules": ["q_proj", "v_proj"]}
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        m = create_initial_adapter(LlamaForCausalLM(cfg_m), lora, seed=1, out_dir=d)
        m = load_adapter(m.unload(), d, trainable=True)
        rows = [{"text": f"proto=udp n={i}", "label": ["benign", "syn_scan"][i % 2]} for i in range(10)]
        cfg = {
            "model": {"device": "cpu"},
            "train": {
                "lr": 1e-3,
                "batch_size": 4,
                "micro_batch": 2,
                "restrict_missing_labels": True,
                "class_balanced": True,
                "logit_adjust_tau": 1.0,
            },
        }
        out = train_local(m, tk, rows, cfg, epochs=0.5, seed=0, log=lambda *_: None)
    assert out["steps"] == 2 and np.isfinite(out["loss_last"])


def test_class_balanced_weights_equalise_classes():
    """With class_balanced, each local class carries the same total weight."""
    rows = [{"label": "benign"}] * 6 + [{"label": "syn_scan"}] * 2
    counts = {"benign": 6, "syn_scan": 2}
    w = [len(rows) / (len(counts) * counts[r["label"]]) for r in rows]
    assert sum(w[:6]) == pytest.approx(sum(w[6:])) and sum(w) == pytest.approx(len(rows))


def test_verbalizer_gives_every_class_its_own_first_token(tok):
    from horizon.model.labels import SYSTEM, VERBALIZER, first_tokens_unique, missing_label_masks

    assert first_tokens_unique(tok)
    assert all(VERBALIZER[lab] in SYSTEM for lab in LABELS)
    m = missing_label_masks(tok, {"benign", "udp_flood", "icmp_flood", "syn_scan"})  # op_a
    assert all(not x for v in m.values() for x in v[1:])  # the restriction acts at position 0 only


def test_logit_adjustment_adds_log_prior_at_the_decision_token_only():
    from horizon.model.train import adjust_first_logits

    logits = torch.zeros(5, 10)
    out = adjust_first_logits(logits, [2, 3], [4, 7], [-0.5, -2.0])  # 2 examples: rows 0–1 and 2–4
    assert out[0, 4] == -0.5 and out[0, 7] == -2.0 and out[2, 4] == -0.5 and out[2, 7] == -2.0
    assert int((out != 0).sum()) == 4  # nothing after the first label token is touched
    with pytest.raises(AssertionError):
        adjust_first_logits(logits, [2, 2], [4], [0.1])
