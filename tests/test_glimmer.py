"""Muse Glimmer code path on CPU: chat format, logits, LoRA targets, fast-path scoring.

The tokenizer tests need meta-models/Muse-Glimmer-30B @ cec94fc in the HF cache (tokenizer files only,
not gated). The model tests build a tiny random MuseGlimmerForConditionalGeneration (real module
structure, real vocabulary size) or the full-size architecture on the meta device (no memory).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from peft import get_peft_model

from horizon.data.serialize import LABELS
from horizon.model import labels as L
from horizon.model.lora import b_params, freeze_a, lora_config, n_b_values

GLIMMER_CFG = yaml.safe_load(open("configs/5g-nidd-glimmer.yaml"))
LLAMA_CFG = yaml.safe_load(open("configs/5g-nidd-llama.yaml"))
TOK_ID = GLIMMER_CFG["model"]["tokenizer_id"]
TOK_REV = GLIMMER_CFG["model"]["tokenizer_revision"]
FLOW = "proto=tcp state=CON dur=1.99 pkts=6/11 ttl=63/59"
EOT_ID = 200008  # id of <|eot|> in the Glimmer vocabulary


@pytest.fixture(scope="module")
def gtok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TOK_ID, revision=TOK_REV, local_files_only=True)
    except OSError:
        pytest.skip("Muse Glimmer tokenizer not in the HF cache")


@pytest.fixture(scope="module")
def ltok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Llama-3.2 tokenizer not in the HF cache")


def test_glimmer_prompt_forces_the_user_channel_and_targets_end_with_eot(gtok):
    assert L.chat_format(gtok) == (" to=user<|message|>", "<|eot|>")
    p = L.prompt_text(gtok, FLOW)
    assert p.startswith("<|begin_of_text|><|start|>system<|message|>" + L.SYSTEM)
    assert p.endswith("<|start|>assistant to=user<|message|>")
    assert "Reasoning strength: low." in p and "Current date" not in p  # no date stamp → deterministic
    for lab in LABELS:
        ids, lab_ids = L.encode_example(gtok, FLOW, lab)
        tgt = [t for t in lab_ids if t != -100]
        assert tgt[-1] == EOT_ID and ids[-len(tgt) :] == tgt
        assert gtok.decode(tgt) == L.VERBALIZER[lab] + "<|eot|>"
        assert ids.count(gtok.bos_token_id) == 1 and ids[0] == gtok.bos_token_id
        # separate tokenisation of prompt and target = tokenising the joined string (no boundary merge)
        joined = gtok(L.prompt_text(gtok, FLOW) + L.target_text(gtok, lab), add_special_tokens=False)
        assert joined["input_ids"] == ids


def test_glimmer_prompt_plus_target_is_the_templates_own_reply(gtok):
    """What we train on is exactly how the model's template renders a one-turn reply to the user."""
    for lab in LABELS:
        msgs = [
            {"role": "system", "content": L.SYSTEM},
            {"role": "user", "content": FLOW},
            {"role": "assistant", "content": L.VERBALIZER[lab]},
        ]
        rendered = gtok.apply_chat_template(msgs, tokenize=False, reasoning_strength=L.REASONING_STRENGTH)
        assert rendered == L.prompt_text(gtok, FLOW) + L.target_text(gtok, lab)


def test_glimmer_v2_labels_have_unique_first_tokens_and_one_restriction_position(gtok):
    assert L.first_tokens_unique(gtok)
    own = {"benign", "udp_flood", "syn_flood", "syn_scan"}
    masks = L.missing_label_masks(gtok, own)
    firsts = {lab: gtok(L.target_text(gtok, lab), add_special_tokens=False)["input_ids"][0] for lab in LABELS}
    for lab in own:
        assert sorted(masks[lab][0]) == sorted(firsts[m] for m in LABELS if m not in own)
        assert all(not m for m in masks[lab][1:])  # unique first tokens → only the decision is masked


def test_llama_encoding_is_unchanged(ltok):
    """Golden fingerprint of the Llama prompt, targets, ids and masks: Glimmer support must not change
    the Llama encoding, so the Llama results stay reproducible."""
    assert L.chat_format(ltok) == ("", "<|eot_id|>")
    blob = {
        "prompt": L.prompt_text(ltok, FLOW),
        "targets": [L.target_text(ltok, lab) for lab in LABELS],
        "enc": [L.encode_example(ltok, FLOW, lab) for lab in LABELS],
        "masks": L.missing_label_masks(ltok, {"benign", "udp_flood", "syn_scan"}),
    }
    h = hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()
    assert h == "ece76db26055623862fd28711f695961595c5c5fa351754fbc5f5c34e59cc925"


def _glimmer_config(tiny: bool):
    from transformers import MuseGlimmerConfig

    if tiny:
        text = dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=202048,
        )
        vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2)
        return MuseGlimmerConfig(
            text_config=text, vision_config=vision, out_hidden_size=128, projector_hidden_size=64
        )
    # the 30B checkpoint's dimensions (config.json @ ecedda3 / cec94fc)
    text = dict(
        hidden_size=6656,
        intermediate_size=19968,
        num_hidden_layers=52,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=202048,
    )
    return MuseGlimmerConfig(text_config=text)


@pytest.fixture(scope="module")
def tiny():
    from transformers import MuseGlimmerForConditionalGeneration

    torch.manual_seed(0)
    m = MuseGlimmerForConditionalGeneration(_glimmer_config(tiny=True)).float().eval()
    m.config.use_cache = False
    return m


def test_lora_regex_targets_language_attention_only_and_payload_is_4685824():
    from transformers import MuseGlimmerForConditionalGeneration

    with torch.device("meta"):
        m = MuseGlimmerForConditionalGeneration(_glimmer_config(tiny=False))
        pm = get_peft_model(m, lora_config(GLIMMER_CFG["lora"]))
    lora_mods = {n.rsplit(".lora_B", 1)[0] for n, _ in b_params(pm)}
    assert len(lora_mods) == 52 * 4
    assert all(".language_model.layers." in n and ".self_attn." in n for n in lora_mods)
    assert not any("vision" in n for n, _ in pm.named_parameters() if "lora_" in n)
    assert n_b_values(pm) == 4_685_824  # = the payload the crypto layer was sized and benchmarked for


def test_label_logits_match_the_wrappers_softcapped_logits(tiny):
    """label_logits calls the decoder + head directly, so it must redo Glimmer's T·tanh(m·z/T)."""
    torch.manual_seed(1)
    ids = torch.randint(0, 200000, (2, 12))
    att = torch.ones_like(ids)
    lab = torch.full_like(ids, -100)
    lab[:, 9:] = ids[:, 9:]
    got, tgt = L.label_logits(tiny, ids, att, lab)
    with torch.no_grad():
        ref = tiny(input_ids=ids, attention_mask=att).logits[:, :-1][lab[:, 1:] != -100]
    assert tgt.tolist() == lab[:, 1:][lab[:, 1:] != -100].tolist()
    assert ref.abs().max() <= 20.0 + 1e-4  # soft-capped at T = 20
    torch.testing.assert_close(got, ref.float(), atol=1e-4, rtol=1e-4)


def test_first_token_fast_path_equals_full_decision_logprobs(gtok, tiny):
    flows = [FLOW, "proto=udp state=INT dur=0.00 pkts=1/0 ttl=64/na"]
    fast = L.score_first_tokens(tiny, gtok, flows, "cpu", batch_flows=2)
    firsts = [gtok(L.target_text(gtok, lab), add_special_tokens=False)["input_ids"][0] for lab in LABELS]
    for i, f in enumerate(flows):
        p = gtok(L.prompt_text(gtok, f), add_special_tokens=False, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            lp = F.log_softmax(tiny(input_ids=p).logits[0, -1].float(), -1)
        np.testing.assert_allclose(fast[i], lp[firsts].double().numpy(), atol=1e-4)
    full = L.score_labels(tiny, gtok, flows, "cpu", batch_flows=2)
    assert full.shape == fast.shape == (2, 9)
    assert np.all(full <= fast + 1e-6)  # a full label's log-prob ≤ its first token's


def test_glimmer_local_training_moves_only_b_with_the_golden_recipe(gtok, tiny):
    from horizon.model.train import train_local

    pm = get_peft_model(tiny, lora_config(GLIMMER_CFG["lora"] | {"r": 2}))
    freeze_a(pm)
    a0 = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_A" in n}
    rows = [{"text": FLOW, "label": lab} for lab in ("benign", "benign", "udp_flood", "syn_scan")]
    cfg = {"model": {"device": "cpu"}, "train": GLIMMER_CFG["train"] | {"batch_size": 4, "micro_batch": 2}}
    tr = train_local(pm, gtok, rows, cfg, epochs=1, seed=0)
    assert tr["steps"] == 1 and np.isfinite(tr["loss_last"])
    assert any(p.abs().sum() > 0 for _, p in b_params(pm))  # B left zero init
    assert all(torch.equal(a0[n], p) for n, p in pm.named_parameters() if "lora_A" in n)
