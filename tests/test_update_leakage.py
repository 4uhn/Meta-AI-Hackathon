"""Update-leakage attack: geometry, metrics and the weight reader, on small synthetic inputs (no network)."""

import itertools
import json
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file

from horizon.eval.update_leakage import (
    auc,
    bf16_to_f32,
    exact_topk,
    layer_scores,
    o_proj_layout,
    permutation_p,
    pooled_auc,
    read_tensor,
    readout_directions,
    safetensors_header,
    total_score,
    zscore_rows,
)

D, D_IN, R, LAYERS, LABELS = 64, 32, 4, 3, 9


def toy(seed=0):
    rng = np.random.default_rng(seed)
    w = {
        "lm_head_rows": rng.normal(size=(LABELS, D)),
        "final_norm": 1 + 0.1 * rng.normal(size=D),
        "post_attn_norm": 0.1 * rng.normal(size=(LAYERS, D)),
    }
    a = {ly: rng.normal(size=(R, D_IN)) for ly in range(LAYERS)}
    blocks = [(ly, ly * D * R, D * R) for ly in range(LAYERS)]
    return rng, w, a, blocks


def planted_update(rng, readout, a, held, strength=1.0, noise=0.05):
    """ΔB_ℓ whose columns write along the held classes' readout directions (what a gradient step does)."""
    out = []
    for ly in range(LAYERS):
        dirs = readout[ly][held]
        db = dirs.T @ rng.normal(size=(len(held), R)) * strength / np.linalg.norm(dirs, axis=1).mean()
        out.append((db + noise * rng.normal(size=db.shape)).reshape(-1))
    return np.concatenate(out)


def test_planted_classes_are_recovered_and_shuffled_are_not():
    rng, w, a, blocks = toy()
    ro = readout_directions(w)
    held_sets = [[1, 2, 6], [4, 5, 7], [3, 8, 6]]
    s, s_null = [], []
    for held in held_sets:
        d = planted_update(rng, ro, a, held)
        s.append(total_score(layer_scores(d, blocks, a, ro)))
        sh = d.copy()
        for _, off, size in blocks:
            sh[off : off + size] = rng.permutation(sh[off : off + size])
        s_null.append(total_score(layer_scores(sh, blocks, a, ro)))
    truth = np.zeros((3, LABELS), bool)
    for i, h in enumerate(held_sets):
        truth[i, h] = True
    assert exact_topk(np.stack(s), truth).all()
    assert pooled_auc(zscore_rows(np.stack(s)), truth) == 1.0
    assert pooled_auc(zscore_rows(np.stack(s_null)), truth) < 0.9


def test_layer_score_matches_explicit_delta_w():
    rng, w, a, blocks = toy(1)
    ro = readout_directions(w)
    d = rng.normal(size=LAYERS * D * R)
    got = layer_scores(d, blocks, a, ro)
    for ly, off, size in blocks:
        dw = d[off : off + size].reshape(D, R) @ a[ly]  # ΔW = ΔB·A, (d_model, d_in)
        want = np.linalg.norm(ro[ly] @ dw, axis=1) / np.linalg.norm(ro[ly], axis=1)
        np.testing.assert_allclose(got[ly], want, rtol=1e-10)


def test_readout_is_centred_and_scaled_per_layer():
    _, w, _, _ = toy(2)
    ro = readout_directions(w)
    np.testing.assert_allclose(ro.sum(1), 0, atol=1e-10)  # centred across labels in every layer
    raw = w["lm_head_rows"] * w["final_norm"]
    np.testing.assert_allclose(ro[1], (1 + w["post_attn_norm"][1]) * (raw - raw.mean(0)), rtol=1e-12)


def test_auc_matches_pair_counting():
    rng = np.random.default_rng(3)
    pos, neg = rng.integers(0, 5, 7).astype(float), rng.integers(0, 5, 9).astype(float)
    pairs = [(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)]
    assert auc(pos, neg) == pytest.approx(np.mean(pairs))


def test_permutation_p_is_exact_on_a_small_case():
    s = np.array([[3.0, 2.0, 1.0, 0.0]])
    truth = np.array([[True, False, False, False]])
    p, n = permutation_p(lambda t: pooled_auc(s, t), truth)
    assert n == 24 and p == pytest.approx(6 / 24)  # class 0 is top under 6 of the 24 relabelings


def test_bf16_reader_reads_rows_by_range(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 6)).astype(np.float32)
    bf = (x.view(np.uint32) >> 16).astype("<u2")  # truncate to bf16
    data = bf.tobytes()
    hdr = json.dumps({"w": {"dtype": "BF16", "shape": [5, 6], "data_offsets": [0, len(data)]}}).encode()
    blob = struct.pack("<Q", len(hdr)) + hdr + data

    def read(a, b):
        return blob[a : b + 1]

    h, base = safetensors_header(read)
    want = bf16_to_f32(bf.tobytes()).reshape(5, 6)
    np.testing.assert_array_equal(read_tensor(read, h, base, "w", rows=[3, 1]), want[[3, 1]])
    np.testing.assert_array_equal(read_tensor(read, h, base, "w"), want)
    assert np.abs(want - x).max() < 1e-2 * np.abs(x).max()


def test_o_proj_layout_follows_the_flat_b_order(tmp_path):
    rng = np.random.default_rng(5)
    t = {}
    for ly in [0, 1, 10, 2]:
        for m, rows in [("k_proj", 3), ("o_proj", 7), ("q_proj", 5), ("v_proj", 3)]:
            pre = f"base_model.model.model.language_model.layers.{ly}.self_attn.{m}"
            t[f"{pre}.lora_B.weight"] = rng.normal(size=(rows, R)).astype(np.float32)
            t[f"{pre}.lora_A.weight"] = rng.normal(size=(R, 6)).astype(np.float32)
    save_file(t, str(tmp_path / "adapter_model.safetensors"))
    blocks, a, total = o_proj_layout(tmp_path)
    from horizon.model.lora import read_b_file

    flat = read_b_file(tmp_path)
    assert total == flat.size
    for layer, off, size in blocks:
        key = f"base_model.model.model.language_model.layers.{layer}.self_attn.o_proj.lora_B.weight"
        np.testing.assert_array_equal(flat[off : off + size].reshape(7, R), t[key])
        np.testing.assert_array_equal(a[layer], t[key.replace("lora_B", "lora_A")])
    assert [b[0] for b in blocks] == [0, 1, 2, 10]


def test_class_gradient_matches_finite_differences_and_is_micro_batch_invariant():
    """∂L/∂B from probe_gradients: central differences along a random direction, any micro-batch split."""
    import tempfile

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from horizon.eval.probe_gradients import class_gradient
    from horizon.model.labels import collate, encode_example, label_logits
    from horizon.model.lora import create_initial_adapter, flatten_b, load_adapter, unflatten_b

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
        torch_dtype="float64",
    )
    lora = {"r": 2, "alpha": 4, "target_modules": ["q_proj", "o_proj"]}
    rows = [{"text": f"proto=udp n={i}", "label": "syn_scan"} for i in range(5)]
    with tempfile.TemporaryDirectory() as d:
        m = create_initial_adapter(LlamaForCausalLM(cfg_m).double(), lora, seed=1, out_dir=d)
        m = load_adapter(m.unload(), d, trainable=True).double()
        rng = np.random.default_rng(0)
        b0 = rng.normal(0, 0.05, flatten_b(m).size)
        unflatten_b(m, b0)
        g1 = class_gradient(m, tk, rows, micro=5, device="cpu")
        g2 = class_gradient(m, tk, rows, micro=2, device="cpu")
        np.testing.assert_allclose(g1, g2, rtol=1e-9, atol=1e-12)

        data = [encode_example(tk, r["text"], r["label"]) for r in rows]
        b = collate(data, tk.eos_token_id)

        def loss(vec):
            unflatten_b(m, vec)
            with torch.no_grad():
                lg, tgt = label_logits(m, b["input_ids"], b["attention_mask"], b["labels"])
                return float(F.cross_entropy(lg.double(), tgt, reduction="mean"))

        v = rng.normal(size=b0.size)
        # label_logits returns fp32 logits, so the step must be large enough for fp32 loss values to resolve
        # it; the O(eps²) truncation error is checked by the two step sizes agreeing
        fd = [(loss(b0 + e * v) - loss(b0 - e * v)) / (2 * e) for e in (3e-2, 1e-2)]
        assert fd[0] == pytest.approx(fd[1], rel=2e-3)
        assert float(g1 @ v) == pytest.approx(fd[1], rel=2e-3)


def test_select_rows_is_seeded_and_per_class():
    from horizon.data.serialize import LABELS
    from horizon.eval.probe_gradients import select_rows

    rows = [{"label": lab} for lab in LABELS for _ in range(5)]
    a, b = select_rows(rows, 3, 7), select_rows(rows, 3, 7)
    assert a == b and all(len(v) == 3 and all(rows[i]["label"] == k for i in v) for k, v in a.items())
    with pytest.raises(ValueError):
        select_rows(rows, 6, 7)


def test_nnls_matches_brute_force_over_active_sets():
    from horizon.eval.update_leakage_probe import nnls

    rng = np.random.default_rng(6)
    g, y = rng.normal(size=(4, 30)), rng.normal(size=30)
    got = nnls(g @ g.T, g @ y)
    best = (np.inf, None)
    for mask in itertools.product([0, 1], repeat=4):  # exact: least squares on every support, keep feasible
        idx = [i for i in range(4) if mask[i]]
        a = np.zeros(4)
        if idx:
            a[idx] = np.linalg.lstsq(g[idx].T, y, rcond=None)[0]
        if (a >= 0).all():
            best = min(best, (float(np.sum((y - g.T @ a) ** 2)), a.tolist()), key=lambda t: t[0])
    np.testing.assert_allclose(got, best[1], atol=1e-8)


def test_probe_scores_recover_a_planted_gradient_mixture():
    from horizon.eval.update_leakage_probe import scores

    # signal grows like d and chance correlations like √d; d must be well above the latter (real: 4.7M)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        d = 5000
        common = rng.normal(size=d)
        g = 3 * common + rng.normal(size=(9, d))  # a large shared component, as in real label-loss gradients
        held = sorted(rng.choice(9, size=4, replace=False).tolist())
        delta = -(g[held].sum(0) + 0.1 * rng.normal(size=d))
        for name, s in scores(delta, g).items():
            assert set(np.argsort(-s)[:4]) == set(held), (seed, name)
