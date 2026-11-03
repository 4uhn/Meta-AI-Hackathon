"""Chat formatting and full-label log-likelihood scoring.

Training and evaluation build prompts with the same function. Each label continuation `word<turn end>`
is scored by summed token log-prob under teacher forcing (`score_labels`); with unique first tokens,
`score_first_tokens` is a fast path (one forward per flow instead of one per label).

Llama 3.x replies after the generation prompt and ends with eos `<|eot_id|>`; its template stamps
"Today Date", so the date is pinned. Muse Glimmer's template addresses every assistant message to a
recipient (`to=user` replies, `to=self` opens reasoning), so the prompt is forced past that choice and
the target ends with `<|eot|>`, not eos.

VERBALIZER gives every class its own first token (`first_tokens_unique`). The dataset names
`udp_flood`/`udp_scan` and `syn_flood`/`syn_scan` share first tokens, which let flood labels absorb
scans under federated training; the renames are standard names for the same attacks. Unique first
tokens also make the restricted softmax act at a single position.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from horizon.data.serialize import LABELS

VERBALIZER = {lab: lab for lab in LABELS} | {
    "syn_scan": "half_open_scan",  # a SYN scan is the half-open scan
    "udp_scan": "datagram_scan",  # UDP = User Datagram Protocol
}
SYSTEM = (
    "You are a 5G network intrusion classifier. Classify the flow. Answer with exactly one label: "
    + ", ".join(VERBALIZER[lab] for lab in LABELS)
    + "."
)
DATE_STRING = "26 Oct 2026"
# Glimmer: answer directly; the template otherwise appends "Reasoning strength: high." to the system turn
REASONING_STRENGTH = "low"
ATEM_REPLY_PREFIX, ATEM_TURN_END = " to=user<|message|>", "<|eot|>"


def chat_format(tok) -> tuple[str, str]:
    """(text forced after the generation prompt, end-of-turn string) for this tokenizer's template."""
    tmpl = getattr(tok, "chat_template", None) or ""
    if "<|start|>assistant" in tmpl and " to=" in tmpl:
        eot = tok.convert_tokens_to_ids(ATEM_TURN_END)  # None / unk when the vocabulary lacks it
        if eot is not None and eot != tok.unk_token_id:
            return ATEM_REPLY_PREFIX, ATEM_TURN_END
    return "", tok.eos_token


def prompt_text(tok, flow: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": flow}]
    text = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=False,
        date_string=DATE_STRING,  # Llama 3.x; Glimmer stamps no date when a system turn is given
        reasoning_strength=REASONING_STRENGTH,  # Glimmer only; unused by the Llama template
    )
    return text + chat_format(tok)[0]


def target_text(tok, label: str) -> str:
    return VERBALIZER[label] + chat_format(tok)[1]


def first_tokens_unique(tok) -> bool:
    firsts = [tok(target_text(tok, lab), add_special_tokens=False)["input_ids"][0] for lab in LABELS]
    return len(set(firsts)) == len(LABELS)


def encode_example(tok, flow: str, label: str) -> tuple[list[int], list[int]]:
    """input_ids and labels (-100 on the prompt: loss on the label and end token only)."""
    p = tok(prompt_text(tok, flow), add_special_tokens=False)["input_ids"]
    t = tok(target_text(tok, label), add_special_tokens=False)["input_ids"]
    return p + t, [-100] * len(p) + t


def missing_label_masks(tok, own: set[str]) -> dict[str, list[list[int]]]:
    """FedRS-style restricted softmax (Li & Zhan, KDD'21), α → 0 limit, on the label prefix tree.

    For each own label and target position: the next-token ids that lead only to labels this operator
    does not hold. Excluding them from the local softmax stops operators from teaching the shared model
    to suppress classes they lack. Uses only the operator's own label set and the public label list.
    """
    seqs = {lab: tok(target_text(tok, lab), add_special_tokens=False)["input_ids"] for lab in LABELS}
    out = {}
    for lab in own:
        t, per = seqs[lab], []
        for j in range(len(t)):
            nxt = {m: s[j] for m, s in seqs.items() if len(s) > j and s[:j] == t[:j]}
            per.append(
                sorted({v for m, v in nxt.items() if m not in own} - {v for m, v in nxt.items() if m in own})
            )
        out[lab] = per
    return out


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad to the longest sequence in the batch."""
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        lab[i, : len(y)] = torch.tensor(y)
        att[i, : len(x)] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def label_logits(model, input_ids, attention_mask, labels) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits (fp32) only at the positions that predict a label token, plus those target ids.

    The output head is applied only where the next token is a label, so the full batch × seq × vocab
    logits tensor is never built. Calling the decoder directly skips the wrapper's forward, so its
    post-head transform is redone in `output_logits`.
    """
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = core.get_decoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    head = core.get_output_embeddings()
    # Explicit matmul, not head(...): on MPS (torch 2.14.1, bf16) F.linear's backward intermittently
    # writes NaN into the last row of the input gradient, poisoning every lora_B grad. matmul's
    # backward is a plain mm and stays finite.
    logits = h[:, :-1][mask] @ head.weight.T
    if head.bias is not None:
        logits = logits + head.bias
    return output_logits(core, logits.float()), tgt[mask]


def output_logits(core, z: torch.Tensor) -> torch.Tensor:
    """Post-lm_head transform: Muse Glimmer computes T·tanh(m·z / T) with m = output_multiplier,
    T = final_logit_softcapping (transformers 5.21). Identity for models without them (Llama). fp32."""
    tc = getattr(core.config, "text_config", None) or core.config
    mult = getattr(tc, "output_multiplier", None)
    cap = getattr(tc, "final_logit_softcapping", None)
    if mult is not None:
        z = z * mult
    if cap:
        z = cap * torch.tanh(z / cap)
    return z


@torch.no_grad()
def score_labels(model, tok, flows: list[str], device: str, batch_flows: int = 8) -> np.ndarray:
    """(n_flows, n_labels) summed label log-probs; argmax over axis 1 is the prediction."""
    model.eval()
    label_ids = [tok(target_text(tok, lab), add_special_tokens=False)["input_ids"] for lab in LABELS]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = np.zeros((len(flows), len(LABELS)), dtype=np.float64)
    for s in range(0, len(flows), batch_flows):
        chunk = flows[s : s + batch_flows]
        seqs = []
        for f in chunk:
            p = tok(prompt_text(tok, f), add_special_tokens=False)["input_ids"]
            seqs += [(p + t, [-100] * len(p) + t) for t in label_ids]
        b = {k: v.to(device) for k, v in collate(seqs, pad).items()}
        logits, tgt = label_logits(model, b["input_ids"], b["attention_mask"], b["labels"])
        tok_lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # label positions come out in row-major order, so a segment-sum per sequence recovers each score
        seq_of = (b["labels"][:, 1:] != -100).nonzero()[:, 0]
        per_seq = torch.zeros(len(seqs), device=tok_lp.device).index_add_(0, seq_of, tok_lp)
        out[s : s + len(chunk)] = per_seq.view(len(chunk), len(LABELS)).cpu().double().numpy()
    return out


@torch.no_grad()
def score_first_tokens(model, tok, flows: list[str], device: str, batch_flows: int = 8) -> np.ndarray:
    """Fast path: (n_flows, n_labels) log-probs of each label's first token at the decision position.

    Matches the `score_labels` argmax when continuations after the first token are near-certain. Needs
    unique first tokens; validate against `score_labels` before relying on it.
    """
    if not first_tokens_unique(tok):
        raise ValueError("first-token scoring needs one first token per class")
    model.eval()
    first = [tok(target_text(tok, lab), add_special_tokens=False)["input_ids"][0] for lab in LABELS]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = np.zeros((len(flows), len(LABELS)), dtype=np.float64)
    for s in range(0, len(flows), batch_flows):
        chunk = flows[s : s + batch_flows]
        seqs = []
        for f in chunk:  # one dummy target token; its logits are the decision
            p = tok(prompt_text(tok, f), add_special_tokens=False)["input_ids"]
            seqs.append((p + [first[0]], [-100] * len(p) + [first[0]]))
        b = {k: v.to(device) for k, v in collate(seqs, pad).items()}
        logits, _ = label_logits(model, b["input_ids"], b["attention_mask"], b["labels"])
        lp = F.log_softmax(logits, dim=-1)[:, first]
        out[s : s + len(chunk)] = lp.cpu().double().numpy()
    return out
