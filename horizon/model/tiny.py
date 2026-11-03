"""A tiny, randomly initialised Llama-architecture model for pipeline tests and the local quickstart.

It exercises every step (tokenisation, chat template, LoRA, training, encryption, evaluation) on a CPU
in minutes. Its weights are random, so its accuracy means nothing. The default tokenizer is Muse
Glimmer's, which is public: building the model needs no Hugging Face account.

    python -m horizon.model.tiny --out models/tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOKENIZER = "meta-models/Muse-Glimmer-30B"
TOKENIZER_REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"


def build(out: str | Path, tokenizer: str = TOKENIZER, revision: str | None = TOKENIZER_REVISION) -> Path:
    """Write the tiny model and `tokenizer`'s files to `out` (a local directory a config can point to)."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(tokenizer, revision=revision)
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    out = Path(out)
    LlamaForCausalLM(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="models/tiny")
    ap.add_argument("--tokenizer", default=TOKENIZER, help="any Hugging Face tokenizer with a chat template")
    ap.add_argument("--revision", default=None, help="tokenizer revision (default: pinned for the default)")
    a = ap.parse_args()
    rev = a.revision or (TOKENIZER_REVISION if a.tokenizer == TOKENIZER else None)
    print(json.dumps({"model": str(build(a.out, a.tokenizer, rev)), "tokenizer": a.tokenizer}))


if __name__ == "__main__":
    main()
