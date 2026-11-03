# Configuration reference

One YAML file describes a run: the task, the model, the training recipe, the consortium and, for GPU
runs, the Modal resources. `configs/` holds the configs our published results came from. Copy one
and change what you need.

## `task`

A built-in name (`nidd5g`, the default; `asrs`) or a definition:

```yaml
task:
  name: support_tickets          # [a-z0-9_], up to 24 characters; also names the Modal volumes
  labels: [billing, outage, account]
  instructions: "You are a support analyst. Classify the ticket by its topic."
  verbalizer: {outage: service_outage}   # optional: the text the model answers with (default: the label)
  shared_class: null             # optional: a class every organisation holds; not counted as a blind spot
```

The system prompt is `instructions` followed by "Answer with exactly one label: …".

## `model`

| Key | Meaning |
|---|---|
| `id`, `revision` | Hugging Face repo (or a local directory) and the commit to pin |
| `tokenizer_id`, `tokenizer_revision` | when the tokenizer lives in another repo (Glimmer's 4-bit checkpoint) |
| `loader` | `causal_lm` (default) or `image_text_to_text` (multimodal checkpoints such as Muse Glimmer) |
| `quantized` | `true` for a pre-quantised (bitsandbytes) checkpoint |
| `gradient_checkpointing` | trade compute for memory on large models |
| `dtype`, `device` | `float32` / `bfloat16`; `cpu`, `mps` or `cuda` |

## `lora`

`r`, `alpha`, `target_modules` (a list of module names, or a regex that PEFT matches in full). Only
the B matrices train (FFA-LoRA): A stays frozen at a seed every organisation shares, so averaging B
across organisations is exact.

## `train`

| Key | Meaning |
|---|---|
| `rounds` | federated rounds |
| `local_epochs` | epochs over the organisation's own shard per round (fractions allowed) |
| `lr`, `batch_size`, `micro_batch` | AdamW learning rate; the effective batch; rows per forward pass (memory only: gradient accumulation makes up the difference) |
| `seed`, `adapter_seed` | data order; the shared initial adapter (A) |
| `restrict_missing_labels` | restricted softmax (FedRS-style): each organisation's softmax over the answer tokens leaves out labels it doesn't hold, so its update doesn't push down classes it never sees (recommended under label skew) |
| `logit_adjust_tau` | logit adjustment by the organisation's own label frequencies (0 = off) |

## `fl`

| Key | Meaning |
|---|---|
| `operators` | `[op_a, op_b, …]` in order (default three) |
| `threshold` | t: how many organisations decrypt each sum (default 2). Rules: 3 ≤ N ≤ 5 (6–7 with `unverified`), 2t > N, t ≤ N − 1; each sum needs at least t + 1 contributors |
| `unverified` | `true` to allow N = 6 or 7 after running `horizon-he check … --unverified` |
| `server_lr` | step size applied to the decrypted average (1.0 = plain FedAvg) |

## `eval`

| Key | Meaning |
|---|---|
| `split` | `test` (default), or `val`: federated validation, each organisation scores its own val split and returns only a confusion matrix |
| `every_round` | evaluate the global model after every round |
| `batch_flows` | inputs per forward pass (each scored against every label) |
| `scoring`, `round_scoring` | `full` scores every label's whole answer; `first_token` scores only its first token (faster; needs distinct first tokens) |
| `compare_fast_final` | also score the final model with `first_token` and record the agreement |

## Paths and Modal

| Key | Meaning |
|---|---|
| `data_dir` | `op_x/train.jsonl`, `op_x/val.jsonl` per organisation, plus `test.jsonl` |
| `results_dir` | where results go when `--out` isn't given |
| `modal.gpu`, `cpu`, `memory_mb`, `timeout_s` | container resources; `gpu: null` runs on CPU |
| `modal.image` | `default`, or `glimmer` (adds bitsandbytes and accelerate) |
| `modal.require_device` | the exact GPU model to accept (`auto`: whatever the first container got) |
