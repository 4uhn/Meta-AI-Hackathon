# Bring your own data and model

Horizon fine-tunes an LLM to classify text into a fixed set of labels. To use it on your own problem
you need one data directory and one config. No code changes are needed. `examples/custom-dataset/`
is a working template.

## 1. Data

One directory per organisation, plus a shared test set:

```
my_data/
  op_a/train.jsonl   op_a/val.jsonl
  op_b/train.jsonl   op_b/val.jsonl
  op_c/train.jsonl   op_c/val.jsonl
  test.jsonl
```

Every line is `{"text": "…", "label": "…"}`, with a label from your task. Organisations are named
`op_a`, `op_b`, … in order. Each organisation's directory is the only data that organisation's
process (or Modal container) can read. `test.jsonl` is read only by the evaluator.

Split carefully, because leakage here inflates every number:
- choose the test set first;
- keep near-duplicates and records of the same event out of training;
- check exact and near-duplicate overlap.

`horizon/data/asrs.py` shows the guards we used.

## 2. Task

Start from `examples/custom-dataset/config.yaml` (the tiny model) and change its `task`, `data_dir`
and `fl` (full reference: [configuration.md](configuration.md)):

```yaml
task:
  name: support_tickets
  labels: [billing, outage, account, shipping]
  instructions: "You are a support analyst. Classify the customer ticket by its topic."
data_dir: my_data
fl: {operators: [op_a, op_b, op_c], threshold: 2}
```

The model answers with one label. Use a `verbalizer` when a label name is cryptic or when two labels
start with the same token: label scoring and logit adjustment work best when each answer starts
with its own token.

## 3. Run locally first

The tiny random model runs the whole encrypted loop on a laptop CPU in about a minute:

```bash
python -m horizon.model.tiny --out models/tiny
python -m horizon.fl.run --config my_config.yaml --mode encrypted --fresh --out runs/mine/results
```

Its scores mean nothing. This step checks the data, the task and the pipeline.

## 4. Choose a model

Copy `model`, `lora`, `train`, `eval` and `modal` from one of our configs. Keep your `task`,
`data_dir` and `fl`.

| Start from | Model | GPU | Notes |
|---|---|---|---|
| `configs/5g-nidd-llama.yaml` | Llama-3.2-1B-Instruct | A10 | cheap; gated on Hugging Face (accept the licence, add a token) |
| `configs/5g-nidd-glimmer.yaml` | Muse Glimmer-30B, 4-bit | L40S | our headline model; not gated |

We have run Llama-3.2 and Muse Glimmer. For another Hugging Face chat model:
- set `model.id` and `revision`, and the attention module names in `lora.target_modules`;
- if the architecture differs, you may also need `loader`, `quantized` or a separate
  `tokenizer_id` (see the Glimmer config).

Then run step 3's tiny-model check against its tokenizer first (`python -m horizon.model.tiny
--tokenizer <repo>`).

## 5. Run on Modal

```bash
python -m modal_app.provision --config my_config.yaml   # your data → one volume per organisation; weights
python -m horizon.fl.run --config my_config.yaml --backend modal --mode encrypted --fresh --out runs/mine/results
```

Each organisation's training runs in its own container, which mounts only that organisation's data
volume. See `docs/run-on-modal.md` for accounts, costs and the volume prefix.

## 6. Compare

Optionally, add the baselines and the per-class report:

- `python -m modal_app.baselines --config my_config.yaml --exp E0 E1a E1b E1c E4 --out runs/mine/results`
  trains the base model, each organisation alone, and all data pooled.
- `python -m horizon.eval.report --results runs/mine/results --config my_config.yaml` writes the
  per-class recall table and, once the baselines exist, each organisation's blind-spot recall. Blind
  spots are the classes an organisation never saw in training, read from its shard.
