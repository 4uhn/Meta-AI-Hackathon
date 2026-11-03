# Your own dataset

A template for using Horizon on a new problem. The task is defined in the config, with no code
changes. The toy dataset is support tickets in four topics, and three companies each see different
topics.

**Needs:** `make setup` and `make he`. No account and no GPU (tiny model on CPU, about a minute).

## Run

```bash
python examples/custom-dataset/make_toy_data.py          # → runs/examples/custom-dataset/data/
python -m horizon.model.tiny --out models/tiny           # tiny random model, public tokenizer
python -m horizon.fl.run --config examples/custom-dataset/config.yaml --mode encrypted --fresh \
    --run-root runs/examples/custom-dataset/runs --out runs/examples/custom-dataset/results
python -m horizon.eval.report --results runs/examples/custom-dataset/results \
    --config examples/custom-dataset/config.yaml --title "Per-class recall"
```

This runs three encrypted rounds. Each round's sum is decrypted by two of the three companies, and
the results go to `runs/examples/custom-dataset/results/`. The tiny model's weights are random, so
its scores mean nothing: this checks your data, task and pipeline.

## Files

| File | |
|---|---|
| `make_toy_data.py` | writes the input format: `op_x/train.jsonl`, `op_x/val.jsonl`, `test.jsonl`, one `{"text", "label"}` per line |
| `config.yaml` | the `task:` definition (name, labels, instructions), the consortium (`fl`), the tiny model |

## Adapt it

1. Put your data in the same layout and point `data_dir` at it.
2. Edit `task:` (labels and instructions) and `fl:` (organisations and threshold).
3. For a real model, copy `model`, `lora`, `train`, `eval` and `modal` from
   `configs/5g-nidd-llama.yaml` or `configs/5g-nidd-glimmer.yaml`.
4. Run with `--backend modal` after `python -m modal_app.provision --config <yours>`.

The full guide is `docs/bring-your-own-data.md`.
