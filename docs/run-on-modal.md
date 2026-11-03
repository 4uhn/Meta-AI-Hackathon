# Run on your own Modal account

Real models train on GPUs through [Modal](https://modal.com). Each organisation's training step runs
in its own container, which mounts only that organisation's data. Key setup, aggregation and
decryption shares stay on your machine. Nothing in the code is tied to our account.

## Set up (once)

```bash
make setup && make he                 # Python env with the Modal client; crypto CLI (+ its Linux build for containers)
.venv/bin/modal setup                 # sign in (opens a browser); GPU runs need a payment method on the account
export HORIZON_MODAL_PREFIX=myteam    # optional: your own app and volume names (default "horizon")
make doctor                           # checks all of the above and prints the next command
```

**Hugging Face token: only for gated models.** Muse Glimmer is not gated and needs no token.
Llama-3.2 is gated: accept its licence on huggingface.co, then store a read token as a Modal
secret:

```bash
.venv/bin/modal secret create huggingface HF_TOKEN=hf_...     # name: $HORIZON_HF_SECRET, default "huggingface"
```

Provisioning checks before launching. If a model is gated and the secret is missing, it stops and
prints this command.

## Tiers

Run them in order. Each tier checks what the next one needs.

| Tier | Command | What runs | HF token | Measured time | Measured cost |
|---|---|---|---|---|---|
| 0 | `python examples/federated-fine-tuning/run.py` | the whole encrypted loop on your CPU, tiny model | no | about a minute | free |
| 1 | `… run.py --on modal --model tiny` | the same in Modal CPU containers: image, volumes, isolation, secrets | no | 4 min | $0.03 |
| 2 | `… run.py --on modal --model llama` | Llama-3.2-1B, 6 rounds on A10 GPUs | **yes** | 40 min | $0.67 |
| 3 | `… run.py --on modal --model glimmer` | Muse Glimmer-30B, 6 rounds on L40S GPUs: our headline run | no | 62 min | $3.37 |

Add `--baselines` to tier 3 for the comparison rows (base model, each organisation alone, pooled
data): 54 min and $5.73 more. The aviation runs (`--dataset asrs`, `--dataset asrs4`) took 117
and 120 minutes and cost $7.42 and $8.71; their reports are longer texts.

Times are wall-clock for a whole run, from its own logs. Costs are what Modal billed for that run's
app (`modal billing report`), collected by `scripts/modal_costs.py` into
`results/modal_costs.json`. One-time setup is extra but small: the data upload, and the weights
download, which runs in a small CPU container.

Before any GPU container starts you are asked to confirm, with the GPU type, its list price and the
measured cost of that config. `--yes` or `HORIZON_YES=1` skips the question, for scripts and
detached runs.

## What the runner does

`examples/federated-fine-tuning/run.py` prints each command before running it. Running the same
commands yourself:

```bash
make data                                                                   # download (sha256-checked) and split
python -m modal_app.provision --config configs/5g-nidd-glimmer.yaml           # data volumes + weights
python -m horizon.fl.run --config configs/5g-nidd-glimmer.yaml --backend modal --mode encrypted \
    --fresh --out runs/mine/results                                         # E3, encrypted
python -m modal_app.baselines --config configs/5g-nidd-glimmer.yaml --exp E0 E1a E1b E1c E4 --out runs/mine/results
python -m horizon.eval.report --results runs/mine/results --partition configs/partitions/5g-nidd.yaml
```

Volumes created, named `<prefix>-…`:

| Volume | Holds | Mounted by |
|---|---|---|
| `<prefix>-data-op_a` (one per organisation) | that organisation's `train.jsonl`, `val.jsonl` | only that organisation's containers |
| `<prefix>-eval` | `test.jsonl` | the evaluator and baselines |
| `<prefix>-models` | Hugging Face weights cache, tiny model | every container (public weights) |

Other datasets get their own volumes: `<prefix>-data-<task>-op_a`, `<prefix>-eval-<task>`.

## Reproducing our numbers

Tiers 2 and 3 run our published configs unchanged and compare with `results/5g-nidd-llama/E3.json` and
`results/5g-nidd-glimmer/E3.json`. Predictions match exactly only on the same GPU model and library
versions. Modal's `A10` request can land on an NVIDIA A10 or an A10G, which use different kernels.
The run pins the first GPU model it gets (`modal.require_device: auto`) and records every step
that had to run elsewhere (`device_mismatches` in `E3.json`). On different hardware, compare the
metrics instead.

We measured the effect by rerunning tier 2 from scratch under a fresh prefix:
- the published run (`results/5g-nidd-llama/E3.json`) ran mostly on A10s; the rerun (`results/5g-nidd-llama-rerun/E3.json`)
  mostly on A10Gs;
- the decrypted aggregates differ from round 1, and macro-F1 per round tracks closely until the end;
- the final scores are 0.930 published and 0.882 rerun.

A single run's final score moves by about this much across hardware, because bf16 training
amplifies tiny numerical differences.

The encryption itself is not the source. Replaying the plaintext run's updates through the real
threshold path gives the same aggregate to within 2.1e-6 per value, every round, and identical test
predictions (`results/5g-nidd-llama/equivalence.json`, `replay`).

## Practical notes

- One experiment at a time keeps within Modal's default GPU limits. FL rounds start N containers
  at once.
- Your machine runs the key ceremony, aggregation and decryption between rounds. Keep it awake
  (`caffeinate -is …` on macOS) for long runs.
- Long single containers can be pre-empted and restarted by Modal. FL steps are short, around
  minutes each.
