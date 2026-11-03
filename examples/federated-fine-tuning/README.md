# Encrypted federated fine-tuning, end to end

Our published training, runnable on your machine or your Modal account. Three mobile operators each
see different attack families in 5G network traffic (5G-NIDD). None can share its logs. Together
they fine-tune one model that detects every family, and no one, including the server, can decrypt
an operator's update.

`run.py` does every step and prints each command as it runs it:
1. download and split the data;
2. upload it to one Modal volume per operator (Modal only);
3. run the encrypted training (E3);
4. write the per-class report and compare with our published result (real models only).

## Run

```bash
make setup && make he && source .venv/bin/activate
python examples/federated-fine-tuning/run.py                              # laptop CPU, tiny model, no account
python examples/federated-fine-tuning/run.py --on modal --model tiny      # your Modal account, CPU containers
python examples/federated-fine-tuning/run.py --on modal --model llama     # Llama-3.2-1B, A10 (HF token: gated)
python examples/federated-fine-tuning/run.py --on modal --model glimmer   # Muse Glimmer-30B, L40S
```

| Option | |
|---|---|
| `--dataset asrs` / `asrs4` | aviation safety reports: 3 organisations (airline, air traffic control, general aviation) or 4 at 3-of-4 |
| `--baselines` | also train the base model, each organisation alone and all data pooled (the comparison rows) |
| `--plaintext-reference` | also run without encryption and check the two runs agree |
| `--yes` | don't ask before starting GPU containers |

Set up Modal first: `docs/run-on-modal.md` (sign-in, the optional volume prefix, the token for
Llama).

## Expected results

| Run | Time | Cost (billed) | Result |
|---|---|---|---|
| laptop, tiny model | about a minute | free | 3 rounds, each sum decrypted by 2 of 3, one share alone gives noise. Scores are meaningless (random weights) |
| Modal, tiny model | 4 min | $0.03 | the same in Modal containers |
| Modal, Llama-3.2-1B | 40 min | $0.67 | macro-F1 0.930 published (`results/5g-nidd-llama/`); our rerun under a fresh prefix gave 0.882 (`results/5g-nidd-llama-rerun/`): it ran on A10G instead of A10, and bf16 training diverged from round 1 |
| Modal, Muse Glimmer-30B | 62 min | $3.37 | macro-F1 0.949; with `--baselines` (+54 min, +$5.73): 0.955 pooled, so Horizon recovers 99.4% |

The Glimmer numbers come from `results/5g-nidd-glimmer/` and `results/modal_costs.json`. Outputs go to
`runs/examples/<dataset>-<model>/results/`. The runner compares them with our file. Predictions match
exactly only on the same GPU model and library versions (see `docs/run-on-modal.md`).

## Adapt it

- **Other data:** `examples/custom-dataset/` and `docs/bring-your-own-data.md`.
- **Other t-of-N:** set `fl: {operators: [...], threshold: t}` in a copy of the config. Check the
  shape first with `make check-tofn PARTIES=N THRESHOLD=t`.
- **Other recipe:** every training setting is in the config (`docs/configuration.md`).
