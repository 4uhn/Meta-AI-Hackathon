# results/

Every number we report comes from a file here, written by code in this repository. Model runs went
to Modal; crypto measurements ran on an Apple M2 Pro (10 cores). Raw run directories (keys,
ciphertexts, adapters, score arrays) are not published.

## Reproduce the headline numbers

Commands run from the repository root. `run.py` is `examples/federated-fine-tuning/run.py`; Modal
setup is in `docs/run-on-modal.md`.

| Number | File | Command |
|---|---|---|
| 5G-NIDD on Glimmer: 0.949 encrypted, 0.955 pooled; blind spots 0.000 → 0.943 | `5g-nidd-glimmer/E3.json`, `E4.json`, `metrics.json` | `run.py --on modal --model glimmer --baselines` |
| Aviation reports: 0.910 encrypted, 0.899 pooled; loss of control 0.08 → 0.92 | `asrs-glimmer/` | `run.py --on modal --model glimmer --dataset asrs --baselines` |
| 3-of-4: 0.905; the new member 0.847 alone | `asrs4-glimmer/` | `run.py --on modal --model glimmer --dataset asrs4 --baselines` |
| Encrypted = plaintext within 2.14e-6 per value (replay) | `5g-nidd-glimmer/equivalence.json` | `run.py --on modal --model glimmer --plaintext-reference` (the same check runs free with the tiny model: `run.py --plaintext-reference`) |
| One plaintext update leaks (AUC 0.993); the sum does not (0.508) | `5g-nidd-glimmer/update_leakage*.json` | see `docs/why-encrypt.md` |
| Membership inference 0.505 / 0.541 | `*/membership_inference.json` | `python -m modal_app.membership --suite 5g-nidd-glimmer` (or `asrs-glimmer`) `--e3-run DIR --baselines-run DIR` |
| t-of-N scaling; 150.6 MB of ciphertext per update | `crypto/scaling.json`, `crypto/bench.json` | `make crypto-scaling`, `make bench` (CPU, free) |

`run.py` writes its runs to `runs/examples/<dataset>-glimmer/runs/` (for example
`5g-nidd-glimmer_E3` and `5g-nidd-glimmer_baselines`). The analysis tools default to the run
directories recorded in the published JSON, which are not published; pass yours with `--e3-run`,
`--baselines-run` or `--ref-run`.

## Experiments

Every model folder uses the same experiment names:

| File | Experiment | Produced by |
|---|---|---|
| `E0.json` | base model, no fine-tuning | `horizon.fl.baseline --exp E0` (on Modal: `modal_app.baselines`) |
| `E1a.json` … `E1d.json` | one organisation alone, on its own data | `horizon.fl.baseline --exp E1a` … |
| `E2.json` | federated, **plaintext** reference (`--insecure-reference` updates) | `horizon.fl.run --mode plaintext` |
| `E3.json` | federated, **encrypted**: Horizon | `horizon.fl.run --mode encrypted` |
| `E4.json` | all data pooled in one place (the centralised ceiling) | `horizon.fl.baseline --exp E4` |
| `E5.json` | encrypted, with a skipped round and an organisation offline at decryption | `horizon.fl.run --scenario e5` |
| `metrics.json`, `detection_matrix.png` | per-class recall table, blind-spot recall | `horizon.eval.report` |
| `equivalence.json` | encrypted aggregate = plaintext aggregate, replayed through the real threshold path | `horizon.eval.equivalence --replay` |
| `membership_inference.json` | membership-inference attack on the final model | `modal_app.membership` |
| `leakage_subset.json` | scores re-computed on the test rows with no near-duplicate in training | `python -m scripts.leakage_subset_eval` |

Each `E*.json` records:
- the run name, model, base-model revision, LoRA layout hash and frozen-A hash;
- per-round crypto details (contributors, decrypting set, errors);
- the final metrics and `pred_sha256`, a hash of every test prediction, so two runs can be compared
  exactly.

## Folders

| Folder | Model | Data | Organisations |
|---|---|---|---|
| `5g-nidd-glimmer/` | Muse Glimmer-30B (4-bit, L40S) | 5G-NIDD intrusion detection | 3, 2-of-3 |
| `asrs-glimmer/` | Muse Glimmer-30B | NASA ASRS aviation safety reports | 3, 2-of-3 |
| `asrs4-glimmer/` | Muse Glimmer-30B | ASRS + business aviation | 4, 3-of-4 |
| `5g-nidd-llama/` | Llama-3.2-1B (A10) | 5G-NIDD | 3, 2-of-3 |
| `5g-nidd-llama-rerun/` | the same config, rerun from scratch under a fresh Modal prefix (mostly A10G) | 5G-NIDD | 3, 2-of-3 |

The training recipe was tuned on the dev model with federated validation splits (each organisation
scores its own validation data), then frozen and applied to Glimmer and to the aviation data without
further tuning.
`asrs4-glimmer/REUSED_BASELINES.json` explains why its E0 and E1a–c are the 3-organisation runs:
their shards and test set are byte-identical (hashes listed).

## Privacy measurements

| File | What it measures | Produced by |
|---|---|---|
| `5g-nidd-glimmer/update_leakage.json`, `.png` | what a server would learn from one organisation's **plaintext** update: which attack classes it holds (logit lens, public weights only). Includes the decrypted-aggregate control | `horizon.eval.update_leakage` |
| `5g-nidd-glimmer/update_leakage_probe.json` | the same question by gradient matching | `horizon.eval.update_leakage_probe` (`modal_app.leakage probe`) |
| `5g-nidd-glimmer/update_leakage_ablation.json` | the leak without the restricted softmax | `horizon.eval.update_leakage_ablation` (`modal_app.leakage ablation`) |
| `*/membership_inference.json` | whether the final model reveals which rows were trained on | `modal_app.membership` |
| `5g-nidd-split-check.json` | exact and near-duplicate overlap between 5G-NIDD test and training rows | `scripts/leakage_check.py` |

## Crypto

| File | What | Produced by |
|---|---|---|
| `crypto/scaling.json`, `.txt` | 2-of-3, 3-of-4, 3-of-5, 4-of-5 at the Glimmer payload: selftest + timings + error | `make crypto-scaling` |
| `crypto/bench.json`, `crypto/bench_10M.json` | encrypt / aggregate / decrypt timings and bytes at 4,685,825 and 10M values | `make bench` |
| `crypto/selftest.json`, `.txt` | full protocol and attack checks at the Glimmer payload | `crypto/bin/horizon-he selftest` |
| `crypto/selftest_200k.json`, `.txt`, `crypto/tests.txt` | the same at 200k values; noise, flooding and precision tests | `make crypto-report` |
| `modal_costs.json` | measured Modal cost and time per config (from `modal billing report`) | see `docs/run-on-modal.md` |

## demo/

`demo/live_*.json` is the recording of the live re-decryption shown in the project video (`make demo`,
a replay of the Glimmer run's final round from its recorded ciphertexts, which are not published).
The video plays this recording at 1.9× speed; the "Done in 4 seconds" it shows is the measured
crypto time.
