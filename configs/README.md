# configs/

Each file describes one run (keys: `docs/configuration.md`). Names are `<dataset>-<model>`. The
published results were produced from these files. Their recorded run names and config paths
(`runs/modal_glimmer_E3`, `configs/modal_glimmer.yaml`, etc.) use the files' earlier names.

| Config | Model | Data | Where | Results |
|---|---|---|---|---|
| `5g-nidd-glimmer.yaml` | Muse Glimmer-30B (4-bit) | 5G-NIDD, 3 organisations | Modal L40S | `results/5g-nidd-glimmer/` |
| `asrs-glimmer.yaml` | Muse Glimmer-30B | NASA ASRS, 3 organisations | Modal L40S | `results/asrs-glimmer/` |
| `asrs4-glimmer.yaml` | Muse Glimmer-30B | NASA ASRS, 4 organisations, 3-of-4 | Modal L40S | `results/asrs4-glimmer/` |
| `5g-nidd-glimmer-no-restriction.yaml` | Muse Glimmer-30B | 5G-NIDD, restricted softmax off (leakage ablation) | Modal L40S | `results/5g-nidd-glimmer/update_leakage_ablation.json` |
| `5g-nidd-llama.yaml` | Llama-3.2-1B-Instruct | 5G-NIDD | Modal A10 | `results/5g-nidd-llama/` |
| `5g-nidd-llama-fp32-eval.yaml` | the same, fp32 inference | 5G-NIDD | Modal A10 | `results/5g-nidd-llama/equivalence.json` |
| `5g-nidd-tiny-modal.yaml` | tiny random model | 5G-NIDD | Modal CPU | pipeline test only |
| `5g-nidd-tiny.yaml` | tiny random model | 5G-NIDD | this machine, CPU | pipeline test only |

**Recipe.** The training recipe was tuned on the dev model (`5g-nidd-llama.yaml`) using federated
validation only (`tuning/`, `eval.split: val`). It was then frozen and copied unchanged into the
Glimmer configs. Only the model, precision and hardware settings differ.

**Partitions.** `partitions/*.yaml` split a raw dataset into one shard per organisation plus a test
set:
- `partitions/5g-nidd.yaml`: 5G-NIDD, where each operator sees different attack families;
- `partitions/asrs.yaml`: ASRS, split naturally by reporter organisation (airline, air traffic
  control, general aviation);
- `partitions/asrs4.yaml`: adds business aviation as a fourth organisation and keeps the other
  three shards and the test set byte-identical.

Each pins its raw file by SHA-256. `make data` and `make data-asrs` download and split them.
