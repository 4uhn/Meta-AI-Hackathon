# horizon/ (Python package)

| Module | What it does |
|---|---|
| `he_bridge.py` | `HorizonHE`: drives `horizon-he` from Python, one process per party step. Usable on its own for any vector (`examples/secure-aggregation/`) |
| `tasks.py` | classification tasks: built-in (`nidd5g`, `asrs`) or defined in a config |
| `fl/run.py` | the federated loop: setup, rounds, aggregation, decryption, evaluation. `--mode encrypted` (E3), `plaintext` (E2, reference), `--scenario e5` (skip / offline) |
| `fl/operator.py` | one organisation's step: train on its own shard, encrypt and sign its update |
| `fl/aggregator.py` | the server's step: initial adapter, sum, request decryption, apply the decrypted average |
| `fl/baseline.py` | non-federated comparisons: base model (E0), one organisation alone (E1x), pooled data (E4) |
| `fl/evaluate.py`, `fl/rundir.py` | evaluation of a global adapter; run-directory layout and the consortium (N, t) |
| `model/` | LoRA set-up (FFA-LoRA: B trains, A frozen), flattening B, training, label scoring; `tiny.py` builds the tiny test model |
| `data/` | dataset download (`download.py`), the 5G-NIDD and ASRS splitters, serialisation |
| `eval/` | per-class report, encrypted-vs-plaintext equivalence, membership inference, update-leakage attacks |
| `events.py` | the structured event log, `runs/<run>/events.jsonl` |
| `doctor.py` | checks the machine and the Modal account, prints the next command (`make doctor`) |
| `demo/` | the narrated live re-decryption shown in the project video (`make demo`; needs the recorded run) |

Every model step runs as its own process. The orchestrator never loads a model, and an operator
reads only its own shard and its own `parties/<op>/` directory. Every step appends a structured
event to `runs/<run>/events.jsonl`.

Install: `pip install -e .` for the encryption layer only, `pip install -e '.[train]'` to train,
`'.[modal]'` for Modal. `make setup` installs everything into `.venv`.
