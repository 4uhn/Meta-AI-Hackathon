# Examples

| Example | What it shows | Runs on | Account | Time, cost |
|---|---|---|---|---|
| [`secure-aggregation/`](secure-aggregation/) | the encryption layer alone: N parties, any t decrypt the sum of their vectors | laptop CPU | none | seconds, free |
| [`federated-fine-tuning/`](federated-fine-tuning/) | our 5G intrusion-detection (or aviation) training end to end, as published | laptop CPU (tiny model) or your Modal account | none / Modal | see its README |
| [`custom-dataset/`](custom-dataset/) | a new task defined in a config, on your own JSONL data | laptop CPU (tiny model) | none | about a minute, free |

All three need `make setup` and `make he` (Go ≥ 1.24). Run them from the repository root with the
project environment active (`source .venv/bin/activate`). Outputs go to `runs/examples/`, never to
`results/`. Run `make doctor` to see what is missing.
