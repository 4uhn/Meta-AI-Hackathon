# modal_app/ (GPU backend)

Runs each model step in its own [Modal](https://modal.com) container. Crypto, aggregation and
decryption shares stay on your machine. Setup, costs and the quickstart are in
`docs/run-on-modal.md`.

| File | What it does |
|---|---|
| `app.py` | the images (default; `glimmer` adds bitsandbytes and accelerate), the volume names, and the container side of every step |
| `backend.py` | `ModalBackend`: decides what each container may see, starts it, writes back its new files, logs cost; asks before any GPU launch |
| `provision.py` | creates and fills the volumes: data (one per organisation), test set, weights, tiny model |
| `baselines.py` | E0, E1x, E4 in parallel containers |
| `membership.py`, `leakage.py` | membership inference; the gradient-matching leakage probe and its ablation |

**Isolation.** Operator X's container mounts only the `<prefix>-data-op_x` volume and receives only
X's signing key, as a per-call secret. The evaluator mounts only `<prefix>-eval`. A container returns
only the files its step created, never anything under a `parties/` directory.

**Names.** Every app and volume name starts with `$HORIZON_MODAL_PREFIX` (default `horizon`). A
Hugging Face token is needed only for gated models, as a Modal secret named by `$HORIZON_HF_SECRET`
(default `huggingface`).
