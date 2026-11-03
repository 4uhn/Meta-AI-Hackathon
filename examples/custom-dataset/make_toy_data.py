"""Write a small toy dataset in Horizon's input format, as a template for your own data.

    python examples/custom-dataset/make_toy_data.py      # → runs/examples/custom-dataset/data/

The format: one directory per organisation holding `train.jsonl` and `val.jsonl`, plus one shared
`test.jsonl`. Every line is {"text": "...", "label": "..."} with a label from the config's task. Here
three companies each see different ticket topics, the situation federated training is meant for.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "runs" / "examples" / "custom-dataset" / "data"
TEMPLATES = {
    "billing": ["I was charged twice for {x}", "my invoice for {x} is wrong", "refund the {x} payment"],
    "outage": ["{x} has been down since this morning", "cannot reach {x} at all", "{x} keeps timing out"],
    "account": ["reset the password for {x}", "I am locked out of {x}", "change the email on {x}"],
    "shipping": ["my {x} order has not arrived", "the {x} parcel is damaged", "track the {x} delivery"],
}
THINGS = ["the app", "my plan", "the router", "the website", "my subscription", "the dashboard", "order 1182"]
HOLDS = {"op_a": ["billing", "outage"], "op_b": ["outage", "account"], "op_c": ["account", "shipping"]}


def rows(labels: list[str], n: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n):
        lab = labels[i % len(labels)]
        out.append({"text": rng.choice(TEMPLATES[lab]).format(x=rng.choice(THINGS)), "label": lab})
    return out


def write(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in data))


def main() -> None:
    rng = random.Random(0)
    for op, labels in HOLDS.items():
        write(OUT / op / "train.jsonl", rows(labels, 40, rng))
        write(OUT / op / "val.jsonl", rows(labels, 8, rng))
    write(OUT / "test.jsonl", rows(list(TEMPLATES), 40, rng))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
