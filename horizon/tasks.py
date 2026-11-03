"""Classification tasks (domains). The config's `task:` selects one; the default is 5G-NIDD.

    nidd5g  5G network intrusion detection (5G-NIDD)
    asrs    aviation safety reports (NASA ASRS)

`task:` can also define a new task inline (see `from_spec`), so a new dataset needs no code change.

A task fixes the labels, the verbalizer and the system prompt. Downstream code reads them from
`serialize.LABELS` and `labels.VERBALIZER / SYSTEM`; `activate(cfg)` at each entry point rewrites those
objects in place, so modules that imported them see the active task. 5G-NIDD is captured at import,
before any activation, so its prompts stay byte-identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from horizon.data import serialize
from horizon.model import labels as label_fmt


@dataclass(frozen=True)
class Task:
    name: str
    labels: tuple[str, ...]
    verbalizer: dict[str, str]
    system: str
    shared_class: str | None  # held by every party (5G-NIDD: benign), so never a blind spot


def _system(intro: str, labels: tuple[str, ...], verbalizer: dict[str, str]) -> str:
    return intro + " Answer with exactly one label: " + ", ".join(verbalizer[lab] for lab in labels) + "."


NIDD5G = Task(
    name="nidd5g",
    labels=tuple(serialize.LABELS),
    verbalizer=dict(label_fmt.VERBALIZER),
    system=label_fmt.SYSTEM,
    shared_class="benign",
)

# ASRS event classes (horizon/data/asrs.py maps the "Anomaly" field onto them). Each verbalizer starts
# with a distinct first token under the Llama-3.2 and Glimmer tokenizers (tested).
ASRS_LABELS = (
    "equipment_failure",
    "traffic_conflict",
    "ground_conflict",
    "turbulence",
    "terrain_warning",
    "airspace_violation",
    "loss_of_control",
    "hazmat",
    "passenger_misconduct",
)
ASRS_VERBALIZER = {lab: lab for lab in ASRS_LABELS}
ASRS = Task(
    name="asrs",
    labels=ASRS_LABELS,
    verbalizer=ASRS_VERBALIZER,
    system=_system(
        "You are an aviation safety analyst. Classify the confidential incident report by its event.",
        ASRS_LABELS,
        ASRS_VERBALIZER,
    ),
    shared_class=None,
)

BUILTIN = {t.name: t for t in (NIDD5G, ASRS)}
TASKS = dict(BUILTIN)  # + tasks defined in configs, registered by activate()
DEFAULT = NIDD5G.name


NAME_RE = re.compile(r"[a-z0-9_]{1,24}")  # also names the task's Modal volumes (64-char limit)


def from_spec(spec: dict) -> Task:
    """A task defined in a config instead of in code:

        task:
          name: support_tickets
          labels: [billing, outage, account]
          instructions: "You are a support analyst. Classify the ticket by its topic."
          verbalizer: {}        # optional: label -> text the model answers with (default: the label)
          shared_class: null    # optional: a class every party holds (never counted as a blind spot)

    The system prompt is `instructions` followed by the list of answers, as for the built-in tasks.
    """
    unknown = set(spec) - {"name", "labels", "instructions", "verbalizer", "shared_class"}
    if unknown:
        raise ValueError(f"task: unknown keys {sorted(unknown)}")
    name, labels = spec.get("name"), tuple(spec.get("labels") or ())
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError(f"task.name must match {NAME_RE.pattern}, got {name!r}")
    if name in BUILTIN:
        raise ValueError(f"task.name {name!r} is a built-in task; pick another name")
    if not all(isinstance(x, str) and x for x in labels):
        raise ValueError("task.labels must be non-empty strings")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("task.labels must list at least 2 distinct labels")
    if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
        raise ValueError("task.instructions must be a non-empty string")
    verbalizer = {lab: lab for lab in labels} | dict(spec.get("verbalizer") or {})
    if set(verbalizer) != set(labels):
        raise ValueError("task.verbalizer keys must be task labels")
    if len(set(verbalizer.values())) != len(labels):
        raise ValueError("task.verbalizer must map labels to distinct answers")
    shared = spec.get("shared_class")
    if shared is not None and shared not in labels:
        raise ValueError(f"task.shared_class {shared!r} is not a label")
    return Task(name, labels, verbalizer, _system(spec["instructions"].strip(), labels, verbalizer), shared)


def task_name(cfg: dict | None) -> str:
    t = (cfg or {}).get("task", DEFAULT)
    return t["name"] if isinstance(t, dict) else t


def activate(cfg: dict | str | None) -> Task:
    """Make the config's task the active one in this process (in place; see module docstring)."""
    spec = None if isinstance(cfg, str) else (cfg or {}).get("task")
    if isinstance(spec, dict):
        t = from_spec(spec)
        TASKS[t.name] = t
    name = cfg if isinstance(cfg, str) else task_name(cfg)
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    t = TASKS[name]
    serialize.LABELS[:] = t.labels
    label_fmt.VERBALIZER.clear()
    label_fmt.VERBALIZER.update(t.verbalizer)
    label_fmt.SYSTEM = t.system
    return t


def active() -> Task:
    """The task whose labels are currently active."""
    for t in TASKS.values():
        if tuple(serialize.LABELS) == t.labels and label_fmt.SYSTEM == t.system:
            return t
    raise RuntimeError("active labels match no registered task")
