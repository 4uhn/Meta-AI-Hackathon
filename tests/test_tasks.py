"""Tasks defined in a config: validation, activation, and no effect on the built-in tasks."""

from __future__ import annotations

import pytest

from horizon import tasks
from horizon.data import serialize
from horizon.model import labels as L

SPEC = {
    "name": "support_tickets",
    "labels": ["billing", "outage", "account"],
    "instructions": "You are a support analyst. Classify the ticket by its topic.",
    "verbalizer": {"outage": "service_outage"},
}


@pytest.fixture
def restore():
    yield
    tasks.TASKS.pop(SPEC["name"], None)
    tasks.activate("nidd5g")


def test_config_task_activates_like_a_builtin(restore):
    nidd = (tuple(serialize.LABELS), dict(L.VERBALIZER), L.SYSTEM)
    t = tasks.activate({"task": SPEC})
    assert tasks.task_name({"task": SPEC}) == "support_tickets" == tasks.active().name
    assert list(serialize.LABELS) == SPEC["labels"]
    assert L.VERBALIZER == {"billing": "billing", "outage": "service_outage", "account": "account"}
    assert (
        L.SYSTEM
        == t.system
        == (
            "You are a support analyst. Classify the ticket by its topic. "
            "Answer with exactly one label: billing, service_outage, account."
        )
    )
    tasks.activate("nidd5g")
    assert (tuple(serialize.LABELS), dict(L.VERBALIZER), L.SYSTEM) == nidd


def test_builtin_tasks_are_unchanged(restore):
    assert set(tasks.BUILTIN) == {"nidd5g", "asrs"}
    for name in ("nidd5g", "asrs"):
        assert tasks.activate(name) is tasks.BUILTIN[name]
        assert tasks.activate({"task": name}) is tasks.BUILTIN[name]
    assert tasks.task_name({}) == tasks.task_name(None) == "nidd5g"


@pytest.mark.parametrize(
    "change, msg",
    [
        ({"name": "Bad Name"}, "task.name"),
        ({"name": "asrs"}, "built-in"),
        ({"labels": ["only_one"]}, "at least 2"),
        ({"labels": ["a", "a"]}, "at least 2 distinct"),
        ({"labels": ["a", ""]}, "non-empty"),
        ({"instructions": " "}, "instructions"),
        ({"verbalizer": {"nope": "x"}}, "verbalizer keys"),
        ({"verbalizer": {"billing": "account"}}, "distinct answers"),
        ({"shared_class": "nope"}, "shared_class"),
        ({"labelz": []}, "unknown keys"),
    ],
)
def test_bad_specs_fail_loudly(change, msg, restore):
    with pytest.raises(ValueError, match=msg):
        tasks.activate({"task": SPEC | change})
