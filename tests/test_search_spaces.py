from typing import Any

import pytest

from trainpipe.api.schemas import ExperimentSpec, SearchSpaceEntry
from trainpipe.autoresearch.search_spaces import (
    apply_overrides,
    sample_spec,
    sample_value,
)


class _FakeTrial:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.calls: list[tuple[str, str, dict]] = []

    def suggest_categorical(self, name, choices):
        self.calls.append((name, "categorical", {"choices": list(choices)}))
        return self.plan[name]

    def suggest_float(self, name, low, high, *, log=False):
        self.calls.append((name, "float", {"low": low, "high": high, "log": log}))
        return self.plan[name]

    def suggest_int(self, name, low, high):
        self.calls.append((name, "int", {"low": low, "high": high}))
        return self.plan[name]


def test_sample_value_categorical():
    trial = _FakeTrial({"x": "b"})
    entry = SearchSpaceEntry(kind="categorical", choices=["a", "b", "c"])
    assert sample_value(trial, "x", entry) == "b"
    assert trial.calls == [("x", "categorical", {"choices": ["a", "b", "c"]})]


def test_sample_value_uniform():
    trial = _FakeTrial({"y": 0.42})
    entry = SearchSpaceEntry(kind="uniform", low=0.0, high=1.0)
    assert sample_value(trial, "y", entry) == 0.42
    assert trial.calls[0][2] == {"low": 0.0, "high": 1.0, "log": False}


def test_sample_value_loguniform_sets_log_flag():
    trial = _FakeTrial({"lr": 1e-4})
    entry = SearchSpaceEntry(kind="loguniform", low=1e-5, high=1e-2)
    assert sample_value(trial, "lr", entry) == 1e-4
    assert trial.calls[0][2]["log"] is True


def test_sample_value_int():
    trial = _FakeTrial({"r": 8})
    entry = SearchSpaceEntry(kind="int", low=4, high=16)
    assert sample_value(trial, "r", entry) == 8


def test_sample_value_missing_choices_raises():
    trial = _FakeTrial({})
    with pytest.raises(ValueError, match="choices"):
        sample_value(trial, "x", SearchSpaceEntry(kind="categorical"))


def test_sample_value_missing_range_raises():
    trial = _FakeTrial({})
    with pytest.raises(ValueError, match="low and high"):
        sample_value(trial, "y", SearchSpaceEntry(kind="uniform"))


def test_sample_value_unknown_kind_raises():
    trial = _FakeTrial({})
    entry = SearchSpaceEntry.model_construct(kind="bogus")  # bypass literal check
    with pytest.raises(ValueError, match="unknown search-space kind"):
        sample_value(trial, "z", entry)


def test_apply_overrides_top_level():
    base = ExperimentSpec(model="m", dataset=["d"])
    new = apply_overrides(base, {"gpu_count": 4})
    assert new.gpu_count == 4
    assert new.model == "m"


def test_apply_overrides_nested():
    base = ExperimentSpec(model="m", dataset=["d"])
    new = apply_overrides(
        base,
        {
            "hyperparameters.learning_rate": 2e-4,
            "hyperparameters.lora_rank": 16,
        },
    )
    assert new.hyperparameters.learning_rate == 2e-4
    assert new.hyperparameters.lora_rank == 16
    # Untouched fields keep defaults
    assert new.hyperparameters.num_train_epochs == 1


def test_apply_overrides_returns_new_instance():
    base = ExperimentSpec(model="m", dataset=["d"])
    new = apply_overrides(base, {"gpu_count": 4})
    assert new is not base
    assert base.gpu_count == 1


def test_sample_spec_end_to_end():
    base = ExperimentSpec(model="m", dataset=["d"])
    trial = _FakeTrial(
        {
            "hyperparameters.learning_rate": 1e-3,
            "hyperparameters.lora_rank": 16,
        }
    )
    space = {
        "hyperparameters.learning_rate": SearchSpaceEntry(
            kind="loguniform", low=1e-5, high=1e-2
        ),
        "hyperparameters.lora_rank": SearchSpaceEntry(
            kind="categorical", choices=[4, 8, 16]
        ),
    }
    spec, sampled = sample_spec(trial, base, space)
    assert spec.hyperparameters.learning_rate == 1e-3
    assert spec.hyperparameters.lora_rank == 16
    assert sampled == {
        "hyperparameters.learning_rate": 1e-3,
        "hyperparameters.lora_rank": 16,
    }
