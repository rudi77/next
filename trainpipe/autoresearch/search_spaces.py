"""Translate a SearchSpaceEntry into an Optuna ``trial.suggest_*`` call and
project the sampled values into an ExperimentSpec via dotted-path overrides.

The module is intentionally optuna-free at import time so tests can pass a
duck-typed trial stub.
"""

from typing import Any, Protocol

from ..api.schemas import ExperimentSpec, SearchSpaceEntry


class TrialProtocol(Protocol):
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...
    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = ...
    ) -> float: ...
    def suggest_int(self, name: str, low: int, high: int) -> int: ...


def sample_value(trial: TrialProtocol, name: str, entry: SearchSpaceEntry) -> Any:
    kind = entry.kind
    if kind == "categorical":
        if entry.choices is None:
            raise ValueError(f"categorical entry '{name}' requires 'choices'")
        return trial.suggest_categorical(name, entry.choices)
    if kind in ("uniform", "loguniform", "int"):
        if entry.low is None or entry.high is None:
            raise ValueError(f"entry '{name}' kind={kind} requires low and high")
        if kind == "uniform":
            return trial.suggest_float(name, entry.low, entry.high)
        if kind == "loguniform":
            return trial.suggest_float(name, entry.low, entry.high, log=True)
        return trial.suggest_int(name, int(entry.low), int(entry.high))
    raise ValueError(f"unknown search-space kind: {kind}")


def apply_overrides(base: ExperimentSpec, overrides: dict[str, Any]) -> ExperimentSpec:
    """Merge dotted-path overrides into ``base`` and return a new ExperimentSpec."""
    data = base.model_dump()
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cur: Any = data
        for p in parts[:-1]:
            existing = cur.get(p)
            if existing is None:
                existing = {}
                cur[p] = existing
            if not isinstance(existing, dict):
                raise ValueError(
                    f"override path '{dotted}' traverses non-dict at '{p}'"
                )
            cur = existing
        cur[parts[-1]] = value
    return ExperimentSpec.model_validate(data)


def sample_spec(
    trial: TrialProtocol,
    base: ExperimentSpec,
    search_space: dict[str, SearchSpaceEntry],
) -> tuple[ExperimentSpec, dict[str, Any]]:
    """Return (spec_with_overrides, sampled_values)."""
    overrides = {
        name: sample_value(trial, name, entry) for name, entry in search_space.items()
    }
    return apply_overrides(base, overrides), overrides
