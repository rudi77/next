"""Base class + decorator for eval metrics."""

from abc import ABC, abstractmethod
from math import sqrt
from typing import Any, ClassVar

from ...api.schemas import MetricAggregate

_REGISTRY: dict[str, type["Metric"]] = {}


def register(metric_cls: type["Metric"]) -> type["Metric"]:
    """Register a Metric subclass under its ``kind`` attribute.

    Decorator form::

        @register
        class FooMetric(Metric):
            kind = "foo"
            ...

    Re-registering the same class is a no-op (lets the package scanner
    re-import modules without complaining). Re-registering a *different*
    class under the same kind raises ``ValueError`` — that almost always
    means a typo or a copy-paste mistake in a metric file.
    """
    if not metric_cls.kind:
        raise ValueError(
            f"{metric_cls.__name__} must set a non-empty 'kind' class attribute "
            "before @register"
        )
    existing = _REGISTRY.get(metric_cls.kind)
    if existing is not None and existing is not metric_cls:
        raise ValueError(
            f"metric kind '{metric_cls.kind}' is already registered to "
            f"{existing.__module__}.{existing.__name__}"
        )
    _REGISTRY[metric_cls.kind] = metric_cls
    return metric_cls


class Metric(ABC):
    """Abstract base for a single eval metric.

    Subclasses must:

    1. Set the ``kind`` class attribute to a unique string (matches
       :class:`MetricConfig.kind`).
    2. Implement :meth:`score`, returning a float per sample. Higher is
       better by default — see ``higher_is_better`` for exceptions.

    Config validation (optional) goes in :meth:`_validate_config`, which
    runs in ``__init__`` so misconfigured metrics fail at suite-creation
    time rather than mid-eval.
    """

    kind: ClassVar[str] = ""
    higher_is_better: ClassVar[bool] = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._validate_config()

    def _validate_config(self) -> None:
        """Override to validate ``self.config``. Raise ``ValueError`` on bad input."""
        return None

    @abstractmethod
    def score(self, prediction: str, sample: dict[str, Any]) -> float:
        """Score one prediction against the sample's gold data."""

    def aggregate(self, scores: list[float]) -> MetricAggregate:
        """Mean + sample-std over per-sample scores.

        Subclasses can override to include extras (e.g. per-class breakdown
        for classification metrics) in :class:`MetricAggregate.extras`.
        """
        if not scores:
            return MetricAggregate(mean=0.0, std=0.0, count=0)
        n = len(scores)
        mean = sum(scores) / n
        if n > 1:
            var = sum((s - mean) ** 2 for s in scores) / (n - 1)
            std = sqrt(var)
        else:
            std = 0.0
        return MetricAggregate(mean=mean, std=std, count=n)
