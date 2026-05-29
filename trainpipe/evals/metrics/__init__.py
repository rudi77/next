"""Metric plugin registry.

Each ``.py`` file under this package that defines a :class:`Metric`
subclass and decorates it with :func:`register` (or sets a non-empty
``kind`` class attribute) is auto-loaded on first lookup. No
``pyproject`` entry points required.

Lookup flow:

1. ``get_metric_class("exact_match")`` triggers a one-shot scan of this
   package (``_scan_once``) — every sibling module is imported, which
   side-effects register their classes.
2. The returned class is instantiated with the per-metric ``config`` dict
   from :class:`MetricConfig` and used to score each sample.

Adding a new metric: drop a file in this directory that subclasses
``Metric``, sets ``kind = "your_kind"``, and implements
``score(prediction, sample)``. No registration call needed.
"""

import importlib
import pkgutil
from typing import ClassVar


class UnknownMetricKind(ValueError):
    """Raised when a MetricConfig.kind doesn't match any discovered plugin."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(f"unknown metric kind: '{kind}'")


# Imported below UnknownMetricKind so that plugin modules can `from .base
# import …` at the top without a circular hazard.
from .base import _REGISTRY, Metric, register  # noqa: E402

_SCANNED = False


def _scan_once() -> None:
    global _SCANNED
    if _SCANNED:
        return
    pkg_path = __path__  # type: ignore[name-defined]
    for _, modname, _ in pkgutil.iter_modules(pkg_path):
        if modname == "base":
            continue
        importlib.import_module(f"{__name__}.{modname}")
    _SCANNED = True


def get_metric_class(kind: str) -> type[Metric]:
    """Resolve a metric kind to its implementation class.

    Triggers a one-shot package scan on first call. Raises
    :class:`UnknownMetricKind` if no plugin claims the kind.
    """
    _scan_once()
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise UnknownMetricKind(kind)
    return cls


def list_metric_kinds() -> list[str]:
    """All registered metric kinds, sorted. Triggers the scan."""
    _scan_once()
    return sorted(_REGISTRY)


__all__: ClassVar = [
    "Metric",
    "UnknownMetricKind",
    "get_metric_class",
    "list_metric_kinds",
    "register",
]
