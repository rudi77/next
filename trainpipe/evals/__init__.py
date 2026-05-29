"""Eval framework (Phase 6).

Subpackages:

* ``metrics`` — pluggable scoring functions. Each module under
  ``trainpipe/evals/metrics/`` that subclasses :class:`Metric` and uses
  :func:`register` is discovered automatically on first lookup.
"""
