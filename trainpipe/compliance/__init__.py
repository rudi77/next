"""GDPR compliance helpers (Phase 15 follow-up).

The flagship operation is the "forget user Y" workflow: an operator
supplies an identifier (email, customer id, account id, anything that
might appear verbatim in training data), and the script scans every
registered dataset's JSONL on disk for hits, then walks the lineage to
report which models trained on a dataset that contains those hits and
therefore need to be retrained.

This module does NOT redact anything itself — it is a *reporting*
tool. The operator follows up with ``POST /datasets/{id}/redact`` (or a
manual edit) and re-trains. Auto-redact is dangerous because the
matching is purely substring/regex; a false positive would silently
mutate training data.
"""

from .forget import (
    DatasetHit,
    ForgetReport,
    ModelImpact,
    scan_datasets_for_term,
)

__all__ = [
    "DatasetHit",
    "ForgetReport",
    "ModelImpact",
    "scan_datasets_for_term",
]
