"""PII redaction (Phase 15).

A lightweight regex-based redactor that covers the most common PII
entities (email, phone, IBAN, credit-card-shape, German Tax-ID). It is
*not* a substitute for Microsoft Presidio for high-stakes deployments;
the architecture allows swapping in a Presidio-backed implementation
without changing call sites.
"""

from .redactor import (
    DEFAULT_ENTITY_KINDS,
    EntityKind,
    redact_jsonl,
    redact_text,
)

__all__ = [
    "DEFAULT_ENTITY_KINDS",
    "EntityKind",
    "redact_jsonl",
    "redact_text",
]
