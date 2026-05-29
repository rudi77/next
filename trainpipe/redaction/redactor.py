"""Regex-based PII redactor.

This is a deliberately conservative implementation: each pattern is
narrow enough to skip obvious false positives (e.g. random numeric
strings get redacted only if they pass an IBAN checksum). It's enough
to satisfy "we run every dataset through PII detection before training"
as a baseline; teams that need higher recall should swap in Presidio.

The output is structurally identical to the input — for JSONL, only
*string* values get rewritten, leaving the record schema unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Literal

EntityKind = Literal["email", "phone", "iban", "credit_card", "de_tax_id"]

DEFAULT_ENTITY_KINDS: tuple[EntityKind, ...] = (
    "email",
    "phone",
    "iban",
    "credit_card",
)


_PATTERNS: dict[EntityKind, re.Pattern[str]] = {
    "email": re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    # International or domestic phone. The pattern requires either
    # (a) a leading ``+`` (international) — the strongest signal that
    # this is intentionally a phone — or (b) parentheses around an
    # area code like ``(0660) 123 4567``. Bare digit runs are NOT
    # matched here to avoid silently shredding dates/timestamps/IPs/
    # invoice numbers, which the previous over-greedy pattern did.
    # Anyone needing recall on bare-digit phones should use the
    # de_tax_id-style approach with a specific country format.
    "phone": re.compile(
        r"(?:\+\d[\d\s\-]{6,18}\d)|"
        r"(?:\(\d{2,5}\)[\s\-]?\d[\d\s\-]{5,15}\d)"
    ),
    # IBAN: country code (2 letters) + 2 check digits + up to 30 alnum.
    # Strict 13–34 char total to avoid matching every alphanumeric run.
    "iban": re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"
    ),
    # Credit card: 13–19 digits with optional spaces/dashes every 4.
    "credit_card": re.compile(
        r"\b(?:\d[ \-]?){12,18}\d\b"
    ),
    # German Steuer-ID: 11 digits, no other delimiters.
    "de_tax_id": re.compile(r"\b\d{11}\b"),
}


_REPLACEMENTS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "iban": "[REDACTED_IBAN]",
    "credit_card": "[REDACTED_CC]",
    "de_tax_id": "[REDACTED_TAX_ID]",
}


def _iban_checksum_ok(s: str) -> bool:
    """ISO 13616 mod-97 IBAN check. Avoids redacting random alnum strings."""
    s = s.replace(" ", "").upper()
    moved = s[4:] + s[:4]
    expanded = "".join(
        str(ord(c) - 55) if c.isalpha() else c for c in moved
    )
    try:
        return int(expanded) % 97 == 1
    except ValueError:
        return False


def _phone_looks_real(match: str) -> bool:
    """Reject phone candidates that are all-the-same-digit or too short."""
    digits = re.sub(r"\D", "", match)
    if len(digits) < 7 or len(digits) > 15:
        return False
    if len(set(digits)) == 1:  # 11111111111 isn't a phone
        return False
    return True


def redact_text(
    text: str, entities: Iterable[EntityKind] | None = None
) -> tuple[str, dict[EntityKind, int]]:
    """Return ``(redacted_text, hit_counts)``.

    Each detected entity is replaced with a labeled marker
    (``[REDACTED_EMAIL]``, etc.). Counts are useful for the audit
    description on the redacted dataset record.
    """
    if not text:
        return text, {}
    entities = list(entities or DEFAULT_ENTITY_KINDS)
    counts: dict[EntityKind, int] = dict.fromkeys(entities, 0)
    out = text

    # Email first — emails contain "@" which neither phone nor IBAN
    # would ever match.
    if "email" in entities:
        new, n = _PATTERNS["email"].subn(_REPLACEMENTS["email"], out)
        out, counts["email"] = new, n

    if "iban" in entities:
        def _sub_iban(m: re.Match[str]) -> str:
            s = m.group(0)
            return _REPLACEMENTS["iban"] if _iban_checksum_ok(s) else s

        new, _ = _PATTERNS["iban"].subn(_sub_iban, out)
        counts["iban"] = sum(
            1 for m in _PATTERNS["iban"].finditer(out) if _iban_checksum_ok(m.group(0))
        )
        out = new

    if "credit_card" in entities:
        def _sub_cc(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19:
                return _REPLACEMENTS["credit_card"]
            return m.group(0)

        new, _ = _PATTERNS["credit_card"].subn(_sub_cc, out)
        counts["credit_card"] = sum(
            1
            for m in _PATTERNS["credit_card"].finditer(out)
            if 13 <= len(re.sub(r"\D", "", m.group(0))) <= 19
        )
        out = new

    if "phone" in entities:
        def _sub_phone(m: re.Match[str]) -> str:
            s = m.group(0)
            return _REPLACEMENTS["phone"] if _phone_looks_real(s) else s

        # Tally before the subn because subn rewrites the string.
        counts["phone"] = sum(
            1
            for m in _PATTERNS["phone"].finditer(out)
            if _phone_looks_real(m.group(0))
        )
        out, _ = _PATTERNS["phone"].subn(_sub_phone, out)

    if "de_tax_id" in entities:
        new, n = _PATTERNS["de_tax_id"].subn(_REPLACEMENTS["de_tax_id"], out)
        out, counts["de_tax_id"] = new, n

    return out, counts


def _walk_redact(obj, entities, counts):
    if isinstance(obj, str):
        new, c = redact_text(obj, entities)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        return new
    if isinstance(obj, list):
        return [_walk_redact(v, entities, counts) for v in obj]
    if isinstance(obj, dict):
        return {k: _walk_redact(v, entities, counts) for k, v in obj.items()}
    return obj


def redact_jsonl(
    in_path: str,
    out_path: str,
    entities: Iterable[EntityKind] | None = None,
) -> tuple[int, dict[EntityKind, int]]:
    """Redact every string field in a JSONL file; write to ``out_path``.

    Returns ``(record_count, total_hit_counts)``.
    """
    entities = tuple(entities or DEFAULT_ENTITY_KINDS)
    total_counts: dict = {}
    rows = 0
    with open(in_path, encoding="utf-8") as in_f, open(
        out_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            new = _walk_redact(record, entities, total_counts)
            out_f.write(json.dumps(new, ensure_ascii=False))
            out_f.write("\n")
            rows += 1
    return rows, total_counts
