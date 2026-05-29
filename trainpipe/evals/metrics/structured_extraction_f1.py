"""Structured extraction F1 for Doc-AI tasks (Phase 9).

This is a stricter, schema-aware sibling of ``field_level_f1`` aimed at
document extraction:

* It expects both prediction and gold to be JSON objects with a known
  set of *expected fields* (configured via the suite). Fields outside
  the schema in the prediction are penalized as FP; missing schema
  fields are FN.
* Nested objects flatten with a configurable separator (default ``"."``)
* Numeric near-equality is allowed within a tolerance (default 0 — exact).

The result per sample is F1 over (TP, FP, FN) across all *expected*
fields. The contract is intentionally narrower than ``field_level_f1``:
the suite declares the schema, the metric grades against it. That makes
runs across different gold dialects (one annotator adds extra ``meta``,
the other doesn't) comparable.

Config:

* ``schema_fields`` (list[str]) — required, non-empty. Field names in
  dotted-path form, e.g. ``["invoice_number", "total.amount",
  "total.currency"]``.
* ``gold_field`` (str, default ``"gold"``).
* ``case_insensitive`` (bool, default True) — strings compared case-folded.
* ``numeric_tolerance`` (float, default 0.0) — abs diff allowed for
  numeric comparisons.
* ``flatten_separator`` (str, default ``"."``).
"""

import json
from typing import Any

from .base import Metric, register


def _flatten(obj: Any, prefix: str, sep: str, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        if not obj:
            out[prefix] = obj
            return
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            _flatten(v, key, sep, out)
    elif isinstance(obj, list):
        if not obj:
            out[prefix] = obj
            return
        for i, v in enumerate(obj):
            key = f"{prefix}{sep}{i}" if prefix else str(i)
            _flatten(v, key, sep, out)
    else:
        out[prefix] = obj


def _values_equal(
    a: Any, b: Any, *, case_insensitive: bool, num_tol: float
) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(float(a) - float(b)) <= num_tol
    sa = str(a).strip()
    sb = str(b).strip()
    if case_insensitive:
        sa, sb = sa.casefold(), sb.casefold()
    return sa == sb


@register
class StructuredExtractionF1Metric(Metric):
    kind = "structured_extraction_f1"

    def _validate_config(self) -> None:
        fields = self.config.get("schema_fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("schema_fields must be a non-empty list of strings")
        if not all(isinstance(f, str) and f for f in fields):
            raise ValueError("schema_fields entries must be non-empty strings")
        gf = self.config.get("gold_field", "gold")
        if not isinstance(gf, str) or not gf:
            raise ValueError("gold_field must be a non-empty string")
        tol = self.config.get("numeric_tolerance", 0.0)
        if not isinstance(tol, int | float) or tol < 0:
            raise ValueError("numeric_tolerance must be a non-negative number")

    def score(self, prediction: str, sample: dict[str, Any]) -> float:
        schema: list[str] = list(self.config["schema_fields"])
        gold_field: str = self.config.get("gold_field", "gold")
        sep: str = self.config.get("flatten_separator", ".")
        ci: bool = bool(self.config.get("case_insensitive", True))
        tol: float = float(self.config.get("numeric_tolerance", 0.0))

        try:
            pred_obj = json.loads(prediction) if prediction.strip() else {}
        except json.JSONDecodeError:
            return 0.0
        if not isinstance(pred_obj, dict):
            return 0.0
        gold_obj = sample.get(gold_field) or {}
        if not isinstance(gold_obj, dict):
            return 0.0

        pred_flat: dict[str, Any] = {}
        gold_flat: dict[str, Any] = {}
        _flatten(pred_obj, "", sep, pred_flat)
        _flatten(gold_obj, "", sep, gold_flat)

        # FP: prediction keys not in schema (model invented fields).
        # We grade only against schema, so out-of-schema pred entries
        # count as FP precisely once each.
        fp_unknown = sum(1 for k in pred_flat if k not in schema)

        tp = 0
        fp = fp_unknown
        fn = 0
        for field in schema:
            p_has = field in pred_flat
            g_has = field in gold_flat
            if g_has and p_has:
                if _values_equal(
                    pred_flat[field],
                    gold_flat[field],
                    case_insensitive=ci,
                    num_tol=tol,
                ):
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif g_has and not p_has:
                fn += 1
            elif p_has and not g_has:
                fp += 1

        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
