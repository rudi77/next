"""Field-level F1 for structured-extraction tasks.

The prediction is expected to be a JSON object (string). It is parsed and
compared field-by-field against the gold dict at ``sample[gold_field]``.

Per field (dotted path after flattening), we count:

* TP — both sides present and equal
* FP — only in prediction
* FN — only in gold

Then ``F1 = 2 * P * R / (P + R)``. Empty intersection → 0.

This matches the way human reviewers grade invoice / form extraction:
each field is its own pass/fail, the score is the harmonic mean of
precision and recall over the full schema.

Config:

* ``gold_field`` (str, default ``"gold"``) — sample key with the gold dict.
* ``ignore_keys`` (list[str], default ``[]``) — flattened keys to skip
  entirely (e.g. ``["meta.timestamp"]``).
* ``case_insensitive`` (bool, default ``True``) — string values compared
  case-insensitively after strip.
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


def _normalize_scalar(v: Any, *, case_insensitive: bool) -> Any:
    if isinstance(v, str):
        s = v.strip()
        return s.lower() if case_insensitive else s
    return v


@register
class FieldLevelF1Metric(Metric):
    kind = "field_level_f1"

    def _validate_config(self) -> None:
        self.gold_field: str = self.config.get("gold_field", "gold")
        self.ignore_keys: set[str] = set(self.config.get("ignore_keys", []))
        self.case_insensitive: bool = bool(self.config.get("case_insensitive", True))
        self.flatten_separator: str = str(self.config.get("flatten_separator", "."))
        if not isinstance(self.gold_field, str) or not self.gold_field:
            raise ValueError("field_level_f1.gold_field must be a non-empty string")
        if not self.flatten_separator:
            raise ValueError("field_level_f1.flatten_separator must be non-empty")

    def _parse_prediction(self, prediction: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(prediction)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def _flatten_for_comparison(self, obj: dict[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        _flatten(obj, "", self.flatten_separator, flat)
        return {k: v for k, v in flat.items() if k not in self.ignore_keys}

    def score(self, prediction: str, sample: dict[str, Any]) -> float:
        gold_raw = sample.get(self.gold_field)
        if not isinstance(gold_raw, dict):
            return 0.0
        pred = self._parse_prediction(prediction)
        if pred is None:
            return 0.0

        pred_flat = self._flatten_for_comparison(pred)
        gold_flat = self._flatten_for_comparison(gold_raw)

        pred_keys = set(pred_flat)
        gold_keys = set(gold_flat)
        common = pred_keys & gold_keys

        tp = 0
        for k in common:
            if _normalize_scalar(
                pred_flat[k], case_insensitive=self.case_insensitive
            ) == _normalize_scalar(
                gold_flat[k], case_insensitive=self.case_insensitive
            ):
                tp += 1

        fp = len(pred_keys - gold_keys) + (len(common) - tp)
        fn = len(gold_keys - pred_keys) + (len(common) - tp)

        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)
