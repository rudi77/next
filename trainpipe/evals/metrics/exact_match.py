"""Exact-match metric.

Compares the model's prediction string to the gold answer extracted from
``sample[gold_field]``. Useful for short-answer extraction, classification,
and any task where the gold response is a single canonical string.

Config:

* ``gold_field`` (str, default ``"gold"``) — which sample field holds the
  gold answer. For ms-swift JSONL with ``response``, set this to
  ``"response"``.
* ``normalize`` (bool, default ``True``) — lowercase + strip whitespace
  on both sides before comparison.
* ``strip_punctuation`` (bool, default ``False``) — also drop trailing
  ``.``, ``!``, ``?``.
"""

import string
from typing import Any

from .base import Metric, register


@register
class ExactMatchMetric(Metric):
    kind = "exact_match"

    def _validate_config(self) -> None:
        self.gold_field: str = self.config.get("gold_field", "gold")
        self.normalize: bool = bool(self.config.get("normalize", True))
        self.strip_punctuation: bool = bool(
            self.config.get("strip_punctuation", False)
        )
        if not isinstance(self.gold_field, str) or not self.gold_field:
            raise ValueError("exact_match.gold_field must be a non-empty string")

    def _norm(self, text: str) -> str:
        if self.normalize:
            text = text.strip().lower()
        if self.strip_punctuation:
            text = text.rstrip(string.punctuation + string.whitespace)
        return text

    def score(self, prediction: str, sample: dict[str, Any]) -> float:
        gold = sample.get(self.gold_field)
        if gold is None:
            return 0.0
        pred = self._norm(prediction)
        gold_str = self._norm(str(gold))
        return 1.0 if pred == gold_str else 0.0
