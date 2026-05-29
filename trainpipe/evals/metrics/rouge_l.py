"""ROUGE-L (Longest Common Subsequence F1) over whitespace tokens.

For free-form chat/Q&A responses where exact match is too strict.
Implements ROUGE-L F1 directly (no rouge_score dep) so the runtime
stays light. See Lin (2004) for the original definition.

Config:

* ``gold_field`` (str, default ``"gold"``).
* ``case_insensitive`` (bool, default ``True``).
* ``beta`` (float, default ``1.0``) — F-beta weighting of recall vs.
  precision. The original ROUGE-L paper uses beta=1 (harmonic mean).
"""

from typing import Any

from .base import Metric, register


def _tokenize(text: str, *, case_insensitive: bool) -> list[str]:
    if case_insensitive:
        text = text.lower()
    return text.split()


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    # space-optimized DP: O(len(a)) memory
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        for j in range(len(b) + 1):
            curr[j] = 0
    return prev[len(b)]


@register
class RougeLMetric(Metric):
    kind = "rouge_l"

    def _validate_config(self) -> None:
        self.gold_field: str = self.config.get("gold_field", "gold")
        self.case_insensitive: bool = bool(self.config.get("case_insensitive", True))
        self.beta: float = float(self.config.get("beta", 1.0))
        if not isinstance(self.gold_field, str) or not self.gold_field:
            raise ValueError("rouge_l.gold_field must be a non-empty string")
        if self.beta <= 0:
            raise ValueError("rouge_l.beta must be positive")

    def score(self, prediction: str, sample: dict[str, Any]) -> float:
        gold = sample.get(self.gold_field)
        if gold is None:
            return 0.0
        pred_tokens = _tokenize(prediction, case_insensitive=self.case_insensitive)
        gold_tokens = _tokenize(str(gold), case_insensitive=self.case_insensitive)
        if not pred_tokens or not gold_tokens:
            return 0.0
        lcs = _lcs_length(pred_tokens, gold_tokens)
        if lcs == 0:
            return 0.0
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold_tokens)
        b2 = self.beta * self.beta
        denom = recall + b2 * precision
        if denom == 0:
            return 0.0
        return (1 + b2) * precision * recall / denom
