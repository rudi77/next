"""Sentence-level BLEU (n-gram precision with brevity penalty).

Self-contained implementation — no ``sacrebleu`` / ``nltk`` dependency.
We use sentence-BLEU rather than corpus-BLEU because the eval runner
scores one prediction at a time; per-sample scores then aggregate via
mean (the standard :meth:`Metric.aggregate`).

Sentence-BLEU degenerates to 0 whenever a higher-order n-gram has zero
matches, which is common on short outputs. Chen & Cherry (2014)
``smoothing-1`` adds a small epsilon to the matched count whenever it
would otherwise be 0; this is on by default and matches what most
practical sentence-BLEU implementations do.

Config:

* ``gold_field`` (str, default ``"gold"``).
* ``max_n`` (int, default 4) — highest n-gram order to score.
* ``smoothing`` (bool, default ``True``) — Chen & Cherry smoothing-1.
* ``case_insensitive`` (bool, default ``True``).
* ``weights`` (list[float], optional) — per-n weights. Length must equal
  ``max_n``; default is uniform ``1/max_n``.
"""

import math
from collections import Counter
from typing import Any

from .base import Metric, register

_SMOOTH_EPSILON = 0.1


def _tokenize(text: str, *, case_insensitive: bool) -> list[str]:
    if case_insensitive:
        text = text.lower()
    return text.split()


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(
    prediction: list[str],
    reference: list[str],
    *,
    max_n: int,
    smoothing: bool,
    weights: list[float],
) -> float:
    if not prediction or not reference:
        return 0.0

    # Effective order: drop max_n to whatever the prediction (or reference)
    # can actually produce. Otherwise short outputs always score 0 because
    # we'd have no 4-grams to count.
    effective_n = min(max_n, len(prediction), len(reference))
    if effective_n < 1:
        return 0.0

    # Re-normalize the weights when we shrink the order so they still sum to 1.
    active_weights = weights[:effective_n]
    wsum = sum(active_weights)
    active_weights = [w / wsum for w in active_weights] if wsum > 0 else []

    log_precisions: list[float] = []
    for n in range(1, effective_n + 1):
        pred_ng = _ngram_counts(prediction, n)
        ref_ng = _ngram_counts(reference, n)
        total = sum(pred_ng.values())
        if total == 0:
            return 0.0
        clipped = sum(min(c, ref_ng[ng]) for ng, c in pred_ng.items())
        if clipped == 0:
            if not smoothing:
                return 0.0
            clipped_eff = _SMOOTH_EPSILON
        else:
            clipped_eff = float(clipped)
        log_precisions.append(math.log(clipped_eff / total))

    weighted_log_prec = sum(
        w * lp for w, lp in zip(active_weights, log_precisions, strict=False)
    )
    geo_mean = math.exp(weighted_log_prec)

    pred_len = len(prediction)
    ref_len = len(reference)
    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / pred_len)
    return bp * geo_mean


@register
class BleuMetric(Metric):
    kind = "bleu"

    def _validate_config(self) -> None:
        self.gold_field: str = self.config.get("gold_field", "gold")
        self.max_n: int = int(self.config.get("max_n", 4))
        self.smoothing: bool = bool(self.config.get("smoothing", True))
        self.case_insensitive: bool = bool(
            self.config.get("case_insensitive", True)
        )
        if not isinstance(self.gold_field, str) or not self.gold_field:
            raise ValueError("bleu.gold_field must be a non-empty string")
        if self.max_n < 1 or self.max_n > 10:
            raise ValueError("bleu.max_n must be between 1 and 10")
        weights = self.config.get("weights")
        if weights is None:
            self.weights = [1.0 / self.max_n] * self.max_n
        else:
            if len(weights) != self.max_n:
                raise ValueError(
                    f"bleu.weights length ({len(weights)}) must equal max_n "
                    f"({self.max_n})"
                )
            total = sum(weights)
            if total <= 0:
                raise ValueError("bleu.weights must sum to a positive value")
            self.weights = [float(w) for w in weights]

    def score(self, prediction: str, sample: dict[str, Any]) -> float:
        gold = sample.get(self.gold_field)
        if gold is None:
            return 0.0
        pred_tokens = _tokenize(prediction, case_insensitive=self.case_insensitive)
        ref_tokens = _tokenize(str(gold), case_insensitive=self.case_insensitive)
        return sentence_bleu(
            pred_tokens,
            ref_tokens,
            max_n=self.max_n,
            smoothing=self.smoothing,
            weights=self.weights,
        )
