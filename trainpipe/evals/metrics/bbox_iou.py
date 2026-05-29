"""Bounding-box IoU for document-layout tasks (Phase 9).

The prediction is expected to be a JSON list of boxes, where each box is
either:

* ``[x1, y1, x2, y2]`` — corner coords, OR
* ``{"box": [x1, y1, x2, y2], "label": "..."}``

Gold has the same shape under ``sample[gold_field]``.

We compute a *greedy matching* IoU @ a configurable threshold:

1. For each pred box, find the gold box with highest IoU.
2. If IoU >= threshold AND (labels match if both present), it's a true
   positive; remove both from the pool.
3. Remaining preds = FP, remaining golds = FN.

Score per sample = F1 over (TP, FP, FN). 0.0 if both sides empty: a
"perfect score for emitting nothing" would let a broken model that
predicts ``[]`` everywhere look good on any sparsely-annotated suite.

Config:

* ``gold_field`` (str, default ``"gold_boxes"``).
* ``iou_threshold`` (float, default 0.5).
* ``label_strict`` (bool, default True) — if True and both sides supply a
  ``label``, mismatched labels disqualify the match even at high IoU.
"""

import json
from typing import Any

from .base import Metric, register


def _box_of(item: Any) -> tuple[list[float], str | None] | None:
    """Normalize an item to ``([x1,y1,x2,y2], label_or_None)``.

    Accepts a 4-tuple/list or a dict with ``box``. Returns None for
    malformed entries so the caller can skip them.
    """
    if isinstance(item, list) and len(item) == 4 and all(
        isinstance(v, int | float) for v in item
    ):
        return [float(v) for v in item], None
    if isinstance(item, dict):
        b = item.get("box")
        if not (isinstance(b, list) and len(b) == 4):
            return None
        if not all(isinstance(v, int | float) for v in b):
            return None
        lbl = item.get("label")
        return [float(v) for v in b], str(lbl) if lbl is not None else None
    return None


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # Tolerate reversed coords.
    ax1, ax2 = sorted((ax1, ax2))
    ay1, ay2 = sorted((ay1, ay2))
    bx1, bx2 = sorted((bx1, bx2))
    by1, by2 = sorted((by1, by2))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@register
class BboxIoUMetric(Metric):
    kind = "bounding_box_iou"

    def _validate_config(self) -> None:
        thr = self.config.get("iou_threshold", 0.5)
        if not isinstance(thr, int | float) or not 0.0 <= thr <= 1.0:
            raise ValueError("iou_threshold must be a number in [0, 1]")
        ls = self.config.get("label_strict", True)
        if not isinstance(ls, bool):
            raise ValueError("label_strict must be a bool")
        gf = self.config.get("gold_field", "gold_boxes")
        if not isinstance(gf, str) or not gf:
            raise ValueError("gold_field must be a non-empty string")

    def score(self, prediction: str, sample: dict[str, Any]) -> float:
        gold_field = self.config.get("gold_field", "gold_boxes")
        threshold = float(self.config.get("iou_threshold", 0.5))
        label_strict = bool(self.config.get("label_strict", True))

        try:
            pred_raw = json.loads(prediction) if prediction.strip() else []
        except json.JSONDecodeError:
            return 0.0
        if not isinstance(pred_raw, list):
            return 0.0
        gold_raw = sample.get(gold_field) or []
        if not isinstance(gold_raw, list):
            return 0.0

        preds = [b for b in (_box_of(x) for x in pred_raw) if b is not None]
        golds = [b for b in (_box_of(x) for x in gold_raw) if b is not None]

        if not preds and not golds:
            # Returning 1.0 here would reward a model that predicts ``[]``
            # on every sample. Anchor at 0.0 instead — see module docstring.
            return 0.0

        used_gold: set[int] = set()
        tp = 0
        for p_box, p_label in preds:
            best_iou = -1.0
            best_idx = -1
            for i, (g_box, g_label) in enumerate(golds):
                if i in used_gold:
                    continue
                if label_strict and p_label and g_label and p_label != g_label:
                    continue
                iou = _iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= threshold:
                tp += 1
                used_gold.add(best_idx)
        fp = len(preds) - tp
        fn = len(golds) - tp
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
