"""Active-learning scoring + queue surfacing.

A *run* loads a model, iterates an unlabeled JSONL dataset, scores
each sample with an :class:`UncertaintyScorer`, and persists the top-N
into ``annotation_queue_items`` for downstream review (manual or via
Label Studio push).

Two stock scorers ship here; both work with any
:class:`InferenceBackend` (mock or transformers):

* :class:`DoublePassScorer` — runs the prompt twice at temperature 0.7
  and treats the normalized character-level difference of the two outputs
  as the uncertainty signal. High variance = the model isn't confident
  in what to emit. Two backend calls per sample, so it's the more
  expensive of the two but the more informative.
* :class:`LengthZScoreScorer` — runs once at T=0 and scores the absolute
  z-score of the response's character length across the run. Cheap, but
  only a proxy for uncertainty — useful as a baseline / fallback when
  the backend doesn't support stochastic sampling.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..api.schemas import InferenceParams
from ..evals.inference import InferenceBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample iteration
# ---------------------------------------------------------------------------


def iter_jsonl_samples(path: Path, limit: int | None = None) -> Iterable[
    tuple[int, dict[str, Any]]
]:
    """Stream ``(sample_index, record)`` from a JSONL file.

    Malformed lines are logged and skipped. ``limit`` caps the number
    of records yielded (sampled from the head, mirroring how the eval
    runner does it — keeps successive runs comparable).
    """
    yielded = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                logger.warning(
                    "active learning: skipping malformed line %d: %s",
                    idx,
                    e,
                )
                continue
            yield idx, record if isinstance(record, dict) else {"sample": record}
            yielded += 1
            if limit is not None and yielded >= limit:
                return


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


@dataclass
class ScoredSample:
    sample_index: int
    sample: dict[str, Any]
    prediction: str
    uncertainty: float


class UncertaintyScorer(ABC):
    """Score one sample's uncertainty given an inference backend.

    The ``async score()`` returns a :class:`ScoredSample` — the
    prediction the model produced (so callers don't re-run inference
    just for display) and the uncertainty number (higher = more
    informative for the annotator).
    """

    @abstractmethod
    async def score(
        self,
        backend: InferenceBackend,
        sample_index: int,
        sample: dict[str, Any],
        params: InferenceParams,
    ) -> ScoredSample: ...

    async def finalize(
        self, scored: list[ScoredSample]
    ) -> list[ScoredSample]:
        """Post-process all scored samples (e.g. compute z-scores).

        Default: return as-is.
        """
        return scored


def _char_diff_distance(a: str, b: str) -> float:
    """Normalized character-difference distance in [0, 1].

    A simple Levenshtein-distance implementation would be more accurate
    but blows up on long strings (O(n*m)). We use the symmetric
    Sørensen–Dice coefficient over the character bigram sets — fast and
    correlates with edit distance for short-to-medium responses.
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    def _bigrams(s: str) -> set[str]:
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    sa, sb = _bigrams(a), _bigrams(b)
    inter = len(sa & sb)
    dice = 2 * inter / (len(sa) + len(sb)) if (len(sa) + len(sb)) else 0.0
    return max(0.0, 1.0 - dice)


class DoublePassScorer(UncertaintyScorer):
    """Two T=0.7 passes; uncertainty = normalized Dice-distance of outputs."""

    def __init__(self, temperature: float = 0.7) -> None:
        self.temperature = temperature

    async def score(
        self,
        backend: InferenceBackend,
        sample_index: int,
        sample: dict[str, Any],
        params: InferenceParams,
    ) -> ScoredSample:
        # Override temperature for both passes; keep everything else as
        # supplied by the caller.
        p = params.model_copy(update={"temperature": self.temperature})
        first = await backend.predict(sample, p)
        second = await backend.predict(sample, p)
        unc = _char_diff_distance(first, second)
        return ScoredSample(
            sample_index=sample_index,
            sample=sample,
            prediction=first,
            uncertainty=unc,
        )


class LengthZScoreScorer(UncertaintyScorer):
    """One T=0 pass; uncertainty = |z-score(response length)|."""

    async def score(
        self,
        backend: InferenceBackend,
        sample_index: int,
        sample: dict[str, Any],
        params: InferenceParams,
    ) -> ScoredSample:
        p = params.model_copy(update={"temperature": 0.0})
        pred = await backend.predict(sample, p)
        return ScoredSample(
            sample_index=sample_index,
            sample=sample,
            prediction=pred,
            uncertainty=float(len(pred)),  # provisional, finalized below
        )

    async def finalize(
        self, scored: list[ScoredSample]
    ) -> list[ScoredSample]:
        if not scored:
            return scored
        lengths = [s.uncertainty for s in scored]
        mean = statistics.fmean(lengths)
        std = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        if std == 0:
            for s in scored:
                s.uncertainty = 0.0
            return scored
        for s in scored:
            s.uncertainty = abs((s.uncertainty - mean) / std)
        return scored


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclass
class ActiveLearningResult:
    scored_count: int
    queued_count: int
    top_items: list[ScoredSample] = field(default_factory=list)


async def run_active_learning(
    *,
    backend: InferenceBackend,
    dataset_path: Path,
    scorer: UncertaintyScorer,
    top_n: int,
    inference_params: InferenceParams | None = None,
    sample_limit: int | None = None,
) -> ActiveLearningResult:
    """Score every sample, rank, return the top ``top_n``.

    The backend is assumed to already be ``open()``ed (callers manage
    lifecycle so the same backend can be reused across runs).
    """
    params = inference_params or InferenceParams(temperature=0.0)
    scored: list[ScoredSample] = []
    for idx, record in iter_jsonl_samples(dataset_path, sample_limit):
        try:
            result = await scorer.score(backend, idx, record, params)
        except Exception:
            logger.exception(
                "active learning: scorer failed on sample %d", idx
            )
            continue
        scored.append(result)
    scored = await scorer.finalize(scored)
    # Sort by uncertainty descending, then sample_index for stability.
    scored.sort(key=lambda s: (-s.uncertainty, s.sample_index))
    top = scored[: max(0, top_n)]
    return ActiveLearningResult(
        scored_count=len(scored),
        queued_count=len(top),
        top_items=top,
    )


def make_scorer(name: str) -> UncertaintyScorer:
    if name == "double_pass":
        return DoublePassScorer()
    if name == "length_zscore":
        return LengthZScoreScorer()
    # Should never reach here — Pydantic Literal validates upstream.
    raise ValueError(f"unknown scorer: {name!r}")


# Suppress unused-import warning if math is not actually used; keep
# available for future scorers that want entropy etc.
_ = math
