"""Implementation of the "forget user Y" scan.

The public surface is :func:`scan_datasets_for_term`. It takes a
:class:`aiosqlite.Connection`, a search term, and a list of dataset
formats to scan (default: ``jsonl``). For each match it records the
dataset id + name + hit count + a few line samples, then resolves
recursively which registered models trained on that dataset (or any
derived dataset thereof) so the operator has the full impact list in
one report.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

from ..core import repository

logger = logging.getLogger(__name__)


@dataclass
class DatasetHit:
    """One dataset whose on-disk content matched the search term."""

    dataset_id: str
    dataset_name: str
    path: str
    line_count: int | None
    hit_count: int
    # First few matched line numbers (1-indexed). Helps the operator
    # verify the match is real before redacting.
    sample_line_numbers: list[int] = field(default_factory=list)


@dataclass
class ModelImpact:
    """A registered model whose training data ultimately contained hits."""

    model_id: str
    name: str
    version: int
    description: str | None
    # Dataset ids in this model's lineage that had hits — the operator
    # can drill into each to decide whether to redact + retrain.
    via_dataset_ids: list[str] = field(default_factory=list)


@dataclass
class ForgetReport:
    """Result of one scan. JSON-serializable via :meth:`to_dict`."""

    term: str
    is_regex: bool
    case_sensitive: bool
    scanned_datasets: int
    skipped_datasets: list[str] = field(default_factory=list)
    hits: list[DatasetHit] = field(default_factory=list)
    impacted_models: list[ModelImpact] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "is_regex": self.is_regex,
            "case_sensitive": self.case_sensitive,
            "scanned_datasets": self.scanned_datasets,
            "skipped_datasets": self.skipped_datasets,
            "hits": [
                {
                    "dataset_id": h.dataset_id,
                    "dataset_name": h.dataset_name,
                    "path": h.path,
                    "line_count": h.line_count,
                    "hit_count": h.hit_count,
                    "sample_line_numbers": h.sample_line_numbers,
                }
                for h in self.hits
            ],
            "impacted_models": [
                {
                    "model_id": m.model_id,
                    "name": m.name,
                    "version": m.version,
                    "description": m.description,
                    "via_dataset_ids": m.via_dataset_ids,
                }
                for m in self.impacted_models
            ],
        }


def _build_matcher(
    term: str, *, is_regex: bool, case_sensitive: bool
) -> "Matcher":
    if is_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(term, flags)
        except re.error as e:
            raise ValueError(f"invalid regex: {e}") from None

        def regex_match(line: str) -> bool:
            return bool(pattern.search(line))

        return regex_match
    needle = term if case_sensitive else term.casefold()
    if case_sensitive:
        def substr_match(line: str) -> bool:
            return needle in line
    else:
        def substr_match(line: str) -> bool:
            return needle in line.casefold()
    return substr_match


# Matcher is a simple callable predicate.
Matcher = "callable[[str], bool]"


def _scan_one(
    path: Path, matcher, *, max_samples: int = 10, max_line_bytes: int = 1024 * 1024
) -> tuple[int, list[int]]:
    """Return ``(hit_count, sample_line_numbers)`` for one file.

    ``max_line_bytes`` is a defensive cap on per-line read size — a
    pathologically long JSONL record (someone embedded a 4GB base64
    blob) shouldn't blow up memory while we're trying to scan a
    50k-row dataset.
    """
    hits = 0
    samples: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if len(line) > max_line_bytes:
                # Truncate before matching; an embedded blob can still
                # contain the search term but we don't want OOM.
                line = line[:max_line_bytes]
            if matcher(line):
                hits += 1
                if len(samples) < max_samples:
                    samples.append(lineno)
    return hits, samples


async def scan_datasets_for_term(
    conn: aiosqlite.Connection,
    term: str,
    *,
    is_regex: bool = False,
    case_sensitive: bool = False,
    formats: tuple[str, ...] = ("jsonl",),
) -> ForgetReport:
    """Find every registered dataset whose on-disk content matches ``term``
    and the models that ultimately trained on those datasets.

    ``term`` is a substring by default; pass ``is_regex=True`` to switch
    to ``re.search``. Only datasets whose ``format`` is in ``formats``
    are scanned — parquet etc. are intentionally skipped (the scanner
    can't read them without extra deps) and listed in
    ``skipped_datasets``.
    """
    matcher = _build_matcher(
        term, is_regex=is_regex, case_sensitive=case_sensitive
    )
    report = ForgetReport(
        term=term, is_regex=is_regex, case_sensitive=case_sensitive,
        scanned_datasets=0,
    )

    datasets = await repository.list_datasets(conn)
    for rec in datasets:
        if rec.format not in formats:
            report.skipped_datasets.append(
                f"{rec.id} ({rec.format} not in {sorted(formats)})"
            )
            continue
        report.scanned_datasets += 1
        p = Path(rec.path)
        if not p.is_file():
            report.skipped_datasets.append(f"{rec.id} (file missing on disk)")
            continue
        try:
            hit_count, samples = _scan_one(p, matcher)
        except Exception:
            logger.exception("scan failed for %s", rec.id)
            report.skipped_datasets.append(f"{rec.id} (scan error)")
            continue
        if hit_count == 0:
            continue
        report.hits.append(
            DatasetHit(
                dataset_id=rec.id,
                dataset_name=rec.name,
                path=rec.path,
                line_count=rec.line_count,
                hit_count=hit_count,
                sample_line_numbers=samples,
            )
        )

    # Resolve recursive model lineage for the hit set.
    impacted: dict[str, ModelImpact] = {}
    for hit in report.hits:
        model_ids = await repository.models_using_dataset_recursive(
            conn, hit.dataset_id
        )
        for mid in model_ids:
            model = await repository.get_model(conn, mid)
            if model is None:
                continue
            entry = impacted.get(mid)
            if entry is None:
                entry = ModelImpact(
                    model_id=mid,
                    name=model.name,
                    version=model.version,
                    description=model.description,
                )
                impacted[mid] = entry
            if hit.dataset_id not in entry.via_dataset_ids:
                entry.via_dataset_ids.append(hit.dataset_id)
    report.impacted_models = sorted(
        impacted.values(), key=lambda m: (m.name, m.version)
    )
    return report
