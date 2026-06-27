"""Lightweight format detection + validation for uploaded datasets.

We trust the file extension to pick the parser, then sample the first ~100
records to ensure the file is well-formed. For JSONL/CSV we also count total
lines, which means one full pass over the file — combined with the SHA256 pass
during upload that's two reads of the whole file, so expect upload latency to
scale with size for multi-GB datasets. Parquet validation just opens the
metadata footer.

These checks are *not* a schema validator — ms-swift will detect missing
fields when it actually trains. The point is to fail at upload time when
the file is obviously garbage (wrong extension, broken JSON, empty file).
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

_SAMPLE_LINES = 100


class DatasetFormatError(ValueError):
    """Raised when the uploaded file doesn't parse as its declared format."""


@dataclass
class FormatInfo:
    """Result of dataset format detection."""

    format: str
    line_count: int | None = None
    # JSONL only: ordered list of media kinds detected by sampling
    # (``["images"]``, ``["images", "videos"]``, etc.). Empty for text-only.
    media_kinds: list[str] = field(default_factory=list)
    # JSONL only: True if every sampled record has ``prompt`` /
    # ``chosen`` / ``rejected`` keys — that's the DPO / preference shape
    # that ``swift rlhf`` expects (Phase 13). Detection is sample-based
    # so a partially-converted dataset returns False.
    is_preference: bool = False
    # JSONL only: True if every sampled record is a single non-empty
    # ``text`` field — the raw-text shape ``swift pt`` (continued
    # pretraining) expects. Sample-based like ``is_preference``.
    is_pretrain: bool = False


def detect_and_validate(path: Path) -> tuple[str, int | None]:
    """Backward-compatible: ``(format, line_count)`` for ``path``.

    For multimodal-aware code use :func:`detect_and_validate_info` instead
    — it returns a :class:`FormatInfo` with the sniffed media kinds.
    """
    info = detect_and_validate_info(path)
    return info.format, info.line_count


def detect_and_validate_info(path: Path) -> FormatInfo:
    """Return :class:`FormatInfo` for ``path``.

    JSONL inputs are sampled to detect ``images`` / ``videos`` fields
    (Phase 9 multimodal). Other formats return only ``format`` + line
    count; multimodality isn't a JSONL-only restriction but our trainers
    only consume multimodal samples from JSONL today.
    """
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("jsonl", "ndjson"):
        line_count, media_kinds, is_preference, is_pretrain = _validate_jsonl(path)
        return FormatInfo(
            "jsonl",
            line_count,
            media_kinds=media_kinds,
            is_preference=is_preference,
            is_pretrain=is_pretrain,
        )
    if suffix == "json":
        return FormatInfo("json", _validate_json(path))
    if suffix == "csv":
        return FormatInfo("csv", _validate_delimited(path, ","))
    if suffix == "tsv":
        return FormatInfo("tsv", _validate_delimited(path, "\t"))
    if suffix == "parquet":
        _validate_parquet(path)
        return FormatInfo("parquet", None)
    raise DatasetFormatError(
        f"unsupported extension '.{suffix}'. supported: jsonl, json, csv, tsv, parquet"
    )


_MEDIA_FIELDS = ("images", "videos", "audios")
_PREFERENCE_FIELDS = ("prompt", "chosen", "rejected")


def _validate_jsonl(path: Path) -> tuple[int, list[str], bool, bool]:
    total = 0
    seen_media: set[str] = set()
    sampled = 0
    preference_hits = 0
    pretrain_hits = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if lineno <= _SAMPLE_LINES:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetFormatError(
                        f"jsonl line {lineno} is not valid JSON: {e}"
                    ) from None
                if isinstance(record, dict):
                    sampled += 1
                    for kind in _MEDIA_FIELDS:
                        v = record.get(kind)
                        if isinstance(v, list) and v:
                            seen_media.add(kind)
                    if all(
                        isinstance(record.get(k), str) and record.get(k)
                        for k in _PREFERENCE_FIELDS
                    ):
                        preference_hits += 1
                    text = record.get("text")
                    if isinstance(text, str) and text:
                        pretrain_hits += 1
            total += 1
    if total == 0:
        raise DatasetFormatError("jsonl file is empty")
    # A shape is claimed only if *every* sampled record matches — a mixed
    # file is almost always a sign of accidental concatenation.
    is_preference = sampled > 0 and preference_hits == sampled
    is_pretrain = sampled > 0 and pretrain_hits == sampled
    return (
        total,
        [k for k in _MEDIA_FIELDS if k in seen_media],
        is_preference,
        is_pretrain,
    )


def _validate_json(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetFormatError(f"json file is not valid JSON: {e}") from None
    if not isinstance(data, list):
        raise DatasetFormatError("json file must contain a top-level list of records")
    if not data:
        raise DatasetFormatError("json file is an empty list")
    return len(data)


def _validate_delimited(path: Path, delim: str) -> int:
    total = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        try:
            header = next(reader)
        except StopIteration:
            raise DatasetFormatError("csv/tsv file has no header row") from None
        if not header:
            raise DatasetFormatError("csv/tsv header row is empty")
        for _ in reader:
            total += 1
    if total == 0:
        raise DatasetFormatError("csv/tsv file has only a header, no data rows")
    return total


def _validate_parquet(path: Path) -> None:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as e:
        raise DatasetFormatError(
            "parquet upload requires pyarrow but it isn't installed"
        ) from e
    try:
        pq.read_metadata(str(path))
    except Exception as e:
        raise DatasetFormatError(f"parquet file is unreadable: {e}") from None
