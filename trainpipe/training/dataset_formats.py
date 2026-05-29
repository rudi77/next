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
from pathlib import Path

_SAMPLE_LINES = 100


class DatasetFormatError(ValueError):
    """Raised when the uploaded file doesn't parse as its declared format."""


def detect_and_validate(path: Path) -> tuple[str, int | None]:
    """Return ``(format, line_count)`` for the file at ``path``.

    ``line_count`` is None for binary/columnar formats where lines are not
    meaningful (parquet).
    """
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("jsonl", "ndjson"):
        return "jsonl", _validate_jsonl(path)
    if suffix == "json":
        return "json", _validate_json(path)
    if suffix == "csv":
        return "csv", _validate_delimited(path, ",")
    if suffix == "tsv":
        return "tsv", _validate_delimited(path, "\t")
    if suffix == "parquet":
        _validate_parquet(path)
        return "parquet", None
    raise DatasetFormatError(
        f"unsupported extension '.{suffix}'. supported: jsonl, json, csv, tsv, parquet"
    )


def _validate_jsonl(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if lineno <= _SAMPLE_LINES:
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetFormatError(
                        f"jsonl line {lineno} is not valid JSON: {e}"
                    ) from None
            total += 1
    if total == 0:
        raise DatasetFormatError("jsonl file is empty")
    return total


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
