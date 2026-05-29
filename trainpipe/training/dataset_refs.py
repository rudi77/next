"""Resolve ``ds:<id>`` dataset references in an ExperimentSpec.

Resolution happens at submit time (not at dispatch). The stored spec_json
carries plain filesystem paths, so the scheduler and the swift_builder
don't need DB access. Sub-sample suffixes (``#500``) are preserved.
"""

import re
from collections.abc import Iterable

import aiosqlite

from ..api.schemas import ExperimentSpec
from ..core import repository

_REF = re.compile(r"^ds:([0-9a-fA-F]+)(@v\d+)?(#.*)?$")


class UnknownDatasetRef(ValueError):
    def __init__(self, ref_id: str) -> None:
        super().__init__(f"unknown dataset: ds:{ref_id}")
        self.ref_id = ref_id


class MalformedDatasetRef(ValueError):
    def __init__(self, raw: str) -> None:
        super().__init__(f"malformed dataset reference: {raw!r}")
        self.raw = raw


def parse_ref(s: str) -> tuple[str, str] | None:
    """Return ``(dataset_id, suffix)`` or None if ``s`` is not a ref.

    The ``@vN`` segment is stripped here and validated against the actual
    persisted ``version`` in ``resolve_single`` — keeping the regex match
    permissive lets the resolver give a clearer error than "malformed".
    The suffix returned here always starts with ``#`` (or is empty).
    """
    m = _REF.match(s)
    if not m:
        return None
    return m.group(1), m.group(3) or ""


def parse_ref_with_version(s: str) -> tuple[str, int | None, str] | None:
    """Return ``(dataset_id, version_or_None, suffix)`` for ``ds:<id>@vN#K``."""
    m = _REF.match(s)
    if not m:
        return None
    version: int | None = None
    if m.group(2):
        version = int(m.group(2)[2:])
    return m.group(1), version, m.group(3) or ""


def is_ref(s: str) -> bool:
    return parse_ref(s) is not None


async def _resolve_list(
    items: Iterable[str], conn: aiosqlite.Connection
) -> list[str]:
    out: list[str] = []
    for raw in items:
        out.append(await resolve_single(raw, conn))
    return out


async def resolve_single(raw: str, conn: aiosqlite.Connection) -> str:
    """Resolve a single ``ds:<id>(@vN)?(#suffix)?`` ref to its filesystem path.

    Non-ref strings (anything that doesn't start with ``ds:``) are passed
    through unchanged. ``ds:``-prefixed strings that don't match the ref
    grammar raise :class:`MalformedDatasetRef`. If a ``@vN`` segment is
    present, it must match the dataset's persisted ``version``.
    """
    parsed = parse_ref_with_version(raw)
    if parsed is None:
        if raw.startswith("ds:"):
            raise MalformedDatasetRef(raw)
        return raw
    ds_id, requested_version, suffix = parsed
    rec = await repository.get_dataset(conn, ds_id)
    if rec is None:
        raise UnknownDatasetRef(ds_id)
    if requested_version is not None and rec.version != requested_version:
        # Wrong version — surface as malformed with a clear message rather
        # than silently using a different version's file. Eventually this
        # should be a separate VersionMismatch error class.
        raise MalformedDatasetRef(
            f"{raw} (registered version is v{rec.version})"
        )
    return f"{rec.path}{suffix}"


async def resolve_spec(
    spec: ExperimentSpec, conn: aiosqlite.Connection
) -> ExperimentSpec:
    """Return a new spec with ``ds:<id>`` entries replaced by their paths."""
    dataset = await _resolve_list(spec.dataset, conn)
    val_dataset = await _resolve_list(spec.val_dataset, conn)
    if dataset == spec.dataset and val_dataset == spec.val_dataset:
        return spec
    return spec.model_copy(update={"dataset": dataset, "val_dataset": val_dataset})
