"""Resolve ``ds:<id>`` dataset references in an ExperimentSpec.

Resolution happens at submit time (not at dispatch). The stored spec_json
carries plain filesystem paths, so the scheduler and the swift_builder
don't need DB access. Sub-sample suffixes (``#500``) are preserved.
"""

import re
from typing import Iterable

import aiosqlite

from ..api.schemas import ExperimentSpec
from ..core import repository

_REF = re.compile(r"^ds:([0-9a-fA-F]+)(#.*)?$")


class UnknownDatasetRef(ValueError):
    def __init__(self, ref_id: str) -> None:
        super().__init__(f"unknown dataset: ds:{ref_id}")
        self.ref_id = ref_id


def parse_ref(s: str) -> tuple[str, str] | None:
    """Return ``(dataset_id, suffix)`` or None if ``s`` is not a ref."""
    m = _REF.match(s)
    if not m:
        return None
    return m.group(1), m.group(2) or ""


def is_ref(s: str) -> bool:
    return parse_ref(s) is not None


async def _resolve_list(
    items: Iterable[str], conn: aiosqlite.Connection
) -> list[str]:
    out: list[str] = []
    for raw in items:
        parsed = parse_ref(raw)
        if parsed is None:
            out.append(raw)
            continue
        ds_id, suffix = parsed
        rec = await repository.get_dataset(conn, ds_id)
        if rec is None:
            raise UnknownDatasetRef(ds_id)
        out.append(f"{rec.path}{suffix}")
    return out


async def resolve_spec(
    spec: ExperimentSpec, conn: aiosqlite.Connection
) -> ExperimentSpec:
    """Return a new spec with ``ds:<id>`` entries replaced by their paths."""
    dataset = await _resolve_list(spec.dataset, conn)
    val_dataset = await _resolve_list(spec.val_dataset, conn)
    if dataset == spec.dataset and val_dataset == spec.val_dataset:
        return spec
    return spec.model_copy(update={"dataset": dataset, "val_dataset": val_dataset})
