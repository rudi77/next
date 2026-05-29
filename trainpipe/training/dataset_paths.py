"""Heuristic + existence check for ExperimentSpec dataset entries.

ms-swift accepts several kinds of dataset reference in --dataset:
  - HuggingFace repo IDs ("org/name")
  - ms-swift registry shortcuts ("AI-ModelScope/...")
  - Local files ("/abs/path.jsonl", "./train.jsonl", "C:/data/train.jsonl")
  - Local directories
  - Any of the above with a "#N" subsample suffix

Only local paths are validated here; remote refs are accepted blindly
(they fail at trainer load-time if wrong). The heuristic is permissive
on the local side — a false positive ("this looks local but isn't") at
worst rejects a valid remote name, which the caller can fix by passing
a more obviously remote-looking string.
"""

from dataclasses import dataclass
from pathlib import Path

from ..api.schemas import ExperimentSpec

_DATA_SUFFIXES = {
    ".jsonl",
    ".json",
    ".csv",
    ".tsv",
    ".parquet",
    ".txt",
    ".arrow",
}


def _strip_subsample(s: str) -> str:
    return s.split("#", 1)[0]


def looks_like_local_path(raw: str) -> bool:
    core = _strip_subsample(raw).strip()
    if not core:
        return False
    if core.startswith(("/", "./", "../", "~/", "~\\")):
        return True
    # Windows-style absolute (C:\... or C:/...)
    if len(core) >= 3 and core[1] == ":" and core[2] in ("\\", "/"):
        return True
    if Path(core).suffix.lower() in _DATA_SUFFIXES:
        return True
    return False


@dataclass(frozen=True)
class MissingPath:
    spec_index: int
    field: str
    path: str


def _missing(
    paths: list[str], *, field: str, spec_index: int
) -> list[MissingPath]:
    out: list[MissingPath] = []
    for raw in paths:
        if not looks_like_local_path(raw):
            continue
        core = _strip_subsample(raw)
        if not Path(core).expanduser().exists():
            out.append(MissingPath(spec_index=spec_index, field=field, path=raw))
    return out


def missing_for_specs(specs: list[ExperimentSpec]) -> list[MissingPath]:
    """Return all locally-looking dataset/val_dataset paths that don't exist."""
    out: list[MissingPath] = []
    for i, spec in enumerate(specs):
        out.extend(_missing(spec.dataset, field="dataset", spec_index=i))
        out.extend(_missing(spec.val_dataset, field="val_dataset", spec_index=i))
    return out
