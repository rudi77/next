"""FastAPI-layer wrappers around training/dataset_paths."""

from fastapi import HTTPException

from ..training.dataset_paths import missing_for_specs
from .schemas import ExperimentSpec


def enforce_dataset_paths_exist(specs: list[ExperimentSpec]) -> None:
    """Raise 422 if any locally-looking dataset path in any spec is missing.

    The error detail lists every failure (with spec_index for batches) so
    an agent can fix all of them in one round trip.
    """
    bad = missing_for_specs(specs)
    if not bad:
        return
    raise HTTPException(
        status_code=422,
        detail={
            "error": "missing_local_paths",
            "missing": [
                {"spec_index": m.spec_index, "field": m.field, "path": m.path}
                for m in bad
            ],
        },
    )
