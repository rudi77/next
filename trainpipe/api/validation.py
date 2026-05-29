"""FastAPI-layer wrappers around training/dataset_paths."""

from fastapi import HTTPException

from ..training.dataset_paths import missing_for_specs
from .schemas import ExperimentSpec


def enforce_dataset_not_empty(specs: list[ExperimentSpec]) -> None:
    """Reject submits with no training dataset.

    ms-swift would otherwise fail with ``self.dataset: []. Please input
    the training dataset.`` after we've already burned a GPU lease and an
    MLflow run. The check lives here (route-level) rather than in the
    Pydantic schema so legacy DB rows with empty datasets can still be
    deserialised by the list endpoints.
    """
    for i, spec in enumerate(specs):
        if not spec.dataset:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "empty_dataset",
                    "spec_index": i,
                    "detail": (
                        "spec.dataset must contain at least one entry "
                        "(HF id, registry shortcut, local path, or ds:<id>)"
                    ),
                },
            )


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
