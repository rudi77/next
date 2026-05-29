from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ...autoresearch.manager import StudyManager
from ...core import repository
from ...core.db import Database
from ...training.dataset_refs import (
    MalformedDatasetRef,
    UnknownDatasetRef,
    resolve_spec,
)
from ..auth import require_api_key
from ..deps import get_db, get_study_manager
from ..schemas import StudyConfig, StudyRecord
from ..validation import enforce_dataset_not_empty, enforce_dataset_paths_exist

router = APIRouter(
    prefix="/studies",
    tags=["studies"],
    dependencies=[Depends(require_api_key)],
)


@router.post("", status_code=201)
async def create_study(
    config: StudyConfig,
    db: Annotated[Database, Depends(get_db)],
    manager: Annotated[StudyManager, Depends(get_study_manager)],
) -> dict[str, str]:
    enforce_dataset_not_empty([config.base_spec])
    # Resolve ds:<id> refs in the base spec before kicking off trials.
    async with db.connect() as conn:
        try:
            resolved_base = await resolve_spec(config.base_spec, conn)
        except UnknownDatasetRef as e:
            raise HTTPException(
                422,
                {"error": "unknown_dataset_ref", "ref_id": e.ref_id},
            ) from None
        except MalformedDatasetRef as e:
            raise HTTPException(
                422,
                {"error": "malformed_dataset_ref", "value": e.raw},
            ) from None
    config = config.model_copy(update={"base_spec": resolved_base})
    enforce_dataset_paths_exist([config.base_spec])
    study_id = await manager.create_and_start(config)
    return {"study_id": study_id}


@router.get("")
async def list_studies(
    db: Annotated[Database, Depends(get_db)],
) -> list[StudyRecord]:
    async with db.connect() as conn:
        return await repository.list_studies(conn)


@router.get("/{study_id}")
async def get_study(
    study_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> StudyRecord:
    async with db.connect() as conn:
        rec = await repository.get_study(conn, study_id)
    if rec is None:
        raise HTTPException(404, "study not found")
    return rec


@router.post("/{study_id}/cancel")
async def cancel_study(
    study_id: str,
    db: Annotated[Database, Depends(get_db)],
    manager: Annotated[StudyManager, Depends(get_study_manager)],
) -> dict[str, str]:
    async with db.connect() as conn:
        rec = await repository.get_study(conn, study_id)
    if rec is None:
        raise HTTPException(404, "study not found")
    cancelled = await manager.cancel(study_id)
    return {"status": "cancelled" if cancelled else "not_active"}
