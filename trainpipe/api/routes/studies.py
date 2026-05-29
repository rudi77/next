"""Read-only study routes for Phase 3. Phase 4 adds POST + driver wiring."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ...core import repository
from ...core.db import Database
from ..auth import require_api_key
from ..deps import get_db
from ..schemas import StudyRecord

router = APIRouter(
    prefix="/studies",
    tags=["studies"],
    dependencies=[Depends(require_api_key)],
)


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
