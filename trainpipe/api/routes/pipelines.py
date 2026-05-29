"""REST API for multi-stage training pipelines (Phase 12)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ...core import repository
from ...core.db import Database
from ...pipelines.manager import PipelineManager
from ..auth import require_api_key
from ..deps import get_db, get_pipeline_manager
from ..schemas import Pipeline, PipelineConfig

router = APIRouter(
    prefix="/pipelines",
    tags=["pipelines"],
    dependencies=[Depends(require_api_key)],
)


@router.post("", status_code=201)
async def create_pipeline(
    config: PipelineConfig,
    db: Annotated[Database, Depends(get_db)],
    manager: Annotated[PipelineManager, Depends(get_pipeline_manager)],
) -> Pipeline:
    try:
        pipeline_id = await manager.create_and_start(config.name, config)
    except ValueError as e:
        raise HTTPException(
            422, {"error": "invalid_pipeline", "detail": str(e)}
        ) from None
    async with db.connect() as conn:
        pipeline = await repository.get_pipeline(conn, pipeline_id)
    assert pipeline is not None
    return pipeline


@router.get("")
async def list_pipelines(
    db: Annotated[Database, Depends(get_db)],
) -> list[Pipeline]:
    async with db.connect() as conn:
        return await repository.list_pipelines(conn)


@router.get("/{pipeline_id}")
async def get_pipeline(
    pipeline_id: str, db: Annotated[Database, Depends(get_db)]
) -> Pipeline:
    async with db.connect() as conn:
        pipeline = await repository.get_pipeline(conn, pipeline_id)
    if pipeline is None:
        raise HTTPException(404, "pipeline not found")
    return pipeline


@router.post("/{pipeline_id}/cancel")
async def cancel_pipeline(
    pipeline_id: str,
    manager: Annotated[PipelineManager, Depends(get_pipeline_manager)],
) -> dict[str, str]:
    ok = await manager.cancel(pipeline_id)
    if not ok:
        return {"status": "noop"}
    return {"status": "cancelled"}
