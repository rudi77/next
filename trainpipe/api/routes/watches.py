"""REST API for continuous-training watches (Phase 17)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ...core import repository
from ...core.db import Database
from ..auth import require_api_key
from ..deps import get_db
from ..schemas import Watch, WatchCreateRequest

router = APIRouter(
    prefix="/watches",
    tags=["watches"],
    dependencies=[Depends(require_api_key)],
)


def _validate(req: WatchCreateRequest) -> None:
    if req.kind == "interval":
        if req.interval_seconds is None:
            raise HTTPException(
                422,
                {
                    "error": "missing_field",
                    "detail": "kind=interval requires interval_seconds",
                },
            )
    if req.kind == "metric_threshold":
        missing = [
            f
            for f, v in (
                ("suite_id", req.suite_id),
                ("metric_name", req.metric_name),
                ("threshold", req.threshold),
            )
            if v is None
        ]
        if missing:
            raise HTTPException(
                422,
                {
                    "error": "missing_fields",
                    "fields": missing,
                    "detail": "metric_threshold needs suite_id, metric_name, threshold",
                },
            )


@router.post("", status_code=201)
async def create_watch(
    request: WatchCreateRequest,
    db: Annotated[Database, Depends(get_db)],
) -> Watch:
    _validate(request)
    async with db.connect() as conn:
        watch_id = await repository.create_watch(
            conn,
            name=request.name,
            kind=request.kind,
            pipeline_config=request.pipeline_config,
            interval_seconds=request.interval_seconds,
            model_name=request.model_name,
            suite_id=request.suite_id,
            metric_name=request.metric_name,
            threshold=request.threshold,
        )
        watch = await repository.get_watch(conn, watch_id)
    assert watch is not None
    return watch


@router.get("")
async def list_watches(
    db: Annotated[Database, Depends(get_db)],
    enabled: bool | None = None,
) -> list[Watch]:
    async with db.connect() as conn:
        return await repository.list_watches(
            conn, only_enabled=bool(enabled) if enabled is not None else False
        )


@router.get("/{watch_id}")
async def get_watch(
    watch_id: str, db: Annotated[Database, Depends(get_db)]
) -> Watch:
    async with db.connect() as conn:
        w = await repository.get_watch(conn, watch_id)
    if w is None:
        raise HTTPException(404, "watch not found")
    return w


@router.post("/{watch_id}/enable")
async def enable_watch(
    watch_id: str, db: Annotated[Database, Depends(get_db)]
) -> dict[str, bool]:
    async with db.connect() as conn:
        ok = await repository.set_watch_enabled(conn, watch_id, True)
    if not ok:
        raise HTTPException(404, "watch not found")
    return {"enabled": True}


@router.post("/{watch_id}/disable")
async def disable_watch(
    watch_id: str, db: Annotated[Database, Depends(get_db)]
) -> dict[str, bool]:
    async with db.connect() as conn:
        ok = await repository.set_watch_enabled(conn, watch_id, False)
    if not ok:
        raise HTTPException(404, "watch not found")
    return {"enabled": False}


@router.delete("/{watch_id}", status_code=200)
async def delete_watch(
    watch_id: str, db: Annotated[Database, Depends(get_db)]
) -> dict[str, bool]:
    async with db.connect() as conn:
        ok = await repository.delete_watch(conn, watch_id)
    return {"deleted": ok}
