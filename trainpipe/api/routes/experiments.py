import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from ...core import repository
from ...core.db import Database
from ..auth import require_api_key
from ..deps import get_db, get_scheduler
from ..schemas import ExperimentRecord, ExperimentSpec, ExperimentStatus

router = APIRouter(
    prefix="/experiments",
    tags=["experiments"],
    dependencies=[Depends(require_api_key)],
)


_TERMINAL = {
    ExperimentStatus.COMPLETED,
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
}


@router.post("", status_code=201)
async def submit(
    spec: ExperimentSpec,
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, str]:
    async with db.connect() as conn:
        experiment_id = await repository.create_experiment(conn, spec)
    return {"experiment_id": experiment_id}


@router.post("/batch", status_code=201)
async def submit_batch(
    specs: list[ExperimentSpec],
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, list[str]]:
    if not specs:
        raise HTTPException(422, "specs must contain at least one item")
    ids: list[str] = []
    async with db.connect() as conn:
        for s in specs:
            ids.append(await repository.create_experiment(conn, s))
    return {"experiment_ids": ids}


@router.get("")
async def list_experiments(
    db: Annotated[Database, Depends(get_db)],
    status: ExperimentStatus | None = None,
    study_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[ExperimentRecord]:
    async with db.connect() as conn:
        return await repository.list_experiments(
            conn,
            status=status,
            study_id=study_id,
            limit=limit,
            offset=offset,
        )


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> ExperimentRecord:
    async with db.connect() as conn:
        rec = await repository.get_experiment(conn, experiment_id)
    if rec is None:
        raise HTTPException(404, "experiment not found")
    return rec


@router.post("/{experiment_id}/cancel")
async def cancel_experiment(
    experiment_id: str,
    db: Annotated[Database, Depends(get_db)],
    scheduler: Annotated[object, Depends(get_scheduler)],
) -> dict[str, str]:
    async with db.connect() as conn:
        result = await repository.request_cancel(conn, experiment_id)
    if result == "not_found":
        raise HTTPException(404, "experiment not found")
    if result == "running":
        # Scheduler has the live RunningProcess; signal it.
        await scheduler.cancel_experiment(experiment_id)
        return {"status": "cancelling"}
    return {"status": result}


@router.get("/{experiment_id}/logs")
async def download_logs(
    experiment_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> PlainTextResponse:
    async with db.connect() as conn:
        rec = await repository.get_experiment(conn, experiment_id)
    if rec is None:
        raise HTTPException(404, "experiment not found")
    if not rec.log_path:
        return PlainTextResponse("")
    path = Path(rec.log_path)
    if not path.exists():
        return PlainTextResponse("")
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


@router.get("/{experiment_id}/logs/stream")
async def stream_logs(
    experiment_id: str,
    request: Request,
    db: Annotated[Database, Depends(get_db)],
) -> EventSourceResponse:
    async with db.connect() as conn:
        rec = await repository.get_experiment(conn, experiment_id)
    if rec is None:
        raise HTTPException(404, "experiment not found")

    async def event_source():
        last_size = 0
        while True:
            if await request.is_disconnected():
                return
            async with db.connect() as conn:
                current = await repository.get_experiment(conn, experiment_id)
            if current is None:
                yield {"event": "end", "data": "not_found"}
                return
            if current.log_path:
                path = Path(current.log_path)
                if path.exists():
                    size = path.stat().st_size
                    if size > last_size:
                        with path.open("rb") as f:
                            f.seek(last_size)
                            chunk = f.read()
                        last_size = size
                        if chunk:
                            yield {
                                "event": "log",
                                "data": chunk.decode("utf-8", errors="replace"),
                            }
            if current.status in _TERMINAL:
                yield {"event": "end", "data": current.status.value}
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_source())
