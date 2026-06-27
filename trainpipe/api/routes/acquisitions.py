"""REST API for agentic data acquisition (Phase 22).

Builds a training dataset from a natural-language brief via a long-running,
phased background job (see docs/spec/agentic-data-acquisition.md). Unlike
``/active-learning/runs`` (synchronous in-request), ``POST /acquisitions``
returns immediately with a ``queued`` run — the work proceeds in an
:class:`AcquisitionDriver` task and the client polls ``GET`` for progress.

Clarifying questions support both paths from the design doc: the interactive
(MCP) path supplies a pre-filled ``spec`` on POST so the run starts already
clarified; the async path lets the run park in ``awaiting_input`` and the
operator answers via ``PATCH …/answers``.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from ...acquisition.manager import AcquisitionManager
from ...core import repository
from ...core.db import Database
from ..auth import require_api_key
from ..deps import get_acquisition_manager, get_db
from ..schemas import (
    AcquisitionAnswers,
    AcquisitionRequest,
    AcquisitionRun,
    AcquisitionSource,
    AcquisitionStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/acquisitions",
    tags=["acquisitions"],
    dependencies=[Depends(require_api_key)],
)


@router.post("", status_code=201)
async def create_acquisition(
    request: AcquisitionRequest,
    manager: Annotated[AcquisitionManager, Depends(get_acquisition_manager)],
) -> AcquisitionRun:
    # A pre-filled spec (interactive/MCP path) wins: target_count comes from
    # the spec so the two can't disagree. Otherwise the request's count seeds
    # the intake phase.
    target_count = request.spec.target_count if request.spec else request.target_count
    return await manager.create_and_start(
        name=request.name,
        brief=request.brief,
        provider=request.provider,
        model=request.model,
        target_count=target_count,
        spec=request.spec,
    )


@router.get("")
async def list_acquisitions(
    db: Annotated[Database, Depends(get_db)],
    status: AcquisitionStatus | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> list[AcquisitionRun]:
    async with db.connect() as conn:
        return await repository.list_acquisition_runs(
            conn, status=status, limit=limit
        )


@router.get("/{run_id}")
async def get_acquisition(
    run_id: str, db: Annotated[Database, Depends(get_db)]
) -> AcquisitionRun:
    async with db.connect() as conn:
        run = await repository.get_acquisition_run(conn, run_id)
    if run is None:
        raise HTTPException(404, "acquisition run not found")
    return run


@router.get("/{run_id}/sources")
async def get_acquisition_sources(
    run_id: str, db: Annotated[Database, Depends(get_db)]
) -> list[AcquisitionSource]:
    async with db.connect() as conn:
        run = await repository.get_acquisition_run(conn, run_id)
        if run is None:
            raise HTTPException(404, "acquisition run not found")
        return await repository.list_acquisition_sources(conn, run_id)


@router.patch("/{run_id}/answers")
async def answer_acquisition(
    run_id: str,
    request: AcquisitionAnswers,
    db: Annotated[Database, Depends(get_db)],
    manager: Annotated[AcquisitionManager, Depends(get_acquisition_manager)],
) -> AcquisitionRun:
    run = await manager.answer(run_id, request.answers)
    if run is not None:
        return run
    # Answer was rejected — distinguish "no such run" (404) from "exists but
    # not parked" (409). This re-read only runs on the error path.
    async with db.connect() as conn:
        existing = await repository.get_acquisition_run(conn, run_id)
    if existing is None:
        raise HTTPException(404, "acquisition run not found")
    raise HTTPException(
        409,
        {
            "error": "not_awaiting_input",
            "status": existing.status.value,
            "detail": "run is not parked waiting for answers",
        },
    )


@router.post("/{run_id}/cancel")
async def cancel_acquisition(
    run_id: str,
    manager: Annotated[AcquisitionManager, Depends(get_acquisition_manager)],
) -> AcquisitionRun:
    # cancel() returns the run after the attempt (unchanged if it was already
    # terminal), or None if there's no such run.
    run = await manager.cancel(run_id)
    if run is None:
        raise HTTPException(404, "acquisition run not found")
    return run
