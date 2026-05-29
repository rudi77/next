"""REST API for active-learning runs (Phase 11).

Endpoints
---------

* ``POST /active-learning/runs`` — start a run synchronously (returns
  201 with the completed run record). Heavy lifting happens in an
  ``asyncio.to_thread``-style background task; the route only stores
  the request and dispatches.
* ``GET  /active-learning/runs`` — list, newest first.
* ``GET  /active-learning/runs/{id}`` — single.
* ``GET  /active-learning/runs/{id}/queue`` — surfaced annotation queue
  (top-N samples sorted by uncertainty descending). ``only_unannotated``
  filters out items the user has already marked done.
* ``POST /active-learning/runs/{id}/queue/{item_id}/annotated`` — mark
  one queue item as annotated (idempotent).
* ``POST /active-learning/runs/{id}/push-labelstudio`` — push the queue
  into a Label Studio project as pre-annotations.

The runner uses the same :class:`InferenceService` as the playground,
so model loads are LRU-cached and shared with ``POST /inferences``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from ...active_learning.runner import (
    make_scorer,
    run_active_learning,
)
from ...core import repository
from ...core.db import Database
from ...inference.service import InferenceService, UnknownModelRef
from ...integrations import labelstudio as ls
from ...training.dataset_refs import (
    MalformedDatasetRef,
    UnknownDatasetRef,
    resolve_single,
)
from ..auth import require_api_key
from ..deps import get_db, get_inference_service
from ..schemas import (
    ActiveLearningRun,
    ActiveLearningRunRequest,
    ALRunStatus,
    AnnotationQueueItem,
    InferenceParams,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/active-learning",
    tags=["active-learning"],
    dependencies=[Depends(require_api_key)],
)


class LSPushRequest(BaseModel):
    """Push the top-N annotation queue into a Label Studio project.

    Each item lands as one task. For the conversation shape we also
    include the model's prediction as a pre-annotation textarea so the
    annotator only has to edit / accept.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., min_length=1)
    token: str = Field(..., min_length=1)
    project_id: int = Field(..., ge=1)


@router.post("/runs", status_code=201)
async def create_run(
    request: ActiveLearningRunRequest,
    db: Annotated[Database, Depends(get_db)],
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> ActiveLearningRun:
    async with db.connect() as conn:
        try:
            dataset_path = await resolve_single(request.dataset, conn)
        except UnknownDatasetRef as e:
            raise HTTPException(
                422,
                {"error": "unknown_dataset_ref", "ref_id": e.ref_id},
            ) from None
        except MalformedDatasetRef as e:
            raise HTTPException(
                422, {"error": "malformed_dataset_ref", "value": e.raw}
            ) from None
        if not Path(dataset_path.split("#", 1)[0]).is_file():
            raise HTTPException(
                422,
                {"error": "dataset_file_missing", "path": dataset_path},
            )

        # Validate the model ref early so we can fail with 422 before
        # marking the run running.
        try:
            resolved_ref = await service.resolve(request.model_ref)
        except UnknownModelRef as e:
            raise HTTPException(
                422,
                {"error": "unknown_model_ref", "ref": e.raw, "reason": e.reason},
            ) from None

        run_id = await repository.create_al_run(
            conn,
            model_ref=request.model_ref,
            dataset_path=dataset_path,
            top_n=request.top_n,
            sample_limit=request.sample_limit,
        )

    # Run synchronously in this request (training pipeline UX: caller
    # waits, no separate dispatcher loop required). For very large
    # datasets the right answer is to background this in a worker task
    # the way the EvalDispatcher does it — left for a follow-up if
    # users start complaining about long POSTs.
    await _execute_run(
        db, service, run_id, request, resolved_ref, dataset_path
    )

    async with db.connect() as conn:
        run = await repository.get_al_run(conn, run_id)
    assert run is not None
    return run


async def _execute_run(
    db: Database,
    service: InferenceService,
    run_id: str,
    request: ActiveLearningRunRequest,
    resolved_ref,
    dataset_path: str,
) -> None:
    async with db.connect() as conn:
        await repository.update_al_run_status(
            conn, run_id, status=ALRunStatus.RUNNING
        )
    try:
        backend = await service.get(resolved_ref)
        scorer = make_scorer(request.scorer)
        # Drop sub-sample suffix at path level (#N) — the runner reads
        # the raw JSONL and applies sample_limit itself.
        ds_file = Path(dataset_path.split("#", 1)[0])
        result = await run_active_learning(
            backend=backend,
            dataset_path=ds_file,
            scorer=scorer,
            top_n=request.top_n,
            inference_params=InferenceParams(),
            sample_limit=request.sample_limit,
        )
    except BaseException as e:
        # Catch ``BaseException`` (NOT ``Exception``) so an
        # ``asyncio.CancelledError`` — fired when the HTTP request is
        # killed mid-scoring — also flips the row to FAILED instead of
        # stranding it in 'running'. We re-raise the cancellation so
        # the surrounding task tree still unwinds.
        logger.exception("active learning run %s failed", run_id)
        try:
            async with db.connect() as conn:
                await repository.update_al_run_status(
                    conn,
                    run_id,
                    status=ALRunStatus.CANCELLED
                    if isinstance(e, asyncio.CancelledError)
                    else ALRunStatus.FAILED,
                    error=str(e) or type(e).__name__,
                )
        except Exception:
            logger.exception(
                "active learning: also failed to persist FAILED state for %s",
                run_id,
            )
        if isinstance(e, asyncio.CancelledError):
            raise
        return

    async with db.connect() as conn:
        for item in result.top_items:
            await repository.add_queue_item(
                conn,
                run_id=run_id,
                sample_index=item.sample_index,
                input=item.sample,
                prediction=item.prediction,
                uncertainty=item.uncertainty,
            )
        await repository.update_al_run_status(
            conn,
            run_id,
            status=ALRunStatus.COMPLETED,
            scored_count=result.scored_count,
            queued_count=result.queued_count,
        )


async def recover_stale_runs(db: Database) -> int:
    """Flip any 'running' AL runs to FAILED on startup.

    Mirrors the experiment crash-recovery in ``Scheduler.start()``. AL
    runs are synchronous-in-request, so on a process restart anything
    still marked running is definitely orphaned and will never resume.
    """
    async with db.connect() as conn:
        cur = await conn.execute(
            "UPDATE active_learning_runs SET status = 'failed', "
            "finished_at = ?, error = 'orphaned by process restart' "
            "WHERE status IN ('queued', 'running')",
            (datetime.now(timezone.utc).isoformat(),),
        )
        await conn.commit()
        return cur.rowcount or 0


@router.get("/runs")
async def list_runs(
    db: Annotated[Database, Depends(get_db)],
    status: ALRunStatus | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> list[ActiveLearningRun]:
    async with db.connect() as conn:
        return await repository.list_al_runs(conn, status=status, limit=limit)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str, db: Annotated[Database, Depends(get_db)]
) -> ActiveLearningRun:
    async with db.connect() as conn:
        run = await repository.get_al_run(conn, run_id)
    if run is None:
        raise HTTPException(404, "active learning run not found")
    return run


@router.get("/runs/{run_id}/queue")
async def get_queue(
    run_id: str,
    db: Annotated[Database, Depends(get_db)],
    only_unannotated: bool = False,
    limit: int = Query(500, ge=1, le=10000),
) -> list[AnnotationQueueItem]:
    async with db.connect() as conn:
        run = await repository.get_al_run(conn, run_id)
        if run is None:
            raise HTTPException(404, "active learning run not found")
        return await repository.list_queue_items(
            conn, run_id, only_unannotated=only_unannotated, limit=limit
        )


@router.post("/runs/{run_id}/queue/{item_id}/annotated")
async def mark_annotated(
    run_id: str,
    item_id: int,
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, bool]:
    async with db.connect() as conn:
        run = await repository.get_al_run(conn, run_id)
        if run is None:
            raise HTTPException(404, "active learning run not found")
        ok = await repository.mark_queue_annotated(conn, item_id)
    return {"updated": ok}


@router.post("/runs/{run_id}/push-labelstudio")
async def push_to_labelstudio(
    run_id: str,
    request: LSPushRequest,
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, int]:
    """Push every queue item as a new LS task with the model prediction
    as a pre-annotation textarea."""
    async with db.connect() as conn:
        run = await repository.get_al_run(conn, run_id)
        if run is None:
            raise HTTPException(404, "active learning run not found")
        items = await repository.list_queue_items(
            conn, run_id, only_unannotated=True, limit=10000
        )
    try:
        canonical = ls._validate_base_url(request.base_url)
    except ls.LabelStudioError as e:
        raise HTTPException(
            422, {"error": "labelstudio_error", "detail": str(e)}
        ) from None

    pushed = await asyncio.to_thread(
        _push_items_blocking, canonical, request.token, request.project_id, items
    )
    return {"pushed": pushed}


def _push_items_blocking(
    base_url: str,
    token: str,
    project_id: int,
    items: list[AnnotationQueueItem],
) -> int:
    """Synchronous LS push. Wrapped in ``to_thread`` to keep the loop free."""
    import httpx

    pushed = 0
    with httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Token {token}"},
        timeout=httpx.Timeout(30.0, connect=5.0),
        follow_redirects=False,
    ) as client:
        for item in items:
            payload = {
                "data": item.input,
                "predictions": [
                    {
                        "result": [
                            {
                                "from_name": "response",
                                "to_name": "prompt",
                                "type": "textarea",
                                "value": {"text": [item.prediction]},
                            }
                        ],
                    }
                ],
            }
            resp = client.post(
                f"/api/projects/{project_id}/import",
                json=[payload],
            )
            if resp.is_error:
                logger.warning(
                    "ls push failed item=%s status=%s",
                    item.id,
                    resp.status_code,
                )
                continue
            pushed += 1
    return pushed
