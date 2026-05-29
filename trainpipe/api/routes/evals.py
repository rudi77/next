"""REST API for the eval framework.

Endpoints
---------

Suites (reusable eval configs):

* ``POST /evals/suites`` — create. ``ds:<id>`` refs are resolved at
  submit time, the dataset file must exist, and every metric is
  instantiated (validates its config) before persisting.
* ``GET /evals/suites`` — list newest first.
* ``GET /evals/suites/{id}`` — single.
* ``DELETE /evals/suites/{id}`` — 409 if any non-terminal eval_run
  references it; ``?force=true`` overrides.

Runs (one execution against one model target):

* ``POST /evals/runs`` — enqueue an eval run for ``suite_id`` against
  ``experiment_id``. The dispatcher picks it up on its next tick.
* ``GET /evals/runs`` — list, optional filters ``suite_id``,
  ``experiment_id``, ``status``.
* ``GET /evals/runs/{id}`` — single.
* ``GET /evals/runs/{id}/results`` — per-sample predictions + scores.
* ``POST /evals/runs/{id}/cancel`` — request cancel.

Compare (Δ between N runs against the same suite):

* ``GET /evals/compare?run_ids=a,b,c`` — n-way comparison: aggregate
  per metric per run, plus the list of samples where at least one run
  scored lower than another (regressions).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from ...core import repository
from ...core.db import Database
from ...evals.metrics import UnknownMetricKind, get_metric_class
from ...training.dataset_refs import (
    MalformedDatasetRef,
    UnknownDatasetRef,
    resolve_single,
)
from ..auth import require_api_key
from ..deps import get_db
from ..schemas import (
    EvalComparison,
    EvalComparisonSample,
    EvalResult,
    EvalRun,
    EvalRunRequest,
    EvalRunStatus,
    EvalSuite,
    EvalSuiteSpec,
)

router = APIRouter(
    prefix="/evals",
    tags=["evals"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/suites", status_code=201)
async def create_suite(
    spec: EvalSuiteSpec,
    db: Annotated[Database, Depends(get_db)],
) -> EvalSuite:
    async with db.connect() as conn:
        existing = await repository.get_eval_suite_by_name(conn, spec.name)
        if existing is not None:
            raise HTTPException(
                409, {"error": "name_exists", "name": spec.name},
            )

        try:
            dataset_path = await resolve_single(spec.dataset, conn)
        except UnknownDatasetRef as e:
            raise HTTPException(
                422, {"error": "unknown_dataset_ref", "ref_id": e.ref_id},
            ) from None
        except MalformedDatasetRef as e:
            raise HTTPException(
                422, {"error": "malformed_dataset_ref", "value": e.raw},
            ) from None

        for cfg in spec.metrics:
            try:
                metric_cls = get_metric_class(cfg.kind)
            except UnknownMetricKind:
                raise HTTPException(
                    422, {"error": "unknown_metric_kind", "kind": cfg.kind},
                ) from None
            try:
                metric_cls(cfg.config)
            except ValueError as e:
                raise HTTPException(
                    422,
                    {
                        "error": "metric_config_invalid",
                        "metric": cfg.metric_name,
                        "detail": str(e),
                    },
                ) from None

        suite_id = await repository.create_eval_suite(
            conn,
            name=spec.name,
            description=spec.description,
            dataset_path=dataset_path,
            metrics=spec.metrics,
            inference_params=spec.inference_params,
        )
        suite = await repository.get_eval_suite(conn, suite_id)
    assert suite is not None
    return suite


@router.get("/suites")
async def list_suites(
    db: Annotated[Database, Depends(get_db)],
) -> list[EvalSuite]:
    async with db.connect() as conn:
        return await repository.list_eval_suites(conn)


@router.get("/suites/{suite_id}")
async def get_suite(
    suite_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> EvalSuite:
    async with db.connect() as conn:
        suite = await repository.get_eval_suite(conn, suite_id)
    if suite is None:
        raise HTTPException(404, "eval suite not found")
    return suite


@router.delete("/suites/{suite_id}")
async def delete_suite(
    suite_id: str,
    db: Annotated[Database, Depends(get_db)],
    force: bool = False,
) -> dict[str, str]:
    async with db.connect() as conn:
        suite = await repository.get_eval_suite(conn, suite_id)
        if suite is None:
            raise HTTPException(404, "eval suite not found")
        if not force:
            active = await repository.active_eval_runs_for_suite(conn, suite_id)
            if active:
                raise HTTPException(
                    409,
                    {
                        "error": "suite_in_use",
                        "active_runs": active,
                        "hint": "pass ?force=true to delete anyway",
                    },
                )
        deleted = await repository.delete_eval_suite(conn, suite_id)
    return {"status": "deleted" if deleted else "not_found"}


@router.post("/runs", status_code=201)
async def create_run(
    request: EvalRunRequest,
    db: Annotated[Database, Depends(get_db)],
) -> EvalRun:
    async with db.connect() as conn:
        suite = await repository.get_eval_suite(conn, request.suite_id)
        if suite is None:
            raise HTTPException(
                422, {"error": "unknown_suite", "suite_id": request.suite_id},
            )
        exp = await repository.get_experiment(conn, request.experiment_id)
        if exp is None:
            raise HTTPException(
                422,
                {
                    "error": "unknown_experiment",
                    "experiment_id": request.experiment_id,
                },
            )
        model_ref = exp.spec.name or request.experiment_id
        run_id = await repository.create_eval_run(
            conn,
            suite_id=request.suite_id,
            experiment_id=request.experiment_id,
            model_ref=model_ref,
            triggered_by=request.triggered_by,
        )
        run = await repository.get_eval_run(conn, run_id)
    assert run is not None
    return run


@router.get("/runs")
async def list_runs(
    db: Annotated[Database, Depends(get_db)],
    suite_id: str | None = None,
    experiment_id: str | None = None,
    status: EvalRunStatus | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[EvalRun]:
    async with db.connect() as conn:
        return await repository.list_eval_runs(
            conn,
            suite_id=suite_id,
            experiment_id=experiment_id,
            status=status,
            limit=limit,
            offset=offset,
        )


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> EvalRun:
    async with db.connect() as conn:
        run = await repository.get_eval_run(conn, run_id)
    if run is None:
        raise HTTPException(404, "eval run not found")
    return run


@router.get("/runs/{run_id}/results")
async def list_results(
    run_id: str,
    db: Annotated[Database, Depends(get_db)],
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> list[EvalResult]:
    async with db.connect() as conn:
        run = await repository.get_eval_run(conn, run_id)
        if run is None:
            raise HTTPException(404, "eval run not found")
        return await repository.list_eval_results(
            conn, run_id, limit=limit, offset=offset,
        )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, str]:
    async with db.connect() as conn:
        result = await repository.request_cancel_eval_run(conn, run_id)
    if result == "not_found":
        raise HTTPException(404, "eval run not found")
    return {"status": result}


@router.get("/compare", response_model=EvalComparison)
async def compare_runs(
    db: Annotated[Database, Depends(get_db)],
    run_ids: str = Query(
        ..., description="Comma-separated eval run IDs (2+ runs against the same suite)"
    ),
) -> EvalComparison:
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if len(ids) < 2:
        raise HTTPException(422, "compare requires at least two run_ids")

    async with db.connect() as conn:
        runs: list[EvalRun] = []
        for rid in ids:
            r = await repository.get_eval_run(conn, rid)
            if r is None:
                raise HTTPException(
                    404, {"error": "unknown_run", "run_id": rid},
                )
            runs.append(r)

        suite_ids = {r.suite_id for r in runs}
        if len(suite_ids) > 1:
            raise HTTPException(
                422,
                {
                    "error": "suite_mismatch",
                    "detail": "all runs must target the same eval suite",
                    "suite_ids": sorted(suite_ids),
                },
            )

        aggregate_delta: dict[str, dict[str, float]] = {}
        for r in runs:
            if r.aggregate:
                for name, agg in r.aggregate.items():
                    aggregate_delta.setdefault(name, {})[r.id] = agg.mean

        per_run_results: dict[str, dict[int, dict]] = {}
        for r in runs:
            rows = await repository.list_eval_results(conn, r.id, limit=5000)
            per_run_results[r.id] = {
                row.sample_index: {
                    "prediction": row.prediction,
                    "scores": row.scores,
                    "error": row.error,
                    "input": row.input,
                    "gold": row.gold,
                }
                for row in rows
            }

    regressions: list[EvalComparisonSample] = []
    all_sample_indices: set[int] = set()
    for d in per_run_results.values():
        all_sample_indices.update(d.keys())

    for idx in sorted(all_sample_indices):
        per_run_entry: dict[str, dict] = {}
        score_vectors: list[tuple[str, dict[str, float]]] = []
        sample_inputs: dict = {}
        sample_gold = None
        for r in runs:
            row = per_run_results[r.id].get(idx)
            if row is None:
                continue
            per_run_entry[r.id] = {
                "prediction": row["prediction"],
                "scores": row["scores"],
                "error": row["error"],
            }
            score_vectors.append((r.id, row["scores"]))
            if not sample_inputs:
                sample_inputs = row["input"]
                sample_gold = row["gold"]

        if len(score_vectors) < 2:
            continue
        metric_names = set().union(*(set(s) for _, s in score_vectors))
        is_regression = False
        for name in metric_names:
            vals = [s.get(name) for _, s in score_vectors if name in s]
            if len(vals) >= 2 and max(vals) - min(vals) > 1e-9:
                is_regression = True
                break
        if is_regression:
            regressions.append(
                EvalComparisonSample(
                    sample_index=idx,
                    input=sample_inputs,
                    gold=sample_gold,
                    per_run=per_run_entry,
                )
            )

    return EvalComparison(
        suite_id=next(iter(suite_ids)),
        runs=runs,
        aggregate_delta=aggregate_delta,
        regressions=regressions,
    )
