"""CRUD helpers for experiments, studies, and events.

All functions take an aiosqlite.Connection so the caller controls transaction
scope. Repository functions commit only when they perform a single self-contained
mutation; multi-step orchestrations (scheduler dispatch) should commit themselves.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from ..api.schemas import (
    ACQUISITION_TERMINAL_STATUSES,
    AcquisitionRun,
    AcquisitionSource,
    AcquisitionSpec,
    AcquisitionStatus,
    ActiveLearningRun,
    ALRunStatus,
    AnnotationQueueItem,
    Dataset,
    EvalResult,
    EvalRun,
    EvalRunStatus,
    EvalSuite,
    ExperimentRecord,
    ExperimentSpec,
    ExperimentStatus,
    InferenceParams,
    MetricAggregate,
    MetricConfig,
    Pipeline,
    PipelineConfig,
    PipelineStage,
    PipelineStatus,
    RegisteredModel,
    StageStatus,
    StudyConfig,
    StudyRecord,
    StudyStatus,
    Watch,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


def _row_to_experiment_record(row: aiosqlite.Row) -> ExperimentRecord:
    def _maybe_get(name: str):
        # Older schema versions don't have these columns; tolerate that
        # in repository reads so test fixtures with pre-v11 DBs still work.
        try:
            return row[name]
        except (KeyError, IndexError):
            return None

    return ExperimentRecord(
        id=row["id"],
        spec=ExperimentSpec.model_validate_json(row["spec_json"]),
        status=ExperimentStatus(row["status"]),
        priority=row["priority"],
        study_id=row["study_id"],
        trial_number=row["trial_number"],
        gpu_ids=json.loads(row["gpu_ids"]) if row["gpu_ids"] else None,
        mlflow_run_id=row["mlflow_run_id"],
        mlflow_experiment_id=row["mlflow_experiment_id"],
        log_path=row["log_path"],
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        queued_at=datetime.fromisoformat(row["queued_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        last_heartbeat_at=(
            datetime.fromisoformat(row["last_heartbeat_at"])
            if row["last_heartbeat_at"]
            else None
        ),
        gpu_seconds=_maybe_get("gpu_seconds"),
        peak_vram_mb=_maybe_get("peak_vram_mb"),
        energy_wh=_maybe_get("energy_wh"),
    )


async def study_cost_summary(
    conn: aiosqlite.Connection, study_id: str
) -> dict[str, Any]:
    """Aggregate GPU/VRAM/energy for one study's experiments.

    Returns sums + counts so the UI's "Cost vs Best-Metric" plot can
    render a single point per study without round-tripping every trial.
    Only ``completed`` experiments contribute — in-progress ones don't
    have a final ``gpu_seconds`` and a failed run's partial cost is
    excluded by design (don't penalize learnings).
    """
    cur = await conn.execute(
        "SELECT "
        "COUNT(*) AS n_trials, "
        "COALESCE(SUM(gpu_seconds), 0.0) AS total_gpu_seconds, "
        "MAX(peak_vram_mb) AS peak_vram_mb, "
        "COALESCE(SUM(energy_wh), 0.0) AS total_energy_wh "
        "FROM experiments WHERE study_id = ? AND status = 'completed'",
        (study_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return {
            "n_trials": 0,
            "total_gpu_seconds": 0.0,
            "peak_vram_mb": None,
            "total_energy_wh": 0.0,
        }
    return {
        "n_trials": int(row["n_trials"] or 0),
        "total_gpu_seconds": float(row["total_gpu_seconds"] or 0.0),
        "peak_vram_mb": (
            float(row["peak_vram_mb"]) if row["peak_vram_mb"] is not None else None
        ),
        "total_energy_wh": float(row["total_energy_wh"] or 0.0),
    }


async def set_experiment_resource_usage(
    conn: aiosqlite.Connection,
    experiment_id: str,
    *,
    gpu_seconds: float | None = None,
    peak_vram_mb: float | None = None,
    energy_wh: float | None = None,
) -> None:
    """Write Phase 20 cost columns. NULL inputs leave the column alone."""
    fields: list[str] = []
    args: list[Any] = []
    if gpu_seconds is not None:
        fields.append("gpu_seconds = ?")
        args.append(gpu_seconds)
    if peak_vram_mb is not None:
        fields.append("peak_vram_mb = ?")
        args.append(peak_vram_mb)
    if energy_wh is not None:
        fields.append("energy_wh = ?")
        args.append(energy_wh)
    if not fields:
        return
    args.append(experiment_id)
    await conn.execute(
        f"UPDATE experiments SET {', '.join(fields)} WHERE id = ?",
        args,
    )
    await conn.commit()


async def create_experiment(
    conn: aiosqlite.Connection,
    spec: ExperimentSpec,
    *,
    study_id: str | None = None,
    trial_number: int | None = None,
) -> str:
    experiment_id = uuid.uuid4().hex
    now = utcnow_iso()
    await conn.execute(
        "INSERT INTO experiments (id, spec_json, status, priority, study_id, trial_number, "
        "created_at, queued_at) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)",
        (
            experiment_id,
            spec.model_dump_json(),
            spec.priority,
            study_id,
            trial_number,
            now,
            now,
        ),
    )
    await conn.execute(
        "INSERT INTO events (experiment_id, study_id, kind, payload_json, created_at) "
        "VALUES (?, ?, 'queued', ?, ?)",
        (experiment_id, study_id, "{}", now),
    )
    await conn.commit()
    return experiment_id


async def get_experiment(
    conn: aiosqlite.Connection, experiment_id: str
) -> ExperimentRecord | None:
    cur = await conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = await cur.fetchone()
    return _row_to_experiment_record(row) if row else None


async def list_experiments(
    conn: aiosqlite.Connection,
    *,
    status: ExperimentStatus | None = None,
    study_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[ExperimentRecord]:
    sql = "SELECT * FROM experiments WHERE 1=1"
    args: list[Any] = []
    if status:
        sql += " AND status = ?"
        args.append(status.value)
    if study_id:
        sql += " AND study_id = ?"
        args.append(study_id)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    cur = await conn.execute(sql, args)
    rows = await cur.fetchall()
    return [_row_to_experiment_record(r) for r in rows]


async def request_cancel(conn: aiosqlite.Connection, experiment_id: str) -> str:
    """Mark a queued experiment cancelled; signal caller for running cases.

    Return values:
        'not_found' | 'cancelled' | 'running' | <other terminal status>
    """
    cur = await conn.execute("SELECT status FROM experiments WHERE id = ?", (experiment_id,))
    row = await cur.fetchone()
    if not row:
        return "not_found"
    status = row[0]
    if status == ExperimentStatus.QUEUED.value:
        now = utcnow_iso()
        await conn.execute(
            "UPDATE experiments SET status = 'cancelled', finished_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (now, experiment_id),
        )
        await conn.commit()
        return "cancelled"
    if status == ExperimentStatus.RUNNING.value:
        return "running"
    return status


async def log_event(
    conn: aiosqlite.Connection,
    *,
    experiment_id: str | None,
    study_id: str | None,
    kind: str,
    payload: dict[str, Any] | None = None,
) -> None:
    await conn.execute(
        "INSERT INTO events (experiment_id, study_id, kind, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            experiment_id,
            study_id,
            kind,
            json.dumps(payload or {}),
            utcnow_iso(),
        ),
    )


def _row_to_study_record(row: aiosqlite.Row) -> StudyRecord:
    return StudyRecord(
        id=row["id"],
        name=row["name"],
        config=StudyConfig.model_validate_json(row["config_json"]),
        status=StudyStatus(row["status"]),
        optuna_storage=row["optuna_storage"],
        n_trials_target=row["n_trials_target"],
        n_trials_completed=row["n_trials_completed"],
        best_value=row["best_value"],
        best_trial_id=row["best_trial_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


async def create_study(
    conn: aiosqlite.Connection,
    config: StudyConfig,
    optuna_storage: str,
    *,
    study_id: str | None = None,
) -> str:
    if study_id is None:
        study_id = uuid.uuid4().hex
    now = utcnow_iso()
    await conn.execute(
        "INSERT INTO studies (id, name, config_json, status, optuna_storage, n_trials_target, "
        "created_at, updated_at) VALUES (?, ?, ?, 'active', ?, ?, ?, ?)",
        (
            study_id,
            config.name,
            config.model_dump_json(),
            optuna_storage,
            config.n_trials,
            now,
            now,
        ),
    )
    await conn.commit()
    return study_id


async def get_study(conn: aiosqlite.Connection, study_id: str) -> StudyRecord | None:
    cur = await conn.execute("SELECT * FROM studies WHERE id = ?", (study_id,))
    row = await cur.fetchone()
    return _row_to_study_record(row) if row else None


async def list_studies(conn: aiosqlite.Connection) -> list[StudyRecord]:
    cur = await conn.execute("SELECT * FROM studies ORDER BY created_at DESC")
    rows = await cur.fetchall()
    return [_row_to_study_record(r) for r in rows]


async def list_active_studies(conn: aiosqlite.Connection) -> list[StudyRecord]:
    cur = await conn.execute(
        "SELECT * FROM studies WHERE status = ? ORDER BY created_at",
        (StudyStatus.ACTIVE.value,),
    )
    rows = await cur.fetchall()
    return [_row_to_study_record(r) for r in rows]


async def update_study_progress(
    conn: aiosqlite.Connection,
    study_id: str,
    *,
    n_completed: int,
    best_value: float | None,
    best_trial_id: str | None,
) -> None:
    await conn.execute(
        "UPDATE studies SET n_trials_completed = ?, best_value = ?, best_trial_id = ?, "
        "updated_at = ? WHERE id = ?",
        (n_completed, best_value, best_trial_id, utcnow_iso(), study_id),
    )
    await conn.commit()


async def set_study_status(
    conn: aiosqlite.Connection, study_id: str, status: StudyStatus
) -> None:
    await conn.execute(
        "UPDATE studies SET status = ?, updated_at = ? WHERE id = ?",
        (status.value, utcnow_iso(), study_id),
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _row_to_dataset(row: aiosqlite.Row) -> Dataset:
    media_kinds: list[str] = []
    # Migration v5 columns may not be loaded by row_to_dict on older rows;
    # access with try/except KeyError so the function tolerates pre-v5
    # snapshots in test fixtures.
    try:
        raw = row["media_kinds_json"]
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                media_kinds = [str(k) for k in parsed]
    except (KeyError, IndexError):
        pass
    try:
        image_root = row["image_root"]
    except (KeyError, IndexError):
        image_root = None
    try:
        version = row["version"] or 1
    except (KeyError, IndexError):
        version = 1
    try:
        derived_from = row["derived_from"]
    except (KeyError, IndexError):
        derived_from = None
    return Dataset(
        id=row["id"],
        name=row["name"],
        path=row["path"],
        format=row["format"],
        line_count=row["line_count"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        description=row["description"],
        created_at=datetime.fromisoformat(row["created_at"]),
        media_kinds=media_kinds,
        image_root=image_root,
        version=version,
        derived_from=derived_from,
    )


async def create_dataset(
    conn: aiosqlite.Connection,
    *,
    name: str,
    path: str,
    fmt: str,
    size_bytes: int,
    sha256: str,
    line_count: int | None = None,
    description: str | None = None,
    dataset_id: str | None = None,
    media_kinds: list[str] | None = None,
    image_root: str | None = None,
    version: int = 1,
    derived_from: str | None = None,
) -> str:
    if dataset_id is None:
        dataset_id = uuid.uuid4().hex
    await conn.execute(
        "INSERT INTO datasets (id, name, path, format, line_count, size_bytes, "
        "sha256, description, media_kinds_json, image_root, version, "
        "derived_from, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            dataset_id,
            name,
            path,
            fmt,
            line_count,
            size_bytes,
            sha256,
            description,
            json.dumps(media_kinds) if media_kinds else None,
            image_root,
            version,
            derived_from,
            utcnow_iso(),
        ),
    )
    await conn.commit()
    return dataset_id


async def get_dataset(conn: aiosqlite.Connection, dataset_id: str) -> Dataset | None:
    cur = await conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = await cur.fetchone()
    return _row_to_dataset(row) if row else None


async def get_dataset_by_sha(
    conn: aiosqlite.Connection, sha256: str
) -> Dataset | None:
    cur = await conn.execute("SELECT * FROM datasets WHERE sha256 = ? LIMIT 1", (sha256,))
    row = await cur.fetchone()
    return _row_to_dataset(row) if row else None


async def list_datasets(conn: aiosqlite.Connection) -> list[Dataset]:
    cur = await conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    rows = await cur.fetchall()
    return [_row_to_dataset(r) for r in rows]


async def delete_dataset(conn: aiosqlite.Connection, dataset_id: str) -> bool:
    cur = await conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    await conn.commit()
    return cur.rowcount > 0


async def active_experiments_referencing_path(
    conn: aiosqlite.Connection, path: str
) -> list[str]:
    """Return ids of non-terminal experiments whose spec references ``path``.

    ``ds:<id>`` refs are resolved to real paths at submit time and frozen into
    spec_json, so a queued/running experiment carries the bare path (possibly
    with a ``#N`` subsample suffix). Deleting the underlying dataset would
    break those jobs at dispatch, so callers use this to guard the delete.
    """
    cur = await conn.execute(
        "SELECT id, spec_json FROM experiments WHERE status IN ('queued', 'running')"
    )
    rows = await cur.fetchall()
    hits: list[str] = []
    for row in rows:
        spec = ExperimentSpec.model_validate_json(row["spec_json"])
        refs = list(spec.dataset) + list(spec.val_dataset)
        if any(r.split("#", 1)[0] == path for r in refs):
            hits.append(row["id"])
    return hits


# ---------------------------------------------------------------------------
# Eval framework
# ---------------------------------------------------------------------------


def _row_to_eval_suite(row: aiosqlite.Row) -> EvalSuite:
    metrics_raw = json.loads(row["metrics_json"])
    return EvalSuite(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        dataset_path=row["dataset_path"],
        metrics=[MetricConfig.model_validate(m) for m in metrics_raw],
        inference_params=InferenceParams.model_validate_json(
            row["inference_params_json"]
        ),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


async def create_eval_suite(
    conn: aiosqlite.Connection,
    *,
    name: str,
    description: str | None,
    dataset_path: str,
    metrics: list[MetricConfig],
    inference_params: InferenceParams,
    suite_id: str | None = None,
) -> str:
    if suite_id is None:
        suite_id = uuid.uuid4().hex
    await conn.execute(
        "INSERT INTO eval_suites (id, name, description, dataset_path, "
        "metrics_json, inference_params_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            suite_id,
            name,
            description,
            dataset_path,
            json.dumps([m.model_dump() for m in metrics]),
            inference_params.model_dump_json(),
            utcnow_iso(),
        ),
    )
    await conn.commit()
    return suite_id


async def get_eval_suite(
    conn: aiosqlite.Connection, suite_id: str
) -> EvalSuite | None:
    cur = await conn.execute("SELECT * FROM eval_suites WHERE id = ?", (suite_id,))
    row = await cur.fetchone()
    return _row_to_eval_suite(row) if row else None


async def get_eval_suite_by_name(
    conn: aiosqlite.Connection, name: str
) -> EvalSuite | None:
    cur = await conn.execute(
        "SELECT * FROM eval_suites WHERE name = ? LIMIT 1", (name,)
    )
    row = await cur.fetchone()
    return _row_to_eval_suite(row) if row else None


async def list_eval_suites(conn: aiosqlite.Connection) -> list[EvalSuite]:
    cur = await conn.execute("SELECT * FROM eval_suites ORDER BY created_at DESC")
    rows = await cur.fetchall()
    return [_row_to_eval_suite(r) for r in rows]


async def delete_eval_suite(conn: aiosqlite.Connection, suite_id: str) -> bool:
    cur = await conn.execute("DELETE FROM eval_suites WHERE id = ?", (suite_id,))
    await conn.commit()
    return cur.rowcount > 0


async def active_eval_runs_for_suite(
    conn: aiosqlite.Connection, suite_id: str
) -> list[str]:
    """Return ids of non-terminal eval runs against ``suite_id``."""
    cur = await conn.execute(
        "SELECT id FROM eval_runs WHERE suite_id = ? AND status IN ('queued', 'running')",
        (suite_id,),
    )
    rows = await cur.fetchall()
    return [r["id"] for r in rows]


def _row_to_eval_run(row: aiosqlite.Row) -> EvalRun:
    aggregate_raw = row["aggregate_json"]
    aggregate: dict[str, MetricAggregate] | None = None
    if aggregate_raw:
        parsed = json.loads(aggregate_raw)
        aggregate = {k: MetricAggregate.model_validate(v) for k, v in parsed.items()}
    return EvalRun(
        id=row["id"],
        suite_id=row["suite_id"],
        experiment_id=row["experiment_id"],
        model_ref=row["model_ref"],
        status=EvalRunStatus(row["status"]),
        gpu_ids=json.loads(row["gpu_ids"]) if row["gpu_ids"] else None,
        log_path=row["log_path"],
        error=row["error"],
        aggregate=aggregate,
        sample_count=row["sample_count"],
        triggered_by=row["triggered_by"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )


async def create_eval_run(
    conn: aiosqlite.Connection,
    *,
    suite_id: str,
    experiment_id: str | None,
    model_ref: str,
    triggered_by: str,
    run_id: str | None = None,
) -> str:
    if run_id is None:
        run_id = uuid.uuid4().hex
    now = utcnow_iso()
    await conn.execute(
        "INSERT INTO eval_runs (id, suite_id, experiment_id, model_ref, status, "
        "triggered_by, created_at) VALUES (?, ?, ?, ?, 'queued', ?, ?)",
        (run_id, suite_id, experiment_id, model_ref, triggered_by, now),
    )
    await conn.execute(
        "INSERT INTO events (experiment_id, study_id, kind, payload_json, created_at) "
        "VALUES (?, ?, 'eval_queued', ?, ?)",
        (
            experiment_id,
            None,
            json.dumps({"eval_run_id": run_id, "suite_id": suite_id}),
            now,
        ),
    )
    await conn.commit()
    return run_id


async def get_eval_run(
    conn: aiosqlite.Connection, run_id: str
) -> EvalRun | None:
    cur = await conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
    row = await cur.fetchone()
    return _row_to_eval_run(row) if row else None


async def list_eval_runs(
    conn: aiosqlite.Connection,
    *,
    suite_id: str | None = None,
    experiment_id: str | None = None,
    status: EvalRunStatus | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[EvalRun]:
    sql = "SELECT * FROM eval_runs WHERE 1=1"
    args: list[Any] = []
    if suite_id:
        sql += " AND suite_id = ?"
        args.append(suite_id)
    if experiment_id:
        sql += " AND experiment_id = ?"
        args.append(experiment_id)
    if status:
        sql += " AND status = ?"
        args.append(status.value)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    cur = await conn.execute(sql, args)
    rows = await cur.fetchall()
    return [_row_to_eval_run(r) for r in rows]


async def claim_eval_run(conn: aiosqlite.Connection, run_id: str) -> bool:
    """Atomically flip a queued eval run to 'running'. Returns False if it
    was already taken or cancelled."""
    now = utcnow_iso()
    cur = await conn.execute(
        "UPDATE eval_runs SET status = 'running', started_at = ? "
        "WHERE id = ? AND status = 'queued'",
        (now, run_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def update_eval_run_progress(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    gpu_ids: list[int] | None = None,
    log_path: str | None = None,
    pid: int | None = None,
    sample_count: int | None = None,
) -> None:
    fields: list[str] = []
    args: list[Any] = []
    if gpu_ids is not None:
        fields.append("gpu_ids = ?")
        args.append(json.dumps(gpu_ids))
    if log_path is not None:
        fields.append("log_path = ?")
        args.append(log_path)
    if pid is not None:
        fields.append("pid = ?")
        args.append(pid)
    if sample_count is not None:
        fields.append("sample_count = ?")
        args.append(sample_count)
    if not fields:
        return
    args.append(run_id)
    await conn.execute(
        f"UPDATE eval_runs SET {', '.join(fields)} WHERE id = ?", args
    )
    await conn.commit()


async def finalize_eval_run(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    status: EvalRunStatus,
    aggregate: dict[str, MetricAggregate] | None = None,
    sample_count: int | None = None,
    error: str | None = None,
) -> None:
    aggregate_json = (
        json.dumps({k: v.model_dump() for k, v in aggregate.items()})
        if aggregate is not None
        else None
    )
    await conn.execute(
        "UPDATE eval_runs SET status = ?, finished_at = ?, aggregate_json = ?, "
        "sample_count = COALESCE(?, sample_count), error = ? WHERE id = ?",
        (
            status.value,
            utcnow_iso(),
            aggregate_json,
            sample_count,
            error,
            run_id,
        ),
    )
    await conn.commit()


async def request_cancel_eval_run(
    conn: aiosqlite.Connection, run_id: str
) -> str:
    """Mark a queued eval run cancelled; signal caller for running cases.

    Mirrors ``request_cancel`` for experiments.
    """
    cur = await conn.execute("SELECT status FROM eval_runs WHERE id = ?", (run_id,))
    row = await cur.fetchone()
    if not row:
        return "not_found"
    status = row[0]
    if status == EvalRunStatus.QUEUED.value:
        now = utcnow_iso()
        await conn.execute(
            "UPDATE eval_runs SET status = 'cancelled', finished_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (now, run_id),
        )
        await conn.commit()
        return "cancelled"
    if status == EvalRunStatus.RUNNING.value:
        return "running"
    return status


def _row_to_eval_result(row: aiosqlite.Row) -> EvalResult:
    return EvalResult(
        id=row["id"],
        run_id=row["run_id"],
        sample_index=row["sample_index"],
        input=json.loads(row["input_json"]),
        prediction=row["prediction"],
        gold=json.loads(row["gold_json"]) if row["gold_json"] else None,
        scores=json.loads(row["scores_json"]),
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


async def add_eval_result(
    conn: aiosqlite.Connection,
    *,
    run_id: str,
    sample_index: int,
    input: dict[str, Any],
    prediction: str,
    gold: dict[str, Any] | None,
    scores: dict[str, float],
    error: str | None = None,
) -> int:
    cur = await conn.execute(
        "INSERT INTO eval_results (run_id, sample_index, input_json, prediction, "
        "gold_json, scores_json, error, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            sample_index,
            json.dumps(input),
            prediction,
            json.dumps(gold) if gold is not None else None,
            json.dumps(scores),
            error,
            utcnow_iso(),
        ),
    )
    await conn.commit()
    return cur.lastrowid


async def list_eval_results(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    limit: int = 500,
    offset: int = 0,
) -> list[EvalResult]:
    cur = await conn.execute(
        "SELECT * FROM eval_results WHERE run_id = ? "
        "ORDER BY sample_index LIMIT ? OFFSET ?",
        (run_id, limit, offset),
    )
    rows = await cur.fetchall()
    return [_row_to_eval_result(r) for r in rows]


async def count_eval_results(conn: aiosqlite.Connection, run_id: str) -> int:
    cur = await conn.execute(
        "SELECT COUNT(*) FROM eval_results WHERE run_id = ?", (run_id,)
    )
    row = await cur.fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Model registry (Phase 7)
# ---------------------------------------------------------------------------


async def _aliases_for_model(
    conn: aiosqlite.Connection, model_id: str
) -> list[str]:
    cur = await conn.execute(
        "SELECT alias FROM model_aliases WHERE model_id = ? ORDER BY alias",
        (model_id,),
    )
    rows = await cur.fetchall()
    return [r[0] for r in rows]


async def _row_to_registered_model(
    conn: aiosqlite.Connection, row: aiosqlite.Row
) -> RegisteredModel:
    aliases = await _aliases_for_model(conn, row["id"])
    return RegisteredModel(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        run_id=row["run_id"],
        experiment_id=row["experiment_id"],
        base_model=row["base_model"],
        adapter_path=row["adapter_path"],
        eval_summary=(
            json.loads(row["eval_summary_json"])
            if row["eval_summary_json"]
            else None
        ),
        description=row["description"],
        created_at=datetime.fromisoformat(row["created_at"]),
        aliases=aliases,
    )


async def next_model_version(conn: aiosqlite.Connection, name: str) -> int:
    cur = await conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM models WHERE name = ?",
        (name,),
    )
    row = await cur.fetchone()
    return int(row[0]) if row else 1


async def create_model(
    conn: aiosqlite.Connection,
    *,
    name: str,
    version: int,
    base_model: str,
    adapter_path: str | None,
    experiment_id: str | None,
    run_id: str | None,
    eval_summary: dict[str, Any] | None,
    description: str | None,
    model_id: str | None = None,
) -> str:
    if model_id is None:
        model_id = uuid.uuid4().hex
    await conn.execute(
        "INSERT INTO models (id, name, version, run_id, experiment_id, "
        "base_model, adapter_path, eval_summary_json, description, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            model_id,
            name,
            version,
            run_id,
            experiment_id,
            base_model,
            adapter_path,
            json.dumps(eval_summary) if eval_summary is not None else None,
            description,
            utcnow_iso(),
        ),
    )
    await conn.commit()
    return model_id


async def register_model_atomic(
    conn: aiosqlite.Connection,
    *,
    name: str,
    explicit_version: int | None,
    base_model: str,
    adapter_path: str | None,
    experiment_id: str | None,
    run_id: str | None,
    eval_summary: dict[str, Any] | None,
    description: str | None,
    alias: str | None,
) -> tuple[str, int]:
    """Register a model with auto-version + optional alias, atomically.

    All writes happen in a single transaction so:
      (a) two concurrent registrations of the same ``name`` cannot both
          land on the same auto-incremented version (the loser sees the
          UNIQUE(name, version) constraint and retries),
      (b) a failure during alias assignment cannot leave a model row
          behind with no alias.

    On explicit-version conflict, raises ``ValueError("version_exists")``
    so the route can translate to a 409.
    """
    # SQLite + aiosqlite: connections default to autocommit-on-write via
    # BEGIN ... COMMIT inside .execute(). For a multi-statement atomic
    # block we open an explicit transaction.
    max_attempts = 5
    last_exc: Exception | None = None
    for _attempt in range(max_attempts):
        try:
            await conn.execute("BEGIN IMMEDIATE")
            if explicit_version is None:
                cur = await conn.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM models WHERE name = ?",
                    (name,),
                )
                row = await cur.fetchone()
                version = int(row[0]) if row else 1
            else:
                version = explicit_version
                cur = await conn.execute(
                    "SELECT 1 FROM models WHERE name = ? AND version = ?",
                    (name, version),
                )
                if await cur.fetchone() is not None:
                    await conn.execute("ROLLBACK")
                    raise ValueError("version_exists")

            model_id = uuid.uuid4().hex
            await conn.execute(
                "INSERT INTO models (id, name, version, run_id, experiment_id, "
                "base_model, adapter_path, eval_summary_json, description, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    name,
                    version,
                    run_id,
                    experiment_id,
                    base_model,
                    adapter_path,
                    json.dumps(eval_summary) if eval_summary is not None else None,
                    description,
                    utcnow_iso(),
                ),
            )
            if alias:
                await conn.execute(
                    "INSERT INTO model_aliases (name, alias, model_id, "
                    "updated_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(name, alias) DO UPDATE SET "
                    "model_id = excluded.model_id, "
                    "updated_at = excluded.updated_at",
                    (name, alias, model_id, utcnow_iso()),
                )
            await conn.commit()
            return model_id, version
        except aiosqlite.IntegrityError as e:
            # Auto-version path: another transaction grabbed our version.
            # Roll back and retry with a fresh MAX(version).
            try:
                await conn.execute("ROLLBACK")
            except aiosqlite.Error:
                pass
            if explicit_version is not None:
                # Explicit version race — still surfaces as version_exists.
                raise ValueError("version_exists") from None
            last_exc = e
            continue
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Active learning (Phase 11)
# ---------------------------------------------------------------------------


def _row_to_al_run(row: aiosqlite.Row) -> ActiveLearningRun:
    return ActiveLearningRun(
        id=row["id"],
        model_ref=row["model_ref"],
        dataset_path=row["dataset_path"],
        top_n=row["top_n"],
        sample_limit=row["sample_limit"],
        status=ALRunStatus(row["status"]),
        error=row["error"],
        scored_count=row["scored_count"],
        queued_count=row["queued_count"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )


async def create_al_run(
    conn: aiosqlite.Connection,
    *,
    model_ref: str,
    dataset_path: str,
    top_n: int,
    sample_limit: int | None,
    run_id: str | None = None,
) -> str:
    if run_id is None:
        run_id = uuid.uuid4().hex
    now = utcnow_iso()
    await conn.execute(
        "INSERT INTO active_learning_runs (id, model_ref, dataset_path, "
        "top_n, sample_limit, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'queued', ?)",
        (run_id, model_ref, dataset_path, top_n, sample_limit, now),
    )
    await conn.commit()
    return run_id


async def get_al_run(
    conn: aiosqlite.Connection, run_id: str
) -> ActiveLearningRun | None:
    cur = await conn.execute(
        "SELECT * FROM active_learning_runs WHERE id = ?", (run_id,)
    )
    row = await cur.fetchone()
    return _row_to_al_run(row) if row else None


async def list_al_runs(
    conn: aiosqlite.Connection,
    *,
    status: ALRunStatus | None = None,
    limit: int = 100,
) -> list[ActiveLearningRun]:
    sql = "SELECT * FROM active_learning_runs"
    args: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        args.append(status.value)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    cur = await conn.execute(sql, args)
    rows = await cur.fetchall()
    return [_row_to_al_run(r) for r in rows]


async def update_al_run_status(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    status: ALRunStatus,
    error: str | None = None,
    scored_count: int | None = None,
    queued_count: int | None = None,
) -> None:
    fields = ["status = ?"]
    args: list[Any] = [status.value]
    if status == ALRunStatus.RUNNING:
        fields.append("started_at = ?")
        args.append(utcnow_iso())
    if status in (
        ALRunStatus.COMPLETED,
        ALRunStatus.FAILED,
        ALRunStatus.CANCELLED,
    ):
        fields.append("finished_at = ?")
        args.append(utcnow_iso())
    if error is not None:
        fields.append("error = ?")
        args.append(error)
    if scored_count is not None:
        fields.append("scored_count = ?")
        args.append(scored_count)
    if queued_count is not None:
        fields.append("queued_count = ?")
        args.append(queued_count)
    args.append(run_id)
    await conn.execute(
        f"UPDATE active_learning_runs SET {', '.join(fields)} WHERE id = ?",
        args,
    )
    await conn.commit()


async def add_queue_item(
    conn: aiosqlite.Connection,
    *,
    run_id: str,
    sample_index: int,
    input: dict[str, Any],
    prediction: str,
    uncertainty: float,
) -> int:
    cur = await conn.execute(
        "INSERT INTO annotation_queue_items (run_id, sample_index, "
        "input_json, prediction, uncertainty, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            run_id,
            sample_index,
            json.dumps(input),
            prediction,
            float(uncertainty),
            utcnow_iso(),
        ),
    )
    await conn.commit()
    return cur.lastrowid


def _row_to_queue_item(row: aiosqlite.Row) -> AnnotationQueueItem:
    return AnnotationQueueItem(
        id=row["id"],
        run_id=row["run_id"],
        sample_index=row["sample_index"],
        input=json.loads(row["input_json"]),
        prediction=row["prediction"],
        uncertainty=row["uncertainty"],
        annotated=bool(row["annotated"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


async def list_queue_items(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    only_unannotated: bool = False,
    limit: int = 500,
) -> list[AnnotationQueueItem]:
    sql = "SELECT * FROM annotation_queue_items WHERE run_id = ?"
    args: list[Any] = [run_id]
    if only_unannotated:
        sql += " AND annotated = 0"
    sql += " ORDER BY uncertainty DESC, sample_index ASC LIMIT ?"
    args.append(limit)
    cur = await conn.execute(sql, args)
    rows = await cur.fetchall()
    return [_row_to_queue_item(r) for r in rows]


async def mark_queue_annotated(
    conn: aiosqlite.Connection, item_id: int
) -> bool:
    cur = await conn.execute(
        "UPDATE annotation_queue_items SET annotated = 1 WHERE id = ?",
        (item_id,),
    )
    await conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Pipelines (Phase 12)
# ---------------------------------------------------------------------------


async def create_pipeline(
    conn: aiosqlite.Connection,
    *,
    name: str,
    config: PipelineConfig,
    pipeline_id: str | None = None,
) -> str:
    if pipeline_id is None:
        pipeline_id = uuid.uuid4().hex
    now = utcnow_iso()
    await conn.execute(
        "INSERT INTO pipelines (id, name, config_json, status, created_at, "
        "updated_at) VALUES (?, ?, ?, 'queued', ?, ?)",
        (pipeline_id, name, config.model_dump_json(), now, now),
    )
    for idx, stage in enumerate(config.stages):
        await conn.execute(
            "INSERT INTO pipeline_stages (pipeline_id, stage_name, stage_index, "
            "depends_on_json, status) VALUES (?, ?, ?, ?, 'pending')",
            (
                pipeline_id,
                stage.name,
                idx,
                json.dumps(stage.depends_on),
            ),
        )
    await conn.commit()
    return pipeline_id


def _row_to_stage(row: aiosqlite.Row) -> PipelineStage:
    return PipelineStage(
        stage_name=row["stage_name"],
        stage_index=row["stage_index"],
        depends_on=json.loads(row["depends_on_json"]) if row["depends_on_json"] else [],
        experiment_id=row["experiment_id"],
        status=StageStatus(row["status"]),
        output_dir=row["output_dir"],
        error=row["error"],
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )


async def get_pipeline(
    conn: aiosqlite.Connection, pipeline_id: str
) -> Pipeline | None:
    cur = await conn.execute(
        "SELECT * FROM pipelines WHERE id = ?", (pipeline_id,)
    )
    row = await cur.fetchone()
    if row is None:
        return None
    stage_cur = await conn.execute(
        "SELECT * FROM pipeline_stages WHERE pipeline_id = ? "
        "ORDER BY stage_index",
        (pipeline_id,),
    )
    stage_rows = await stage_cur.fetchall()
    return Pipeline(
        id=row["id"],
        name=row["name"],
        status=PipelineStatus(row["status"]),
        config=PipelineConfig.model_validate_json(row["config_json"]),
        stages=[_row_to_stage(r) for r in stage_rows],
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


async def list_pipelines(conn: aiosqlite.Connection) -> list[Pipeline]:
    cur = await conn.execute(
        "SELECT id FROM pipelines ORDER BY created_at DESC"
    )
    rows = await cur.fetchall()
    out: list[Pipeline] = []
    for row in rows:
        p = await get_pipeline(conn, row["id"])
        if p:
            out.append(p)
    return out


async def list_active_pipelines(
    conn: aiosqlite.Connection,
) -> list[Pipeline]:
    cur = await conn.execute(
        "SELECT id FROM pipelines WHERE status IN ('queued', 'running')"
    )
    rows = await cur.fetchall()
    out: list[Pipeline] = []
    for row in rows:
        p = await get_pipeline(conn, row["id"])
        if p:
            out.append(p)
    return out


async def set_pipeline_status(
    conn: aiosqlite.Connection,
    pipeline_id: str,
    *,
    status: PipelineStatus,
    error: str | None = None,
) -> None:
    fields = ["status = ?", "updated_at = ?"]
    args: list[Any] = [status.value, utcnow_iso()]
    if error is not None:
        fields.append("error = ?")
        args.append(error)
    args.append(pipeline_id)
    await conn.execute(
        f"UPDATE pipelines SET {', '.join(fields)} WHERE id = ?", args
    )
    await conn.commit()


async def enqueue_stage_with_experiment(
    conn: aiosqlite.Connection,
    *,
    pipeline_id: str,
    stage_name: str,
    spec: ExperimentSpec,
    output_dir: str | None,
) -> str:
    """Create the stage's backing experiment AND flip the stage to QUEUED
    in one transaction.

    The two writes were previously split across ``create_experiment`` +
    ``update_stage``, each committing independently — a process crash
    between them would leave an orphan ``queued`` experiment that no
    stage row referenced, and the driver's next tick would silently
    spawn a second experiment for the same stage. ``BEGIN IMMEDIATE``
    here is mostly belt-and-suspenders: aiosqlite already wraps a
    sequence of writes in an implicit transaction, but the explicit
    BEGIN ensures we don't pick up someone else's open write in the
    middle.
    """
    experiment_id = uuid.uuid4().hex
    now = utcnow_iso()
    try:
        await conn.execute("BEGIN IMMEDIATE")
        await conn.execute(
            "INSERT INTO experiments (id, spec_json, status, priority, "
            "study_id, trial_number, created_at, queued_at) "
            "VALUES (?, ?, 'queued', ?, NULL, NULL, ?, ?)",
            (
                experiment_id,
                spec.model_dump_json(),
                spec.priority,
                now,
                now,
            ),
        )
        await conn.execute(
            "INSERT INTO events (experiment_id, study_id, kind, "
            "payload_json, created_at) VALUES (?, NULL, 'queued', ?, ?)",
            (experiment_id, "{}", now),
        )
        await conn.execute(
            "UPDATE pipeline_stages SET status = 'queued', "
            "experiment_id = ?, output_dir = ? "
            "WHERE pipeline_id = ? AND stage_name = ?",
            (experiment_id, output_dir, pipeline_id, stage_name),
        )
        await conn.commit()
    except Exception:
        try:
            await conn.execute("ROLLBACK")
        except aiosqlite.Error:
            pass
        raise
    return experiment_id


async def update_stage(
    conn: aiosqlite.Connection,
    pipeline_id: str,
    stage_name: str,
    *,
    status: StageStatus | None = None,
    experiment_id: str | None = None,
    output_dir: str | None = None,
    error: str | None = None,
) -> None:
    fields: list[str] = []
    args: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        args.append(status.value)
        if status == StageStatus.RUNNING:
            fields.append("started_at = ?")
            args.append(utcnow_iso())
        if status in (
            StageStatus.COMPLETED,
            StageStatus.FAILED,
            StageStatus.CANCELLED,
            StageStatus.SKIPPED,
        ):
            fields.append("finished_at = ?")
            args.append(utcnow_iso())
    if experiment_id is not None:
        fields.append("experiment_id = ?")
        args.append(experiment_id)
    if output_dir is not None:
        fields.append("output_dir = ?")
        args.append(output_dir)
    if error is not None:
        fields.append("error = ?")
        args.append(error)
    if not fields:
        return
    args.extend([pipeline_id, stage_name])
    await conn.execute(
        f"UPDATE pipeline_stages SET {', '.join(fields)} "
        f"WHERE pipeline_id = ? AND stage_name = ?",
        args,
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# Watches (Phase 17)
# ---------------------------------------------------------------------------


def _row_to_watch(row: aiosqlite.Row) -> Watch:
    def _maybe(name: str, default=None):
        try:
            v = row[name]
            return default if v is None else v
        except (KeyError, IndexError):
            return default

    return Watch(
        id=row["id"],
        name=row["name"],
        kind=row["kind"],
        enabled=bool(row["enabled"]),
        interval_seconds=row["interval_seconds"],
        model_name=row["model_name"],
        suite_id=row["suite_id"],
        metric_name=row["metric_name"],
        threshold=row["threshold"],
        pipeline_config=PipelineConfig.model_validate_json(
            row["pipeline_config_json"]
        ),
        last_fired_at=(
            datetime.fromisoformat(row["last_fired_at"])
            if row["last_fired_at"]
            else None
        ),
        last_fired_pipeline_id=row["last_fired_pipeline_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        consecutive_failures=int(_maybe("consecutive_failures", 0)),
        last_error=_maybe("last_error", None),
    )


async def create_watch(
    conn: aiosqlite.Connection,
    *,
    name: str,
    kind: str,
    pipeline_config: PipelineConfig,
    interval_seconds: int | None = None,
    model_name: str | None = None,
    suite_id: str | None = None,
    metric_name: str | None = None,
    threshold: float | None = None,
    watch_id: str | None = None,
) -> str:
    if watch_id is None:
        watch_id = uuid.uuid4().hex
    await conn.execute(
        "INSERT INTO watches (id, name, kind, enabled, interval_seconds, "
        "model_name, suite_id, metric_name, threshold, pipeline_config_json, "
        "created_at) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
        (
            watch_id,
            name,
            kind,
            interval_seconds,
            model_name,
            suite_id,
            metric_name,
            threshold,
            pipeline_config.model_dump_json(),
            utcnow_iso(),
        ),
    )
    await conn.commit()
    return watch_id


async def get_watch(
    conn: aiosqlite.Connection, watch_id: str
) -> Watch | None:
    cur = await conn.execute("SELECT * FROM watches WHERE id = ?", (watch_id,))
    row = await cur.fetchone()
    return _row_to_watch(row) if row else None


async def list_watches(
    conn: aiosqlite.Connection, *, only_enabled: bool = False
) -> list[Watch]:
    sql = "SELECT * FROM watches"
    if only_enabled:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY created_at DESC"
    cur = await conn.execute(sql)
    rows = await cur.fetchall()
    return [_row_to_watch(r) for r in rows]


async def set_watch_enabled(
    conn: aiosqlite.Connection, watch_id: str, enabled: bool
) -> bool:
    cur = await conn.execute(
        "UPDATE watches SET enabled = ? WHERE id = ?",
        (1 if enabled else 0, watch_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def record_watch_fire(
    conn: aiosqlite.Connection, watch_id: str, pipeline_id: str
) -> None:
    """Mark a successful fire — resets the failure counter."""
    await conn.execute(
        "UPDATE watches SET last_fired_at = ?, last_fired_pipeline_id = ?, "
        "consecutive_failures = 0, last_error = NULL WHERE id = ?",
        (utcnow_iso(), pipeline_id, watch_id),
    )
    await conn.commit()


async def record_watch_failure(
    conn: aiosqlite.Connection,
    watch_id: str,
    error: str,
    *,
    disable_threshold: int,
) -> int:
    """Increment the consecutive-failure counter and auto-disable once it
    reaches ``disable_threshold``. Returns the new counter value (handy
    for logging the disable decision)."""
    await conn.execute(
        "UPDATE watches SET consecutive_failures = consecutive_failures + 1, "
        "last_error = ? WHERE id = ?",
        (error[:1024], watch_id),
    )
    cur = await conn.execute(
        "SELECT consecutive_failures FROM watches WHERE id = ?",
        (watch_id,),
    )
    row = await cur.fetchone()
    n = int(row[0]) if row else 0
    if n >= disable_threshold:
        await conn.execute(
            "UPDATE watches SET enabled = 0 WHERE id = ?", (watch_id,)
        )
    await conn.commit()
    return n


async def delete_watch(conn: aiosqlite.Connection, watch_id: str) -> bool:
    cur = await conn.execute(
        "DELETE FROM watches WHERE id = ?", (watch_id,)
    )
    await conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Model lineage (Phase 15)
# ---------------------------------------------------------------------------


async def record_model_lineage(
    conn: aiosqlite.Connection, model_id: str, dataset_ids: list[str]
) -> int:
    """Insert one row per (model, dataset). Idempotent via UPSERT — re-
    registering the same model is harmless."""
    now = utcnow_iso()
    for ds_id in dataset_ids:
        await conn.execute(
            "INSERT INTO model_lineage (model_id, dataset_id, used_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(model_id, dataset_id) DO NOTHING",
            (model_id, ds_id, now),
        )
    await conn.commit()
    return len(dataset_ids)


async def models_using_dataset(
    conn: aiosqlite.Connection, dataset_id: str
) -> list[str]:
    cur = await conn.execute(
        "SELECT model_id FROM model_lineage WHERE dataset_id = ?",
        (dataset_id,),
    )
    rows = await cur.fetchall()
    return [r[0] for r in rows]


async def datasets_used_by_model(
    conn: aiosqlite.Connection, model_id: str
) -> list[str]:
    cur = await conn.execute(
        "SELECT dataset_id FROM model_lineage WHERE model_id = ?",
        (model_id,),
    )
    rows = await cur.fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Dataset lineage (Phase 16 follow-up — N:M parent tracking)
# ---------------------------------------------------------------------------


async def record_dataset_lineage(
    conn: aiosqlite.Connection,
    derived_id: str,
    parent_ids: list[str],
    *,
    role: str,
) -> None:
    """Insert (derived_id, parent_id) rows for each parent. Idempotent.

    ``role`` is a free-form label (``split-of`` / ``mix-of`` /
    ``redacted-from`` / ``synthesized-from``) — used in audit queries to
    distinguish how the derivation happened.
    """
    now = utcnow_iso()
    for pid in parent_ids:
        if pid == derived_id:
            continue  # self-loop guard
        await conn.execute(
            "INSERT INTO dataset_lineage (derived_id, parent_id, role, "
            "recorded_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(derived_id, parent_id) DO NOTHING",
            (derived_id, pid, role, now),
        )
    await conn.commit()


async def dataset_ancestors(
    conn: aiosqlite.Connection, dataset_id: str, *, max_depth: int = 32
) -> set[str]:
    """All ancestors of ``dataset_id`` (transitive parents).

    Walks ``dataset_lineage`` BFS-style. ``max_depth`` is a cycle-safety
    cap (the table allows cycles in principle — we never create them,
    but a corrupt row shouldn't hang the audit query). Returns a set of
    dataset ids; ``dataset_id`` itself is NOT included.
    """
    seen: set[str] = set()
    frontier = {dataset_id}
    for _ in range(max_depth):
        if not frontier:
            break
        placeholders = ",".join("?" for _ in frontier)
        cur = await conn.execute(
            f"SELECT parent_id FROM dataset_lineage WHERE derived_id IN ({placeholders})",
            tuple(frontier),
        )
        rows = await cur.fetchall()
        next_frontier = {r[0] for r in rows} - seen - {dataset_id}
        seen.update(next_frontier)
        frontier = next_frontier
    return seen


async def dataset_descendants(
    conn: aiosqlite.Connection, dataset_id: str, *, max_depth: int = 32
) -> set[str]:
    """All descendants of ``dataset_id`` (transitive children).

    The reverse of :func:`dataset_ancestors`. Used to answer
    "the user asked to forget data in dataset X — which derived
    datasets also need to be redacted?"
    """
    seen: set[str] = set()
    frontier = {dataset_id}
    for _ in range(max_depth):
        if not frontier:
            break
        placeholders = ",".join("?" for _ in frontier)
        cur = await conn.execute(
            f"SELECT derived_id FROM dataset_lineage WHERE parent_id IN ({placeholders})",
            tuple(frontier),
        )
        rows = await cur.fetchall()
        next_frontier = {r[0] for r in rows} - seen - {dataset_id}
        seen.update(next_frontier)
        frontier = next_frontier
    return seen


async def models_using_dataset_recursive(
    conn: aiosqlite.Connection, dataset_id: str
) -> list[str]:
    """All models whose training data ultimately derives from ``dataset_id``.

    GDPR-relevant: ``models_using_dataset`` only catches direct usage,
    so a model trained on a mix or split of ``dataset_id`` would slip
    through. This walks ``dataset_descendants`` first, then unions all
    matching ``model_lineage`` rows. Returns deduplicated model ids.
    """
    descendants = await dataset_descendants(conn, dataset_id)
    candidates = descendants | {dataset_id}
    if not candidates:
        return []
    placeholders = ",".join("?" for _ in candidates)
    cur = await conn.execute(
        f"SELECT DISTINCT model_id FROM model_lineage "
        f"WHERE dataset_id IN ({placeholders})",
        tuple(candidates),
    )
    rows = await cur.fetchall()
    return [r[0] for r in rows]


async def datasets_by_paths(
    conn: aiosqlite.Connection, paths: list[str]
) -> dict[str, str]:
    """Return ``{path: dataset_id}`` for any registered dataset whose
    on-disk path matches one of ``paths``. Used by the model-register
    flow to populate ``model_lineage`` from the spec's resolved paths.
    """
    if not paths:
        return {}
    placeholders = ",".join("?" for _ in paths)
    cur = await conn.execute(
        f"SELECT id, path FROM datasets WHERE path IN ({placeholders})",
        paths,
    )
    rows = await cur.fetchall()
    return {r["path"]: r["id"] for r in rows}


async def active_models_referencing_experiment(
    conn: aiosqlite.Connection, experiment_id: str
) -> list[str]:
    """Model ids that still point at ``experiment_id``.

    Mirrors ``active_experiments_referencing_path`` for datasets. Callers
    use this to block experiment delete when registered models would be
    orphaned.
    """
    cur = await conn.execute(
        "SELECT id FROM models WHERE experiment_id = ?", (experiment_id,)
    )
    rows = await cur.fetchall()
    return [r[0] for r in rows]


async def get_model(
    conn: aiosqlite.Connection, model_id: str
) -> RegisteredModel | None:
    cur = await conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    row = await cur.fetchone()
    return await _row_to_registered_model(conn, row) if row else None


async def get_model_by_name_version(
    conn: aiosqlite.Connection, name: str, version: int
) -> RegisteredModel | None:
    cur = await conn.execute(
        "SELECT * FROM models WHERE name = ? AND version = ?",
        (name, version),
    )
    row = await cur.fetchone()
    return await _row_to_registered_model(conn, row) if row else None


async def list_models(
    conn: aiosqlite.Connection,
    *,
    name: str | None = None,
    alias: str | None = None,
) -> list[RegisteredModel]:
    sql = "SELECT m.* FROM models m"
    args: list[Any] = []
    where: list[str] = []
    if alias:
        sql += " JOIN model_aliases a ON a.model_id = m.id AND a.alias = ?"
        args.append(alias)
    if name:
        where.append("m.name = ?")
        args.append(name)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY m.name, m.version DESC"
    cur = await conn.execute(sql, args)
    rows = await cur.fetchall()
    return [await _row_to_registered_model(conn, r) for r in rows]


async def list_models_by_name(
    conn: aiosqlite.Connection, name: str
) -> list[RegisteredModel]:
    return await list_models(conn, name=name)


async def resolve_model_alias(
    conn: aiosqlite.Connection, name: str, alias: str
) -> RegisteredModel | None:
    cur = await conn.execute(
        "SELECT m.* FROM models m JOIN model_aliases a ON a.model_id = m.id "
        "WHERE a.name = ? AND a.alias = ?",
        (name, alias),
    )
    row = await cur.fetchone()
    return await _row_to_registered_model(conn, row) if row else None


async def set_model_alias(
    conn: aiosqlite.Connection,
    *,
    name: str,
    alias: str,
    model_id: str,
) -> None:
    """Assign or move an alias within a model family. UPSERT semantics."""
    await conn.execute(
        "INSERT INTO model_aliases (name, alias, model_id, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(name, alias) DO UPDATE SET model_id = excluded.model_id, "
        "updated_at = excluded.updated_at",
        (name, alias, model_id, utcnow_iso()),
    )
    await conn.commit()


async def delete_model_alias(
    conn: aiosqlite.Connection, name: str, alias: str
) -> bool:
    cur = await conn.execute(
        "DELETE FROM model_aliases WHERE name = ? AND alias = ?",
        (name, alias),
    )
    await conn.commit()
    return cur.rowcount > 0


async def delete_model(conn: aiosqlite.Connection, model_id: str) -> bool:
    cur = await conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    await conn.commit()
    return cur.rowcount > 0


async def recover_eval_runs(conn: aiosqlite.Connection) -> int:
    """Requeue 'running' eval runs on scheduler start (crash recovery).

    Mirrors the experiment recovery in scheduler.start(). Returns the
    number of rows touched.
    """
    now = utcnow_iso()
    cur = await conn.execute(
        "UPDATE eval_runs SET status = 'queued', started_at = NULL, "
        "gpu_ids = NULL, log_path = NULL, pid = NULL WHERE status = 'running'",
        (),
    )
    # Restore queued_at-equivalent by stamping created_at? No — eval_runs has
    # only created_at, which is immutable. The status flip is enough; the
    # eval driver re-claims in FIFO via created_at.
    await conn.commit()
    _ = now  # reserved for future started_at history table
    return cur.rowcount


# ---------------------------------------------------------------------------
# Agentic data acquisition (Phase 22)
# ---------------------------------------------------------------------------


def _row_to_acquisition_run(row: aiosqlite.Row) -> AcquisitionRun:
    return AcquisitionRun(
        id=row["id"],
        name=row["name"],
        brief=row["brief"],
        provider=row["provider"],
        model=row["model"],
        target_count=row["target_count"],
        search_provider=row["search_provider"],
        max_sources=row["max_sources"],
        spec=(
            AcquisitionSpec.model_validate_json(row["spec_json"])
            if row["spec_json"]
            else None
        ),
        answers=json.loads(row["answers_json"]) if row["answers_json"] else None,
        status=AcquisitionStatus(row["status"]),
        phase=row["phase"],
        dataset_id=row["dataset_id"],
        raw_count=row["raw_count"],
        final_count=row["final_count"],
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )


async def create_acquisition_run(
    conn: aiosqlite.Connection,
    *,
    name: str,
    brief: str,
    provider: str,
    model: str,
    target_count: int,
    search_provider: str = "none",
    max_sources: int = 0,
    spec: AcquisitionSpec | None = None,
    run_id: str | None = None,
) -> str:
    if run_id is None:
        run_id = uuid.uuid4().hex
    now = utcnow_iso()
    await conn.execute(
        "INSERT INTO acquisition_runs (id, name, brief, provider, model, "
        "target_count, search_provider, max_sources, spec_json, status, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)",
        (
            run_id,
            name,
            brief,
            provider,
            model,
            target_count,
            search_provider,
            max_sources,
            spec.model_dump_json() if spec else None,
            now,
        ),
    )
    await conn.commit()
    return run_id


async def get_acquisition_run(
    conn: aiosqlite.Connection, run_id: str
) -> AcquisitionRun | None:
    cur = await conn.execute(
        "SELECT * FROM acquisition_runs WHERE id = ?", (run_id,)
    )
    row = await cur.fetchone()
    return _row_to_acquisition_run(row) if row else None


async def list_acquisition_runs(
    conn: aiosqlite.Connection,
    *,
    status: AcquisitionStatus | None = None,
    limit: int = 100,
) -> list[AcquisitionRun]:
    sql = "SELECT * FROM acquisition_runs"
    args: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        args.append(status.value)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    cur = await conn.execute(sql, args)
    rows = await cur.fetchall()
    return [_row_to_acquisition_run(r) for r in rows]


async def list_active_acquisition_runs(
    conn: aiosqlite.Connection,
) -> list[AcquisitionRun]:
    """Runs that still have a live driver to (re)start: queued/running.

    ``awaiting_input`` is deliberately excluded — those are parked on the
    operator, not on us, so a restart must leave them parked rather than
    resume and re-ask.
    """
    cur = await conn.execute(
        "SELECT * FROM acquisition_runs WHERE status IN ('queued', 'running') "
        "ORDER BY created_at ASC"
    )
    rows = await cur.fetchall()
    return [_row_to_acquisition_run(r) for r in rows]


async def update_acquisition_run(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    status: AcquisitionStatus | None = None,
    phase: str | None = None,
    spec: AcquisitionSpec | None = None,
    answers: dict[str, str] | None = None,
    dataset_id: str | None = None,
    raw_count: int | None = None,
    final_count: int | None = None,
    error: str | None = None,
) -> None:
    fields: list[str] = []
    args: list[Any] = []
    # Plain column → value; None means "leave it". Kept data-driven so a new
    # column is one row, not another copy-pasted if-block.
    columns = {
        "phase": phase,
        "spec_json": spec.model_dump_json() if spec is not None else None,
        "answers_json": json.dumps(answers) if answers is not None else None,
        "dataset_id": dataset_id,
        "raw_count": raw_count,
        "final_count": final_count,
        "error": error,
    }
    if status is not None:
        fields.append("status = ?")
        args.append(status.value)
        if status == AcquisitionStatus.RUNNING:
            # Only stamp started_at the first time we go RUNNING.
            fields.append("started_at = COALESCE(started_at, ?)")
            args.append(utcnow_iso())
        elif status in ACQUISITION_TERMINAL_STATUSES:
            fields.append("finished_at = ?")
            args.append(utcnow_iso())
    for col, val in columns.items():
        if val is not None:
            fields.append(f"{col} = ?")
            args.append(val)
    if not fields:
        return
    args.append(run_id)
    await conn.execute(
        f"UPDATE acquisition_runs SET {', '.join(fields)} WHERE id = ?", args
    )
    await conn.commit()


async def record_acquisition_source(
    conn: aiosqlite.Connection,
    *,
    run_id: str,
    url: str,
    title: str | None = None,
    topic: str | None = None,
    license_status: str = "unknown",
    used: bool = False,
) -> None:
    await conn.execute(
        "INSERT INTO acquisition_sources (run_id, url, title, topic, "
        "license_status, used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, url, title, topic, license_status, int(used), utcnow_iso()),
    )
    await conn.commit()


async def list_acquisition_sources(
    conn: aiosqlite.Connection, run_id: str
) -> list[AcquisitionSource]:
    cur = await conn.execute(
        "SELECT * FROM acquisition_sources WHERE run_id = ? ORDER BY id ASC",
        (run_id,),
    )
    rows = await cur.fetchall()
    return [
        AcquisitionSource(
            id=r["id"],
            url=r["url"],
            title=r["title"],
            topic=r["topic"],
            license_status=r["license_status"],
            used=bool(r["used"]),
            created_at=datetime.fromisoformat(r["created_at"]),
        )
        for r in rows
    ]
