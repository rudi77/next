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
    StudyConfig,
    StudyRecord,
    StudyStatus,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


def _row_to_experiment_record(row: aiosqlite.Row) -> ExperimentRecord:
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
    )


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
) -> str:
    if dataset_id is None:
        dataset_id = uuid.uuid4().hex
    await conn.execute(
        "INSERT INTO datasets (id, name, path, format, line_count, size_bytes, "
        "sha256, description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            dataset_id,
            name,
            path,
            fmt,
            line_count,
            size_bytes,
            sha256,
            description,
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
