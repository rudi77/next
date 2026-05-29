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
    ExperimentRecord,
    ExperimentSpec,
    ExperimentStatus,
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
) -> str:
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
