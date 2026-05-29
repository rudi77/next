"""Main scheduler loop.

One asyncio task polls SQLite for queued experiments. Per tick it walks the
queue under a dispatch lock and *claims* (atomic CAS UPDATE) one experiment
at a time: allocate GPUs, flip status to 'running'. Each claim then fires
a background launch task that does the slow work (MLflow run creation +
subprocess spawn) outside the lock, so an MLflow stall can't serialize
concurrent submits.

A per-experiment monitor task watches the subprocess exit, releases GPUs,
finalizes the MLflow run, and persists final state.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from ..api.schemas import ExperimentSpec
from ..core import repository
from ..core.db import Database
from ..settings import settings
from ..training.swift_builder import build_swift_command
from .gpu_pool import GpuPool
from .runner import RunningProcess, spawn_training_subprocess

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Scheduler:
    def __init__(self, db: Database, gpu_pool: GpuPool) -> None:
        self.db = db
        self.gpu_pool = gpu_pool
        self._running: dict[str, RunningProcess] = {}
        self._monitors: dict[str, asyncio.Task] = {}
        self._launches: set[asyncio.Task] = set()
        self._stop_event = asyncio.Event()
        self._main_task: asyncio.Task | None = None
        self._dispatch_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self.db.connect() as conn:
            # Recover from a crash: requeue 'running' rows BEFORE syncing
            # leases, so sync_leases can free the leases those experiments
            # held. Reset queued_at to now so old crashed experiments don't
            # starve newer submissions in the FIFO tie-break.
            now = _utcnow_iso()
            await conn.execute(
                "UPDATE experiments SET status = 'queued', queued_at = ?, "
                "started_at = NULL, pid = NULL, gpu_ids = NULL, "
                "mlflow_run_id = NULL, mlflow_experiment_id = NULL, "
                "log_path = NULL WHERE status = 'running'",
                (now,),
            )
            await conn.commit()
            await self.gpu_pool.sync_leases(conn)
        self._main_task = asyncio.create_task(self._loop(), name="trainpipe-scheduler")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._main_task is not None:
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        # Wait for in-flight launches to finish (or fail) so their cleanup
        # paths can run before we tear down running processes.
        if self._launches:
            await asyncio.gather(*self._launches, return_exceptions=True)
        for rp in list(self._running.values()):
            await rp.cancel()
        for task in list(self._monitors.values()):
            try:
                await task
            except Exception:
                logger.exception("monitor task raised during shutdown")

    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment by sending SIGTERM. Returns True if found."""
        rp = self._running.get(experiment_id)
        if rp is None:
            return False
        await rp.cancel()
        return True

    async def _loop(self) -> None:
        logger.info("Scheduler started (gpus=%s)", self.gpu_pool.indices)
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("scheduler tick failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=settings.poll_interval_sec
                )
            except asyncio.TimeoutError:
                pass
        logger.info("Scheduler stopped")

    async def _tick(self) -> None:
        if self.gpu_pool.total == 0:
            return
        # Claim as many experiments as the pool allows in one tick. Each claim
        # is a short, atomic DB transaction; the slow launch (MLflow +
        # subprocess) runs as a background task so MLflow latency never
        # serializes claims.
        while not self._stop_event.is_set():
            claim = await self._claim_next()
            if claim is None:
                return
            task = asyncio.create_task(
                self._launch(*claim),
                name=f"trainpipe-launch-{claim[0]}",
            )
            self._launches.add(task)
            task.add_done_callback(self._launches.discard)

    async def _claim_next(
        self,
    ) -> tuple[str, ExperimentSpec, list[int], str | None, int | None] | None:
        """Atomically claim the next dispatchable queued experiment.

        Returns (experiment_id, spec, gpu_indices, study_id, trial_number) or
        None if nothing is currently dispatchable.
        """
        async with self._dispatch_lock:
            async with self.db.connect() as conn:
                cur = await conn.execute(
                    "SELECT id, spec_json, study_id, trial_number FROM experiments "
                    "WHERE status = 'queued' "
                    "ORDER BY priority DESC, queued_at ASC"
                )
                queued = await cur.fetchall()
                for row in queued:
                    exp_id = row[0]
                    spec = ExperimentSpec.model_validate_json(row[1])
                    study_id = row[2]
                    trial_number = row[3]

                    if spec.gpu_count > self.gpu_pool.total:
                        await conn.execute(
                            "UPDATE experiments SET status = 'failed', error = ?, "
                            "finished_at = ? WHERE id = ? AND status = 'queued'",
                            (
                                f"gpu_count={spec.gpu_count} exceeds pool size "
                                f"{self.gpu_pool.total}",
                                _utcnow_iso(),
                                exp_id,
                            ),
                        )
                        await conn.commit()
                        continue

                    gpu_indices = await self.gpu_pool.try_allocate(
                        conn, spec.gpu_count, exp_id
                    )
                    if gpu_indices is None:
                        continue

                    # Atomic claim: only succeeds if the row is still 'queued'
                    # (i.e. nobody cancelled it between our SELECT and now).
                    now = _utcnow_iso()
                    cur2 = await conn.execute(
                        "UPDATE experiments SET status = 'running', started_at = ?, "
                        "gpu_ids = ?, last_heartbeat_at = ? "
                        "WHERE id = ? AND status = 'queued'",
                        (now, json.dumps(gpu_indices), now, exp_id),
                    )
                    if cur2.rowcount == 0:
                        # Lost a race against cancellation.
                        await self.gpu_pool.release(conn, exp_id)
                        await conn.commit()
                        continue
                    await conn.commit()
                    return exp_id, spec, gpu_indices, study_id, trial_number
                return None

    async def _launch(
        self,
        experiment_id: str,
        spec: ExperimentSpec,
        gpu_indices: list[int],
        study_id: str | None,
        trial_number: int | None,
    ) -> None:
        """Slow path: create MLflow run, spawn subprocess. Runs outside the lock."""
        mlflow_experiment_name = spec.tags.get("mlflow_experiment") or "default"
        run_name = spec.name or f"exp-{experiment_id[:8]}"

        try:
            mlflow_exp_id, mlflow_run_id = await asyncio.to_thread(
                _create_mlflow_run,
                mlflow_experiment_name,
                run_name,
                experiment_id,
                study_id,
                trial_number,
                spec.tags,
            )
        except Exception as e:
            logger.exception("failed to create MLflow run for %s", experiment_id)
            await self._abort_launch(experiment_id, study_id, None, str(e))
            return

        output_dir = (
            Path(spec.output_dir)
            if spec.output_dir
            else settings.output_base_dir / experiment_id
        )
        log_path = settings.logs_dir / f"{experiment_id}.log"

        argv, env = build_swift_command(spec, gpu_indices, output_dir)
        env["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
        env["MLFLOW_RUN_ID"] = mlflow_run_id
        env["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name

        try:
            rp = await spawn_training_subprocess(experiment_id, argv, env, log_path)
        except Exception as e:
            # Was previously only FileNotFoundError; PermissionError / mkdir
            # failure / EAGAIN would have leaked GPU leases.
            logger.exception("failed to spawn subprocess for %s", experiment_id)
            await self._abort_launch(experiment_id, study_id, mlflow_run_id, str(e))
            return

        self._running[experiment_id] = rp
        async with self.db.connect() as conn:
            await conn.execute(
                "UPDATE experiments SET mlflow_run_id = ?, mlflow_experiment_id = ?, "
                "log_path = ?, pid = ? WHERE id = ?",
                (
                    mlflow_run_id,
                    mlflow_exp_id,
                    str(log_path),
                    rp.pid,
                    experiment_id,
                ),
            )
            await repository.log_event(
                conn,
                experiment_id=experiment_id,
                study_id=study_id,
                kind="started",
                payload={"gpu_ids": gpu_indices, "pid": rp.pid},
            )
            await conn.commit()

        self._monitors[experiment_id] = asyncio.create_task(
            self._monitor(experiment_id, rp, study_id, mlflow_run_id),
            name=f"trainpipe-monitor-{experiment_id}",
        )
        logger.info(
            "dispatched experiment=%s gpus=%s pid=%s",
            experiment_id,
            gpu_indices,
            rp.pid,
        )

    async def _abort_launch(
        self,
        experiment_id: str,
        study_id: str | None,
        mlflow_run_id: str | None,
        error: str,
    ) -> None:
        """Cleanup when launch fails after claim: release lease, mark failed."""
        async with self.db.connect() as conn:
            await conn.execute(
                "UPDATE experiments SET status = 'failed', finished_at = ?, error = ? "
                "WHERE id = ? AND status IN ('queued', 'running')",
                (_utcnow_iso(), error, experiment_id),
            )
            await repository.log_event(
                conn,
                experiment_id=experiment_id,
                study_id=study_id,
                kind="failed",
                payload={"error": error},
            )
            await self.gpu_pool.release(conn, experiment_id)
            await conn.commit()
        if mlflow_run_id is not None:
            await asyncio.to_thread(_terminate_mlflow_run, mlflow_run_id, "FAILED")

    async def _monitor(
        self,
        experiment_id: str,
        rp: RunningProcess,
        study_id: str | None,
        mlflow_run_id: str,
    ) -> None:
        try:
            return_code = await rp.wait()
            if rp.cancelled:
                status = "cancelled"
                mlflow_status = "KILLED"
                error: str | None = None
            elif return_code == 0:
                status = "completed"
                mlflow_status = "FINISHED"
                error = None
            else:
                status = "failed"
                mlflow_status = "FAILED"
                error = f"swift exited with code {return_code}"

            async with self.db.connect() as conn:
                await conn.execute(
                    "UPDATE experiments SET status = ?, finished_at = ?, error = ? "
                    "WHERE id = ?",
                    (status, _utcnow_iso(), error, experiment_id),
                )
                await repository.log_event(
                    conn,
                    experiment_id=experiment_id,
                    study_id=study_id,
                    kind=status,
                    payload={"return_code": return_code},
                )
                await self.gpu_pool.release(conn, experiment_id)
                await conn.commit()

            await asyncio.to_thread(_terminate_mlflow_run, mlflow_run_id, mlflow_status)
            logger.info(
                "experiment=%s finished status=%s rc=%s",
                experiment_id,
                status,
                return_code,
            )
        except Exception:
            logger.exception("monitor crashed for %s", experiment_id)
        finally:
            self._running.pop(experiment_id, None)
            self._monitors.pop(experiment_id, None)


def _create_mlflow_run(
    experiment_name: str,
    run_name: str,
    experiment_id: str,
    study_id: str | None,
    trial_number: int | None,
    user_tags: dict[str, Any],
) -> tuple[str, str]:
    """Synchronous helper (called via to_thread) to create an MLflow run."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    existing = client.get_experiment_by_name(experiment_name)
    mlflow_experiment_id = (
        existing.experiment_id if existing else client.create_experiment(experiment_name)
    )

    tags = {
        "trainpipe.experiment_id": experiment_id,
        "trainpipe.study_id": study_id or "",
        "trainpipe.trial_number": "" if trial_number is None else str(trial_number),
    }
    for k, v in user_tags.items():
        tags[f"user.{k}"] = v
    run = client.create_run(experiment_id=mlflow_experiment_id, run_name=run_name, tags=tags)
    return mlflow_experiment_id, run.info.run_id


def _terminate_mlflow_run(run_id: str, status: str) -> None:
    """Best-effort finalize an MLflow run. Status: FINISHED | FAILED | KILLED."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        MlflowClient().set_terminated(run_id, status=status)
    except Exception:
        logger.exception("failed to terminate mlflow run %s", run_id)
