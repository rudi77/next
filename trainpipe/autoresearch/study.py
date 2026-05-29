"""Optuna-driven study runner.

Each StudyDriver owns one asyncio task that drives an Optuna study end-to-end:
ask a trial → enqueue a new experiment row → poll until terminal → read the
target metric from MLflow → tell Optuna. Up to ``config.max_concurrent`` trials
run in parallel via a semaphore.

Optuna's storage is a per-study SQLite file under ``data/studies/``; this lets
us resume on API restart without rebuilding state.
"""

import asyncio
import logging
from typing import Any

from ..api.schemas import ExperimentSpec, ExperimentStatus, StudyConfig, StudyStatus
from ..core import repository
from ..core.db import Database
from ..settings import settings
from .search_spaces import sample_spec

logger = logging.getLogger(__name__)

_TERMINAL = {
    ExperimentStatus.COMPLETED,
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
}


def _build_sampler(name: str):
    import optuna

    if name == "tpe":
        return optuna.samplers.TPESampler()
    if name == "random":
        return optuna.samplers.RandomSampler()
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler()
    raise ValueError(f"unknown sampler: {name}")


class StudyDriver:
    def __init__(
        self,
        study_id: str,
        config: StudyConfig,
        optuna_storage: str,
        db: Database,
    ) -> None:
        self.study_id = study_id
        self.config = config
        self.optuna_storage = optuna_storage
        self.db = db
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run(), name=f"trainpipe-study-{self.study_id}"
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        import optuna

        logger.info("study=%s driver starting", self.study_id)
        try:
            study = await asyncio.to_thread(
                optuna.create_study,
                study_name=self.config.name,
                storage=self.optuna_storage,
                load_if_exists=True,
                direction=self.config.direction,
                sampler=_build_sampler(self.config.sampler),
            )

            # If a prior process crashed mid-trial, Optuna keeps those trials
            # in RUNNING state forever and ask() would issue fresh trial
            # numbers in parallel, creating duplicate experiment rows. Fail
            # the orphans so the study state is consistent.
            await self._reconcile_pending_trials(study)

            sem = asyncio.Semaphore(self.config.max_concurrent)
            target = self.config.n_trials
            launched = 0
            pending: set[asyncio.Task] = set()

            while not self._stop.is_set():
                if target is not None and launched >= target:
                    break
                await sem.acquire()
                if self._stop.is_set():
                    sem.release()
                    break
                launched += 1
                task = asyncio.create_task(
                    self._run_trial(study, sem),
                    name=f"trainpipe-trial-{self.study_id}-{launched}",
                )
                pending.add(task)
                task.add_done_callback(pending.discard)

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            async with self.db.connect() as conn:
                await self._update_progress(study, conn)
                if target is not None and launched >= target:
                    await repository.set_study_status(
                        conn, self.study_id, StudyStatus.COMPLETED
                    )
            logger.info("study=%s driver finished", self.study_id)
        except Exception:
            logger.exception("study=%s driver crashed", self.study_id)
            try:
                async with self.db.connect() as conn:
                    await repository.set_study_status(
                        conn, self.study_id, StudyStatus.FAILED
                    )
            except Exception:
                logger.exception("study=%s failed to mark study failed", self.study_id)

    async def _run_trial(self, study, sem: asyncio.Semaphore) -> None:
        import optuna

        try:
            trial = await asyncio.to_thread(study.ask)
            try:
                spec, sampled = sample_spec(
                    trial, self.config.base_spec, self.config.search_space
                )
            except Exception:
                logger.exception(
                    "study=%s trial=%s spec sampling failed",
                    self.study_id,
                    trial.number,
                )
                await asyncio.to_thread(
                    study.tell, trial, state=optuna.trial.TrialState.FAIL
                )
                return

            async with self.db.connect() as conn:
                exp_id = await repository.create_experiment(
                    conn,
                    spec,
                    study_id=self.study_id,
                    trial_number=trial.number,
                )
            await asyncio.to_thread(trial.set_user_attr, "experiment_id", exp_id)

            metric_value = await self._wait_for_metric(exp_id)

            if metric_value is None:
                await asyncio.to_thread(
                    study.tell, trial, state=optuna.trial.TrialState.FAIL
                )
            else:
                await asyncio.to_thread(study.tell, trial, metric_value)

            async with self.db.connect() as conn:
                await self._update_progress(study, conn)
        finally:
            sem.release()

    async def _wait_for_metric(self, exp_id: str) -> float | None:
        while not self._stop.is_set():
            async with self.db.connect() as conn:
                rec = await repository.get_experiment(conn, exp_id)
            if rec is None:
                return None
            if rec.status in _TERMINAL:
                if rec.status != ExperimentStatus.COMPLETED:
                    return None
                if rec.mlflow_run_id is None:
                    return None
                return await asyncio.to_thread(
                    _read_metric, rec.mlflow_run_id, self.config.target_metric
                )
            await asyncio.sleep(2.0)
        return None

    async def _update_progress(self, study, conn) -> None:
        # Optuna's .best_trial and .trials are sync DB-backed property
        # accessors. For large studies on slow disks they can block the loop
        # for hundreds of milliseconds, so push them to a thread.
        best_value, best_trial_id, n_completed = await asyncio.to_thread(
            _summarize_study, study
        )
        await repository.update_study_progress(
            conn,
            self.study_id,
            n_completed=n_completed,
            best_value=best_value,
            best_trial_id=best_trial_id,
        )

    async def _reconcile_pending_trials(self, study) -> None:
        import optuna

        def collect_pending():
            return [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.RUNNING
            ]

        pending = await asyncio.to_thread(collect_pending)
        for t in pending:
            try:
                await asyncio.to_thread(
                    study.tell, t, state=optuna.trial.TrialState.FAIL
                )
                logger.warning(
                    "study=%s reconciled orphaned trial=%s as FAIL "
                    "(experiment_id=%s)",
                    self.study_id,
                    t.number,
                    t.user_attrs.get("experiment_id"),
                )
            except Exception:
                logger.exception(
                    "study=%s could not reconcile trial=%s",
                    self.study_id,
                    t.number,
                )


def _summarize_study(study) -> tuple[float | None, str | None, int]:
    """Read best_trial + completed-count off the Optuna study. Sync, blocks I/O."""
    import optuna

    try:
        best = study.best_trial
        best_value = best.value
        best_trial_id = best.user_attrs.get("experiment_id")
    except ValueError:
        best_value = None
        best_trial_id = None
    n_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    return best_value, best_trial_id, n_completed


def _read_metric(mlflow_run_id: str, metric_name: str) -> float | None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        run = client.get_run(mlflow_run_id)
        return run.data.metrics.get(metric_name)
    except Exception:
        logger.exception(
            "failed to read metric %s from mlflow run %s", metric_name, mlflow_run_id
        )
        return None
