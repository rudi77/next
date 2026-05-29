"""EvalDispatcher — background loop that drains the queued eval_runs table.

Mirrors the experiments :class:`~trainpipe.scheduler.loop.Scheduler` but
much smaller: an eval run takes 1 GPU (or 0 on dev hosts without a pool),
no MLflow run is created, no subprocess is spawned (the inference backend
runs in-process).

Crash recovery: on start, any ``'running'`` rows are flipped back to
``'queued'`` so the dispatcher re-claims them after restart. GPU leases
are released by the shared :meth:`GpuPool.sync_leases` call in the
:class:`Scheduler` startup, which now exempts running eval_runs.
"""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from ..api.schemas import EvalRun, EvalRunStatus, EvalSuite, ExperimentSpec
from ..core import repository
from ..core.db import Database
from ..scheduler.gpu_pool import GpuPool
from ..settings import settings
from .inference import (
    InferenceBackend,
    MockInferenceBackend,
    TransformersInferenceBackend,
)
from .runner import EvalDriver

logger = logging.getLogger(__name__)


BackendFactory = Callable[[EvalRun, EvalSuite], InferenceBackend]


def _default_backend_factory(
    base_model: str | None,
    adapter_path: Path | None,
    gpu_indices: list[int],
) -> InferenceBackend:
    """Production default: load via transformers + peft.

    Fallback if no base_model can be determined: a no-op mock that returns
    empty strings. This keeps the dispatcher graceful when a misconfigured
    experiment is targeted, rather than crashing the whole loop.
    """
    if not base_model:
        logger.warning(
            "no base_model resolved for eval target; using empty-response backend"
        )
        return MockInferenceBackend(default_response="")
    return TransformersInferenceBackend(
        base_model=base_model,
        adapter_path=adapter_path,
        gpu_indices=gpu_indices,
    )


class EvalDispatcher:
    def __init__(
        self,
        db: Database,
        gpu_pool: GpuPool,
        *,
        backend_factory: BackendFactory | None = None,
        gpus_per_run: int = 1,
        poll_interval_sec: float | None = None,
    ) -> None:
        self.db = db
        self.gpu_pool = gpu_pool
        self._backend_factory = backend_factory or self._default_factory
        self._gpus_per_run = max(0, gpus_per_run)
        self._poll = poll_interval_sec or settings.poll_interval_sec
        self._stop = asyncio.Event()
        self._main_task: asyncio.Task | None = None
        self._active: dict[str, asyncio.Task] = {}
        self._dispatch_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self.db.connect() as conn:
            touched = await repository.recover_eval_runs(conn)
        if touched:
            logger.info("requeued %d eval_runs that were 'running' at startup", touched)
        self._main_task = asyncio.create_task(
            self._loop(), name="trainpipe-eval-dispatcher"
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._main_task is not None:
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        if self._active:
            await asyncio.gather(*self._active.values(), return_exceptions=True)

    async def _loop(self) -> None:
        logger.info(
            "EvalDispatcher started (gpus_per_run=%d, pool_size=%d)",
            self._gpus_per_run,
            self.gpu_pool.total,
        )
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("eval dispatcher tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._poll)
            except asyncio.TimeoutError:
                pass
        logger.info("EvalDispatcher stopped")

    async def _tick(self) -> None:
        while not self._stop.is_set():
            claimed = await self._claim_next()
            if claimed is None:
                return
            run_id, gpu_indices = claimed
            task = asyncio.create_task(
                self._run_one(run_id, gpu_indices),
                name=f"trainpipe-eval-run-{run_id}",
            )
            self._active[run_id] = task
            task.add_done_callback(lambda t, rid=run_id: self._active.pop(rid, None))

    async def _claim_next(self) -> tuple[str, list[int]] | None:
        async with self._dispatch_lock:
            async with self.db.connect() as conn:
                cur = await conn.execute(
                    "SELECT id FROM eval_runs WHERE status = 'queued' "
                    "ORDER BY created_at ASC LIMIT 1"
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                run_id = row[0]

                gpu_indices: list[int] = []
                if self._gpus_per_run > 0 and self.gpu_pool.total > 0:
                    allocated = await self.gpu_pool.try_allocate(
                        conn, self._gpus_per_run, run_id
                    )
                    if allocated is None:
                        return None
                    gpu_indices = allocated

                claimed = await repository.claim_eval_run(conn, run_id)
                if not claimed:
                    if gpu_indices:
                        await self.gpu_pool.release(conn, run_id)
                    return None
                if gpu_indices:
                    await repository.update_eval_run_progress(
                        conn, run_id, gpu_ids=gpu_indices
                    )
                return run_id, gpu_indices

    async def _run_one(self, run_id: str, gpu_indices: list[int]) -> None:
        try:
            async with self.db.connect() as conn:
                run = await repository.get_eval_run(conn, run_id)
                if run is None:
                    logger.error("dispatched eval run=%s disappeared", run_id)
                    return
                suite = await repository.get_eval_suite(conn, run.suite_id)
                base_model = await self._resolve_base_model(conn, run)
            if suite is None:
                async with self.db.connect() as conn:
                    await repository.finalize_eval_run(
                        conn,
                        run_id,
                        status=EvalRunStatus.FAILED,
                        error="suite vanished between claim and execute",
                    )
                return

            backend = self._backend_factory(run, suite)
            if isinstance(backend, TransformersInferenceBackend):
                # Production default needs late-bound GPU + model info.
                backend.gpu_indices = gpu_indices
                if base_model and not backend.base_model:
                    backend.base_model = base_model

            driver = EvalDriver(
                db=self.db, run=run, suite=suite, backend=backend,
            )
            await driver.execute()
        finally:
            if gpu_indices:
                async with self.db.connect() as conn:
                    await self.gpu_pool.release(conn, run_id)

    async def _resolve_base_model(self, conn, run: EvalRun) -> str | None:
        if not run.experiment_id:
            return None
        exp = await repository.get_experiment(conn, run.experiment_id)
        if exp is None:
            return None
        try:
            spec = exp.spec if isinstance(exp.spec, ExperimentSpec) else None
        except Exception:
            spec = None
        return spec.model if spec else None

    def _default_factory(self, run: EvalRun, _suite: EvalSuite) -> InferenceBackend:
        adapter_path: Path | None = None
        if run.experiment_id:
            candidate = settings.output_base_dir / run.experiment_id
            if candidate.exists():
                adapter_path = candidate
        return _default_backend_factory(
            base_model=None,  # set by _run_one once the experiment spec is read
            adapter_path=adapter_path,
            gpu_indices=[],
        )
