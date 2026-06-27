"""Tracks all running :class:`AcquisitionDriver` instances.

Same shape as ``pipelines.manager.PipelineManager`` /
``autoresearch.manager.StudyManager``: start on ``create_and_start``, resume
non-terminal runs in ``start_existing`` after a restart, ``stop_all`` on
shutdown. Adds ``answer`` for the async clarification path — supplying
answers to a parked run flips it back to RUNNING and (re)starts its driver.
"""

from __future__ import annotations

import asyncio
import logging

from ..api.schemas import (
    ACQUISITION_TERMINAL_STATUSES,
    AcquisitionRun,
    AcquisitionSpec,
    AcquisitionStatus,
)
from ..core import repository
from ..core.db import Database
from .driver import AcquisitionDriver

logger = logging.getLogger(__name__)


class AcquisitionManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._drivers: dict[str, AcquisitionDriver] = {}
        self._lock = asyncio.Lock()

    async def start_existing(self) -> None:
        """Resume queued/running runs after a process restart.

        ``awaiting_input`` runs are intentionally not resumed (they're parked
        on the operator). A run that was mid-synthesize when the process died
        re-runs that phase from its persisted spec — synthesize is idempotent
        (it overwrites), so no orphan results leak.
        """
        async with self.db.connect() as conn:
            actives = await repository.list_active_acquisition_runs(conn)
        async with self._lock:
            for run in actives:
                self._start_driver_locked(run.id)

    async def create_and_start(
        self,
        *,
        name: str,
        brief: str,
        provider: str,
        model: str,
        target_count: int,
        search_provider: str = "none",
        max_sources: int = 0,
        spec: AcquisitionSpec | None = None,
    ) -> AcquisitionRun:
        async with self.db.connect() as conn:
            run_id = await repository.create_acquisition_run(
                conn,
                name=name,
                brief=brief,
                provider=provider,
                model=model,
                target_count=target_count,
                search_provider=search_provider,
                max_sources=max_sources,
                spec=spec,
            )
            run = await repository.get_acquisition_run(conn, run_id)
        assert run is not None
        async with self._lock:
            self._start_driver_locked(run_id)
        return run

    async def answer(
        self, run_id: str, answers: dict[str, str]
    ) -> AcquisitionRun | None:
        """Async clarification path: answer a parked run's open questions.

        Returns the resumed run, or ``None`` if it doesn't exist or isn't
        awaiting input. Persists the answers, flips the run back to RUNNING,
        and restarts the driver, which resumes at synthesize (spec is already
        on file).
        """
        async with self.db.connect() as conn:
            run = await repository.get_acquisition_run(conn, run_id)
            if run is None or run.status != AcquisitionStatus.AWAITING_INPUT:
                return None
            await repository.update_acquisition_run(
                conn, run_id, answers=answers, status=AcquisitionStatus.RUNNING
            )
            run = await repository.get_acquisition_run(conn, run_id)
        await self._remove_driver(run_id)
        async with self._lock:
            self._start_driver_locked(run_id)
        return run

    async def cancel(self, run_id: str) -> AcquisitionRun | None:
        """Cancel a run. Returns the run after the attempt (unchanged if it
        was already terminal), or ``None`` if it doesn't exist."""
        await self._remove_driver(run_id)
        async with self.db.connect() as conn:
            run = await repository.get_acquisition_run(conn, run_id)
            if run is None or run.status in ACQUISITION_TERMINAL_STATUSES:
                return run
            await repository.update_acquisition_run(
                conn, run_id, status=AcquisitionStatus.CANCELLED
            )
            return await repository.get_acquisition_run(conn, run_id)

    async def _remove_driver(self, run_id: str) -> None:
        """Pop a run's driver (if any) and await its shutdown. Shared by
        cancel and the answer-restart path so driver teardown lives in one
        place."""
        async with self._lock:
            driver = self._drivers.pop(run_id, None)
        if driver is not None:
            await driver.stop()

    async def stop_all(self) -> None:
        async with self._lock:
            drivers = list(self._drivers.values())
            self._drivers.clear()
        await asyncio.gather(
            *(d.stop() for d in drivers), return_exceptions=True
        )

    def _start_driver_locked(self, run_id: str) -> None:
        """Caller must hold ``self._lock``."""
        if run_id in self._drivers:
            return
        driver = AcquisitionDriver(run_id, self.db)
        driver.start()
        self._drivers[run_id] = driver
