"""Tracks all running :class:`PipelineDriver` instances.

Same shape as ``autoresearch.manager.StudyManager`` — start on
``create_and_start``, resume non-terminal pipelines in
``start_existing`` after a restart, ``stop_all`` on shutdown.
"""

from __future__ import annotations

import asyncio
import logging

from ..api.schemas import PipelineConfig, PipelineStatus
from ..core import repository
from ..core.db import Database
from .driver import PipelineDriver

logger = logging.getLogger(__name__)


class PipelineManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._drivers: dict[str, PipelineDriver] = {}
        self._lock = asyncio.Lock()

    async def start_existing(self) -> None:
        async with self.db.connect() as conn:
            actives = await repository.list_active_pipelines(conn)
        for p in actives:
            self._start_driver(p.id)

    async def create_and_start(
        self, name: str, config: PipelineConfig
    ) -> str:
        _validate_dag(config)
        async with self.db.connect() as conn:
            pipeline_id = await repository.create_pipeline(
                conn, name=name, config=config
            )
        self._start_driver(pipeline_id)
        return pipeline_id

    async def cancel(self, pipeline_id: str) -> bool:
        async with self._lock:
            driver = self._drivers.pop(pipeline_id, None)
        if driver is not None:
            await driver.stop()
        async with self.db.connect() as conn:
            pipeline = await repository.get_pipeline(conn, pipeline_id)
            if pipeline is None:
                return False
            if pipeline.status in (
                PipelineStatus.COMPLETED,
                PipelineStatus.FAILED,
                PipelineStatus.CANCELLED,
            ):
                return False
            await repository.set_pipeline_status(
                conn, pipeline_id, status=PipelineStatus.CANCELLED
            )
        return True

    async def stop_all(self) -> None:
        async with self._lock:
            drivers = list(self._drivers.values())
            self._drivers.clear()
        await asyncio.gather(
            *(d.stop() for d in drivers), return_exceptions=True
        )

    def _start_driver(self, pipeline_id: str) -> None:
        driver = PipelineDriver(pipeline_id, self.db)
        driver.start()
        self._drivers[pipeline_id] = driver


def _validate_dag(config: PipelineConfig) -> None:
    """Reject duplicate stage names, dangling dependencies, and cycles.

    Doing this at create-time means the driver never has to deal with
    bad input mid-flight.
    """
    names = [s.name for s in config.stages]
    seen: set[str] = set()
    for n in names:
        if n in seen:
            raise ValueError(f"duplicate stage name: {n!r}")
        seen.add(n)
    valid = set(names)
    for s in config.stages:
        for d in s.depends_on:
            if d not in valid:
                raise ValueError(
                    f"stage {s.name!r}: depends_on references unknown stage {d!r}"
                )
        if s.input_from_stage and s.input_from_stage not in valid:
            raise ValueError(
                f"stage {s.name!r}: input_from_stage references unknown "
                f"stage {s.input_from_stage!r}"
            )
    # Cycle detection via DFS.
    deps = {s.name: list(s.depends_on) for s in config.stages}
    WHITE, GRAY, BLACK = 0, 1, 2
    state = dict.fromkeys(deps, WHITE)

    def visit(n: str) -> None:
        if state[n] == GRAY:
            raise ValueError(f"cycle in stages involving {n!r}")
        if state[n] == BLACK:
            return
        state[n] = GRAY
        for d in deps[n]:
            visit(d)
        state[n] = BLACK

    for n in deps:
        visit(n)
