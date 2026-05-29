"""Watch manager — periodic poll → fire pipelines.

Lightweight design: one async loop that ticks every ``poll_interval``,
walks the enabled watches, and fires anything whose trigger condition is
met. No external scheduler (APScheduler) dependency — for the simple
periodic + metric-threshold cases that's overkill.

Firing means: spawn the watch's stored ``PipelineConfig`` via the
already-existing :class:`PipelineManager`. The fired pipeline ID is
written back to the watch row so the UI can link to it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from ..api.schemas import Watch
from ..core import repository
from ..core.db import Database
from ..pipelines.manager import PipelineManager

logger = logging.getLogger(__name__)


class WatchManager:
    def __init__(
        self,
        db: Database,
        pipeline_manager: PipelineManager,
        *,
        poll_interval_sec: float = 30.0,
    ) -> None:
        self.db = db
        self.pipelines = pipeline_manager
        self.poll_interval_sec = poll_interval_sec
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._loop(), name="trainpipe-watches"
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except RuntimeError as e:
                logger.warning("watches stop crossed loops: %s", e)

    async def _loop(self) -> None:
        logger.info("watches manager started")
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("watches tick failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.poll_interval_sec
                )
            except asyncio.TimeoutError:
                pass
        logger.info("watches manager stopped")

    async def _tick(self) -> None:
        async with self.db.connect() as conn:
            watches = await repository.list_watches(conn, only_enabled=True)
        for watch in watches:
            try:
                if await self._should_fire(watch):
                    await self._fire(watch)
            except Exception:
                logger.exception("watches: error processing %s", watch.id)

    async def _should_fire(self, watch: Watch) -> bool:
        if watch.kind == "interval":
            if watch.interval_seconds is None:
                return False
            if watch.last_fired_at is None:
                return True
            now = datetime.now(timezone.utc)
            elapsed = (now - watch.last_fired_at).total_seconds()
            return elapsed >= watch.interval_seconds
        if watch.kind == "metric_threshold":
            return await self._metric_below_threshold(watch)
        return False

    async def _metric_below_threshold(self, watch: Watch) -> bool:
        if (
            not watch.suite_id
            or not watch.metric_name
            or watch.threshold is None
        ):
            return False
        async with self.db.connect() as conn:
            # Take the most recent completed eval run against the suite,
            # optionally filtered by the watched model's experiments.
            runs = await repository.list_eval_runs(
                conn, suite_id=watch.suite_id, limit=10
            )
        for run in runs:
            if run.aggregate is None:
                continue
            agg = run.aggregate.get(watch.metric_name)
            if agg is None:
                continue
            if agg.mean < watch.threshold:
                # Only fire once per dip — don't refire if the same
                # eval is still the most recent.
                if watch.last_fired_at and run.created_at <= watch.last_fired_at:
                    return False
                return True
        return False

    async def _fire(self, watch: Watch) -> None:
        logger.info("watches: firing %s (%s)", watch.name, watch.kind)
        try:
            pipeline_id = await self.pipelines.create_and_start(
                f"watch:{watch.name}", watch.pipeline_config
            )
        except Exception:
            logger.exception(
                "watches: failed to spawn pipeline for watch=%s", watch.id
            )
            return
        async with self.db.connect() as conn:
            await repository.record_watch_fire(conn, watch.id, pipeline_id)
