"""Pipeline driver — runs one Pipeline's stages in dependency order.

Mirrors :class:`autoresearch.study.StudyDriver` in shape: one async
task watching a sequence of dependent experiments, polling DB state,
moving forward when prerequisites finish. Differences:

* Dependencies are an explicit DAG (``StageSpec.depends_on``) rather
  than "one trial at a time".
* Each stage produces an adapter dir (``ExperimentSpec.output_dir``)
  that downstream stages consume by rewriting ``base_spec.model`` to
  point at that path.

Failure model: if any stage fails, all not-yet-started downstream
stages flip to ``skipped`` and the pipeline status goes ``failed``.
Stages already running when the pipeline is cancelled are left to
finish on the regular scheduler — we just stop dispatching new ones.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..api.schemas import (
    ExperimentSpec,
    ExperimentStatus,
    PipelineStatus,
    StageStatus,
)
from ..core import repository
from ..core.db import Database
from ..settings import settings

logger = logging.getLogger(__name__)


class PipelineDriver:
    """Runs one pipeline. Owned by :class:`PipelineManager`."""

    def __init__(self, pipeline_id: str, db: Database) -> None:
        self.pipeline_id = pipeline_id
        self.db = db
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run(), name=f"trainpipe-pipeline-{self.pipeline_id}"
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            # Best-effort cancel: if the task was created on a different
            # loop (test harness), we can't await it but we can at least
            # request cancellation so it exits cleanly on next tick.
            if not self._task.done():
                try:
                    self._task.cancel()
                except RuntimeError:
                    pass
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except RuntimeError as e:
                logger.warning("pipeline stop crossed loops: %s", e)

    async def _run(self) -> None:
        logger.info("pipeline=%s driver started", self.pipeline_id)
        try:
            async with self.db.connect() as conn:
                pipeline = await repository.get_pipeline(conn, self.pipeline_id)
                if pipeline is None:
                    return
                await repository.set_pipeline_status(
                    conn, self.pipeline_id, status=PipelineStatus.RUNNING
                )

            await self._loop()
        except Exception as e:
            logger.exception(
                "pipeline=%s driver crashed", self.pipeline_id
            )
            async with self.db.connect() as conn:
                await repository.set_pipeline_status(
                    conn,
                    self.pipeline_id,
                    status=PipelineStatus.FAILED,
                    error=str(e),
                )
        finally:
            logger.info("pipeline=%s driver exit", self.pipeline_id)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            async with self.db.connect() as conn:
                pipeline = await repository.get_pipeline(conn, self.pipeline_id)
            if pipeline is None:
                return

            # Index stages by name; map dependency lookups.
            by_name = {s.stage_name: s for s in pipeline.stages}

            # Mark downstream stages skipped if any dependency failed.
            failed_names = {
                s.stage_name
                for s in pipeline.stages
                if s.status == StageStatus.FAILED
            }
            for s in pipeline.stages:
                if s.status not in (StageStatus.PENDING,):
                    continue
                if any(d in failed_names for d in s.depends_on):
                    async with self.db.connect() as conn:
                        await repository.update_stage(
                            conn,
                            self.pipeline_id,
                            s.stage_name,
                            status=StageStatus.SKIPPED,
                            error="upstream stage failed",
                        )

            # Pipeline-level terminality check.
            statuses = {s.status for s in pipeline.stages}
            if statuses.issubset(
                {StageStatus.COMPLETED, StageStatus.SKIPPED, StageStatus.FAILED}
            ):
                final = (
                    PipelineStatus.COMPLETED
                    if statuses == {StageStatus.COMPLETED}
                    else PipelineStatus.FAILED
                )
                async with self.db.connect() as conn:
                    await repository.set_pipeline_status(
                        conn,
                        self.pipeline_id,
                        status=final,
                        error="one or more stages failed"
                        if final == PipelineStatus.FAILED
                        else None,
                    )
                return

            # Find pending stages whose dependencies are all completed →
            # enqueue them as experiments. Order doesn't matter since
            # the scheduler claims FIFO on its own.
            stage_specs_by_name = {
                s.name: s for s in pipeline.config.stages
            }
            for stage in pipeline.stages:
                if self._stop.is_set():
                    return
                if stage.status != StageStatus.PENDING:
                    continue
                if not all(
                    by_name[d].status == StageStatus.COMPLETED
                    for d in stage.depends_on
                ):
                    continue
                spec_template = stage_specs_by_name[stage.stage_name]
                materialized = self._materialize_stage_spec(
                    spec_template, by_name
                )
                async with self.db.connect() as conn:
                    exp_id = await repository.enqueue_stage_with_experiment(
                        conn,
                        pipeline_id=self.pipeline_id,
                        stage_name=stage.stage_name,
                        spec=materialized,
                        output_dir=materialized.output_dir,
                    )
                logger.info(
                    "pipeline=%s enqueued stage=%s as exp=%s",
                    self.pipeline_id,
                    stage.stage_name,
                    exp_id,
                )

            # Poll experiments for terminality of any RUNNING/QUEUED stages.
            async with self.db.connect() as conn:
                refreshed = await repository.get_pipeline(conn, self.pipeline_id)
            if refreshed is None:
                return
            for stage in refreshed.stages:
                if stage.experiment_id is None:
                    continue
                if stage.status in (
                    StageStatus.COMPLETED,
                    StageStatus.FAILED,
                    StageStatus.CANCELLED,
                    StageStatus.SKIPPED,
                ):
                    continue
                async with self.db.connect() as conn:
                    exp = await repository.get_experiment(
                        conn, stage.experiment_id
                    )
                if exp is None:
                    continue
                new_stage_status = _experiment_to_stage_status(exp.status)
                if new_stage_status is None:
                    continue
                async with self.db.connect() as conn:
                    await repository.update_stage(
                        conn,
                        self.pipeline_id,
                        stage.stage_name,
                        status=new_stage_status,
                        error=exp.error,
                    )

            # Tick.
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=settings.poll_interval_sec,
                )
            except asyncio.TimeoutError:
                pass

    def _materialize_stage_spec(
        self, stage_spec, by_name: dict
    ) -> ExperimentSpec:
        """Take the user-supplied StageSpec and produce an enqueuable
        ExperimentSpec.

        * Sets ``output_dir`` to a per-stage directory so downstream
          stages know where to look.
        * If ``input_from_stage`` is set, rewrites ``base_spec.model``
          to that upstream stage's ``output_dir``.
        * Tags the spec with ``trainpipe.pipeline_id`` /
          ``trainpipe.pipeline_stage`` so MLflow shows the relationship.
        """
        spec = stage_spec.base_spec.model_copy(deep=True)
        # Each stage's output goes in data/outputs/pipelines/<id>/<stage>
        stage_out = (
            settings.output_base_dir
            / "pipelines"
            / self.pipeline_id
            / stage_spec.name
        )
        spec = spec.model_copy(update={"output_dir": str(stage_out)})

        if stage_spec.input_from_stage:
            upstream = by_name.get(stage_spec.input_from_stage)
            if upstream is None or upstream.output_dir is None:
                raise RuntimeError(
                    f"stage {stage_spec.name}: input_from_stage references "
                    f"{stage_spec.input_from_stage!r}, which has no output_dir"
                )
            # Adapter chaining: ms-swift accepts ``--model <dir>`` for an
            # existing checkpoint. The previous adapter is now the "base"
            # for this stage.
            spec = spec.model_copy(update={"model": upstream.output_dir})

        # Stamp metadata so MLflow can filter on it.
        tags = dict(spec.tags)
        tags["trainpipe.pipeline_id"] = self.pipeline_id
        tags["trainpipe.pipeline_stage"] = stage_spec.name
        spec = spec.model_copy(update={"tags": tags})
        # The driver assumed the path; make sure the actual dir exists
        # so the trainer can write into it.
        Path(stage_out).mkdir(parents=True, exist_ok=True)
        return spec


def _experiment_to_stage_status(
    es: ExperimentStatus,
) -> StageStatus | None:
    if es == ExperimentStatus.RUNNING:
        return StageStatus.RUNNING
    if es == ExperimentStatus.COMPLETED:
        return StageStatus.COMPLETED
    if es == ExperimentStatus.FAILED:
        return StageStatus.FAILED
    if es == ExperimentStatus.CANCELLED:
        return StageStatus.CANCELLED
    return None
