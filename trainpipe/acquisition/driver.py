"""Acquisition driver — runs one acquisition run's phases in-process.

Shape mirrors :class:`pipelines.driver.PipelineDriver` (one async task per
run, owned by a manager, cancellable, resumable from DB state) but the work
runs *here* rather than being dispatched as experiments: intake →
(research/acquire — MVP no-op) → synthesize → curate → register.

The run row is the state machine. On (re)start the driver reads it and skips
whatever is already done: a row with ``spec`` set has finished intake, so a
restart after the ``awaiting_input`` pause resumes straight at synthesize.

Terminality: the driver writes COMPLETED on success and FAILED on error, but
on cancellation it bails at a phase boundary *without* a terminal write — the
manager owns the CANCELLED transition (same split as the pipeline driver).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid

from ..api.schemas import AcquisitionSpec, AcquisitionStatus
from ..core import repository
from ..core.db import Database
from ..settings import settings
from ..synth.runner import SynthAborted, make_provider
from .runner import curate, intake_spec, synthesize_records, write_records_jsonl

logger = logging.getLogger(__name__)


class AcquisitionDriver:
    """Runs one acquisition run. Owned by :class:`AcquisitionManager`."""

    def __init__(self, run_id: str, db: Database) -> None:
        self.run_id = run_id
        self.db = db
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run(), name=f"trainpipe-acquisition-{self.run_id}"
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
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
                logger.warning("acquisition stop crossed loops: %s", e)

    async def _run(self) -> None:
        logger.info("acquisition=%s driver started", self.run_id)
        try:
            async with self.db.connect() as conn:
                run = await repository.get_acquisition_run(conn, self.run_id)
            if run is None:
                return
            await self._set(status=AcquisitionStatus.RUNNING)
            provider = make_provider(run.provider)

            # --- Phase 1: Intake -------------------------------------------
            spec = run.spec
            if spec is None:
                await self._set(phase="intake")
                spec = await asyncio.to_thread(
                    intake_spec,
                    provider,
                    model=run.model,
                    brief=run.brief,
                    target_count=run.target_count,
                )
                if spec.open_questions and not run.answers:
                    # Park on the operator — async answer path resumes us.
                    await self._set(spec=spec, status=AcquisitionStatus.AWAITING_INPUT)
                    logger.info(
                        "acquisition=%s parked: %d open question(s)",
                        self.run_id,
                        len(spec.open_questions),
                    )
                    return
                await self._set(spec=spec)

            if self._stop.is_set():
                return

            # --- Phase 2/3a: Research / Acquire (real web) -----------------
            # MVP: no real web acquisition yet. Phase markers only so the
            # progress surface is honest about what ran.
            await self._set(phase="research")

            # --- Phase 3b: Synthesize --------------------------------------
            await self._set(phase="synthesize")
            records = await asyncio.to_thread(
                synthesize_records,
                provider,
                model=run.model,
                spec=spec,
                answers=run.answers,
                should_stop=self._stop.is_set,
            )
            await self._set(raw_count=len(records))
            if self._stop.is_set():
                return

            # --- Phase 4: Curate -------------------------------------------
            await self._set(phase="curate")
            curated, dropped = curate(records)
            if dropped:
                logger.info(
                    "acquisition=%s curate dropped %d duplicate(s)",
                    self.run_id,
                    dropped,
                )
            if not curated:
                raise ValueError(
                    "no records survived curation — check provider/model"
                )

            # --- Phase 5: Register -----------------------------------------
            await self._set(phase="register")
            dataset_id = await self._register(
                run.name, run.provider, run.model, spec, curated
            )
            await self._set(
                status=AcquisitionStatus.COMPLETED,
                dataset_id=dataset_id,
                final_count=len(curated),
            )
            logger.info(
                "acquisition=%s completed: ds=%s (%d records)",
                self.run_id,
                dataset_id,
                len(curated),
            )
        except asyncio.CancelledError:
            # Manager-driven cancel — it writes CANCELLED. Re-raise so the
            # task tree unwinds cleanly.
            raise
        except SynthAborted as e:
            logger.warning("acquisition=%s aborted: %s", self.run_id, e)
            await self._set(status=AcquisitionStatus.FAILED, error=str(e))
        except Exception as e:
            logger.exception("acquisition=%s driver crashed", self.run_id)
            await self._set(
                status=AcquisitionStatus.FAILED, error=str(e) or type(e).__name__
            )
        finally:
            logger.info("acquisition=%s driver exit", self.run_id)

    async def _register(
        self,
        name: str,
        provider: str,
        model: str,
        spec: AcquisitionSpec,
        records: list[dict],
    ) -> str:
        """Write the curated JSONL and register it as a dataset.

        Mirrors ``routes/synth.py``: sha256 dedup against existing datasets,
        format validation, provenance description. Runs the (small) file +
        hash work inline — these aren't network calls.
        """
        from ..training.dataset_formats import detect_and_validate_info

        dataset_id = uuid.uuid4().hex
        target_dir = settings.datasets_dir / dataset_id
        target_path = target_dir / f"{name}.jsonl"
        write_records_jsonl(records, target_path)

        sha = hashlib.sha256()
        with target_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
        size = target_path.stat().st_size
        info = detect_and_validate_info(target_path)

        provenance = (
            f"acquired via {provider}:{model} for domain={spec.domain!r} "
            f"({len(records)} records, format={spec.format})"
        )

        async with self.db.connect() as conn:
            existing = await repository.get_dataset_by_sha(conn, digest)
            if existing is not None:
                # Identical content already registered — drop our copy and
                # point the run at the existing dataset.
                try:
                    target_path.unlink(missing_ok=True)
                    target_dir.rmdir()
                except OSError:
                    pass
                return existing.id
            await repository.create_dataset(
                conn,
                name=name,
                path=str(target_path),
                fmt=info.format,
                size_bytes=size,
                sha256=digest,
                line_count=info.line_count,
                description=provenance,
                dataset_id=dataset_id,
                media_kinds=info.media_kinds,
            )
        return dataset_id

    async def _set(self, **kwargs) -> None:
        async with self.db.connect() as conn:
            await repository.update_acquisition_run(conn, self.run_id, **kwargs)
