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
from ..synth.runner import SynthAborted, SynthProvider, make_provider
from .runner import (
    acquire_records,
    curate,
    intake_spec,
    research_sources,
    synthesize_records,
    write_records_jsonl,
)
from .web import (
    make_extractor,
    make_fetch_gate,
    make_search_provider,
    make_text_fetcher,
)

logger = logging.getLogger(__name__)


class _BudgetProvider(SynthProvider):
    """Wraps a provider to count generate() calls and enforce a per-run cap.

    The phases poll :meth:`exhausted` via their ``should_stop`` hook and stop
    once the budget is reached, so ``max_llm_calls`` bounds total teacher-LLM
    spend across synthesize + acquire. ``max_calls=0`` means unlimited.
    """

    def __init__(self, inner: SynthProvider, max_calls: int) -> None:
        self.inner = inner
        self.name = inner.name
        self.max_calls = max_calls
        self.calls = 0

    def generate(self, prompt: str, *, model: str, max_tokens: int) -> str:
        self.calls += 1
        return self.inner.generate(prompt, model=model, max_tokens=max_tokens)

    def exhausted(self) -> bool:
        return self.max_calls > 0 and self.calls >= self.max_calls


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
            provider = _BudgetProvider(make_provider(run.provider), run.max_llm_calls)

            # budget_stop ends *generating* once the call budget is spent; the
            # inter-phase `self._stop` checks below are raw cancellation only
            # (a spent budget shouldn't abort the pipeline, just cap new calls).
            def budget_stop() -> bool:
                return self._stop.is_set() or provider.exhausted()

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

            # --- Phase 2/3a: Research + Acquire (real web) -----------------
            # Only when a search provider is configured; 'none' stays
            # synth-only with no network. Real records are combined with the
            # synthesized ones below and deduped in curate.
            real_records: list[dict] = []
            if run.search_provider != "none":
                real_records = await self._research_and_acquire(
                    run, spec, provider, budget_stop
                )
                if self._stop.is_set():
                    return

            # --- Phase 3b: Synthesize --------------------------------------
            # target_count is the whole-dataset budget: synthesize only the
            # remainder the web phase didn't already supply, so turning on web
            # research doesn't inflate the output past target_count.
            await self._set(phase="synthesize")
            remaining = max(0, run.target_count - len(real_records))
            synth_spec = spec.model_copy(update={"target_count": remaining})
            records = await asyncio.to_thread(
                synthesize_records,
                provider,
                model=run.model,
                spec=synth_spec,
                answers=run.answers,
                should_stop=budget_stop,
            )
            records = real_records + records
            await self._set(raw_count=len(records))
            if self._stop.is_set():
                return

            # --- Phase 4: Curate (mandatory PII-redaction + dedup) ---------
            await self._set(phase="curate")
            curated, stats = curate(records)
            await self._set(redaction=stats.redaction)
            if stats.dropped or stats.redaction:
                logger.info(
                    "acquisition=%s curate dropped %d dup(s), redacted %d PII hit(s)",
                    self.run_id,
                    stats.dropped,
                    sum(stats.redaction.values()),
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

    async def _research_and_acquire(
        self, run, spec, provider, should_stop
    ) -> list[dict]:
        """Phases 2+3a: find candidate sources, gate them, fetch the allowed
        ones, and distil grounded records. Persists the full source ledger
        (allowed and skipped) with the per-source ``used`` flag."""
        await self._set(phase="research")
        search = make_search_provider(run.search_provider)
        gate = make_fetch_gate(strict_license=run.strict_license)
        sources = await asyncio.to_thread(
            research_sources, search, gate, spec, max_sources=run.max_sources
        )

        await self._set(phase="acquire")
        fetch_text = make_text_fetcher(make_extractor())
        records = await asyncio.to_thread(
            acquire_records,
            provider,
            model=run.model,
            sources=sources,
            spec=spec,
            fetch_text=fetch_text,
            should_stop=should_stop,
        )

        # acquire_records marked each fetched source's ``used`` flag in place.
        async with self.db.connect() as conn:
            for src in sources:
                await repository.record_acquisition_source(
                    conn,
                    run_id=self.run_id,
                    url=src.url,
                    title=src.title,
                    topic=src.topic,
                    license_status=src.license_status,
                    used=src.used,
                )
        logger.info(
            "acquisition=%s acquired %d real record(s) from %d/%d source(s)",
            self.run_id,
            len(records),
            sum(1 for s in sources if s.used),
            len(sources),
        )
        return records

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
            f"({len(records)} records, format={spec.format}; "
            f"acquisition_run={self.run_id})"
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
