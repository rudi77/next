import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ..autoresearch.manager import StudyManager
from ..core.db import Database
from ..evals.dispatcher import EvalDispatcher
from ..inference.service import InferenceService
from ..pipelines.manager import PipelineManager
from ..scheduler.gpu_pool import GpuPool, detect_gpus
from ..scheduler.loop import Scheduler
from ..settings import settings
from ..watches.manager import WatchManager
from .routes import (
    active_learning,
    compliance,
    datasets,
    evals,
    experiments,
    gpus,
    inferences,
    models,
    pipelines,
    studies,
    synth,
    watches,
)

_UI_INDEX = Path(__file__).resolve().parent.parent / "ui" / "index.html"


def _public_mlflow_uri() -> str:
    """The tracking URI with any embedded user:password stripped.

    ``/ui/config`` is served unauthenticated, so credentials accidentally
    baked into the URI (``http://user:pass@host``) must never leak to the SPA.
    """
    parts = urlsplit(settings.mlflow_tracking_uri)
    if parts.username or parts.password:
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        parts = parts._replace(netloc=netloc)
    return urlunsplit(parts)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.output_base_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)

    db = Database(settings.sqlite_path)
    await db.init()

    detected = detect_gpus(settings.visible_gpus)
    gpu_pool = GpuPool(detected)

    scheduler = Scheduler(db, gpu_pool)
    await scheduler.start()

    study_manager = StudyManager(db)
    await study_manager.start_existing()

    eval_dispatcher = EvalDispatcher(db, gpu_pool)
    await eval_dispatcher.start()

    inference_service = InferenceService(db)

    # Phase 11: clear out any 'running' active-learning rows left by a
    # process restart. AL runs are synchronous-in-request and have no
    # resume path, so an orphan is always failed.
    await active_learning.recover_stale_runs(db)

    pipeline_manager = PipelineManager(db)
    await pipeline_manager.start_existing()

    watch_manager = WatchManager(db, pipeline_manager)
    await watch_manager.start()

    app.state.db = db
    app.state.gpu_pool = gpu_pool
    app.state.scheduler = scheduler
    app.state.study_manager = study_manager
    app.state.eval_dispatcher = eval_dispatcher
    app.state.inference_service = inference_service
    app.state.pipeline_manager = pipeline_manager
    app.state.watch_manager = watch_manager

    try:
        yield
    finally:
        await watch_manager.stop()
        await pipeline_manager.stop_all()
        await inference_service.close_all()
        await eval_dispatcher.stop()
        await study_manager.stop_all()
        await scheduler.stop()


app = FastAPI(title="trainpipe", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui/config")
async def ui_config() -> dict[str, str]:
    """Frontend-safe config (no secrets). Used by the SPA on startup."""
    return {"mlflow_tracking_uri": _public_mlflow_uri()}


@app.get("/", include_in_schema=False)
async def ui_root() -> FileResponse:
    return FileResponse(_UI_INDEX, media_type="text/html")


app.include_router(experiments.router)
app.include_router(gpus.router)
app.include_router(studies.router)
app.include_router(datasets.router)
app.include_router(evals.router)
app.include_router(models.router)
app.include_router(inferences.router)
app.include_router(active_learning.router)
app.include_router(pipelines.router)
app.include_router(synth.router)
app.include_router(watches.router)
app.include_router(compliance.router)
