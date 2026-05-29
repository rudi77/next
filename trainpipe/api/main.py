import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ..autoresearch.manager import StudyManager
from ..core.db import Database
from ..evals.dispatcher import EvalDispatcher
from ..scheduler.gpu_pool import GpuPool, detect_gpus
from ..scheduler.loop import Scheduler
from ..settings import settings
from .routes import datasets, evals, experiments, gpus, studies

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

    app.state.db = db
    app.state.gpu_pool = gpu_pool
    app.state.scheduler = scheduler
    app.state.study_manager = study_manager
    app.state.eval_dispatcher = eval_dispatcher

    try:
        yield
    finally:
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
