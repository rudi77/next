import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ..autoresearch.manager import StudyManager
from ..core.db import Database
from ..scheduler.gpu_pool import GpuPool, detect_gpus
from ..scheduler.loop import Scheduler
from ..settings import settings
from .routes import datasets, experiments, gpus, studies

_UI_INDEX = Path(__file__).resolve().parent.parent / "ui" / "index.html"

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

    app.state.db = db
    app.state.gpu_pool = gpu_pool
    app.state.scheduler = scheduler
    app.state.study_manager = study_manager

    try:
        yield
    finally:
        await study_manager.stop_all()
        await scheduler.stop()


app = FastAPI(title="trainpipe", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui/config")
async def ui_config() -> dict[str, str]:
    """Frontend-safe config (no secrets). Used by the SPA on startup."""
    return {"mlflow_tracking_uri": settings.mlflow_tracking_uri}


@app.get("/", include_in_schema=False)
async def ui_root() -> FileResponse:
    return FileResponse(_UI_INDEX, media_type="text/html")


app.include_router(experiments.router)
app.include_router(gpus.router)
app.include_router(studies.router)
app.include_router(datasets.router)
