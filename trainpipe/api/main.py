import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..autoresearch.manager import StudyManager
from ..core.db import Database
from ..scheduler.gpu_pool import GpuPool, detect_gpus
from ..scheduler.loop import Scheduler
from ..settings import settings
from .routes import experiments, gpus, studies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.output_base_dir.mkdir(parents=True, exist_ok=True)

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


app.include_router(experiments.router)
app.include_router(gpus.router)
app.include_router(studies.router)
