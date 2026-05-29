import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..core.db import Database
from ..scheduler.gpu_pool import GpuPool, detect_gpus
from ..scheduler.loop import Scheduler
from ..settings import settings

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

    gpus = detect_gpus(settings.visible_gpus)
    gpu_pool = GpuPool(gpus)

    scheduler = Scheduler(db, gpu_pool)
    await scheduler.start()

    app.state.db = db
    app.state.gpu_pool = gpu_pool
    app.state.scheduler = scheduler

    try:
        yield
    finally:
        await scheduler.stop()


app = FastAPI(title="trainpipe", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
