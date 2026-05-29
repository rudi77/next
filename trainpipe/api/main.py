from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..core.db import Database
from ..settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.output_base_dir.mkdir(parents=True, exist_ok=True)

    db = Database(settings.sqlite_path)
    await db.init()
    app.state.db = db
    yield


app = FastAPI(title="trainpipe", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
