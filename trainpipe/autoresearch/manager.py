"""Owns all live StudyDriver instances inside the API process.

Started in the FastAPI lifespan: resumes any study rows with status='active'
that were running prior to a restart, and creates+starts new drivers on
POST /studies.
"""

import asyncio
import logging

from ..api.schemas import StudyConfig, StudyStatus
from ..core import repository
from ..core.db import Database
from ..settings import settings
from .study import StudyDriver

logger = logging.getLogger(__name__)


class StudyManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._drivers: dict[str, StudyDriver] = {}
        self._lock = asyncio.Lock()

    async def start_existing(self) -> None:
        async with self.db.connect() as conn:
            actives = await repository.list_active_studies(conn)
        for rec in actives:
            await self._start_driver(rec.id, rec.config, rec.optuna_storage)

    async def create_and_start(self, config: StudyConfig) -> str:
        async with self.db.connect() as conn:
            study_id = await repository.create_study(conn, config, "pending")
        storage = self._optuna_url(study_id)
        async with self.db.connect() as conn:
            await conn.execute(
                "UPDATE studies SET optuna_storage = ? WHERE id = ?",
                (storage, study_id),
            )
            await conn.commit()
        await self._start_driver(study_id, config, storage)
        return study_id

    async def cancel(self, study_id: str) -> bool:
        async with self._lock:
            driver = self._drivers.pop(study_id, None)
        if driver is None:
            return False
        await driver.stop()
        async with self.db.connect() as conn:
            await repository.set_study_status(conn, study_id, StudyStatus.COMPLETED)
        return True

    async def stop_all(self) -> None:
        async with self._lock:
            drivers = list(self._drivers.values())
            self._drivers.clear()
        await asyncio.gather(*(d.stop() for d in drivers), return_exceptions=True)

    async def _start_driver(
        self, study_id: str, config: StudyConfig, storage: str
    ) -> None:
        driver = StudyDriver(study_id, config, storage, self.db)
        driver.start()
        async with self._lock:
            self._drivers[study_id] = driver

    def _optuna_url(self, study_id: str) -> str:
        path = settings.data_dir / "studies" / f"{study_id}.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"
