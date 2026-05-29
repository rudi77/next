from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

MIGRATIONS: list[str] = [
    # v1: initial schema
    """
    CREATE TABLE experiments (
        id TEXT PRIMARY KEY,
        spec_json TEXT NOT NULL,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 0,
        study_id TEXT,
        trial_number INTEGER,
        gpu_ids TEXT,
        mlflow_run_id TEXT,
        mlflow_experiment_id TEXT,
        log_path TEXT,
        error TEXT,
        pid INTEGER,
        created_at TEXT NOT NULL,
        queued_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        last_heartbeat_at TEXT
    );
    CREATE INDEX idx_experiments_status ON experiments(status);
    CREATE INDEX idx_experiments_study ON experiments(study_id);
    CREATE INDEX idx_experiments_queue
        ON experiments(status, priority DESC, queued_at ASC);

    CREATE TABLE studies (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        config_json TEXT NOT NULL,
        status TEXT NOT NULL,
        optuna_storage TEXT NOT NULL,
        n_trials_target INTEGER,
        n_trials_completed INTEGER NOT NULL DEFAULT 0,
        best_value REAL,
        best_trial_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE gpu_leases (
        gpu_index INTEGER PRIMARY KEY,
        experiment_id TEXT,
        leased_at TEXT
    );

    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT,
        study_id TEXT,
        kind TEXT NOT NULL,
        payload_json TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_events_exp ON events(experiment_id, created_at);
    """,
]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as conn:
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA synchronous = NORMAL")
            await conn.execute("PRAGMA foreign_keys = ON")
            await self._migrate(conn)
            await conn.commit()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        async with aiosqlite.connect(self.path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            yield conn

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)"
        )
        cur = await conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
        row = await cur.fetchone()
        current = int(row[0]) if row else 0

        for version, sql in enumerate(MIGRATIONS, start=1):
            if version <= current:
                continue
            await conn.executescript(sql)
            await conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
