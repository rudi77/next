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
    # v2: datasets registry
    """
    CREATE TABLE datasets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        path TEXT NOT NULL UNIQUE,
        format TEXT NOT NULL,
        line_count INTEGER,
        size_bytes INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_datasets_name ON datasets(name);
    CREATE INDEX idx_datasets_sha ON datasets(sha256);
    """,
    # v3: eval framework — suites (reusable config), runs (one execution
    # against one model target), results (per-sample prediction + scores).
    """
    CREATE TABLE eval_suites (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        dataset_path TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        inference_params_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_eval_suites_name ON eval_suites(name);

    CREATE TABLE eval_runs (
        id TEXT PRIMARY KEY,
        suite_id TEXT NOT NULL,
        experiment_id TEXT,
        model_ref TEXT NOT NULL,
        status TEXT NOT NULL,
        gpu_ids TEXT,
        log_path TEXT,
        error TEXT,
        pid INTEGER,
        aggregate_json TEXT,
        sample_count INTEGER,
        triggered_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        FOREIGN KEY (suite_id) REFERENCES eval_suites(id) ON DELETE CASCADE,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE SET NULL
    );
    CREATE INDEX idx_eval_runs_suite ON eval_runs(suite_id);
    CREATE INDEX idx_eval_runs_experiment ON eval_runs(experiment_id);
    CREATE INDEX idx_eval_runs_status ON eval_runs(status);

    CREATE TABLE eval_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        sample_index INTEGER NOT NULL,
        input_json TEXT NOT NULL,
        prediction TEXT NOT NULL,
        gold_json TEXT,
        scores_json TEXT NOT NULL,
        error TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES eval_runs(id) ON DELETE CASCADE
    );
    CREATE INDEX idx_eval_results_run ON eval_results(run_id);
    CREATE UNIQUE INDEX idx_eval_results_run_sample
        ON eval_results(run_id, sample_index);
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
