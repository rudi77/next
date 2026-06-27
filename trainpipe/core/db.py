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
    # v4: model registry (Phase 7). A "model" is a named, versioned pointer
    # to one experiment's adapter output dir + the eval summary at that
    # point. Aliases ("production", "staging") are mutable labels that move
    # between versions; ``UNIQUE(name, alias)`` enforces at-most-one model
    # per alias per family.
    """
    CREATE TABLE models (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version INTEGER NOT NULL,
        run_id TEXT,
        experiment_id TEXT,
        base_model TEXT NOT NULL,
        adapter_path TEXT,
        eval_summary_json TEXT,
        description TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE SET NULL,
        UNIQUE(name, version)
    );
    CREATE INDEX idx_models_name ON models(name);
    CREATE INDEX idx_models_experiment ON models(experiment_id);

    CREATE TABLE model_aliases (
        name TEXT NOT NULL,
        alias TEXT NOT NULL,
        model_id TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (name, alias),
        FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
    );
    CREATE INDEX idx_model_aliases_model ON model_aliases(model_id);
    """,
    # v5: multimodal dataset support (Phase 9). ``media_kinds_json`` is a
    # JSON array like ``["images"]`` / ``["images","videos"]`` recorded
    # when a JSONL upload contains those fields. ``image_root`` is the
    # bundle's image directory (relative paths in samples resolve against
    # it). Both NULL for text-only datasets.
    #
    # NB: combining both ALTERs in one migration entry means a crash mid-
    # script could leave v5 half-applied on disk while ``schema_version``
    # still reads 4. Mitigation: on a fresh boot the failing ALTER would
    # also be the one that re-runs, so the only durable bad state is
    # "first ALTER succeeded, second failed AND schema_version recorded
    # 5". executescript runs in autocommit so this is theoretically
    # possible. If it ever happens in practice, recover by manually
    # dropping the orphaned column and re-running. Keeping both here
    # respects the codebase's migration-immutability rule (CLAUDE.md).
    """
    ALTER TABLE datasets ADD COLUMN media_kinds_json TEXT;
    ALTER TABLE datasets ADD COLUMN image_root TEXT;
    """,
    # v6: active learning (Phase 11).
    #
    # ``active_learning_runs`` is one execution of "score every unlabeled
    # sample, rank by uncertainty, surface the top N as an annotation
    # queue." Status transitions queued → running → completed/failed.
    #
    # ``annotation_queue_items`` is the surfaced ranked list — one row
    # per sample with the model's prediction, an uncertainty score, and
    # the raw input. Annotators consume the queue (in trainpipe's UI or
    # pushed to Label Studio via the helper in routes/active_learning.py).
    """
    CREATE TABLE active_learning_runs (
        id TEXT PRIMARY KEY,
        model_ref TEXT NOT NULL,
        dataset_path TEXT NOT NULL,
        top_n INTEGER NOT NULL,
        sample_limit INTEGER,
        status TEXT NOT NULL,
        error TEXT,
        scored_count INTEGER,
        queued_count INTEGER,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    );
    CREATE INDEX idx_al_runs_status ON active_learning_runs(status);

    CREATE TABLE annotation_queue_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        sample_index INTEGER NOT NULL,
        input_json TEXT NOT NULL,
        prediction TEXT NOT NULL,
        uncertainty REAL NOT NULL,
        annotated INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES active_learning_runs(id) ON DELETE CASCADE
    );
    CREATE INDEX idx_al_queue_run ON annotation_queue_items(run_id);
    CREATE INDEX idx_al_queue_run_uncert
        ON annotation_queue_items(run_id, uncertainty DESC);
    """,
    # v7: multi-stage pipelines (Phase 12). A pipeline is a sequence of
    # named stages (CPT → SFT → DPO, or whatever shape the user wants).
    # Each stage produces an adapter dir that the next stage consumes
    # via its ``input_from_stage`` ref. The driver watches dependencies,
    # spawns each stage as a regular experiment, and propagates output
    # paths.
    """
    CREATE TABLE pipelines (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        config_json TEXT NOT NULL,
        status TEXT NOT NULL,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX idx_pipelines_status ON pipelines(status);

    CREATE TABLE pipeline_stages (
        pipeline_id TEXT NOT NULL,
        stage_name TEXT NOT NULL,
        stage_index INTEGER NOT NULL,
        depends_on_json TEXT,
        experiment_id TEXT,
        status TEXT NOT NULL,
        output_dir TEXT,
        error TEXT,
        started_at TEXT,
        finished_at TEXT,
        PRIMARY KEY (pipeline_id, stage_name),
        FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE SET NULL
    );
    CREATE INDEX idx_pipeline_stages_status
        ON pipeline_stages(pipeline_id, status);
    """,
    # v8: model lineage (Phase 15). One row per (model, dataset) usage —
    # populated when a model is registered, by walking the experiment's
    # spec.dataset / spec.val_dataset and matching paths against the
    # registered datasets. Enables the "which models used dataset X"
    # audit and the GDPR "forget user Y" workflow.
    """
    CREATE TABLE model_lineage (
        model_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        used_at TEXT NOT NULL,
        PRIMARY KEY (model_id, dataset_id),
        FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
    );
    CREATE INDEX idx_lineage_dataset ON model_lineage(dataset_id);
    """,
    # v9: dataset versioning (Phase 16). Datasets become immutable after
    # create; ``version`` is bumped only by explicit derivation
    # (split/mix/redact). Default 1 for everything that already exists.
    # ``derived_from`` records the parent dataset for audit; NULL for
    # uploaded originals.
    """
    ALTER TABLE datasets ADD COLUMN version INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE datasets ADD COLUMN derived_from TEXT;
    """,
    # v10: watches (Phase 17). A watch is a stored rule that fires a
    # pipeline when its trigger condition is met. Two trigger kinds:
    #
    #  * ``interval`` — periodic re-train ("every N seconds"). The
    #    ``interval_seconds`` column carries the period; ``last_fired``
    #    is the bookkeeping for when we last actually triggered.
    #  * ``metric_threshold`` — drift detection. ``model_name`` /
    #    ``suite_id`` / ``metric_name`` / ``threshold`` identify which
    #    eval mean to watch; when it dips below ``threshold`` we fire.
    #
    # ``pipeline_config_json`` is the PipelineConfig that gets spawned
    # on fire (kept inline so the watch doesn't break if the user
    # rewrites their pipelines).
    """
    CREATE TABLE watches (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        interval_seconds INTEGER,
        model_name TEXT,
        suite_id TEXT,
        metric_name TEXT,
        threshold REAL,
        pipeline_config_json TEXT NOT NULL,
        last_fired_at TEXT,
        last_fired_pipeline_id TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_watches_enabled ON watches(enabled);
    """,
    # v11: cost + resource tracking (Phase 20). Columns on experiments
    # so the cost-vs-metric query is a single row read. ``gpu_seconds``
    # is gpu_count * wall_clock_seconds; ``peak_vram_mb`` is the max
    # nvml reading during the run; ``energy_wh`` is integrated power.
    # All NULL until the monitor finalizes the run.
    """
    ALTER TABLE experiments ADD COLUMN gpu_seconds REAL;
    ALTER TABLE experiments ADD COLUMN peak_vram_mb REAL;
    ALTER TABLE experiments ADD COLUMN energy_wh REAL;
    """,
    # v12: watch failure bookkeeping (follow-up to Phase 17). A watch
    # with a malformed pipeline_config used to swallow its ValueError
    # every tick and silently spam the log forever. Now we count
    # consecutive failures and auto-disable after a threshold.
    """
    ALTER TABLE watches ADD COLUMN consecutive_failures INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE watches ADD COLUMN last_error TEXT;
    """,
    # v13: N-source dataset lineage (follow-up to Phase 16). The
    # ``datasets.derived_from`` column is 1:1; a mix of two source
    # datasets loses one parent in the audit chain. ``dataset_lineage``
    # is N:M so mix(a, b) gets two rows pointing at both parents, and
    # the recursive "which models trained on data ultimately derived
    # from X" query (GDPR) returns correct results. ``role`` lets us
    # distinguish "split-of" / "mix-of" / "redacted-from" for audit.
    """
    CREATE TABLE dataset_lineage (
        derived_id TEXT NOT NULL,
        parent_id TEXT NOT NULL,
        role TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY (derived_id, parent_id),
        FOREIGN KEY (derived_id) REFERENCES datasets(id) ON DELETE CASCADE,
        FOREIGN KEY (parent_id) REFERENCES datasets(id) ON DELETE CASCADE
    );
    CREATE INDEX idx_lineage_parent ON dataset_lineage(parent_id);
    """,
    # v14: agentic data acquisition (Phase 22). One ``acquisition_runs``
    # row is a single "build me a training set from this brief" job that
    # walks phases intake → research → acquire/synthesize → curate →
    # register (see docs/spec/agentic-data-acquisition.md). Unlike a
    # pipeline it dispatches no experiments — the work runs in-process in
    # an AcquisitionDriver task, so the row IS the state machine: ``phase``
    # is where the driver currently is, ``spec_json`` is the structured
    # intake result (NULL until intake runs), ``answers_json`` holds the
    # operator's replies to ``spec.open_questions`` for the awaiting_input
    # pause/resume path. ``dataset_id`` is the registered result.
    #
    # ``acquisition_sources`` is the per-run source ledger the research /
    # acquire phases populate (URL + topic + license decision), kept
    # separate so the audit trail survives even when a source is skipped.
    # Empty in the MVP (no real web yet) but the table ships now so the
    # later phase is a pure additive change, not another migration of the
    # runs table.
    """
    CREATE TABLE acquisition_runs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        brief TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        target_count INTEGER NOT NULL,
        spec_json TEXT,
        answers_json TEXT,
        status TEXT NOT NULL,
        phase TEXT,
        dataset_id TEXT,
        raw_count INTEGER,
        final_count INTEGER,
        error TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    );
    CREATE INDEX idx_acquisition_runs_status ON acquisition_runs(status);

    CREATE TABLE acquisition_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        url TEXT NOT NULL,
        title TEXT,
        topic TEXT,
        license_status TEXT NOT NULL DEFAULT 'unknown',
        used INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES acquisition_runs(id) ON DELETE CASCADE
    );
    CREATE INDEX idx_acquisition_sources_run ON acquisition_sources(run_id);
    """,
    # v15: web research/acquisition config on the run (Phase 22 stage 3).
    # ``search_provider`` selects how the research phase finds candidate
    # sources ('none' = synth-only, the MVP default; 'mock' / 'tavily');
    # ``max_sources`` caps how many candidate URLs the research phase gates.
    # Defaults keep every pre-stage-3 run behaving exactly as before.
    """
    ALTER TABLE acquisition_runs ADD COLUMN search_provider TEXT NOT NULL DEFAULT 'none';
    ALTER TABLE acquisition_runs ADD COLUMN max_sources INTEGER NOT NULL DEFAULT 0;
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
