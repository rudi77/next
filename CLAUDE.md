# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


Read the following file for additional important information:
- [CLAUDE_BEHAVIORAL.md](CLAUDE_BEHAVIORAL.md)

---

## Commands

```bash
# Install (Linux deployment target; on Windows dev see "Dev quirks" below)
pip install -e ".[dev]"                # tests + lint
pip install -e ".[training]"           # add ms-swift + torch (heavy)
pip install -e ".[mcp]"                # add MCP SDK

# Tests
pytest                                 # full suite
pytest tests/test_swift_builder.py     # single file
pytest tests/test_api.py::test_submit_and_get   # single test
pytest -q --tb=short                   # what CI-shaped output looks like

# Lint
ruff check trainpipe tests

# Run the stack
docker compose up -d                   # MLflow on :5000
trainpipe                              # FastAPI on :8080 (uvicorn entry: trainpipe.cli:main)
trainpipe-mcp                          # MCP server (stdio) — needs TRAINPIPE_API_KEY
```

The trainpipe entry point relies on `_resolve_swift_binary()` finding `swift` via
`shutil.which` → `Path(sys.executable).parent / "swift"`. Launching with
`.venv/bin/trainpipe` is enough; no PATH munging needed.

## Architecture: the parts that take multiple files to see

### Three concurrent async loops own the lifecycle

1. **`scheduler.loop.Scheduler._loop()`** polls SQLite every
   `poll_interval_sec` for queued experiments. Per tick it walks the queue
   under `_dispatch_lock`, **but only to claim** (`_claim_next` does an
   atomic CAS `UPDATE … SET status='running' WHERE id=? AND status='queued'`
   so a concurrent cancellation isn't lost). The slow work — MLflow run
   creation + subprocess spawn — runs in `_launch` as a background task
   **outside the lock**, so MLflow latency doesn't serialize concurrent
   submits. Don't move MLflow calls back into the lock.
2. **Per-experiment `_monitor` task** waits on the subprocess, then in one
   transaction marks the row terminal, releases GPU leases, finalizes the
   MLflow run.
3. **Per-study `autoresearch.study.StudyDriver._run()`** drives Optuna:
   ask → sample spec via dotted-path overrides → enqueue as an experiment
   → poll until terminal → read metric from MLflow → tell. Up to
   `config.max_concurrent` trials in parallel via a semaphore.

### Crash recovery has a specific order

`Scheduler.start()` must requeue `'running'` rows **before** `sync_leases`,
otherwise `sync_leases` keeps the leases (they're still 'running' at that
moment) and they orphan after the next UPDATE flips status. The requeue
also resets `queued_at = now` so old crashed experiments don't permanently
win the FIFO tie-break and starve newer submissions.

`StudyDriver._reconcile_pending_trials()` fails any Optuna trial left in
`RUNNING` state at start so the driver doesn't `ask()` new trials in
parallel with orphaned ones.

### Dataset references resolve at submit time

`ds:<hex>(#suffix)?` in `ExperimentSpec.dataset` / `val_dataset` is
resolved to the registered file path in `routes/experiments.py` and
`routes/studies.py` via `training.dataset_refs.resolve_spec()` **before**
the spec is persisted. The scheduler and `swift_builder` never see `ds:`
strings. Malformed (`ds:` empty, non-hex) raise `MalformedDatasetRef` →
422 — do **not** loosen this; the live-test bug that caused us to add it
silently passed `dataset: ["ds:"]` to the trainer.

### swift_builder is the version-translation layer

ms-swift renames its CLI flags between major versions. Public
`ExperimentSpec` field names (`model`, `sft_type`, `lora_target_modules`)
stay stable; `training/swift_builder.py` maps them to the current ms-swift
flag names. Current mapping (ms-swift v4): `--model`, `--tuner_type`,
`--target_modules`. If ms-swift renames again, this is the **only** file
to touch.

### Database is migration-versioned, append-only

`MIGRATIONS: list[str]` in `core/db.py`. The index in the list is the
version number; `_migrate()` applies any not yet recorded in
`schema_version`. **Never edit a previous entry** — add a new one.
Tests assert `len(MIGRATIONS) == MAX(version)`.

`Database.connect()` opens a fresh aiosqlite connection per call. WAL
mode + `synchronous=NORMAL` allows concurrent reads while a writer is
staging changes. Keep transactions short.

### GPU pool degrades gracefully

`scheduler.gpu_pool.detect_gpus()` returns `[]` when pynvml or the NVIDIA
driver isn't available, so the API still boots on dev machines without
GPUs. `GpuPool([])` is also the test default. Real allocation goes
through `try_allocate` which performs an `executemany` UPDATE on the
`gpu_leases` table under an asyncio lock.

### Auth + UI

- API-key check via `require_api_key` (constant-time compare) attached at
  router level for every router except UI helpers.
- `GET /`, `GET /health`, `GET /ui/config` are public. `/ui/config`
  strips embedded credentials (`http://user:pass@host`) from the MLflow
  tracking URI before exposing it — never serve `settings.mlflow_tracking_uri`
  raw from an unauthenticated route.
- UI is a single self-contained `trainpipe/ui/index.html` (Tailwind +
  Alpine via CDN, no build step). API key lives in `localStorage`,
  travels only as `X-API-Key` header (never URL param — keeps it out of
  server logs).

### Dataset upload safety

- `POST /datasets` dedupes by sha256 — uploading the same content twice
  returns the existing record with **200** (not 201). Callers that
  branch on status code should handle both.
- `DELETE /datasets/{id}` returns **409** if any queued/running
  experiment references the dataset path; pass `?force=true` to override.

### MCP server uses lazy client init

`trainpipe/mcp.py` imports cleanly **without** `TRAINPIPE_API_KEY` set —
the FastMCP instance is built at import (so `@mcp.tool()` decorators
work), but the httpx client is constructed on first call via
`_get_client()`. Tests rely on this.

## Testing patterns

- `pyproject.toml` sets `asyncio_mode = "auto"` so `async def test_*`
  works without `@pytest.mark.asyncio`. Async fixtures use
  `@pytest_asyncio.fixture`.
- Settings overrides via monkeypatch: `monkeypatch.setattr(
  "trainpipe.settings.settings.data_dir", tmp_path)`. Don't patch the
  computed properties (`datasets_dir`, `sqlite_path`) — patch
  `data_dir` and let them recompute.
- API tests use `fastapi.testclient.TestClient(app)` with
  `app.dependency_overrides[get_db / get_scheduler / get_gpu_pool /
  get_study_manager]`. The `state` fixture wires all four; tear-down
  calls `app.dependency_overrides.clear()`.
- For DB mutations inside sync test bodies, call `asyncio.run(coro)`
  via the `_run` helper rather than mixing async/sync fixtures.

## Dev quirks

- **Linux-only runtime.** The scheduler uses `os.setsid` for POSIX
  process groups and `os.killpg` for cancel. There are `os.name ==
  "posix"` guards, but real training only works on Linux.
  Recommended dev path on Windows: WSL2 + Ubuntu, see README's
  Deployment / WSL section.
- **WSL2 + torch CUDA**: pip will pull the latest cu1XX wheel which
  may be ahead of your NVIDIA driver. Force the matching version
  with `--index-url https://download.pytorch.org/whl/cu128` (or
  whatever your driver's CUDA reports via `nvidia-smi`).
- **Two MLflow imports**: `_create_mlflow_run` / `_terminate_mlflow_run`
  in `scheduler/loop.py` and `_read_metric` in `autoresearch/study.py`
  each lazy-import mlflow inside the function. Keeps the import cost
  off the cold path and lets tests skip MLflow setup.

  ## Smoke Tests
  Smike tests shall be executed to guarantee a certain level of software quality
  
  - [SMOKE.md](SMOKE.md)
