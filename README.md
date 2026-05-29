# trainpipe

An AI training pipeline for a single Linux box with 1-N NVIDIA GPUs. Submit
`ms-swift` fine-tuning jobs (LoRA / full / qlora / longlora / adalora /
ia3), watch them stream live to MLflow, queue more than you have GPUs for,
and drive hyperparameter sweeps via Optuna — all through a small REST API
that an agent (or a human) can drive.

## Architecture

```
┌──────────────────────── Linux host (≥1 GPU) ────────────────────────┐
│                                                                     │
│  ┌────────────┐        ┌──────────────────┐                         │
│  │  FastAPI   │ ──────►│  MLflow server   │◄── browser UI           │
│  │  + auth    │ create │  (sqlite + fs    │                         │
│  │            │  run   │   artifacts)     │                         │
│  └─────┬──────┘        └──────────────────┘                         │
│        │                          ▲ metrics + checkpoints           │
│        ▼                          │ (HF Trainer → MLflowCallback)   │
│  ┌────────────┐         ┌─────────┴────────┐                        │
│  │  SQLite    │◄───────►│   Scheduler      │                        │
│  │  queue +   │ status  │  (asyncio loop,  │                        │
│  │  studies + │         │   GPU pool,      │                        │
│  │  events    │         │   subprocess mgr)│                        │
│  └────────────┘         └─────────┬────────┘                        │
│                                   │ spawns one process per run     │
│                  ┌─────────┬──────┼──────┬─────────┐                │
│                  ▼         ▼      ▼      ▼         ▼                │
│                GPU 0     GPU 1  GPU 2  GPU 3    (idle)              │
│                swift sft (CUDA_VISIBLE_DEVICES + MLFLOW_RUN_ID)     │
│                                                                     │
│  ┌────────────┐  ask trial → enqueue exp → wait → read metric       │
│  │  Optuna    │  tell trial. Up to max_concurrent in parallel.      │
│  │  drivers   │  Per-study sqlite under data/studies/.              │
│  └────────────┘                                                     │
└─────────────────────────────────────────────────────────────────────┘
        ▲ X-API-Key            ▲ http                ▲ ssh / tailscale
   agent / CLI            MLflow UI               remote dev
```

## What it does

- Queue 1..N concurrent ms-swift training runs across the local GPUs.
- One MLflow run per experiment, with our `trainpipe.experiment_id` /
  `trainpipe.study_id` / `trainpipe.trial_number` tags so the UI groups
  related runs.
- Live log streaming over Server-Sent Events.
- Crash recovery: a process restart releases stale GPU leases and
  requeues experiments that were running pre-crash.
- Hyperparameter sweeps via Optuna with a JSON-path-based search-space
  DSL — submit one `StudyConfig` and trials get enqueued automatically.
- Single API surface for both humans and agents.

## Setup

```bash
# 1. Python deps
python -m venv .venv
source .venv/bin/activate          # Linux: deployment target
pip install -e ".[training]"        # add `,dev` for tests + linting

# 2. MLflow tracking server
docker compose up -d
# → http://localhost:5000

# 3. Configure
cp .env.example .env
# edit TRAINPIPE_API_KEY, optionally TRAINPIPE_VISIBLE_GPUS

# 4. Run
trainpipe                           # uvicorn on :8080
```

Health check: `curl http://localhost:8080/health`.

## Datasets

`dataset` and `val_dataset` accept any of:

- HuggingFace repo IDs: `"meta-llama/Llama-3.1-8B"`
- ms-swift registry shortcuts: `"AI-ModelScope/alpaca-gpt4-data-en"`
- Local files: `"/srv/data/train.jsonl"`, `"./train.jsonl"`, `"C:/data/train.jsonl"`
- Local directories: `"/srv/data/my-dataset/"`
- **Uploaded dataset by id**: `"ds:<dataset_id>"`
- Any of the above with a sub-sample suffix: `"/srv/data/train.jsonl#500"`, `"ds:abc123#500"`

### Uploading your own data

Instead of placing files on the server by hand, push them through the API:

```bash
curl -H "X-API-Key: $TRAINPIPE_API_KEY" \
  -F "file=@./train.jsonl" \
  -F "name=my-training-set" \
  http://server:8080/datasets
# → {"id": "abc123…", "format": "jsonl", "line_count": 500, "sha256": "…", …}
```

The server validates the format (samples the first 100 records — bad JSON,
empty file, etc. is rejected with 422), computes a sha256, and stores under
`data/datasets/<id>/`. Then reference it from a spec:

```json
{"model": "...", "dataset": ["ds:abc123"], "val_dataset": ["ds:abc123#50"]}
```

Endpoints: `GET /datasets`, `GET /datasets/{id}`,
`GET /datasets/{id}/preview?n=10` (text formats only), `DELETE /datasets/{id}`.

### Path validation

Local-looking paths are validated at submit time — `POST /experiments`,
`POST /experiments/batch`, and `POST /studies` all return **422** with a
`missing_local_paths` detail listing every offending entry, so an agent
can fix all of them in one round-trip. Remote refs (HF, registry) are
accepted blindly and fail at trainer load if wrong. Malformed `ds:` strings
(no hex, unknown id) return **422** with `error: malformed_dataset_ref` or
`error: unknown_dataset_ref`.

Expected ms-swift JSONL formats:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"query": "...", "response": "..."}                                   # legacy
{"messages": [...], "images": ["/path/to/img.jpg"]}                   # multimodal
```

## Web UI

The API serves a single-page UI at **http://server:8080/** (no separate
build step — Tailwind + Alpine.js via CDN). Tabs:

- **Experiments** — submit form, table with status badges, detail panel
  with live log tail, MLflow run link, cancel button
- **Studies** — Optuna sweep submit form + progress table
- **Datasets** — drag-and-drop upload, click-to-copy `ds:<id>` ref,
  preview, delete
- **GPUs** — card per device with lease state

API key is stored in browser `localStorage` — first visit prompts for
it. Polling refreshes every 4 s; the detail panel polls logs every
2.5 s while open.

## Submitting an experiment

```bash
curl -X POST http://localhost:8080/experiments \
  -H "X-API-Key: $TRAINPIPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen2-vl-lora",
    "model": "qwen/Qwen2-VL-2B-Instruct",
    "sft_type": "lora",
    "dataset": ["AI-ModelScope/alpaca-gpt4-data-en"],
    "gpu_count": 1,
    "hyperparameters": {
      "num_train_epochs": 3,
      "learning_rate": 1e-4,
      "lora_rank": 8
    },
    "tags": {"mlflow_experiment": "vlm-explore"}
  }'
# → {"experiment_id": "..."}
```

Watch live logs:

```bash
curl -N http://localhost:8080/experiments/<id>/logs/stream \
  -H "X-API-Key: $TRAINPIPE_API_KEY"
```

GPU state:

```bash
curl http://localhost:8080/gpus -H "X-API-Key: $TRAINPIPE_API_KEY"
```

## Agent-driven autoresearch

A study is a Pydantic spec: `base_spec` (an `ExperimentSpec`), `search_space`
(dotted paths into the spec → range), `target_metric` (read from MLflow on
trial completion), `direction`, `n_trials`, `max_concurrent`, `sampler`.

```bash
curl -X POST http://localhost:8080/studies \
  -H "X-API-Key: $TRAINPIPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "lr-rank-sweep",
    "base_spec": {
      "model": "qwen/Qwen2-VL-2B-Instruct",
      "dataset": ["AI-ModelScope/alpaca-gpt4-data-en"],
      "sft_type": "lora"
    },
    "search_space": {
      "hyperparameters.learning_rate": {"kind": "loguniform", "low": 1e-5, "high": 1e-3},
      "hyperparameters.lora_rank":     {"kind": "categorical", "choices": [4, 8, 16, 32]}
    },
    "target_metric": "eval/loss",
    "direction": "minimize",
    "n_trials": 20,
    "max_concurrent": 4,
    "sampler": "tpe"
  }'
```

The driver `ask()`s Optuna, samples a spec, enqueues it as an experiment,
waits for it to terminate, reads `eval/loss` from MLflow, then `tell()`s
Optuna. Up to `max_concurrent` trials run in parallel — capped at whatever
the GPU pool allows.

For an agent doing freer-form autoresearch (not just a fixed search space),
talk directly to `POST /experiments` in a loop, read run metrics from MLflow,
and pick the next spec yourself.

## REST API

| Method | Path                                  | Purpose                          |
| ------ | ------------------------------------- | -------------------------------- |
| GET    | `/health`                             | Liveness (no auth)               |
| POST   | `/experiments`                        | Submit one experiment            |
| POST   | `/experiments/batch`                  | Submit a list                    |
| GET    | `/experiments`                        | List (filter: status, study_id)  |
| GET    | `/experiments/{id}`                   | Detail                           |
| POST   | `/experiments/{id}/cancel`            | Cancel (queued or running)       |
| GET    | `/experiments/{id}/logs`              | Download full log                |
| GET    | `/experiments/{id}/logs/stream`       | SSE live tail                    |
| GET    | `/gpus`                               | Pool state with leases           |
| POST   | `/studies`                            | Create + start a sweep           |
| GET    | `/studies`                            | List studies                     |
| GET    | `/studies/{id}`                       | Detail (best_value, best_trial)  |
| POST   | `/studies/{id}/cancel`                | Stop driver, mark completed      |

All routes except `/health` require the `X-API-Key` header.

## MCP integration

trainpipe ships an MCP server that mirrors the REST surface as
Claude-Code-friendly tools. Install once:

```bash
pip install -e ".[mcp]"
```

Then register with Claude Code (run trainpipe locally first):

```bash
claude mcp add trainpipe -- env \
  TRAINPIPE_API_KEY=$TRAINPIPE_API_KEY \
  TRAINPIPE_BASE_URL=http://localhost:8080 \
  python -m trainpipe.mcp
```

Tools exposed: `submit_experiment`, `get_experiment`, `list_experiments`,
`cancel_experiment`, `tail_logs`, `submit_study`, `list_studies`,
`get_study`, `cancel_study`, `gpu_status`, `upload_dataset`,
`list_datasets`, `get_dataset`, `preview_dataset`, `delete_dataset`.

Auth never leaks into the model context — the API key stays inside the
MCP server process; the agent only sees tool calls and their results.

## Configuration

All settings are prefixed `TRAINPIPE_` and loaded from `.env` or the
environment.

| Var                          | Default                  | Notes                                  |
| ---------------------------- | ------------------------ | -------------------------------------- |
| `TRAINPIPE_API_KEY`          | `dev-key-change-me`      | Required for every non-health route    |
| `TRAINPIPE_HOST`             | `0.0.0.0`                |                                        |
| `TRAINPIPE_PORT`             | `8080`                   |                                        |
| `TRAINPIPE_DATA_DIR`         | `./data`                 | sqlite, logs, outputs, study storage   |
| `TRAINPIPE_MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server                       |
| `TRAINPIPE_VISIBLE_GPUS`     | unset                    | JSON list, e.g. `[0,1]`. Default: all  |
| `TRAINPIPE_POLL_INTERVAL_SEC` | `1.0`                    | Scheduler tick                         |
| `TRAINPIPE_HEARTBEAT_INTERVAL_SEC` | `5.0`              | Reserved                               |

## Project layout

```
trainpipe/
├── api/
│   ├── main.py               FastAPI app + lifespan
│   ├── auth.py               X-API-Key middleware
│   ├── deps.py               typed accessors from app.state
│   ├── schemas.py            ExperimentSpec, StudyConfig, …
│   └── routes/{experiments,gpus,studies}.py
├── core/
│   ├── db.py                 aiosqlite, WAL, versioned migrations
│   └── repository.py         CRUD for experiments, studies, events
├── scheduler/
│   ├── gpu_pool.py           pynvml detection + SQLite-backed leases
│   ├── runner.py             asyncio subprocess + POSIX process group
│   └── loop.py               dispatch + monitor + MLflow run creation
├── training/
│   └── swift_builder.py      ExperimentSpec → (argv, env)
├── autoresearch/
│   ├── search_spaces.py      dotted-path overrides + suggest_* dispatch
│   ├── study.py              StudyDriver: ask → enqueue → wait → tell
│   └── manager.py            owns drivers in the API process
├── settings.py
└── cli.py                    `trainpipe` entry point
```

## Development

```bash
pip install -e ".[dev]"
pytest                              # 51 unit tests, all should pass
ruff check trainpipe tests
```

## License

MIT.
