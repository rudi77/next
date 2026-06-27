# trainpipe

An AI training pipeline for a single Linux box with 1-N NVIDIA GPUs. Submit
`ms-swift` jobs — supervised fine-tuning (LoRA / full / qlora / longlora /
adalora / ia3), (continued) pretraining, and preference/RL training
(DPO / KTO / PPO / GRPO) — watch them stream live to MLflow, queue more
than you have GPUs for, and drive hyperparameter sweeps via Optuna. Around
that core sits a full lifecycle: a dataset registry, an eval framework, a
model registry with aliases, quantization, an inference playground, and an
**agentic data-acquisition** module that builds a training set from a
natural-language brief. Everything is reachable through a single REST API
that an agent (via MCP or the `trainpipe` CLI) or a human (via the web UI)
can drive.

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
│            swift sft/pt/rlhf (CUDA_VISIBLE_DEVICES + MLFLOW_RUN_ID) │
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

- Queue 1..N concurrent ms-swift training runs across the local GPUs —
  SFT, (continued) pretraining (`train_kind=pt`), and preference/RL
  (`dpo` / `kto` / `ppo` / `grpo`) all from one `ExperimentSpec`.
- One MLflow run per experiment, with our `trainpipe.experiment_id` /
  `trainpipe.study_id` / `trainpipe.trial_number` tags so the UI groups
  related runs.
- Live log streaming over Server-Sent Events.
- Crash recovery: a process restart releases stale GPU leases and
  requeues experiments that were running pre-crash.
- Hyperparameter sweeps via Optuna with a JSON-path-based search-space
  DSL — submit one `StudyConfig` and trials get enqueued automatically.
- A **dataset registry** (upload, dedup by sha256, split / mix / redact /
  bundle, Label-Studio import, lineage queries) addressed by `ds:<id>` refs.
- An **eval framework** (suites, runs, 7 metrics incl. LLM-as-judge,
  compare for regressions) and a **model registry** (versions, aliases,
  dataset lineage, quantization).
- An **inference playground** (sync / streaming predict, N-way compare).
- **Agentic data acquisition**: turn a natural-language brief into a
  registered, PII-redacted training set — research + synthesize, with
  cost-budget, strict-license, and human-in-the-loop clarification.
- A single API surface for both humans and agents — driven from the web
  UI, the `trainpipe` CLI, or 40 MCP tools.

## Setup

On the Linux deploy box the quickest path is the lifecycle script, which
creates the venv, installs the package, seeds `.env`, and runs the server as
a backgrounded process tracked by a PID file:

```bash
./.scripts/next.sh install      # .venv + pip install -e ".[training,dev]" + seed .env
docker compose up -d            # MLflow tracking server → http://localhost:5000
# edit .env: TRAINPIPE_API_KEY, optionally TRAINPIPE_VISIBLE_GPUS
./.scripts/next.sh start        # uvicorn on :8080, health-checked
./.scripts/next.sh status       # PID + HTTP health probe
./.scripts/next.sh logs -f      # follow the server log
./.scripts/next.sh stop         # stop (restart = stop + start)
```

GPU training also needs a CUDA torch build: `./.scripts/install-torch-cu128.sh`.

<details>
<summary>Manual setup (equivalent steps)</summary>

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
trainpipe                           # uvicorn on :8080 (no subcommand = serve)
```
</details>

Health check: `curl http://localhost:8080/health`.

### Driving it from the terminal

`trainpipe` is also an operative client over the REST API — the same surface
the MCP server gives agents — so you can run the full train → eval → improve
loop without hand-writing `curl`. Operative subcommands need
`TRAINPIPE_API_KEY` (and optionally `TRAINPIPE_BASE_URL`) set; output is JSON
on stdout for piping into `jq`.

```bash
trainpipe submit --model Qwen/Qwen2.5-0.5B --dataset ds:ab12 --train-kind sft
trainpipe experiments --status running
trainpipe logs <exp-id> -n 50
trainpipe register-model --name my-model --experiment <exp-id> --alias staging
trainpipe run-eval --suite <suite-id> --experiment <exp-id>
trainpipe compare-evals <run-a> <run-b>
trainpipe inference my-model@staging "Summarize: ..."
trainpipe api GET /datasets/<id>/models   # generic escape hatch: any endpoint
```

## Deployment

trainpipe is a FastAPI + SQLite + MLflow stack with a **single shared
API key** as its only auth. That's fine for a private workstation; for
anything beyond that you need to think about transport (TLS) and access
control. Three patterns, in increasing order of exposure:

### 1. Local-only (default)

```bash
trainpipe                       # binds 0.0.0.0:8080
docker compose up -d            # MLflow on 0.0.0.0:5000
```

OK on a private laptop or single-user box. **Don't open these ports on
a public machine** — `X-API-Key` is sent in plain text without TLS.

### 2. Tailscale (recommended for remote access)

Lowest-friction "I want my server reachable from somewhere else". No
certs, no DNS, traffic is encrypted, only devices on your tailnet can
connect. The host config:

```bash
# .env
TRAINPIPE_HOST=127.0.0.1
TRAINPIPE_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
```

Then bind the docker-compose MLflow service to localhost too (edit
`docker-compose.yml` → `ports: ["127.0.0.1:5000:5000"]`), and expose
both via Tailscale:

```bash
tailscale serve --bg --https=8443 --set-path=/  http://127.0.0.1:8080
tailscale serve --bg --https=5443 --set-path=/  http://127.0.0.1:5000
```

From any other device on your tailnet:
`https://<host>.<tailnet>.ts.net:8443`. If you also need access from
outside your tailnet, swap `serve` for `funnel` — Tailscale will give
you a public HTTPS URL, still gated by your API key.

### 3. Public IP via Caddy + Let's Encrypt

If you really need a public hostname (CI, team access without
Tailscale), put trainpipe behind a reverse proxy that terminates TLS
and provisions certs automatically. Caddy is the lightest path.

1. Bind trainpipe and MLflow to **127.0.0.1** so they're not directly
   reachable:
   ```
   TRAINPIPE_HOST=127.0.0.1
   ```
   And in `docker-compose.yml`: `ports: ["127.0.0.1:5000:5000"]`.

2. Generate a strong key (don't reuse the dev one):
   ```bash
   python -c 'import secrets; print(secrets.token_urlsafe(32))'
   ```

3. `/etc/caddy/Caddyfile`:
   ```
   trainpipe.example.com {
       reverse_proxy 127.0.0.1:8080
       # Optional IP allowlist:
       # @allowed remote_ip 203.0.113.0/24
       # handle @allowed { reverse_proxy 127.0.0.1:8080 }
       # handle { abort }
   }

   mlflow.example.com {
       reverse_proxy 127.0.0.1:5000
       # MLflow has no built-in auth — at minimum add HTTP basic:
       basic_auth {
           admin <bcrypt-hash-via-`caddy hash-password`>
       }
   }
   ```

4. Open ports **80 and 443 only** in your firewall — never 8080 or
   5000.

5. `sudo systemctl enable --now caddy`. Certs auto-renew.

Caveats to be honest about:

- The API key is a single bearer secret. Anyone with it gets full
  access (submit, cancel, delete datasets). Rotate by editing `.env`
  and restarting trainpipe.
- The UI's localStorage is per-browser-profile — don't share a profile
  if you don't want shared access.
- For a multi-team setup, run multiple trainpipe instances with
  distinct keys rather than retrofitting multi-user auth onto a
  shared-secret model.
- MLflow's artifact store paths are exposed in the `path` field of
  dataset records. Don't put secrets on filenames.

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

- **Experiments** — submit form (SFT / pretraining / RL-GRPO controls),
  table with status badges, detail panel with live log tail, MLflow run
  link, cancel button
- **Studies** — Optuna sweep submit form + progress table + cost plot
- **Datasets** — drag-and-drop upload, click-to-copy `ds:<id>` ref,
  preview, delete
- **Evals** — suites, runs, aggregate scores, compare
- **Models** — registry with versions and aliases
- **Pipelines** — multi-stage workflow creation form + DAG view
- **Acquisition** — start a data-acquisition run from a brief, answer
  clarifying questions, watch it register a dataset
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

## Agent-driven training

Any HTTP-capable agent can drive trainpipe. There are three flavors,
each fine for a different use case.

### a) REST + Bash (Claude Code, Cursor, anything that can shell out)

The whole API is documented above. The friction in a Claude Code
session is the per-call permission prompt — allowlist the common
hosts up-front:

```jsonc
// .claude/settings.json  (or project-level .claude/settings.local.json)
{
  "permissions": {
    "allow": [
      "Bash(curl http://localhost:8080/*)",
      "Bash(curl https://trainpipe.example.com/*)",
      "Bash(curl http://localhost:5000/*)"
    ]
  }
}
```

Then prompts like *"upload `./my-train.jsonl`, submit a LoRA job on it,
watch logs, ping me when it finishes"* work directly. The agent strings
together: `POST /datasets` → `POST /experiments` → poll
`GET /experiments/{id}` → `GET /experiments/{id}/logs`.

### b) MCP server (Claude Code, Cursor, Claude Desktop)

The MCP layer hides the API key from the model context and exposes
each operation as a structured tool. See the [MCP integration](#mcp-integration)
section below for the `claude mcp add` command — for Cursor, drop the
same command into `~/.cursor/mcp.json` under `mcpServers`. After
registration the agent sees 40 typed tools instead of curl:
`submit_experiment(spec)`, `upload_dataset(name, filename, content_b64)`,
`tail_logs(id, n_lines)`, `run_eval(suite_id, experiment_id)`,
`start_acquisition(name, brief, ...)`, ...

When in doubt, prefer MCP for **repeated** use (cleaner tool calls,
schemas guide the model) and Bash+curl for **one-offs** or for keys
the agent shouldn't ever know.

### c) Autoresearch loop

The interesting use case: the agent isn't just running *a* job, it's
*iterating*. Two patterns, both supported.

**Pattern 1 — Optuna-driven** (best for a well-defined search space
and fixed budget):

```bash
curl -X POST https://trainpipe.example.com/studies \
  -H "X-API-Key: $TRAINPIPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "lr-rank-sweep",
    "base_spec": {
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "dataset": ["ds:abc123"],
      "sft_type": "lora"
    },
    "search_space": {
      "hyperparameters.learning_rate": {"kind": "loguniform", "low": 1e-5, "high": 1e-3},
      "hyperparameters.lora_rank":     {"kind": "categorical", "choices": [4, 8, 16, 32]}
    },
    "target_metric": "eval/loss",
    "direction": "minimize",
    "n_trials": 16, "max_concurrent": 4, "sampler": "tpe"
  }'
```

The driver `ask()`s Optuna, samples a spec, enqueues it, waits for
terminal status, reads the metric from MLflow, calls `tell()`. Up to
`max_concurrent` trials run in parallel, capped at GPU pool size.

**Pattern 2 — LLM-in-the-loop** (best when "what to try next"
requires actual reasoning over results — e.g. *"loss plateaued at step
50, try a higher LR with longer warmup"* or *"the model overfits on
short prompts, mix in a longer-context dataset"*):

```python
# Sketch of what an agent might generate via the MCP tools.
goal = "eval/loss < 1.0 on Qwen2.5-1.5B + my dataset"
max_experiments = 8
history = []

for i in range(max_experiments):
    spec = propose_next_spec(history, goal)            # ← LLM call
    eid = mcp.submit_experiment(spec)["experiment_id"]

    while True:
        rec = mcp.get_experiment(eid)
        if rec["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(15)

    metric = read_mlflow_metric(rec["mlflow_run_id"], "eval/loss")
    history.append({"spec": spec, "metric": metric, "status": rec["status"]})
    if metric is not None and metric < 1.0:
        break
```

What the agent has leverage over (cheap, just a spec field):

- `learning_rate`, `lora_rank`, `lora_alpha`, `lora_dropout`,
  `warmup_ratio`, `weight_decay`, `lr_scheduler_type`
- `max_length`, `gradient_accumulation_steps` (memory-bound)
- `sft_type` (`lora` ↔ `qlora` when VRAM is tight; `full` if it fits)
- `lora_target_modules`
- Dataset mix and `#N` sub-sampling, train/val split via separate uploads
- Multi-task training: list multiple datasets

What the agent **cannot** vary from a spec (would need a code change):

- Model architecture, tokenizer
- Loss function, custom trainer hooks
- Hardware topology

Guardrails worth adding when an agent is running this unattended:

- A budget: `max_experiments` and a wall-clock cap.
- An "abandon ship" rule: stop after K trials without improvement.
- Use `priority` lower than 0 on agent-submitted experiments so a
  human can queue-jump.
- An emergency `cancel_experiment` / `cancel_study` — both are MCP
  tools and one-line curls.
- Run the agent against a **separate** trainpipe instance or use a
  scoped key if you also have production runs you don't want it to see.

### What "improving the LLM" actually means here

The pipeline lets an agent produce a sequence of fine-tuning runs and
pick the best by your target metric — that gives you a checkpoint
that scores well on **the metric you defined**. It can't make the
base model fundamentally smarter, and it can't tell whether your eval
metric actually tracks what you care about in production. Those are
on you. trainpipe's contribution is removing the manual bookkeeping
(queue, GPU allocation, MLflow wiring, log capture) so the agent can
iterate fast on the things it *can* vary.

## REST API

The surface below is grouped by resource. Every route except `/health`
and `/ui/config` requires the `X-API-Key` header. The live, authoritative
contract is the OpenAPI doc at `/docs` (Swagger) / `/openapi.json`.

**Experiments & GPUs**

| Method | Path                            | Purpose                          |
| ------ | ------------------------------- | -------------------------------- |
| GET    | `/health`                       | Liveness (no auth)               |
| POST   | `/experiments`                  | Submit one experiment            |
| POST   | `/experiments/batch`            | Submit a list (atomic)           |
| GET    | `/experiments`                  | List (filter: status, study_id)  |
| GET    | `/experiments/{id}`             | Detail                           |
| POST   | `/experiments/{id}/cancel`      | Cancel (queued or running)       |
| GET    | `/experiments/{id}/logs`        | Download full log                |
| GET    | `/experiments/{id}/logs/stream` | SSE live tail                    |
| GET    | `/gpus`                         | Pool state with leases           |

**Studies (Optuna sweeps)**

| Method | Path                     | Purpose                          |
| ------ | ------------------------ | -------------------------------- |
| POST   | `/studies`               | Create + start a sweep           |
| GET    | `/studies`               | List studies                     |
| GET    | `/studies/cost-summary`  | GPU-seconds / energy per study   |
| GET    | `/studies/{id}`          | Detail (best_value, best_trial)  |
| POST   | `/studies/{id}/cancel`   | Stop driver, mark completed      |

**Datasets**

| Method | Path                          | Purpose                              |
| ------ | ----------------------------- | ------------------------------------ |
| POST   | `/datasets`                   | Upload (dedup by sha256 → 200/201)   |
| GET    | `/datasets`                   | List                                 |
| GET    | `/datasets/{id}`              | Detail                               |
| GET    | `/datasets/{id}/preview`      | First N rows (text formats)          |
| GET    | `/datasets/{id}/media`        | Serve a bundled image                |
| POST   | `/datasets/{id}/split`        | Deterministic train/val split        |
| POST   | `/datasets/mixes`             | Weighted mix of datasets             |
| POST   | `/datasets/{id}/redact`       | PII redaction → new dataset          |
| POST   | `/datasets/bundle`            | Images + JSONL as a ZIP bundle       |
| POST   | `/datasets/from-labelstudio`  | Import a Label-Studio export         |
| GET    | `/datasets/{id}/models`       | Lineage: models trained on this data |
| DELETE | `/datasets/{id}`              | Delete (409 if referenced; `?force`) |

**Evals**

| Method | Path                       | Purpose                          |
| ------ | -------------------------- | -------------------------------- |
| POST   | `/evals/suites`            | Create eval suite                |
| GET    | `/evals/suites`            | List suites                      |
| GET    | `/evals/suites/{id}`       | Suite detail                     |
| DELETE | `/evals/suites/{id}`       | Delete suite (`?force`)          |
| POST   | `/evals/runs`              | Enqueue an eval run              |
| GET    | `/evals/runs`              | List runs (filters)              |
| GET    | `/evals/runs/{id}`         | Run detail + aggregate           |
| GET    | `/evals/runs/{id}/results` | Per-sample results (paginated)   |
| POST   | `/evals/runs/{id}/cancel`  | Cancel a run                     |
| GET    | `/evals/compare`           | Compare runs → deltas            |

**Models & inference**

| Method | Path                              | Purpose                          |
| ------ | --------------------------------- | -------------------------------- |
| POST   | `/models`                         | Register a trained model         |
| GET    | `/models`                         | List (filter: name, alias)       |
| GET    | `/models/{name}`                  | All versions of a family         |
| GET    | `/models/{name}/{alias_or_version}` | Resolve one version            |
| GET    | `/models/{id}/datasets`           | Dataset lineage of a model       |
| POST   | `/models/{name}/aliases/{alias}`  | Move an alias (atomic)           |
| DELETE | `/models/{name}/aliases/{alias}`  | Drop an alias                    |
| POST   | `/models/{id}/quantize`           | Quantize a model                 |
| DELETE | `/models/{id}`                    | Delete (409 if aliased; `?force`)|
| POST   | `/inferences`                     | Sync predict                     |
| POST   | `/inferences/stream`              | Streaming predict (SSE)          |
| POST   | `/inferences/compare`             | N-way playground compare         |
| GET    | `/inferences/cache`               | Loaded-adapter cache state       |

**Pipelines, active learning, watches, synth, compliance**

| Method | Path                                              | Purpose                       |
| ------ | ------------------------------------------------- | ----------------------------- |
| POST   | `/pipelines`                                      | Create + start a pipeline     |
| GET    | `/pipelines` · `/pipelines/{id}`                  | List / detail                 |
| POST   | `/pipelines/{id}/cancel`                          | Cancel                        |
| POST   | `/active-learning/runs`                           | Start an AL run               |
| GET    | `/active-learning/runs` · `/runs/{id}`            | List / detail                 |
| GET    | `/active-learning/runs/{id}/queue`                | Read the labeling queue       |
| POST   | `/active-learning/runs/{id}/queue/{item}/annotated` | Mark item annotated         |
| POST   | `/active-learning/runs/{id}/push-labelstudio`     | Push queue to Label Studio    |
| POST   | `/watches` · `GET` · `DELETE /watches/{id}`       | Continuous-training watches   |
| POST   | `/watches/{id}/enable` · `/disable`               | Toggle a watch                |
| POST   | `/synth`                                          | Synthesize a dataset          |
| POST   | `/compliance/forget-scan`                         | GDPR substring/regex scan     |

**Acquisitions (agentic data acquisition)**

| Method | Path                               | Purpose                             |
| ------ | ---------------------------------- | ----------------------------------- |
| POST   | `/acquisitions`                    | Start a run from a brief            |
| GET    | `/acquisitions`                    | List (filter: status)               |
| GET    | `/acquisitions/{id}`               | Detail (phase, counts, dataset_id)  |
| GET    | `/acquisitions/{id}/sources`       | Web sources considered (audit)      |
| PATCH  | `/acquisitions/{id}/answers`       | Answer clarifying questions         |
| POST   | `/acquisitions/{id}/cancel`        | Cancel a run                        |

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

The 40 tools mirror the REST surface, grouped by resource:

- **Experiments / GPUs**: `submit_experiment`, `get_experiment`,
  `list_experiments`, `cancel_experiment`, `tail_logs`, `gpu_status`
- **Studies**: `submit_study`, `list_studies`, `get_study`, `cancel_study`
- **Datasets**: `upload_dataset`, `list_datasets`, `get_dataset`,
  `preview_dataset`, `delete_dataset`, `synth_dataset`
- **Models / inference**: `register_model`, `list_models`, `get_model`,
  `set_alias`, `delete_model`, `inference`, `inference_compare`
- **Evals**: `create_eval_suite`, `list_eval_suites`, `get_eval_suite`,
  `delete_eval_suite`, `run_eval`, `list_eval_runs`, `get_eval_run`,
  `get_eval_results`, `cancel_eval_run`, `compare_evals`
- **Acquisition**: `start_acquisition`, `get_acquisition`,
  `get_acquisition_sources`, `list_acquisitions`, `answer_acquisition`,
  `cancel_acquisition`
- **Compliance**: `forget_scan`

Auth never leaks into the model context — the API key stays inside the
MCP server process; the agent only sees tool calls and their results.

## Configuration

All settings are prefixed `TRAINPIPE_` and loaded from `.env` or the
environment

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
│   ├── schemas.py            ExperimentSpec, StudyConfig, AcquisitionRequest, …
│   ├── validation.py         submit-time dataset/path checks
│   └── routes/               one module per resource (experiments, gpus,
│                             studies, datasets, evals, models, inferences,
│                             pipelines, active_learning, watches, synth,
│                             compliance, acquisitions)
├── core/
│   ├── db.py                 aiosqlite, WAL, versioned migrations
│   └── repository.py         CRUD for experiments, studies, events
├── scheduler/
│   ├── gpu_pool.py           pynvml detection + SQLite-backed leases
│   ├── runner.py             asyncio subprocess + POSIX process group
│   └── loop.py               dispatch + monitor + MLflow run creation
├── training/
│   ├── swift_builder.py      ExperimentSpec → (argv, env); sft/pt/rlhf
│   ├── dataset_refs.py       resolve ds:<id> refs at submit time
│   └── dataset_formats.py    JSONL/Parquet format detection + preview
├── autoresearch/
│   ├── search_spaces.py      dotted-path overrides + suggest_* dispatch
│   ├── study.py              StudyDriver: ask → enqueue → wait → tell
│   └── manager.py            owns drivers in the API process
├── evals/                    suites, runner, dispatcher, metrics/
├── inference/                adapter cache + predict service
├── quantization/             post-training quantization runner
├── pipelines/                multi-stage workflow driver + manager
├── active_learning/          uncertainty sampling + labeling queue
├── watches/                  continuous-training watches
├── synth/                    synthetic-dataset runner
├── acquisition/              agentic data acquisition
│   ├── driver.py             phase machine: intake→research→acquire→
│   │                         synthesize→curate→register
│   ├── manager.py            owns acquisition drivers + status transitions
│   ├── web.py                search providers, license gate, SSRF/robots
│   └── runner.py             teacher-LLM providers (+ budget wrapper)
├── redaction/                PII redactor (shared by datasets + acquisition)
├── compliance/               GDPR forget-scan + CLI
├── integrations/             Label Studio import/push
├── core/repository.py, settings.py
├── client.py                 shared httpx client (CLI + MCP)
├── cli.py                    `trainpipe` entry point (serve + operative client)
├── mcp.py                    `trainpipe-mcp` — 40 MCP tools over the REST API
└── ui/index.html             single-page web UI
```

## Development

```bash
pip install -e ".[dev]"
pytest                              # the full suite (500+ tests) should pass
ruff check trainpipe tests
```

The behavior-first specs per subsystem live in [docs/spec/](docs/spec/);
the long-form user/operator manual is [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## License

MIT.
