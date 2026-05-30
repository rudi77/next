# Live-Server Smoke Harness

`.scripts/smoke_e2e.py` exercises every user-facing feature of a running
trainpipe deployment in one command. Treats the server as a black box —
only talks REST (and optionally the `trainpipe-forget` CLI).

The 470 in-process pytest suite is the inner ring (run with `pytest`);
this script is the outer ring proving the seams between the FastAPI
process, SQLite, MLflow tags, and the filesystem all work together at
runtime.

## Prerequisites

- Python 3.10+ with `httpx` installed (already a runtime dep of trainpipe).
- A running trainpipe server reachable from the host. On the dev box:
  ```
  wsl -d Ubuntu-24.04 -- /home/rudi/src/next/.scripts/next.sh status
  ```
- The server's API key, either via env or `--key`.

## Run

```bash
python .scripts/smoke_e2e.py \
  --url http://192.168.2.213:8080 \
  --key local-dev-secret
```

Equivalent with env:

```bash
TRAINPIPE_BASE_URL=http://192.168.2.213:8080 \
TRAINPIPE_API_KEY=local-dev-secret \
python .scripts/smoke_e2e.py
```

A green run looks like:

```
trainpipe smoke - http://192.168.2.213:8080
run_id: 20260530070000-a3f4d2

  [s00 health-public               ] PASS  3 steps   0.04s
  [s01 datasets-jsonl              ] PASS  5 steps   0.18s
  [s02 datasets-split              ] PASS  3 steps   0.21s
  ...
  cleaning up 14 resources... 14 ok
==================================================
  20 sections | 19 passed | 0 failed | 1 skipped
  Report: .run/smoke-report.json
```

Exit code:
- **0** all sections passed (skipped count doesn't fail the run)
- **1** any section failed
- **2** server unreachable or no API key

## Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--url` | `$TRAINPIPE_BASE_URL` or `http://127.0.0.1:8080` | Base URL of the live server |
| `--key` | `$TRAINPIPE_API_KEY` | API key for `X-API-Key` header |
| `--only s01,s05` | (all) | Run a subset; dependencies pulled in automatically |
| `--keep` | off | Skip the LIFO cleanup step (for manual inspection) |
| `--report PATH` | `.run/smoke-report.json` | Where to write the JSON report |
| `--timeout 30` | 30 | HTTP read timeout per request, seconds |
| `--with-cli` | off | Also run s20-compliance-cli (`trainpipe-forget`) |
| `--verbose` | off | Per-step error detail + cleanup misses |

## Sections

| # | What it proves |
| --- | --- |
| s00 health-public | `/health`, `/ui/config` reachable without auth; `/gpus` returns 401 without key |
| s01 datasets-jsonl | Upload, list, preview, sha256 dedup, get-by-id |
| s02 datasets-split | `/datasets/{id}/split` produces two derived datasets with `version=2` and seeded determinism |
| s03 datasets-mix | `/datasets/mixes` N-source mix, all parents recorded in `dataset_lineage` |
| s04 datasets-bundle | Zip upload with images, `/media` serves files, `..` and symlink members rejected |
| s05 datasets-redact | PII redactor rewrites emails, new dataset has redaction-from lineage |
| s06 experiments | `gpu_count=99` rejected by scheduler; submit + cancel state machine; `/logs` shape |
| s07 evals-suites | Suite CRUD over a `ds:` ref; metric config validated |
| s08 evals-runs | Manual eval run trigger against an existing completed experiment + cancel |
| s09 models | Register v1 with alias, auto-increment v2, alias move, `/{id}/datasets` lineage |
| s10 models-quantize | `POST /quantize` returns structured error envelope (no swift on test host) |
| s11 inferences-cache | `GET /inferences/cache` shape — no real model load |
| s12 pipelines | 2-stage DAG, driver enqueues stage A, immediate cancel |
| s13 active-learning | Tiny AL run reaches terminal state (loads small model live) |
| s14 watches | Watch CRUD + enable/disable transitions |
| s15 synth-mock | `provider="mock"` generates a registered dataset with `_source` provenance |
| s16 studies | `/studies` + `/studies/cost-summary` shape |
| s17 gpus | `/gpus` shape — total/free/leases |
| s18 ds-ref-versioning | `ds:<id>@v1` resolves; `@v99` returns 422 |
| s19 datasets-gdpr-recursive | `?recursive=true` parameter on `/datasets/{id}/models` returns the right shape |
| s20 compliance-cli | (`--with-cli` only) `trainpipe-forget` runs and emits a valid JSON report |

## What's intentionally skipped

Reported as `status="skip"` with a `covered_by` pointer:

| Feature | Reason | Covered by |
| --- | --- | --- |
| Quantize **happy** path | Server has no in-prod backend override hook | `tests/test_phase19_quantize.py` |
| Label Studio import | SSRF guard blocks loopback in production | `tests/test_phase10_labelstudio.py` |
| Synth real provider | Needs Anthropic/OpenAI API key | `tests/test_phase14_synth.py` |
| Inference real load | Skipped in s11; the AL run in s13 exercises a real load anyway | `tests/test_inference.py` |

## Idempotency contract

- Every created resource is named `smoke-<run_id>-<slug>`. `run_id` is
  `UTC YYYYMMDDHHMMSS + 6 hex chars` — unique per invocation.
- Content always embeds `run_id` so SHA256 dedup never collides across
  runs.
- **Pre-flight** drops any `smoke-*` resource whose embedded timestamp
  is older than 1 hour. Bounds drift from killed runs.
- **Cleanup** runs in LIFO order in the script's exit path. Failures
  are logged (`--verbose`) and don't fail the run — a leftover smoke-*
  resource will be cleaned on the next run's pre-flight.
- `--keep` short-circuits cleanup. Useful when you want to inspect what
  the script created.

A second invocation immediately after the first must produce the same
pass/fail counts and exit 0 — that's the repeatability test.

## Report file shape

```json
{
  "run_id": "20260530070000-a3f4d2",
  "started": "20260530070000",
  "finished": "20260530070043",
  "server": {"url": "http://192.168.2.213:8080"},
  "sections": [
    {
      "id": "s00",
      "name": "health-public",
      "status": "pass",
      "duration_ms": 41.2,
      "steps": [
        {"name": "GET /health (no auth)", "ok": true, "duration_ms": 8.1},
        {"name": "GET /ui/config (no auth, no credentials leaked)", "ok": true, "duration_ms": 5.4}
      ],
      "cleaned": [],
      "skip_reason": null,
      "covered_by": null,
      "error": null
    }
  ],
  "summary": {"passed": 19, "failed": 0, "skipped": 1, "total": 20},
  "cleaned": ["DELETE /datasets/...", ...]
}
```

`jq` over this is a clean way to wire smoke into CI:

```bash
python -c "import json; r=json.load(open('.run/smoke-report.json')); \
  exit(0 if r['summary']['failed']==0 else 1)"
```

## Troubleshooting

**`error: cannot reach <url>`**
The server isn't running, or the LAN IP isn't reachable. From WSL on the
dev box: `wsl -d Ubuntu-24.04 -- /home/rudi/src/next/.scripts/next.sh status`.

**`401 Unauthorized`** on the first real call after `--key` is set
Check the value matches `.env` in the WSL checkout:
```
wsl -d Ubuntu-24.04 -- grep TRAINPIPE_API_KEY /home/rudi/src/next/.env
```

**`s06 experiments fails with "expected failed, got running"`**
Scheduler is slow on a busy box. Re-run when GPU pool is quiet. The
section waits ~10s for the scheduler to mark the bad spec failed.

**`s13 active-learning fails with timeout`**
The default backend tries to load Qwen2.5-0.5B-Instruct. On a 4GB GPU
with another job running, this can OOM. Re-run when GPU is free, or
skip with `--only` listing every section except `s13`.

**Stale `smoke-*` rows after a killed run**
Re-run the script; pre-flight will drop them. Or manually:
```
curl -s -H "X-API-Key: $KEY" "$URL/datasets" | \
  jq -r '.[] | select(.name | startswith("smoke-")) | .id' | \
  xargs -I{} curl -s -X DELETE -H "X-API-Key: $KEY" "$URL/datasets/{}?force=true"
```

## Wiring into CI

```bash
#!/usr/bin/env bash
set -e
python .scripts/smoke_e2e.py --url "$SMOKE_URL" --key "$SMOKE_KEY"
```

Exit 1 on any failure. The JSON report at `.run/smoke-report.json` is a
suitable build artifact.
