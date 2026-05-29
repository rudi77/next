"""MCP server that exposes trainpipe operations as tools.

Wraps the REST API at ``TRAINPIPE_BASE_URL`` with ``TRAINPIPE_API_KEY``.
Tools cover experiments, studies, GPUs, and datasets — the same surface
a human would drive via ``curl``.

Run standalone (e.g. for `claude mcp add`):

    pip install -e ".[mcp]"
    TRAINPIPE_API_KEY=... python -m trainpipe.mcp

Add to Claude Code:

    claude mcp add trainpipe -- \
      env TRAINPIPE_API_KEY=$TRAINPIPE_API_KEY \
          TRAINPIPE_BASE_URL=http://localhost:8080 \
          python -m trainpipe.mcp
"""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "trainpipe.mcp requires the 'mcp' package. "
        "Install with: pip install -e \".[mcp]\""
    ) from exc


def _build_client() -> httpx.Client:
    base_url = os.environ.get("TRAINPIPE_BASE_URL", "http://127.0.0.1:8080")
    api_key = os.environ.get("TRAINPIPE_API_KEY")
    if not api_key:
        raise SystemExit(
            "TRAINPIPE_API_KEY environment variable must be set "
            "(use the same value you put in trainpipe's .env)"
        )
    return httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": api_key},
        timeout=httpx.Timeout(30.0, connect=5.0),
    )


def _unwrap(resp: httpx.Response) -> Any:
    """Raise on HTTP errors; return parsed body for JSON, text otherwise."""
    if resp.is_error:
        # Re-raise with body preserved so MCP clients see the actionable detail.
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        return resp.json()
    return resp.text


mcp = FastMCP("trainpipe")
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """Return the shared httpx client, building it on first use.

    Deferred so that ``import trainpipe.mcp`` doesn't require
    TRAINPIPE_API_KEY (e.g. when introspecting the tool list from tests).
    """
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@mcp.tool()
def submit_experiment(spec: dict) -> dict:
    """Queue a fine-tuning job.

    ``spec`` follows the ExperimentSpec schema. Minimum keys: ``model``
    (HF/ms-swift id or local path) and ``dataset`` (list of dataset
    references — paths, HF ids, or ``ds:<dataset_id>`` to point at an
    uploaded file). Returns ``{"experiment_id": "..."}``.
    """
    return _unwrap(_get_client().post("/experiments", json=spec))


@mcp.tool()
def get_experiment(experiment_id: str) -> dict:
    """Return the current state of one experiment (status, gpu_ids,
    mlflow_run_id, error, timestamps, full spec)."""
    return _unwrap(_get_client().get(f"/experiments/{experiment_id}"))


@mcp.tool()
def list_experiments(
    status: str | None = None,
    study_id: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """List experiments, optionally filtered by status (queued, running,
    completed, failed, cancelled) and/or study_id."""
    params: dict[str, Any] = {"limit": limit}
    if status:
        params["status"] = status
    if study_id:
        params["study_id"] = study_id
    return _unwrap(_get_client().get("/experiments", params=params))


@mcp.tool()
def cancel_experiment(experiment_id: str) -> dict:
    """Cancel a queued or running experiment."""
    return _unwrap(_get_client().post(f"/experiments/{experiment_id}/cancel"))


@mcp.tool()
def tail_logs(experiment_id: str, n_lines: int = 80) -> str:
    """Return the last ``n_lines`` of the experiment's training log."""
    body = _unwrap(_get_client().get(f"/experiments/{experiment_id}/logs"))
    if not isinstance(body, str):
        body = str(body)
    lines = body.splitlines()
    return "\n".join(lines[-n_lines:])


# ---------------------------------------------------------------------------
# Studies (Optuna sweeps)
# ---------------------------------------------------------------------------


@mcp.tool()
def submit_study(config: dict) -> dict:
    """Start an Optuna study. ``config`` follows StudyConfig: name, base_spec,
    search_space (dotted-path → range), target_metric, direction, n_trials,
    max_concurrent, sampler. Returns ``{"study_id": "..."}``."""
    return _unwrap(_get_client().post("/studies", json=config))


@mcp.tool()
def list_studies() -> list[dict]:
    """List all studies and their progress."""
    return _unwrap(_get_client().get("/studies"))


@mcp.tool()
def get_study(study_id: str) -> dict:
    """Return study detail including best_value and best_trial_id."""
    return _unwrap(_get_client().get(f"/studies/{study_id}"))


@mcp.tool()
def cancel_study(study_id: str) -> dict:
    """Stop the study driver and mark it cancelled."""
    return _unwrap(_get_client().post(f"/studies/{study_id}/cancel"))


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


@mcp.tool()
def gpu_status() -> dict:
    """Pool state: total GPUs, list of free indices, lease detail."""
    return _unwrap(_get_client().get("/gpus"))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@mcp.tool()
def upload_dataset(
    name: str,
    filename: str,
    content_b64: str,
    description: str | None = None,
) -> dict:
    """Register a dataset by uploading its base64-encoded content.

    Format is inferred from ``filename``'s extension (jsonl / json / csv /
    tsv / parquet). For files larger than ~10 MB prefer asking the user to
    upload directly via ``curl -F file=@…`` to avoid base64 overhead.
    """
    try:
        raw = base64.b64decode(content_b64, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise RuntimeError(f"content_b64 is not valid base64: {e}") from None
    files = {"file": (filename, raw, "application/octet-stream")}
    data: dict[str, str] = {"name": name}
    if description:
        data["description"] = description
    return _unwrap(_get_client().post("/datasets", files=files, data=data))


@mcp.tool()
def list_datasets() -> list[dict]:
    """List all registered datasets."""
    return _unwrap(_get_client().get("/datasets"))


@mcp.tool()
def get_dataset(dataset_id: str) -> dict:
    """Return dataset detail (path, format, line_count, sha256, ...)."""
    return _unwrap(_get_client().get(f"/datasets/{dataset_id}"))


@mcp.tool()
def preview_dataset(dataset_id: str, n: int = 10) -> str:
    """Return the first ``n`` lines of a text dataset for quick inspection."""
    body = _unwrap(_get_client().get(f"/datasets/{dataset_id}/preview", params={"n": n}))
    return body if isinstance(body, str) else str(body)


@mcp.tool()
def delete_dataset(dataset_id: str) -> dict:
    """Delete a dataset (removes the file and DB row)."""
    return _unwrap(_get_client().delete(f"/datasets/{dataset_id}"))


def main() -> None:
    """Entrypoint for ``python -m trainpipe.mcp`` and the trainpipe-mcp script."""
    mcp.run()


if __name__ == "__main__":
    main()
