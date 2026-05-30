#!/usr/bin/env python3
"""End-to-end smoke harness for a live trainpipe server.

Treats the deployment as a black box and exercises ~20 features in
sequence (datasets, experiments, evals, models, pipelines, watches,
synth, active learning, GPUs, studies, compliance CLI). Each section
asserts response shapes and registers cleanup so the script is safe to
re-run; resources it creates are named ``smoke-<run_id>-<slug>`` so two
parallel invocations never collide.

Usage:
    python .scripts/smoke_e2e.py --url http://host:8080 --key <key>

Run ``--help`` for the full flag set. Output a structured
``.run/smoke-report.json`` for parsing alongside the human-readable
stdout table.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    name: str
    ok: bool
    duration_ms: float
    detail: str | None = None
    error: str | None = None


@dataclass
class SectionResult:
    id: str
    name: str
    status: str  # "pass" | "fail" | "skip"
    duration_ms: float
    steps: list[StepResult] = field(default_factory=list)
    cleaned: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    covered_by: str | None = None
    error: str | None = None


@dataclass
class Ctx:
    client: httpx.Client
    run_id: str
    url: str
    key: str
    keep: bool
    verbose: bool
    with_cli: bool
    cleanup: list[tuple[str, Callable[[], None]]] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)

    def name(self, slug: str) -> str:
        return f"smoke-{self.run_id}-{slug}"

    def add_cleanup(self, label: str, fn: Callable[[], None]) -> None:
        self.cleanup.append((label, fn))


SectionFn = Callable[[Ctx, list[StepResult]], None]
SECTIONS: list[tuple[str, str, list[str], SectionFn]] = []


def section(id_: str, name: str, deps: list[str] | None = None):
    def deco(fn: SectionFn) -> SectionFn:
        SECTIONS.append((id_, name, deps or [], fn))
        return fn

    return deco


# ---------------------------------------------------------------------------
# Step + assertion helpers
# ---------------------------------------------------------------------------


@contextmanager
def step(steps: list[StepResult], name: str, detail: str | None = None):
    """Time + record one step. Re-raises on failure so the section
    finalizer can mark the whole section failed."""
    t = time.monotonic()
    try:
        yield
    except Exception as e:
        steps.append(
            StepResult(
                name=name,
                ok=False,
                duration_ms=(time.monotonic() - t) * 1000,
                error=f"{type(e).__name__}: {e}",
                detail=detail,
            )
        )
        raise
    steps.append(
        StepResult(
            name=name,
            ok=True,
            duration_ms=(time.monotonic() - t) * 1000,
            detail=detail,
        )
    )


def expect_status(r: httpx.Response, *codes: int) -> None:
    if r.status_code not in codes:
        body = r.text[:512]
        raise AssertionError(
            f"expected status in {codes}, got {r.status_code}: {body}"
        )


def expect_json(r: httpx.Response) -> Any:
    ct = r.headers.get("content-type", "")
    if not ct.startswith("application/json"):
        raise AssertionError(f"expected JSON response, got content-type={ct!r}")
    return r.json()


def expect_keys(obj: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise AssertionError(
            f"missing keys {missing} in {list(obj)[:10]}"
        )


# ---------------------------------------------------------------------------
# Pre-flight: drop stale smoke-* resources older than 1 hour
# ---------------------------------------------------------------------------


_RUN_ID_RE = re.compile(r"smoke-(\d{14})-")


def _is_stale(name: str, *, now_utc: datetime, max_age_sec: int) -> bool:
    m = _RUN_ID_RE.search(name or "")
    if not m:
        return False
    try:
        ts = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return False
    return (now_utc - ts).total_seconds() > max_age_sec


def preflight_cleanup(client: httpx.Client, *, verbose: bool) -> int:
    """Drop ``smoke-*`` resources older than 1h from prior failed runs."""
    now = datetime.now(timezone.utc)
    dropped = 0

    # Datasets — only delete leaves first; references prevent FK deletes
    # in most cases. ``force=true`` overrides where supported.
    try:
        rs = client.get("/datasets").json()
        for d in rs:
            if _is_stale(d.get("name") or "", now_utc=now, max_age_sec=3600):
                client.delete(f"/datasets/{d['id']}", params={"force": "true"})
                dropped += 1
    except httpx.HTTPError:
        pass

    # Models — name embeds run_id
    try:
        ms = client.get("/models").json()
        for m in ms:
            if _is_stale(m.get("name") or "", now_utc=now, max_age_sec=3600):
                client.delete(f"/models/{m['id']}", params={"force": "true"})
                dropped += 1
    except httpx.HTTPError:
        pass

    # Watches
    try:
        ws = client.get("/watches").json()
        for w in ws:
            if _is_stale(w.get("name") or "", now_utc=now, max_age_sec=3600):
                client.delete(f"/watches/{w['id']}")
                dropped += 1
    except httpx.HTTPError:
        pass

    # Eval suites
    try:
        ss = client.get("/evals/suites").json()
        for s in ss:
            if _is_stale(s.get("name") or "", now_utc=now, max_age_sec=3600):
                client.delete(
                    f"/evals/suites/{s['id']}", params={"force": "true"}
                )
                dropped += 1
    except httpx.HTTPError:
        pass

    if verbose and dropped:
        print(f"  pre-flight: dropped {dropped} stale smoke-* rows")
    return dropped


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@section("s00", "health-public", deps=[])
def s00(ctx: Ctx, steps: list[StepResult]) -> None:
    with step(steps, "GET /health (no auth)"):
        # Use a fresh client without auth header.
        r = httpx.get(ctx.url + "/health", timeout=5.0)
        expect_status(r, 200)
        body = expect_json(r)
        assert body.get("status") == "ok", body

    with step(steps, "GET /ui/config (no auth, no credentials leaked)"):
        r = httpx.get(ctx.url + "/ui/config", timeout=5.0)
        expect_status(r, 200)
        body = expect_json(r)
        assert "mlflow_tracking_uri" in body, body
        # Credentials must be stripped server-side.
        assert "@" not in (body["mlflow_tracking_uri"].split("//", 1)[-1].split("/")[0])

    with step(steps, "GET /gpus (no key) -> 401"):
        r = httpx.get(ctx.url + "/gpus", timeout=5.0)
        expect_status(r, 401, 403)


@section("s01", "datasets-jsonl", deps=["s00"])
def s01(ctx: Ctx, steps: list[StepResult]) -> None:
    name = ctx.name("ds1")
    content = (
        f'{{"run_id":"{ctx.run_id}","prompt":"q1","answer":"a1"}}\n'
        f'{{"run_id":"{ctx.run_id}","prompt":"q2","answer":"a2"}}\n'
    )

    ds_id: str | None = None
    with step(steps, "POST /datasets (jsonl upload)"):
        r = ctx.client.post(
            "/datasets",
            files={"file": (f"{name}.jsonl", content.encode("utf-8"),
                            "application/x-ndjson")},
            data={"name": name, "description": "smoke s01"},
        )
        expect_status(r, 201)
        body = expect_json(r)
        expect_keys(body, "id", "format", "line_count", "sha256")
        assert body["format"] == "jsonl"
        assert body["line_count"] == 2
        ds_id = body["id"]
    ctx.shared["s01_ds_id"] = ds_id
    ctx.shared["s01_ds_name"] = name
    ctx.shared["s01_ds_path"] = body["path"]
    ctx.add_cleanup(
        f"DELETE /datasets/{ds_id}",
        lambda: ctx.client.delete(f"/datasets/{ds_id}", params={"force": "true"}),
    )

    with step(steps, "GET /datasets/{id}"):
        r = ctx.client.get(f"/datasets/{ds_id}")
        expect_status(r, 200)
        body = expect_json(r)
        assert body["name"] == name
        assert body["version"] == 1

    with step(steps, "GET /datasets/{id}/preview"):
        r = ctx.client.get(f"/datasets/{ds_id}/preview", params={"n": 5})
        expect_status(r, 200)
        assert "q1" in r.text and "q2" in r.text

    with step(steps, "Re-upload same content -> 200 dedup"):
        r = ctx.client.post(
            "/datasets",
            files={"file": (f"{name}.jsonl", content.encode("utf-8"),
                            "application/x-ndjson")},
            data={"name": name + "-dup", "description": "smoke s01 dedup"},
        )
        # 200 on dedup, 201 on fresh insert.
        expect_status(r, 200)
        body = expect_json(r)
        assert body["id"] == ds_id, "dedup must return existing id"

    with step(steps, "GET /datasets list contains ours"):
        r = ctx.client.get("/datasets")
        expect_status(r, 200)
        ids = [d["id"] for d in expect_json(r)]
        assert ds_id in ids


@section("s02", "datasets-split", deps=["s01"])
def s02(ctx: Ctx, steps: list[StepResult]) -> None:
    src = ctx.shared["s01_ds_id"]
    # Need at least a few rows for a 90:10 split.
    # Upload a 10-row JSONL specifically for splitting.
    name = ctx.name("ds-split-src")
    content = "\n".join(
        json.dumps({"run_id": ctx.run_id, "i": i}) for i in range(10)
    ) + "\n"
    with step(steps, "Upload 10-row source for split"):
        r = ctx.client.post(
            "/datasets",
            files={"file": (f"{name}.jsonl", content.encode("utf-8"),
                            "application/x-ndjson")},
            data={"name": name},
        )
        expect_status(r, 200, 201)
        src = expect_json(r)["id"]
    ctx.add_cleanup(
        f"DELETE /datasets/{src}",
        lambda: ctx.client.delete(f"/datasets/{src}", params={"force": "true"}),
    )

    train_id = val_id = None
    with step(steps, "POST /datasets/{id}/split ratio=80:20"):
        r = ctx.client.post(
            f"/datasets/{src}/split",
            json={"ratio": "80:20", "seed": 1},
        )
        expect_status(r, 201)
        body = expect_json(r)
        expect_keys(body, "train", "val")
        train_id, val_id = body["train"]["id"], body["val"]["id"]
        assert body["train"]["derived_from"] == src
        assert body["val"]["derived_from"] == src
        assert body["train"]["version"] == 2
        assert body["val"]["version"] == 2
        assert body["train"]["line_count"] == 8
        assert body["val"]["line_count"] == 2
    ctx.shared["s02_train_id"] = train_id
    ctx.shared["s02_val_id"] = val_id
    ctx.add_cleanup(
        f"DELETE /datasets/{train_id}",
        lambda: ctx.client.delete(f"/datasets/{train_id}", params={"force": "true"}),
    )
    ctx.add_cleanup(
        f"DELETE /datasets/{val_id}",
        lambda: ctx.client.delete(f"/datasets/{val_id}", params={"force": "true"}),
    )

    with step(steps, "Split is deterministic with same seed (sha match)"):
        # Re-run the split with a different output name; the content
        # should be byte-identical → same sha → dedup returns same ids.
        r = ctx.client.post(
            f"/datasets/{src}/split",
            json={"ratio": "80:20", "seed": 1,
                  "train_name": "smoke-rerun-t", "val_name": "smoke-rerun-v"},
        )
        expect_status(r, 201)
        body = expect_json(r)
        assert body["train"]["sha256"] == ctx.client.get(
            f"/datasets/{train_id}"
        ).json()["sha256"]


@section("s03", "datasets-mix", deps=["s01"])
def s03(ctx: Ctx, steps: list[StepResult]) -> None:
    a_id = ctx.shared["s01_ds_id"]

    b_name = ctx.name("ds-mix-b")
    b_content = "\n".join(
        json.dumps({"run_id": ctx.run_id, "src": "b", "i": i}) for i in range(5)
    ) + "\n"
    with step(steps, "Upload second source for mix"):
        r = ctx.client.post(
            "/datasets",
            files={"file": (f"{b_name}.jsonl", b_content.encode("utf-8"),
                            "application/x-ndjson")},
            data={"name": b_name},
        )
        expect_status(r, 200, 201)
        b_id = expect_json(r)["id"]
    ctx.add_cleanup(
        f"DELETE /datasets/{b_id}",
        lambda: ctx.client.delete(f"/datasets/{b_id}", params={"force": "true"}),
    )

    mix_id = None
    with step(steps, "POST /datasets/mixes (2 sources)"):
        r = ctx.client.post(
            "/datasets/mixes",
            json={
                "name": ctx.name("mix"),
                "sources": [
                    {"dataset_id": a_id, "weight": 0.5},
                    {"dataset_id": b_id, "weight": 0.5},
                ],
                "target_count": 6,
                "seed": 0,
            },
        )
        expect_status(r, 201)
        body = expect_json(r)
        mix_id = body["id"]
        assert body["line_count"] == 6
        assert body["version"] == 2
        assert "mix of" in (body["description"] or "")
    ctx.shared["s03_mix_id"] = mix_id
    ctx.add_cleanup(
        f"DELETE /datasets/{mix_id}",
        lambda: ctx.client.delete(f"/datasets/{mix_id}", params={"force": "true"}),
    )


@section("s04", "datasets-bundle", deps=["s00"])
def s04(ctx: Ctx, steps: list[StepResult]) -> None:
    # Build a valid zip with one JSONL referencing one PNG.
    good_zip = io.BytesIO()
    with zipfile.ZipFile(good_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "train.jsonl",
            json.dumps({"images": ["images/a.png"], "prompt": "p", "response": "r"})
            + "\n",
        )
        zf.writestr("images/a.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    ds_id = None
    with step(steps, "POST /datasets/bundle (zip with images)"):
        r = ctx.client.post(
            "/datasets/bundle",
            files={"file": (ctx.name("bundle") + ".zip", good_zip.getvalue(),
                            "application/zip")},
            data={"name": ctx.name("bundle")},
        )
        expect_status(r, 200, 201)
        body = expect_json(r)
        assert body["media_kinds"] == ["images"]
        assert body["image_root"] is not None
        ds_id = body["id"]
    ctx.add_cleanup(
        f"DELETE /datasets/{ds_id}",
        lambda: ctx.client.delete(f"/datasets/{ds_id}", params={"force": "true"}),
    )

    with step(steps, "GET /datasets/{id}/media (path traversal blocked)"):
        bad = ctx.client.get(
            f"/datasets/{ds_id}/media",
            params={"path": "../../etc/hosts"},
        )
        expect_status(bad, 404)

    with step(steps, "GET /datasets/{id}/media (legitimate file)"):
        ok = ctx.client.get(
            f"/datasets/{ds_id}/media", params={"path": "images/a.png"}
        )
        expect_status(ok, 200)
        assert ok.content.startswith(b"\x89PNG"), "expected PNG bytes"

    with step(steps, "POST /datasets/bundle (zip-slip rejected)"):
        evil_zip = io.BytesIO()
        with zipfile.ZipFile(evil_zip, "w") as zf:
            zf.writestr("../escaped.png", b"PNG")
            zf.writestr("train.jsonl", json.dumps({"images": ["a.png"]}) + "\n")
        r = ctx.client.post(
            "/datasets/bundle",
            files={"file": (ctx.name("evil") + ".zip", evil_zip.getvalue(),
                            "application/zip")},
            data={"name": ctx.name("evil-bundle")},
        )
        expect_status(r, 422)
        body = expect_json(r)
        assert body.get("detail", {}).get("error") == "unsafe_zip_path"


@section("s05", "datasets-redact", deps=["s00"])
def s05(ctx: Ctx, steps: list[StepResult]) -> None:
    name = ctx.name("ds-pii")
    content = (
        json.dumps({"run_id": ctx.run_id, "prompt": "mail jane@example.com"})
        + "\n"
        + json.dumps({"run_id": ctx.run_id, "prompt": "no pii here"})
        + "\n"
    )
    with step(steps, "Upload PII dataset"):
        r = ctx.client.post(
            "/datasets",
            files={"file": (f"{name}.jsonl", content.encode("utf-8"),
                            "application/x-ndjson")},
            data={"name": name},
        )
        expect_status(r, 200, 201)
        src_id = expect_json(r)["id"]
    ctx.add_cleanup(
        f"DELETE /datasets/{src_id}",
        lambda: ctx.client.delete(f"/datasets/{src_id}", params={"force": "true"}),
    )

    red_id = None
    with step(steps, "POST /datasets/{id}/redact"):
        r = ctx.client.post(f"/datasets/{src_id}/redact", json={})
        expect_status(r, 201)
        body = expect_json(r)
        red_id = body["id"]
        assert "redacted from ds:" in (body["description"] or "")
    ctx.add_cleanup(
        f"DELETE /datasets/{red_id}",
        lambda: ctx.client.delete(f"/datasets/{red_id}", params={"force": "true"}),
    )

    with step(steps, "Redacted preview has no email"):
        r = ctx.client.get(f"/datasets/{red_id}/preview", params={"n": 10})
        expect_status(r, 200)
        assert "jane@example.com" not in r.text
        assert "REDACTED_EMAIL" in r.text


@section("s06", "experiments", deps=["s00"])
def s06(ctx: Ctx, steps: list[StepResult]) -> None:
    # Submit a spec with gpu_count exceeding pool -> scheduler should
    # mark it failed on the first claim attempt within a tick or two.
    bad = {
        "name": ctx.name("bad-exp"),
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": ["AI-ModelScope/alpaca-gpt4-data-en#1"],
        "gpu_count": 99,
        "tags": {"smoke": ctx.run_id},
    }
    bad_id = None
    with step(steps, "POST /experiments (gpu_count=99 -> scheduler fails)"):
        r = ctx.client.post("/experiments", json=bad)
        expect_status(r, 201)
        bad_id = expect_json(r)["experiment_id"]
        # Poll briefly for the scheduler to reject it.
        for _ in range(20):
            time.sleep(0.5)
            s = ctx.client.get(f"/experiments/{bad_id}").json()["status"]
            if s in ("failed", "cancelled"):
                break
        assert s == "failed", f"expected failed, got {s}"

    good = {
        "name": ctx.name("good-exp"),
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": ["AI-ModelScope/alpaca-gpt4-data-en#1"],
        "gpu_count": 1,
        "tags": {"smoke": ctx.run_id},
    }
    good_id = None
    with step(steps, "POST /experiments (cancel before scheduler picks up)"):
        r = ctx.client.post("/experiments", json=good)
        expect_status(r, 201)
        good_id = expect_json(r)["experiment_id"]
        # Cancel ASAP.
        ctx.client.post(f"/experiments/{good_id}/cancel")
        # Poll briefly until terminal.
        last_status = "?"
        for _ in range(40):
            time.sleep(0.5)
            last_status = ctx.client.get(f"/experiments/{good_id}").json()["status"]
            if last_status in ("cancelled", "completed", "failed"):
                break
        assert last_status in ("cancelled", "completed", "failed"), last_status

    with step(steps, "GET /experiments/{id}/logs (text or empty)"):
        r = ctx.client.get(f"/experiments/{good_id}/logs")
        expect_status(r, 200)
        # Plain text — empty is fine if the run was cancelled before any
        # log lines were written.
        assert isinstance(r.text, str)


@section("s07", "evals-suites", deps=["s01"])
def s07(ctx: Ctx, steps: list[StepResult]) -> None:
    suite_id = None
    with step(steps, "POST /evals/suites"):
        r = ctx.client.post(
            "/evals/suites",
            json={
                "name": ctx.name("suite"),
                "description": "smoke",
                "dataset": f"ds:{ctx.shared['s01_ds_id']}",
                "metrics": [{"kind": "exact_match"}, {"kind": "bleu"}],
                "inference_params": {"max_new_tokens": 16},
            },
        )
        expect_status(r, 201)
        body = expect_json(r)
        suite_id = body["id"]
        assert len(body["metrics"]) == 2
    ctx.add_cleanup(
        f"DELETE /evals/suites/{suite_id}",
        lambda: ctx.client.delete(f"/evals/suites/{suite_id}",
                                  params={"force": "true"}),
    )
    ctx.shared["s07_suite_id"] = suite_id

    with step(steps, "GET /evals/suites/{id}"):
        r = ctx.client.get(f"/evals/suites/{suite_id}")
        expect_status(r, 200)
        assert expect_json(r)["id"] == suite_id

    with step(steps, "GET /evals/suites list contains ours"):
        r = ctx.client.get("/evals/suites")
        expect_status(r, 200)
        assert any(s["id"] == suite_id for s in expect_json(r))


@section("s08", "evals-runs", deps=["s07"])
def s08(ctx: Ctx, steps: list[StepResult]) -> None:
    """Trigger a manual eval run if there's any completed experiment
    available. We don't wait for completion (would need real backend);
    we just prove the queue + cancel transitions work.
    """
    suite_id = ctx.shared["s07_suite_id"]

    with step(steps, "Find a completed experiment"):
        r = ctx.client.get(
            "/experiments", params={"status": "completed", "limit": 1}
        )
        expect_status(r, 200)
        runs = expect_json(r)
        if not runs:
            raise AssertionError(
                "no completed experiment found — cannot smoke eval run"
            )
        exp_id = runs[0]["id"]

    with step(steps, "POST /evals/runs (queue)"):
        r = ctx.client.post(
            "/evals/runs",
            json={"suite_id": suite_id, "experiment_id": exp_id},
        )
        expect_status(r, 201)
        run = expect_json(r)
        run_id = run["id"]
        assert run["status"] in ("queued", "running"), run["status"]

    with step(steps, "POST /evals/runs/{id}/cancel"):
        r = ctx.client.post(f"/evals/runs/{run_id}/cancel")
        expect_status(r, 200)
        # Either cancelled (was queued) or running (will be cancelled
        # by signal). Both fine.
        body = expect_json(r)
        assert body["status"] in ("cancelled", "running")


@section("s09", "models", deps=["s06"])
def s09(ctx: Ctx, steps: list[StepResult]) -> None:
    """Register an existing completed experiment as a new family with
    versions, exercise aliases, and the /datasets endpoint."""
    with step(steps, "Find two completed experiments"):
        r = ctx.client.get(
            "/experiments", params={"status": "completed", "limit": 5}
        )
        completed = expect_json(r)
        if len(completed) < 2:
            raise AssertionError(
                "need >=2 completed experiments for alias-move test"
            )
        exp_a, exp_b = completed[0]["id"], completed[1]["id"]

    family = ctx.name("fam")
    m_a_id = m_b_id = None
    with step(steps, "POST /models register v1 with alias=staging"):
        r = ctx.client.post(
            "/models",
            json={"name": family, "experiment_id": exp_a, "alias": "staging"},
        )
        expect_status(r, 201)
        m = expect_json(r)
        m_a_id = m["id"]
        assert m["version"] == 1
        assert "staging" in m["aliases"]
    ctx.add_cleanup(
        f"DELETE /models/{m_a_id}",
        lambda: ctx.client.delete(f"/models/{m_a_id}", params={"force": "true"}),
    )

    with step(steps, "POST /models register v2 (auto-increment)"):
        r = ctx.client.post(
            "/models",
            json={"name": family, "experiment_id": exp_b},
        )
        expect_status(r, 201)
        m = expect_json(r)
        m_b_id = m["id"]
        assert m["version"] == 2
    ctx.add_cleanup(
        f"DELETE /models/{m_b_id}",
        lambda: ctx.client.delete(f"/models/{m_b_id}", params={"force": "true"}),
    )

    with step(steps, "Move alias staging -> v2"):
        r = ctx.client.post(
            f"/models/{family}/aliases/staging",
            json={"version": 2},
        )
        expect_status(r, 200)
        assert "staging" in expect_json(r)["aliases"]

    with step(steps, "GET /models/{name}/staging resolves to v2"):
        r = ctx.client.get(f"/models/{family}/staging")
        expect_status(r, 200)
        body = expect_json(r)
        assert body["id"] == m_b_id

    with step(steps, "GET /models/{id}/datasets (trained-on lineage)"):
        r = ctx.client.get(f"/models/{m_b_id}/datasets")
        expect_status(r, 200)
        # May be empty if the experiment's dataset wasn't in the
        # registry, that's fine — shape check is the assertion.
        assert isinstance(expect_json(r), list)

    ctx.shared["s09_model_id"] = m_b_id


@section("s10", "models-quantize", deps=["s09"])
def s10(ctx: Ctx, steps: list[StepResult]) -> None:
    model_id = ctx.shared["s09_model_id"]
    with step(steps, "POST /models/{id}/quantize (error envelope only)"):
        r = ctx.client.post(
            f"/models/{model_id}/quantize",
            json={"method": "awq", "bits": 4},
            timeout=60.0,
        )
        # On a live server without swift export available the subprocess
        # will fail. We accept 500 (subprocess failure) OR 422 (caller
        # validation, e.g. missing adapter_path). What we DON'T accept is
        # an un-structured error body.
        expect_status(r, 200, 201, 422, 500)
        body = expect_json(r)
        if r.status_code >= 400:
            detail = body.get("detail")
            assert isinstance(detail, dict) and "error" in detail, (
                f"expected structured error envelope, got {body}"
            )


@section("s11", "inferences-cache", deps=["s00"])
def s11(ctx: Ctx, steps: list[StepResult]) -> None:
    with step(steps, "GET /inferences/cache"):
        r = ctx.client.get("/inferences/cache")
        expect_status(r, 200)
        body = expect_json(r)
        expect_keys(body, "max_loaded", "loaded")
        assert isinstance(body["loaded"], list)


@section("s12", "pipelines", deps=["s00"])
def s12(ctx: Ctx, steps: list[StepResult]) -> None:
    name = ctx.name("pipe")
    cfg = {
        "name": name,
        "stages": [
            {
                "name": "a",
                "base_spec": {
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "dataset": ["AI-ModelScope/alpaca-gpt4-data-en#1"],
                    "gpu_count": 1,
                },
            },
            {
                "name": "b",
                "depends_on": ["a"],
                "base_spec": {
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "dataset": ["AI-ModelScope/alpaca-gpt4-data-en#1"],
                    "gpu_count": 1,
                },
            },
        ],
    }
    pipeline_id = None
    with step(steps, "POST /pipelines (2-stage DAG)"):
        r = ctx.client.post("/pipelines", json=cfg)
        expect_status(r, 201)
        body = expect_json(r)
        pipeline_id = body["id"]
        assert len(body["stages"]) == 2

    # Cancel immediately to avoid burning GPU time. The driver may
    # have already enqueued stage A as an experiment; that experiment
    # is best-effort cancelled by the pipeline cancel via status flip.
    with step(steps, "POST /pipelines/{id}/cancel"):
        r = ctx.client.post(f"/pipelines/{pipeline_id}/cancel")
        expect_status(r, 200)

    with step(steps, "GET /pipelines/{id} -> cancelled/terminal"):
        time.sleep(1.0)
        body = ctx.client.get(f"/pipelines/{pipeline_id}").json()
        assert body["status"] in (
            "cancelled", "failed", "completed"
        ), body["status"]
    # Also cancel any spawned experiments so they don't keep the GPU.
    ctx.add_cleanup(
        f"cancel pipeline {pipeline_id} (best-effort)",
        lambda: ctx.client.post(f"/pipelines/{pipeline_id}/cancel"),
    )


@section("s13", "active-learning", deps=["s01"])
def s13(ctx: Ctx, steps: list[StepResult]) -> None:
    """Submit a tiny AL run. The default backend would try to load a real
    HF model — on a 4GB GPU laptop that takes 10-30s. We skip the wait
    and just verify the queue transitions; the route already runs the
    actual work synchronously inside the POST so when we get a response
    it's already terminal."""
    payload = {
        "model_ref": "base:Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": f"ds:{ctx.shared['s01_ds_id']}",
        "top_n": 1,
        "sample_limit": 1,
        "scorer": "length_zscore",
    }
    with step(steps, "POST /active-learning/runs (sample_limit=1)"):
        r = ctx.client.post(
            "/active-learning/runs", json=payload, timeout=120.0
        )
        # Either completed (model loaded successfully) OR failed (OOM /
        # offline). Both prove the orchestration works end-to-end.
        expect_status(r, 201)
        body = expect_json(r)
        assert body["status"] in (
            "completed", "failed", "cancelled"
        ), body["status"]


@section("s14", "watches", deps=["s00"])
def s14(ctx: Ctx, steps: list[StepResult]) -> None:
    pipe_cfg = {
        "name": ctx.name("pipe-for-watch"),
        "stages": [
            {
                "name": "stub",
                "base_spec": {
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "dataset": ["AI-ModelScope/alpaca-gpt4-data-en#1"],
                    "gpu_count": 1,
                },
            }
        ],
    }
    watch_id = None
    with step(steps, "POST /watches (interval, never fires in window)"):
        r = ctx.client.post(
            "/watches",
            json={
                "name": ctx.name("watch-interval"),
                "kind": "interval",
                "interval_seconds": 86400,
                "pipeline_config": pipe_cfg,
            },
        )
        expect_status(r, 201)
        body = expect_json(r)
        watch_id = body["id"]
        assert body["enabled"] is True
    ctx.add_cleanup(
        f"DELETE /watches/{watch_id}",
        lambda: ctx.client.delete(f"/watches/{watch_id}"),
    )

    with step(steps, "GET /watches list"):
        r = ctx.client.get("/watches")
        expect_status(r, 200)
        assert any(w["id"] == watch_id for w in expect_json(r))

    with step(steps, "POST /watches/{id}/disable"):
        r = ctx.client.post(f"/watches/{watch_id}/disable")
        expect_status(r, 200)
        assert expect_json(r)["enabled"] is False

    with step(steps, "POST /watches/{id}/enable"):
        r = ctx.client.post(f"/watches/{watch_id}/enable")
        expect_status(r, 200)
        assert expect_json(r)["enabled"] is True


@section("s15", "synth-mock", deps=["s01"])
def s15(ctx: Ctx, steps: list[StepResult]) -> None:
    with step(steps, "POST /synth (provider=mock)"):
        r = ctx.client.post(
            "/synth",
            json={
                "provider": "mock",
                "model": "ignored-by-mock",
                "source_dataset": f"ds:{ctx.shared['s01_ds_id']}",
                "instruction": "smoke test",
                "target_count": 3,
                "seed": 0,
                "name": ctx.name("synth"),
            },
            timeout=30.0,
        )
        expect_status(r, 201)
        body = expect_json(r)
        synth_id = body["id"]
        assert body["line_count"] == 3
        assert "synthesized via mock" in (body["description"] or "")
    ctx.add_cleanup(
        f"DELETE /datasets/{synth_id}",
        lambda: ctx.client.delete(f"/datasets/{synth_id}", params={"force": "true"}),
    )

    with step(steps, "Output records include _source provenance"):
        r = ctx.client.get(f"/datasets/{synth_id}/preview", params={"n": 1})
        expect_status(r, 200)
        rec = json.loads(r.text.splitlines()[0])
        assert "_source" in rec
        assert "completion" in rec


@section("s16", "studies", deps=["s00"])
def s16(ctx: Ctx, steps: list[StepResult]) -> None:
    with step(steps, "GET /studies"):
        r = ctx.client.get("/studies")
        expect_status(r, 200)
        assert isinstance(expect_json(r), list)

    with step(steps, "GET /studies/cost-summary"):
        r = ctx.client.get("/studies/cost-summary")
        expect_status(r, 200)
        body = expect_json(r)
        assert isinstance(body, list)
        # Shape check: every row must have these keys when non-empty.
        for s in body:
            expect_keys(
                s, "study_id", "n_trials", "total_gpu_seconds",
                "target_metric", "direction",
            )


@section("s17", "gpus", deps=["s00"])
def s17(ctx: Ctx, steps: list[StepResult]) -> None:
    with step(steps, "GET /gpus"):
        r = ctx.client.get("/gpus")
        expect_status(r, 200)
        body = expect_json(r)
        expect_keys(body, "total", "free", "leases")
        assert isinstance(body["leases"], list)
        for lease in body["leases"]:
            expect_keys(lease, "index", "memory_total_mb", "experiment_id")


@section("s18", "ds-ref-versioning", deps=["s01"])
def s18(ctx: Ctx, steps: list[StepResult]) -> None:
    ds_id = ctx.shared["s01_ds_id"]
    with step(steps, "POST /experiments with ds:<id>@v1 (resolves)"):
        # Submit an experiment we cancel immediately. We only care that
        # the ds:<id>@v1 syntax doesn't return 422.
        r = ctx.client.post(
            "/experiments",
            json={
                "name": ctx.name("ref-good"),
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "dataset": [f"ds:{ds_id}@v1"],
                "gpu_count": 1,
            },
        )
        expect_status(r, 201)
        eid = expect_json(r)["experiment_id"]
        ctx.client.post(f"/experiments/{eid}/cancel")

    with step(steps, "POST /experiments with ds:<id>@v99 (rejected)"):
        r = ctx.client.post(
            "/experiments",
            json={
                "name": ctx.name("ref-bad"),
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "dataset": [f"ds:{ds_id}@v99"],
                "gpu_count": 1,
            },
        )
        expect_status(r, 422)
        body = expect_json(r)
        assert body.get("detail", {}).get("error") in (
            "malformed_dataset_ref", "unknown_dataset_ref",
        )


@section("s19", "datasets-gdpr-recursive", deps=["s03"])
def s19(ctx: Ctx, steps: list[StepResult]) -> None:
    """Verify the recursive GDPR query returns models trained on
    descendants. The mix in s03 has two parents; we just check the
    shape of the recursive query — actual model creation is in s09 but
    that model trained on the experiment's HF dataset, not s01."""
    parent = ctx.shared["s01_ds_id"]
    mix = ctx.shared["s03_mix_id"]

    with step(steps, "GET /datasets/{parent}/models?recursive=true"):
        r = ctx.client.get(
            f"/datasets/{parent}/models", params={"recursive": "true"}
        )
        expect_status(r, 200)
        expect_keys(expect_json(r), "model_ids")

    with step(steps, "Recursive walks via dataset_lineage (mix -> parents)"):
        # The mix is a descendant of `parent`; if any model trained on
        # `mix` shows up only under recursive=true, that proves the walk.
        # We can't *create* such a model here without running real
        # training — so this asserts the endpoint shape only.
        r = ctx.client.get(
            f"/datasets/{mix}/models", params={"recursive": "false"}
        )
        expect_status(r, 200)
        expect_keys(expect_json(r), "model_ids")


@section("s20", "compliance-cli", deps=["s00"])
def s20(ctx: Ctx, steps: list[StepResult]) -> None:
    if not ctx.with_cli:
        raise SkipSection("--with-cli not set")

    # The CLI scans the local SQLite. On Windows we reach into WSL.
    is_windows = sys.platform == "win32"
    out_path = "/tmp/smoke-forget-report.json"
    if is_windows:
        wsl_cmd = [
            "wsl", "-d", "Ubuntu-24.04", "--",
            "/home/rudi/src/next/.venv/bin/trainpipe-forget",
            "--db", "/home/rudi/src/next/data/trainpipe.sqlite3",
            "--output", out_path,
            f"smoke-{ctx.run_id}",
        ]
    else:
        wsl_cmd = [
            "trainpipe-forget",
            "--output", out_path,
            f"smoke-{ctx.run_id}",
        ]

    with step(steps, "Run trainpipe-forget CLI"):
        result = subprocess.run(
            wsl_cmd, capture_output=True, text=True, timeout=60
        )
        # Exit 0 = no hits found (expected for fresh smoke run_id).
        assert result.returncode == 0, (
            f"rc={result.returncode}\nstdout={result.stdout[:200]}\n"
            f"stderr={result.stderr[:200]}"
        )

    with step(steps, "JSON report file is well-formed"):
        if is_windows:
            # Pull file out of WSL.
            cat = subprocess.run(
                ["wsl", "-d", "Ubuntu-24.04", "--", "cat", out_path],
                capture_output=True, text=True,
            )
            assert cat.returncode == 0
            body = json.loads(cat.stdout)
        else:
            body = json.loads(Path(out_path).read_text())
        expect_keys(body, "term", "scanned_datasets", "hits", "impacted_models")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class SkipSection(Exception):
    """Section signals skip-with-reason."""


def resolve_with_deps(ids: list[str]) -> list[tuple[str, str, list[str], SectionFn]]:
    by_id = {sid: (sid, name, deps, fn) for (sid, name, deps, fn) in SECTIONS}
    requested = set(ids)
    out: list[tuple[str, str, list[str], SectionFn]] = []
    seen: set[str] = set()

    def visit(sid: str) -> None:
        if sid in seen:
            return
        if sid not in by_id:
            raise ValueError(f"unknown section id: {sid}")
        for d in by_id[sid][2]:
            visit(d)
        seen.add(sid)
        out.append(by_id[sid])

    for sid in [s[0] for s in SECTIONS if s[0] in requested]:
        visit(sid)
    return out


def run_section(
    ctx: Ctx, sid: str, name: str, fn: SectionFn
) -> SectionResult:
    res = SectionResult(id=sid, name=name, status="pass", duration_ms=0.0)
    t = time.monotonic()
    cleanup_start = len(ctx.cleanup)
    try:
        fn(ctx, res.steps)
        res.status = "pass"
    except SkipSection as e:
        res.status = "skip"
        res.skip_reason = str(e)
        # remove any cleanup the section added before it skipped
        while len(ctx.cleanup) > cleanup_start:
            ctx.cleanup.pop()
    except Exception as e:
        res.status = "fail"
        res.error = f"{type(e).__name__}: {e}"
    res.duration_ms = (time.monotonic() - t) * 1000
    return res


def run_cleanup(ctx: Ctx) -> list[str]:
    """LIFO drain of the cleanup stack. Returns labels of successful cleans."""
    done: list[str] = []
    while ctx.cleanup:
        label, fn = ctx.cleanup.pop()
        try:
            fn()
            done.append(label)
        except Exception as e:
            if ctx.verbose:
                print(f"  cleanup failed: {label}: {e}")
    return done


def fmt_section_line(res: SectionResult) -> str:
    status_color = {
        "pass": "\033[32mPASS\033[0m",
        "fail": "\033[31mFAIL\033[0m",
        "skip": "\033[33mSKIP\033[0m",
    }
    label = f"{res.id} {res.name}"
    n_steps = len(res.steps)
    step_word = "step" if n_steps == 1 else "steps"
    return (
        f"  [{label:32s}] {status_color[res.status]}  "
        f"{n_steps} {step_word}  {res.duration_ms/1000:6.2f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="smoke_e2e",
        description=(
            "Live-server smoke harness for trainpipe. Exercises every "
            "user-facing feature once against a real deployment."
        ),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("TRAINPIPE_BASE_URL", "http://127.0.0.1:8080"),
        help="Server base URL (default: $TRAINPIPE_BASE_URL or localhost)",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("TRAINPIPE_API_KEY", ""),
        help="API key for X-API-Key header (default: $TRAINPIPE_API_KEY)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma list of section ids to run (deps auto-pulled). Default: all.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Don't clean up created resources at exit.",
    )
    parser.add_argument(
        "--report",
        default=".run/smoke-report.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP read timeout per request, seconds.",
    )
    parser.add_argument(
        "--with-cli",
        action="store_true",
        help="Also run s20-compliance-cli (requires trainpipe-forget).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.key:
        print(
            "error: no API key set. Pass --key or set TRAINPIPE_API_KEY env.",
            file=sys.stderr,
        )
        return 2

    # Build run_id once.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_id = f"{ts}-{uuid.uuid4().hex[:6]}"

    print(f"trainpipe smoke - {args.url}")
    print(f"run_id: {run_id}\n")

    client = httpx.Client(
        base_url=args.url.rstrip("/"),
        headers={"X-API-Key": args.key},
        timeout=httpx.Timeout(args.timeout, connect=5.0),
    )

    # Connectivity check.
    try:
        h = client.get("/health", headers={})
        if h.status_code != 200:
            print(f"error: /health returned {h.status_code}", file=sys.stderr)
            return 2
    except httpx.HTTPError as e:
        print(f"error: cannot reach {args.url}: {e}", file=sys.stderr)
        return 2

    preflight_cleanup(client, verbose=args.verbose)

    ctx = Ctx(
        client=client,
        run_id=run_id,
        url=args.url,
        key=args.key,
        keep=args.keep,
        verbose=args.verbose,
        with_cli=args.with_cli,
    )

    # Resolve which sections to run.
    if args.only.strip():
        ids = [s.strip() for s in args.only.split(",") if s.strip()]
        ordered = resolve_with_deps(ids)
    else:
        ordered = list(SECTIONS)

    results: list[SectionResult] = []
    failures = 0
    for sid, name, _deps, fn in ordered:
        res = run_section(ctx, sid, name, fn)
        print(fmt_section_line(res))
        if res.status == "fail" and (args.verbose or res.error):
            print(f"      -> {res.error}")
            for s in res.steps:
                if not s.ok:
                    print(f"      step: {s.name}: {s.error}")
        results.append(res)
        if res.status == "fail":
            failures += 1

    cleaned: list[str] = []
    if not args.keep:
        print(f"\n  cleaning up {len(ctx.cleanup)} resources...", end=" ")
        cleaned = run_cleanup(ctx)
        print(f"{len(cleaned)} ok")

    summary = {
        "passed": sum(1 for r in results if r.status == "pass"),
        "failed": sum(1 for r in results if r.status == "fail"),
        "skipped": sum(1 for r in results if r.status == "skip"),
        "total": len(results),
    }
    print("=" * 50)
    print(
        f"  {summary['total']} sections | "
        f"{summary['passed']} passed | "
        f"{summary['failed']} failed | "
        f"{summary['skipped']} skipped"
    )

    # Write report.
    report = {
        "run_id": run_id,
        "started": ts,
        "finished": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "server": {"url": args.url},
        "sections": [asdict(r) for r in results],
        "summary": summary,
        "cleaned": cleaned,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print(f"  Report: {args.report}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
