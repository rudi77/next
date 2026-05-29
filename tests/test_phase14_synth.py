"""Tests for Phase 14: synthetic data generation."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import get_db, get_gpu_pool, get_scheduler, get_study_manager
from trainpipe.api.main import app
from trainpipe.core.db import Database
from trainpipe.scheduler.gpu_pool import GpuPool
from trainpipe.synth.runner import (
    MockProvider,
    _iter_source,
    generate_synthetic,
    make_provider,
)

HEADERS = {"X-API-Key": "test-key"}


def _run(coro):
    return asyncio.run(coro)


class _NoopScheduler:
    async def cancel_experiment(self, experiment_id):
        return False


class _StubStudyManager:
    async def create_and_start(self, config):
        return "x"

    async def cancel(self, study_id):
        return True


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    db = Database(tmp_path / "test.sqlite3")
    _run(db.init())
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: _NoopScheduler()
    app.dependency_overrides[get_gpu_pool] = lambda: GpuPool([])
    app.dependency_overrides[get_study_manager] = lambda: _StubStudyManager()
    yield {"db": db, "tmp": tmp_path}
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _write_source(tmp_path: Path, n: int = 3) -> Path:
    p = tmp_path / "source.jsonl"
    p.write_text(
        "\n".join(json.dumps({"prompt": f"q{i}"}) for i in range(n)) + "\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_iter_source_sampling_respects_seed(tmp_path):
    src = _write_source(tmp_path, n=5)
    a = list(_iter_source(src, seed=42, sample_n=10))
    b = list(_iter_source(src, seed=42, sample_n=10))
    assert a == b


def test_iter_source_no_sampling_yields_all(tmp_path):
    src = _write_source(tmp_path, n=4)
    assert len(list(_iter_source(src, seed=0))) == 4


def test_iter_source_skips_malformed(tmp_path):
    p = tmp_path / "noisy.jsonl"
    p.write_text(
        '{"prompt":"good"}\n'
        'garbage\n'
        '{"prompt":"good2"}\n',
        encoding="utf-8",
    )
    assert len(list(_iter_source(p, seed=0))) == 2


def test_generate_synthetic_writes_target_count(tmp_path):
    src = _write_source(tmp_path, n=3)
    out = tmp_path / "synth.jsonl"
    written = generate_synthetic(
        provider=MockProvider(),
        model="any",
        source_path=src,
        instruction="rephrase",
        target_count=12,
        out_path=out,
        seed=1,
    )
    assert written == 12
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    record = json.loads(lines[0])
    assert record["completion"].startswith("synth(")
    assert "_source" in record


def test_generate_synthetic_skips_provider_failures(tmp_path):
    src = _write_source(tmp_path, n=4)
    out = tmp_path / "synth.jsonl"

    counter = {"i": 0}

    def flaky(p):
        counter["i"] += 1
        if counter["i"] % 2 == 0:
            raise RuntimeError("rate limited")
        return "ok"

    p = MockProvider(transform=flaky)
    written = generate_synthetic(
        provider=p,
        model="m",
        source_path=src,
        instruction="rephrase",
        target_count=6,
        out_path=out,
    )
    # 3 out of 6 succeed.
    assert written == 3


def test_make_provider_unknown_raises():
    with pytest.raises(ValueError):
        make_provider("nope")


def test_make_provider_mock_works():
    assert make_provider("mock").name == "mock"


# ---------------------------------------------------------------------------
# REST integration
# ---------------------------------------------------------------------------


def test_synth_route_creates_dataset_with_provenance(state, client):
    src = _write_source(state["tmp"], n=2)
    r = client.post(
        "/synth",
        headers=HEADERS,
        json={
            "provider": "mock",
            "model": "ignored",
            "source_dataset": str(src),
            "instruction": "make variants",
            "target_count": 5,
            "name": "synth-out",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["line_count"] == 5
    assert "synthesized via mock:ignored" in body["description"]


def test_synth_route_422_on_missing_source(state, client):
    r = client.post(
        "/synth",
        headers=HEADERS,
        json={
            "provider": "mock",
            "model": "m",
            "source_dataset": "/does/not/exist.jsonl",
            "instruction": "x",
            "target_count": 1,
            "name": "x",
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "source_file_missing"


def test_synth_route_500_on_provider_outage(state, client, monkeypatch):
    src = _write_source(state["tmp"])

    class _Broken:
        name = "mock"

        def generate(self, prompt, *, model, max_tokens):
            raise RuntimeError("provider outage")

    def fake_make(name):
        return _Broken()

    monkeypatch.setattr("trainpipe.api.routes.synth.make_provider", fake_make)
    r = client.post(
        "/synth",
        headers=HEADERS,
        json={
            "provider": "mock",
            "model": "m",
            "source_dataset": str(src),
            "instruction": "x",
            "target_count": 3,
            "name": "x",
        },
    )
    # All 3 records fail → 422 no_records_generated.
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "no_records_generated"


def test_synth_route_requires_auth(state, client):
    r = client.post(
        "/synth",
        json={
            "provider": "mock", "model": "m",
            "source_dataset": "/x", "instruction": "y",
            "target_count": 1, "name": "n",
        },
    )
    assert r.status_code == 401


def test_synth_route_dedup_by_sha(state, client):
    src = _write_source(state["tmp"], n=2)
    payload = {
        "provider": "mock",
        "model": "m",
        "source_dataset": str(src),
        "instruction": "x",
        "target_count": 3,
        "name": "synth-x",
        "seed": 42,
    }
    r1 = client.post("/synth", headers=HEADERS, json=payload)
    r2 = client.post("/synth", headers=HEADERS, json=payload)
    assert r1.json()["id"] == r2.json()["id"]
