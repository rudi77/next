"""Tests for Phase 19: model quantization."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import get_db, get_gpu_pool, get_scheduler, get_study_manager
from trainpipe.api.main import app
from trainpipe.api.routes.models import _set_quantize_backend
from trainpipe.api.schemas import ExperimentSpec
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.quantization.runner import MockQuantizeBackend
from trainpipe.scheduler.gpu_pool import GpuPool

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
    backend = MockQuantizeBackend()
    _set_quantize_backend(backend)
    yield {"db": db, "backend": backend, "tmp": tmp_path}
    _set_quantize_backend(None)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


async def _make_completed_with_adapter(db, tmp_path) -> tuple[str, str]:
    """Create a completed experiment + register as v1 model with an
    adapter_path that exists on disk."""
    adapter = tmp_path / "adapter-v1"
    adapter.mkdir()
    (adapter / "weights").write_text("fake", encoding="utf-8")
    async with db.connect() as conn:
        spec = ExperimentSpec(
            model="qwen/Qwen2-0.5B",
            dataset=["/tmp/ds.jsonl"],
            output_dir=str(adapter),
        )
        exp_id = await repository.create_experiment(conn, spec)
        await conn.execute(
            "UPDATE experiments SET status='completed' WHERE id=?", (exp_id,)
        )
        await conn.commit()
        model_id, _ = await repository.register_model_atomic(
            conn,
            name="fam",
            explicit_version=None,
            base_model="qwen/Qwen2-0.5B",
            adapter_path=str(adapter),
            experiment_id=exp_id,
            run_id=None,
            eval_summary=None,
            description=None,
            alias=None,
        )
    return exp_id, model_id


def test_quantize_creates_new_version(state, client, tmp_path):
    _exp, parent_id = _run(_make_completed_with_adapter(state["db"], tmp_path))
    r = client.post(
        f"/models/{parent_id}/quantize",
        headers=HEADERS,
        json={"method": "awq", "bits": 4},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["version"] == 2
    assert body["name"] == "fam"
    assert "quantized awq:4bit" in body["description"]
    # The backend was actually invoked.
    assert state["backend"].calls
    assert state["backend"].calls[0]["method"] == "awq"
    assert state["backend"].calls[0]["bits"] == 4


def test_quantize_rejects_unsupported_method(state, client, tmp_path):
    _exp, parent_id = _run(_make_completed_with_adapter(state["db"], tmp_path))
    # bits=4 keeps Pydantic happy; method is unknown so the route's
    # explicit check is the one that fires (returns the {error: ...}
    # body — not Pydantic's list-of-validation-errors shape).
    r = client.post(
        f"/models/{parent_id}/quantize",
        headers=HEADERS,
        json={"method": "bitnet", "bits": 4},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unsupported_method"


def test_quantize_404_on_missing_model(state, client):
    r = client.post(
        "/models/no-such-id/quantize",
        headers=HEADERS,
        json={"method": "awq", "bits": 4},
    )
    assert r.status_code == 404


def test_quantize_422_when_parent_has_no_adapter(state, client, tmp_path):
    async def _make():
        async with state["db"].connect() as conn:
            spec = ExperimentSpec(model="m", dataset=["/x"])
            exp_id = await repository.create_experiment(conn, spec)
            await conn.execute(
                "UPDATE experiments SET status='completed' WHERE id=?", (exp_id,)
            )
            await conn.commit()
            # Register without an adapter path.
            mid, _ = await repository.register_model_atomic(
                conn,
                name="hollow",
                explicit_version=None,
                base_model="m",
                adapter_path=None,
                experiment_id=exp_id,
                run_id=None,
                eval_summary=None,
                description=None,
                alias=None,
            )
            return mid

    parent_id = _run(_make())
    r = client.post(
        f"/models/{parent_id}/quantize",
        headers=HEADERS,
        json={"method": "awq", "bits": 4},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "no_adapter_path"


def test_quantize_500_on_backend_failure(state, client, tmp_path):
    _exp, parent_id = _run(_make_completed_with_adapter(state["db"], tmp_path))

    def boom(p):
        raise RuntimeError("OOM during awq calibration")

    state["backend"].on_call = boom
    r = client.post(
        f"/models/{parent_id}/quantize",
        headers=HEADERS,
        json={"method": "awq", "bits": 4},
    )
    assert r.status_code == 500
    assert "OOM" in r.json()["detail"]["detail"]


def test_quantize_inherits_parent_eval_summary(state, client, tmp_path):
    """The quantized version starts with the parent's eval summary as its
    baseline so the UI can show 'before quantize: …'."""
    _exp, parent_id = _run(_make_completed_with_adapter(state["db"], tmp_path))

    async def _add_summary():
        async with state["db"].connect() as conn:
            await conn.execute(
                "UPDATE models SET eval_summary_json = ? WHERE id = ?",
                ('{"suite1":{"exact_match":0.9}}', parent_id),
            )
            await conn.commit()

    _run(_add_summary())
    r = client.post(
        f"/models/{parent_id}/quantize",
        headers=HEADERS,
        json={"method": "gptq", "bits": 4},
    )
    assert r.status_code == 201
    assert r.json()["eval_summary"] == {"suite1": {"exact_match": 0.9}}
