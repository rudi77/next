"""Tests for the Phase 7 model registry REST API."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import get_db, get_gpu_pool, get_scheduler, get_study_manager
from trainpipe.api.main import app
from trainpipe.api.schemas import (
    EvalRunStatus,
    ExperimentSpec,
    InferenceParams,
    MetricAggregate,
    MetricConfig,
)
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.scheduler.gpu_pool import GpuPool

HEADERS = {"X-API-Key": "test-key"}


class _NoopScheduler:
    async def cancel_experiment(self, experiment_id: str) -> bool:
        return False


class _StubStudyManager:
    async def create_and_start(self, config):
        return "stub"

    async def cancel(self, study_id):
        return True


def _run(coro):
    return asyncio.run(coro)


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
    yield {"db": db}
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _make_completed_experiment(db, name: str = "exp1") -> str:
    async def _do():
        async with db.connect() as conn:
            spec = ExperimentSpec(
                name=name, model="qwen/Qwen2-0.5B", dataset=["/tmp/ds.jsonl"]
            )
            exp_id = await repository.create_experiment(conn, spec)
            await conn.execute(
                "UPDATE experiments SET status = 'completed', "
                "finished_at = ?, mlflow_run_id = ? WHERE id = ?",
                ("2026-05-29T00:00:00+00:00", f"mlflow-{exp_id[:8]}", exp_id),
            )
            await conn.commit()
            return exp_id

    return _run(_do())


def _attach_eval_summary(db, experiment_id: str) -> None:
    async def _do():
        async with db.connect() as conn:
            suite_id = await repository.create_eval_suite(
                conn,
                name=f"suite-{experiment_id[:6]}",
                description=None,
                dataset_path="/tmp/eval.jsonl",
                metrics=[MetricConfig(kind="exact_match")],
                inference_params=InferenceParams(),
            )
            run_id = await repository.create_eval_run(
                conn,
                suite_id=suite_id,
                experiment_id=experiment_id,
                model_ref=experiment_id,
                triggered_by="auto",
            )
            await repository.finalize_eval_run(
                conn,
                run_id,
                status=EvalRunStatus.COMPLETED,
                aggregate={
                    "exact_match": MetricAggregate(mean=0.82, std=0.1, count=100)
                },
                sample_count=100,
            )

    _run(_do())


def test_register_model_auto_increments_version(state, client):
    exp_id = _make_completed_experiment(state["db"])
    r1 = client.post(
        "/models",
        json={"name": "invoice-extractor", "experiment_id": exp_id},
        headers=HEADERS,
    )
    assert r1.status_code == 201, r1.text
    assert r1.json()["version"] == 1

    exp2 = _make_completed_experiment(state["db"], name="exp2")
    r2 = client.post(
        "/models",
        json={"name": "invoice-extractor", "experiment_id": exp2},
        headers=HEADERS,
    )
    assert r2.status_code == 201
    assert r2.json()["version"] == 2


def test_register_rejects_non_completed(state, client):
    async def _make_running():
        async with state["db"].connect() as conn:
            spec = ExperimentSpec(model="m", dataset=["/d"])
            return await repository.create_experiment(conn, spec)

    exp_id = _run(_make_running())
    r = client.post(
        "/models",
        json={"name": "x", "experiment_id": exp_id},
        headers=HEADERS,
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "experiment_not_completed"


def test_register_with_explicit_version_conflict(state, client):
    exp_id = _make_completed_experiment(state["db"])
    r1 = client.post(
        "/models",
        json={"name": "fam", "experiment_id": exp_id, "version": 5},
        headers=HEADERS,
    )
    assert r1.status_code == 201
    r2 = client.post(
        "/models",
        json={"name": "fam", "experiment_id": exp_id, "version": 5},
        headers=HEADERS,
    )
    assert r2.status_code == 409


def test_register_eval_summary_picked_up(state, client):
    exp_id = _make_completed_experiment(state["db"])
    _attach_eval_summary(state["db"], exp_id)
    r = client.post(
        "/models",
        json={"name": "with-evals", "experiment_id": exp_id},
        headers=HEADERS,
    )
    assert r.status_code == 201
    body = r.json()
    assert body["eval_summary"] is not None
    summary = body["eval_summary"]
    # one suite, one metric
    suite_summary = next(iter(summary.values()))
    assert pytest_approx(suite_summary["exact_match"], 0.82)


def test_alias_assign_move_and_filter(state, client):
    exp1 = _make_completed_experiment(state["db"], name="e1")
    r1 = client.post(
        "/models",
        json={"name": "fam", "experiment_id": exp1, "alias": "production"},
        headers=HEADERS,
    )
    assert r1.status_code == 201
    m1 = r1.json()
    assert "production" in m1["aliases"]

    # Filter by alias returns just the production model.
    r_prod = client.get("/models", params={"alias": "production"}, headers=HEADERS)
    assert len(r_prod.json()) == 1
    assert r_prod.json()[0]["id"] == m1["id"]

    # Add v2 and move production to it via the explicit alias endpoint.
    exp2 = _make_completed_experiment(state["db"], name="e2")
    r2 = client.post(
        "/models",
        json={"name": "fam", "experiment_id": exp2},
        headers=HEADERS,
    )
    assert r2.json()["version"] == 2
    move = client.post(
        "/models/fam/aliases/production",
        json={"version": 2},
        headers=HEADERS,
    )
    assert move.status_code == 200
    assert "production" in move.json()["aliases"]

    # v1 should no longer hold production.
    v1 = client.get("/models/fam/1", headers=HEADERS).json()
    assert "production" not in v1["aliases"]


def test_resolve_by_alias_and_version(state, client):
    exp_id = _make_completed_experiment(state["db"])
    client.post(
        "/models",
        json={"name": "fam", "experiment_id": exp_id, "alias": "staging"},
        headers=HEADERS,
    )
    by_alias = client.get("/models/fam/staging", headers=HEADERS)
    by_ver = client.get("/models/fam/1", headers=HEADERS)
    assert by_alias.status_code == 200
    assert by_ver.status_code == 200
    assert by_alias.json()["id"] == by_ver.json()["id"]


def test_resolve_missing_404(state, client):
    r = client.get("/models/nope/production", headers=HEADERS)
    assert r.status_code == 404


def test_cross_family_alias_rejected(state, client):
    exp_id = _make_completed_experiment(state["db"])
    r = client.post(
        "/models",
        json={"name": "famA", "experiment_id": exp_id},
        headers=HEADERS,
    )
    model_a = r.json()
    # Try to assign an alias under famB to a model that lives in famA.
    move = client.post(
        "/models/famB/aliases/production",
        json={"model_id": model_a["id"]},
        headers=HEADERS,
    )
    assert move.status_code == 422
    assert move.json()["detail"]["error"] == "cross_family_alias"


def test_delete_blocked_by_alias_unless_forced(state, client):
    exp_id = _make_completed_experiment(state["db"])
    r = client.post(
        "/models",
        json={"name": "fam", "experiment_id": exp_id, "alias": "production"},
        headers=HEADERS,
    )
    mid = r.json()["id"]
    blocked = client.delete(f"/models/{mid}", headers=HEADERS)
    assert blocked.status_code == 409

    forced = client.delete(f"/models/{mid}?force=true", headers=HEADERS)
    assert forced.status_code == 200
    assert forced.json()["deleted"]


def test_remove_alias_endpoint(state, client):
    exp_id = _make_completed_experiment(state["db"])
    client.post(
        "/models",
        json={"name": "fam", "experiment_id": exp_id, "alias": "production"},
        headers=HEADERS,
    )
    rm = client.delete("/models/fam/aliases/production", headers=HEADERS)
    assert rm.status_code == 200
    assert rm.json()["status"] == "deleted"
    # Idempotent: a second call returns not_found.
    rm2 = client.delete("/models/fam/aliases/production", headers=HEADERS)
    assert rm2.json()["status"] == "not_found"


def test_models_require_auth(state, client):
    r = client.get("/models")
    assert r.status_code == 401


async def test_concurrent_auto_version_race(db):
    """Two concurrent register_model_atomic calls for the same family
    must both succeed with distinct sequential versions, not both pick v1
    and one crash on UNIQUE(name, version)."""
    async with db.connect() as conn_a, db.connect() as conn_b:
        # Pre-create one experiment so FK is happy.
        spec = ExperimentSpec(model="m", dataset=["/d"])
        exp_id = await repository.create_experiment(conn_a, spec)
        await conn_a.execute(
            "UPDATE experiments SET status='completed' WHERE id=?", (exp_id,)
        )
        await conn_a.commit()

        async def do_register(conn):
            mid, ver = await repository.register_model_atomic(
                conn,
                name="rc-family",
                explicit_version=None,
                base_model="m",
                adapter_path="/tmp/x",
                experiment_id=exp_id,
                run_id=None,
                eval_summary=None,
                description=None,
                alias=None,
            )
            return mid, ver

        results = await asyncio.gather(
            do_register(conn_a), do_register(conn_b)
        )
    versions = sorted([r[1] for r in results])
    assert versions == [1, 2]
    ids = {r[0] for r in results}
    assert len(ids) == 2


def pytest_approx(actual, expected, tol=1e-6):
    return abs(actual - expected) < tol
