import asyncio

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import (
    get_db,
    get_gpu_pool,
    get_scheduler,
    get_study_manager,
)
from trainpipe.api.main import app
from trainpipe.api.schemas import ExperimentSpec, StudyConfig
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.scheduler.gpu_pool import GpuPool

HEADERS = {"X-API-Key": "test-key"}


class _NoopScheduler:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel_experiment(self, experiment_id: str) -> bool:
        self.cancelled.append(experiment_id)
        return True


class _StubStudyManager:
    def __init__(self) -> None:
        self.created: list[StudyConfig] = []
        self.cancelled_ids: list[str] = []
        self.cancel_outcome: bool = True

    async def create_and_start(self, config: StudyConfig) -> str:
        self.created.append(config)
        return f"stub-{len(self.created)}"

    async def cancel(self, study_id: str) -> bool:
        self.cancelled_ids.append(study_id)
        return self.cancel_outcome


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")

    db = Database(tmp_path / "test.sqlite3")
    _run(db.init())

    pool = GpuPool([])
    scheduler = _NoopScheduler()
    study_manager = _StubStudyManager()

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: scheduler
    app.dependency_overrides[get_gpu_pool] = lambda: pool
    app.dependency_overrides[get_study_manager] = lambda: study_manager

    yield {
        "db": db,
        "pool": pool,
        "scheduler": scheduler,
        "study_manager": study_manager,
    }

    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_no_auth_required(state, client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_submit_requires_auth(state, client):
    r = client.post("/experiments", json={"model": "m", "dataset": ["d"]})
    assert r.status_code == 401


def test_submit_rejects_wrong_key(state, client):
    r = client.post(
        "/experiments",
        json={"model": "m", "dataset": ["d"]},
        headers={"X-API-Key": "nope"},
    )
    assert r.status_code == 401


def test_submit_and_get(state, client):
    r = client.post(
        "/experiments", json={"model": "m1", "dataset": ["d1"]}, headers=HEADERS
    )
    assert r.status_code == 201
    exp_id = r.json()["experiment_id"]

    r = client.get(f"/experiments/{exp_id}", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["spec"]["model"] == "m1"
    assert body["spec"]["dataset"] == ["d1"]


def test_cancel_queued_experiment(state, client):
    r = client.post(
        "/experiments", json={"model": "m", "dataset": ["d"]}, headers=HEADERS
    )
    exp_id = r.json()["experiment_id"]

    r = client.post(f"/experiments/{exp_id}/cancel", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"

    rec = client.get(f"/experiments/{exp_id}", headers=HEADERS).json()
    assert rec["status"] == "cancelled"
    assert rec["finished_at"] is not None


def test_cancel_running_signals_scheduler(state, client):
    r = client.post(
        "/experiments", json={"model": "m", "dataset": ["d"]}, headers=HEADERS
    )
    exp_id = r.json()["experiment_id"]

    async def mark_running():
        async with state["db"].connect() as conn:
            await conn.execute(
                "UPDATE experiments SET status = 'running' WHERE id = ?", (exp_id,)
            )
            await conn.commit()

    _run(mark_running())

    r = client.post(f"/experiments/{exp_id}/cancel", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelling"
    assert exp_id in state["scheduler"].cancelled


def test_list_filter_by_status(state, client):
    a = client.post(
        "/experiments", json={"model": "m", "dataset": ["d"]}, headers=HEADERS
    ).json()["experiment_id"]
    b = client.post(
        "/experiments", json={"model": "m", "dataset": ["d"]}, headers=HEADERS
    ).json()["experiment_id"]

    async def mark_completed():
        async with state["db"].connect() as conn:
            await conn.execute(
                "UPDATE experiments SET status = 'completed' WHERE id = ?", (b,)
            )
            await conn.commit()

    _run(mark_completed())

    queued = client.get("/experiments?status=queued", headers=HEADERS).json()
    assert {x["id"] for x in queued} == {a}
    completed = client.get("/experiments?status=completed", headers=HEADERS).json()
    assert {x["id"] for x in completed} == {b}


def test_batch_submit(state, client):
    payload = [
        {"model": "m", "dataset": ["d"], "name": "a"},
        {"model": "m", "dataset": ["d"], "name": "b"},
    ]
    r = client.post("/experiments/batch", json=payload, headers=HEADERS)
    assert r.status_code == 201
    assert len(r.json()["experiment_ids"]) == 2


def test_batch_rejects_empty(state, client):
    r = client.post("/experiments/batch", json=[], headers=HEADERS)
    assert r.status_code == 422


def test_gpus_empty_pool(state, client):
    r = client.get("/gpus", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["free"] == []
    assert data["leases"] == []


def test_studies_initially_empty(state, client):
    r = client.get("/studies", headers=HEADERS)
    assert r.status_code == 200
    assert r.json() == []


def test_404_on_missing(state, client):
    r = client.get("/experiments/deadbeef", headers=HEADERS)
    assert r.status_code == 404

    r = client.get("/studies/deadbeef", headers=HEADERS)
    assert r.status_code == 404

    r = client.post("/experiments/deadbeef/cancel", headers=HEADERS)
    assert r.status_code == 404


def test_logs_empty_when_no_path(state, client):
    r = client.post(
        "/experiments", json={"model": "m", "dataset": ["d"]}, headers=HEADERS
    )
    exp_id = r.json()["experiment_id"]

    r = client.get(f"/experiments/{exp_id}/logs", headers=HEADERS)
    assert r.status_code == 200
    assert r.text == ""


def test_logs_returns_file_content(state, client, tmp_path):
    log_file = tmp_path / "x.log"
    log_file.write_text("training-line-1\ntraining-line-2\n", encoding="utf-8")

    async def setup():
        async with state["db"].connect() as conn:
            eid = await repository.create_experiment(
                conn, ExperimentSpec(model="m", dataset=["d"])
            )
            await conn.execute(
                "UPDATE experiments SET log_path = ? WHERE id = ?", (str(log_file), eid)
            )
            await conn.commit()
            return eid

    exp_id = _run(setup())
    r = client.get(f"/experiments/{exp_id}/logs", headers=HEADERS)
    assert r.status_code == 200
    assert "training-line-1" in r.text
    assert "training-line-2" in r.text


def test_create_study_returns_id_from_manager(state, client):
    payload = {
        "name": "sweep-1",
        "base_spec": {"model": "m", "dataset": ["d"]},
        "search_space": {
            "hyperparameters.learning_rate": {
                "kind": "loguniform",
                "low": 1e-5,
                "high": 1e-2,
            }
        },
        "target_metric": "eval/loss",
        "n_trials": 5,
    }
    r = client.post("/studies", json=payload, headers=HEADERS)
    assert r.status_code == 201
    assert r.json()["study_id"] == "stub-1"
    assert len(state["study_manager"].created) == 1
    assert state["study_manager"].created[0].name == "sweep-1"


def test_cancel_study_404_when_no_record(state, client):
    r = client.post("/studies/no-such-id/cancel", headers=HEADERS)
    assert r.status_code == 404


def test_cancel_study_invokes_manager_when_record_exists(state, client):
    cfg = StudyConfig(
        name="s",
        base_spec=ExperimentSpec(model="m", dataset=["d"]),
        search_space={
            "hyperparameters.learning_rate": {
                "kind": "loguniform",
                "low": 1e-5,
                "high": 1e-2,
            }
        },
        target_metric="eval/loss",
    )

    async def insert():
        async with state["db"].connect() as conn:
            return await repository.create_study(conn, cfg, "sqlite:///dummy")

    sid = _run(insert())
    r = client.post(f"/studies/{sid}/cancel", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    assert sid in state["study_manager"].cancelled_ids


def test_stream_logs_emits_end_on_terminal_status(state, client, tmp_path):
    log_file = tmp_path / "x.log"
    log_file.write_text("already-done\n", encoding="utf-8")

    async def setup():
        async with state["db"].connect() as conn:
            eid = await repository.create_experiment(
                conn, ExperimentSpec(model="m", dataset=["d"])
            )
            await conn.execute(
                "UPDATE experiments SET status = 'completed', log_path = ? WHERE id = ?",
                (str(log_file), eid),
            )
            await conn.commit()
            return eid

    exp_id = _run(setup())
    with client.stream(
        "GET", f"/experiments/{exp_id}/logs/stream", headers=HEADERS
    ) as r:
        assert r.status_code == 200
        text_chunks: list[str] = []
        end_seen = False
        extra = 0
        for raw in r.iter_lines():
            text_chunks.append(raw)
            if end_seen:
                extra += 1
                if extra >= 2:
                    break
            elif "event: end" in raw:
                end_seen = True
            if len(text_chunks) > 50:
                break
    joined = "\n".join(text_chunks)
    assert "event: end" in joined
    assert "data: completed" in joined
