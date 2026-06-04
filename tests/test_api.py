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


def test_ui_config_public_and_strips_mlflow_credentials(state, client, monkeypatch):
    # /ui/config is one of the three public routes (no X-API-Key) and must
    # never echo embedded user:pass credentials from the MLflow URI.
    monkeypatch.setattr(
        "trainpipe.settings.settings.mlflow_tracking_uri",
        "http://user:s3cret@mlflow.internal:5000/path",
    )
    r = client.get("/ui/config")
    assert r.status_code == 200
    body = r.json()
    assert body["mlflow_tracking_uri"] == "http://mlflow.internal:5000/path"
    assert "s3cret" not in r.text


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


def test_submit_empty_dataset_returns_422(state, client):
    r = client.post(
        "/experiments",
        json={"model": "m", "dataset": []},
        headers=HEADERS,
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "empty_dataset"


def test_batch_submit_empty_dataset_reports_index(state, client):
    r = client.post(
        "/experiments/batch",
        json=[
            {"model": "m", "dataset": ["AI-ModelScope/x"]},
            {"model": "m", "dataset": []},
        ],
        headers=HEADERS,
    )
    assert r.status_code == 422
    assert r.json()["detail"]["spec_index"] == 1


def test_legacy_empty_dataset_row_still_lists(state, client):
    """Reading a row with dataset=[] (legacy / written pre-validation) must
    not 500 the list endpoint."""
    import json as _json

    async def insert_legacy():
        async with state["db"].connect() as conn:
            await conn.execute(
                "INSERT INTO experiments (id, spec_json, status, priority, "
                "created_at, queued_at) VALUES (?, ?, 'failed', 0, ?, ?)",
                (
                    "legacy-empty",
                    _json.dumps({"model": "m", "dataset": []}),
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            await conn.commit()

    _run(insert_legacy())
    r = client.get("/experiments", headers=HEADERS)
    assert r.status_code == 200
    ids = [e["id"] for e in r.json()]
    assert "legacy-empty" in ids


def test_submit_with_missing_local_path_returns_422(state, client, tmp_path):
    missing = tmp_path / "absent.jsonl"
    r = client.post(
        "/experiments",
        json={"model": "m", "dataset": [str(missing)]},
        headers=HEADERS,
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "missing_local_paths"
    paths = [m["path"] for m in detail["missing"]]
    assert str(missing) in paths


def test_submit_with_existing_local_path_ok(state, client, tmp_path):
    present = tmp_path / "train.jsonl"
    present.write_text("{}\n", encoding="utf-8")
    r = client.post(
        "/experiments",
        json={"model": "m", "dataset": [str(present)]},
        headers=HEADERS,
    )
    assert r.status_code == 201


def test_submit_with_remote_name_skips_validation(state, client):
    r = client.post(
        "/experiments",
        json={"model": "m", "dataset": ["meta-llama/Llama-3.1-8B"]},
        headers=HEADERS,
    )
    assert r.status_code == 201


def test_batch_reports_all_missing_paths(state, client, tmp_path):
    bad1 = str(tmp_path / "no1.jsonl")
    bad2 = str(tmp_path / "no2.jsonl")
    r = client.post(
        "/experiments/batch",
        json=[
            {"model": "m", "dataset": [bad1]},
            {"model": "m", "dataset": [bad2]},
        ],
        headers=HEADERS,
    )
    assert r.status_code == 422
    items = r.json()["detail"]["missing"]
    assert {(m["spec_index"], m["path"]) for m in items} == {
        (0, bad1),
        (1, bad2),
    }


def test_val_dataset_missing_path_reported(state, client, tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    missing_val = str(tmp_path / "val-absent.jsonl")
    r = client.post(
        "/experiments",
        json={
            "model": "m",
            "dataset": [str(train)],
            "val_dataset": [missing_val],
        },
        headers=HEADERS,
    )
    assert r.status_code == 422
    items = r.json()["detail"]["missing"]
    fields = {m["field"] for m in items}
    assert fields == {"val_dataset"}


def test_upload_dataset_jsonl(state, client, tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    content = b'{"messages":[{"role":"user","content":"hi"}]}\n'
    r = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("train.jsonl", content, "application/x-ndjson")},
        data={"name": "alpaca-tiny"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "alpaca-tiny"
    assert body["format"] == "jsonl"
    assert body["line_count"] == 1
    assert body["size_bytes"] == len(content)


def test_upload_rejects_bad_jsonl(state, client, tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    r = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("bad.jsonl", b"{not json}\n", "application/x-ndjson")},
        data={"name": "bad"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid_dataset_format"


def test_submit_with_dataset_ref_resolves_to_path(
    state, client, tmp_path, monkeypatch
):
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    # Upload first
    r = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("t.jsonl", b'{"a":1}\n', "application/x-ndjson")},
        data={"name": "t"},
    )
    assert r.status_code == 201
    ds_id = r.json()["id"]

    # Submit referring to the dataset by id
    r = client.post(
        "/experiments",
        headers=HEADERS,
        json={"model": "m", "dataset": [f"ds:{ds_id}#5"]},
    )
    assert r.status_code == 201, r.text
    exp_id = r.json()["experiment_id"]

    # Stored spec should have the resolved path with the #5 suffix
    detail = client.get(f"/experiments/{exp_id}", headers=HEADERS).json()
    paths = detail["spec"]["dataset"]
    assert len(paths) == 1
    assert paths[0].endswith("t.jsonl#5")
    assert "ds:" not in paths[0]


def test_submit_unknown_ds_ref_returns_422(state, client):
    r = client.post(
        "/experiments",
        headers=HEADERS,
        json={"model": "m", "dataset": ["ds:deadbeef99"]},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unknown_dataset_ref"


def test_list_and_delete_dataset(state, client, tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    r = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("x.jsonl", b'{"a":1}\n', "application/x-ndjson")},
        data={"name": "x"},
    )
    ds_id = r.json()["id"]

    r = client.get("/datasets", headers=HEADERS)
    assert r.status_code == 200
    assert any(d["id"] == ds_id for d in r.json())

    r = client.delete(f"/datasets/{ds_id}", headers=HEADERS)
    assert r.status_code == 200
    assert r.json() == {"deleted": True}

    r = client.get(f"/datasets/{ds_id}", headers=HEADERS)
    assert r.status_code == 404


def test_upload_same_content_dedups_to_existing(state, client, tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    content = b'{"a":1}\n'
    first = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("a.jsonl", content, "application/x-ndjson")},
        data={"name": "first"},
    )
    assert first.status_code == 201
    first_id = first.json()["id"]

    # Same bytes, different filename/name → dedup to the existing record (200).
    second = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("b.jsonl", content, "application/x-ndjson")},
        data={"name": "second"},
    )
    assert second.status_code == 200, second.text
    assert second.json()["id"] == first_id
    # Only one row should exist.
    listing = client.get("/datasets", headers=HEADERS).json()
    assert sum(1 for d in listing if d["id"] == first_id) == 1
    assert len(listing) == 1


def test_delete_dataset_blocked_when_referenced(state, client, tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    up = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("t.jsonl", b'{"a":1}\n', "application/x-ndjson")},
        data={"name": "t"},
    )
    ds_id = up.json()["id"]
    # Queued experiment references it → delete must be refused.
    client.post(
        "/experiments",
        headers=HEADERS,
        json={"model": "m", "dataset": [f"ds:{ds_id}"]},
    )

    blocked = client.delete(f"/datasets/{ds_id}", headers=HEADERS)
    assert blocked.status_code == 409
    detail = blocked.json()["detail"]
    assert detail["error"] == "dataset_in_use"
    assert detail["experiment_ids"]

    # force=true overrides.
    forced = client.delete(f"/datasets/{ds_id}?force=true", headers=HEADERS)
    assert forced.status_code == 200
    assert forced.json() == {"deleted": True}


def test_preview_rejects_out_of_range_n(state, client, tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    up = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("p.jsonl", b'{"a":1}\n{"b":2}\n', "application/x-ndjson")},
        data={"name": "p"},
    )
    ds_id = up.json()["id"]
    assert client.get(f"/datasets/{ds_id}/preview?n=0", headers=HEADERS).status_code == 422
    assert (
        client.get(f"/datasets/{ds_id}/preview?n=99999", headers=HEADERS).status_code
        == 422
    )
    ok = client.get(f"/datasets/{ds_id}/preview?n=1", headers=HEADERS)
    assert ok.status_code == 200
    assert ok.text == '{"a":1}'


def test_create_study_validates_base_spec_dataset(state, client, tmp_path):
    bad = str(tmp_path / "absent.jsonl")
    r = client.post(
        "/studies",
        json={
            "name": "s",
            "base_spec": {"model": "m", "dataset": [bad]},
            "search_space": {
                "hyperparameters.learning_rate": {
                    "kind": "loguniform",
                    "low": 1e-5,
                    "high": 1e-2,
                }
            },
            "target_metric": "eval/loss",
        },
        headers=HEADERS,
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "missing_local_paths"


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
