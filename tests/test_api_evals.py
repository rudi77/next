import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import (
    get_db,
    get_eval_dispatcher,
    get_gpu_pool,
    get_scheduler,
    get_study_manager,
)
from trainpipe.api.main import app
from trainpipe.api.schemas import (
    EvalRunStatus,
    ExperimentSpec,
    MetricAggregate,
)
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.scheduler.gpu_pool import GpuPool

HEADERS = {"X-API-Key": "test-key"}


def _run(coro):
    return asyncio.run(coro)


def _in_conn(state, fn):
    """Run `fn(conn)` against a fresh aiosqlite connection from state['db']."""
    async def go():
        async with state["db"].connect() as conn:
            return await fn(conn)
    return _run(go())


@pytest.fixture
def eval_state(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    db = Database(tmp_path / "evals.sqlite3")
    _run(db.init())

    pool = GpuPool([])
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: object()
    app.dependency_overrides[get_gpu_pool] = lambda: pool
    app.dependency_overrides[get_study_manager] = lambda: object()
    app.dependency_overrides[get_eval_dispatcher] = lambda: object()

    yield {"db": db, "tmp_path": tmp_path}
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _make_dataset(tmp_path: Path, name: str = "eval.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"prompt": "x", "gold": "y"}) + "\n", encoding="utf-8")
    return p


def _suite_payload(dataset_path: str, name: str = "s1") -> dict:
    return {
        "name": name,
        "description": "test suite",
        "dataset": dataset_path,
        "metrics": [{"kind": "exact_match"}],
        "inference_params": {"max_new_tokens": 32},
    }


def _make_experiment(state, name: str = "e") -> str:
    return _in_conn(
        state,
        lambda conn: repository.create_experiment(
            conn, ExperimentSpec(name=name, model="m", dataset=["d"]),
        ),
    )


def _finalize_with_results(state, run_id: str, scores: list[tuple[int, float]]) -> None:
    async def go():
        async with state["db"].connect() as conn:
            for idx, s in scores:
                await repository.add_eval_result(
                    conn,
                    run_id=run_id,
                    sample_index=idx,
                    input={"prompt": f"q{idx}"},
                    prediction="y" if s == 1.0 else "wrong",
                    gold={"a": "y"},
                    scores={"exact_match": s},
                )
            mean = sum(s for _, s in scores) / len(scores)
            await repository.finalize_eval_run(
                conn,
                run_id,
                status=EvalRunStatus.COMPLETED,
                aggregate={
                    "exact_match": MetricAggregate(
                        mean=mean, std=0.0, count=len(scores),
                    )
                },
                sample_count=len(scores),
            )
    _run(go())


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def test_create_suite_requires_auth(eval_state, client):
    r = client.post(
        "/evals/suites", json=_suite_payload(str(eval_state["tmp_path"])),
    )
    assert r.status_code == 401


def test_create_suite_happy_path(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    r = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "happy"), headers=HEADERS,
    )
    assert r.status_code == 201, r.json()
    body = r.json()
    assert body["name"] == "happy"
    assert body["dataset_path"] == str(ds)
    assert body["metrics"][0]["kind"] == "exact_match"


def test_create_suite_name_conflict(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    r1 = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "dupe"), headers=HEADERS,
    )
    assert r1.status_code == 201
    r2 = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "dupe"), headers=HEADERS,
    )
    assert r2.status_code == 409


def test_create_suite_unknown_metric_kind(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    payload = _suite_payload(str(ds), "badmetric")
    payload["metrics"] = [{"kind": "never_existed"}]
    r = client.post("/evals/suites", json=payload, headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unknown_metric_kind"


def test_create_suite_invalid_metric_config(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    payload = _suite_payload(str(ds), "badcfg")
    payload["metrics"] = [{"kind": "exact_match", "config": {"gold_field": ""}}]
    r = client.post("/evals/suites", json=payload, headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "metric_config_invalid"


def test_create_suite_resolves_ds_ref(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"], "for-ref.jsonl")
    _in_conn(
        eval_state,
        lambda conn: repository.create_dataset(
            conn,
            name="for-ref",
            path=str(ds),
            fmt="jsonl",
            size_bytes=ds.stat().st_size,
            sha256="abc",
            dataset_id="deadbeef",
        ),
    )
    payload = _suite_payload("ds:deadbeef", "ref-suite")
    r = client.post("/evals/suites", json=payload, headers=HEADERS)
    assert r.status_code == 201, r.json()
    assert r.json()["dataset_path"] == str(ds)


def test_create_suite_unknown_ds_ref(eval_state, client):
    payload = _suite_payload("ds:abc123", "missing-ref")
    r = client.post("/evals/suites", json=payload, headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unknown_dataset_ref"


def test_list_and_get_suite(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    sid = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "list-me"), headers=HEADERS,
    ).json()["id"]
    r = client.get("/evals/suites", headers=HEADERS)
    assert r.status_code == 200
    assert any(s["id"] == sid for s in r.json())

    r2 = client.get(f"/evals/suites/{sid}", headers=HEADERS)
    assert r2.status_code == 200
    assert r2.json()["name"] == "list-me"


def test_delete_suite_blocked_by_active_run(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    sid = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "lock-me"), headers=HEADERS,
    ).json()["id"]
    exp_id = _make_experiment(eval_state)
    client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": exp_id},
        headers=HEADERS,
    )

    r = client.delete(f"/evals/suites/{sid}", headers=HEADERS)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "suite_in_use"

    r2 = client.delete(f"/evals/suites/{sid}?force=true", headers=HEADERS)
    assert r2.status_code == 200
    assert r2.json()["status"] == "deleted"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_create_run_unknown_suite(eval_state, client):
    exp_id = _make_experiment(eval_state)
    r = client.post(
        "/evals/runs",
        json={"suite_id": "no-such-suite", "experiment_id": exp_id},
        headers=HEADERS,
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unknown_suite"


def test_create_run_unknown_experiment(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    sid = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "for-run"), headers=HEADERS,
    ).json()["id"]
    r = client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": "deadbeef"},
        headers=HEADERS,
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unknown_experiment"


def test_create_run_happy_path(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    sid = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "ok-run"), headers=HEADERS,
    ).json()["id"]
    exp_id = _make_experiment(eval_state, name="exp-x")
    r = client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": exp_id},
        headers=HEADERS,
    )
    assert r.status_code == 201, r.json()
    body = r.json()
    assert body["status"] == "queued"
    assert body["suite_id"] == sid
    assert body["experiment_id"] == exp_id
    assert body["model_ref"] == "exp-x"


def test_list_runs_filters(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    sid = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "filter-me"), headers=HEADERS,
    ).json()["id"]
    exp1 = _make_experiment(eval_state, name="e1")
    exp2 = _make_experiment(eval_state, name="e2")
    rid1 = client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": exp1},
        headers=HEADERS,
    ).json()["id"]
    client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": exp2},
        headers=HEADERS,
    )
    r = client.get(f"/evals/runs?experiment_id={exp1}", headers=HEADERS)
    assert r.status_code == 200
    ids = [x["id"] for x in r.json()]
    assert ids == [rid1]


def test_cancel_queued_run(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    sid = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "cancel-me"), headers=HEADERS,
    ).json()["id"]
    exp_id = _make_experiment(eval_state)
    rid = client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": exp_id},
        headers=HEADERS,
    ).json()["id"]
    r = client.post(f"/evals/runs/{rid}/cancel", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    body = client.get(f"/evals/runs/{rid}", headers=HEADERS).json()
    assert body["status"] == "cancelled"


def test_list_results_returns_persisted_rows(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    sid = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "with-results"), headers=HEADERS,
    ).json()["id"]
    exp_id = _make_experiment(eval_state)
    rid = client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": exp_id},
        headers=HEADERS,
    ).json()["id"]
    _in_conn(
        eval_state,
        lambda conn: repository.add_eval_result(
            conn,
            run_id=rid,
            sample_index=0,
            input={"q": "x"},
            prediction="y",
            gold={"a": "y"},
            scores={"exact_match": 1.0},
        ),
    )
    r = client.get(f"/evals/runs/{rid}/results", headers=HEADERS)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["scores"]["exact_match"] == 1.0


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def test_compare_requires_two_runs(eval_state, client):
    r = client.get("/evals/compare?run_ids=only-one", headers=HEADERS)
    assert r.status_code == 422


def test_compare_two_runs_with_delta_and_regression(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"])
    sid = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "cmp-suite"), headers=HEADERS,
    ).json()["id"]
    exp_a = _make_experiment(eval_state, name="A")
    exp_b = _make_experiment(eval_state, name="B")

    rid_a = client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": exp_a},
        headers=HEADERS,
    ).json()["id"]
    rid_b = client.post(
        "/evals/runs",
        json={"suite_id": sid, "experiment_id": exp_b},
        headers=HEADERS,
    ).json()["id"]

    _finalize_with_results(eval_state, rid_a, [(0, 1.0), (1, 1.0), (2, 0.0)])
    _finalize_with_results(eval_state, rid_b, [(0, 1.0), (1, 0.0), (2, 0.0)])

    r = client.get(f"/evals/compare?run_ids={rid_a},{rid_b}", headers=HEADERS)
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["suite_id"] == sid
    assert len(body["runs"]) == 2
    assert rid_a in body["aggregate_delta"]["exact_match"]
    assert rid_b in body["aggregate_delta"]["exact_match"]
    indices = {r["sample_index"] for r in body["regressions"]}
    assert 1 in indices  # diverging sample
    assert 0 not in indices  # both scored 1.0


def test_compare_rejects_suite_mismatch(eval_state, client):
    ds = _make_dataset(eval_state["tmp_path"], "a.jsonl")
    ds2 = _make_dataset(eval_state["tmp_path"], "b.jsonl")
    sid_a = client.post(
        "/evals/suites", json=_suite_payload(str(ds), "mis-a"), headers=HEADERS,
    ).json()["id"]
    sid_b = client.post(
        "/evals/suites", json=_suite_payload(str(ds2), "mis-b"), headers=HEADERS,
    ).json()["id"]
    exp = _make_experiment(eval_state)
    rid_a = client.post(
        "/evals/runs",
        json={"suite_id": sid_a, "experiment_id": exp},
        headers=HEADERS,
    ).json()["id"]
    rid_b = client.post(
        "/evals/runs",
        json={"suite_id": sid_b, "experiment_id": exp},
        headers=HEADERS,
    ).json()["id"]
    r = client.get(f"/evals/compare?run_ids={rid_a},{rid_b}", headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "suite_mismatch"
