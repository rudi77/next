"""End-to-end smoke for Phase 6.

Walks the full Acceptance-Path from ROADMAP.md:

    submit experiment → auto-eval against suite X
    → /evals/compare shows run A vs run B with per-sample Δ

Without spawning a real training subprocess (we stand in for the
scheduler at the hook boundary) and without GPUs (MockInferenceBackend).
"""

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
)
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.evals.dispatcher import EvalDispatcher
from trainpipe.evals.inference import MockInferenceBackend
from trainpipe.scheduler.gpu_pool import GpuPool
from trainpipe.scheduler.loop import Scheduler

HEADERS = {"X-API-Key": "test-key"}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def e2e(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    db = Database(tmp_path / "e2e.sqlite3")
    _run(db.init())

    pool = GpuPool([])
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: object()
    app.dependency_overrides[get_gpu_pool] = lambda: pool
    app.dependency_overrides[get_study_manager] = lambda: object()
    app.dependency_overrides[get_eval_dispatcher] = lambda: object()

    yield {"db": db, "tmp_path": tmp_path, "pool": pool}
    app.dependency_overrides.clear()


def _eval_dataset(tmp_path: Path) -> Path:
    p = tmp_path / "eval.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "capital of France", "gold": "Paris"}),
                json.dumps({"prompt": "capital of Germany", "gold": "Berlin"}),
                json.dumps({"prompt": "capital of Italy", "gold": "Rome"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


def test_full_phase6_acceptance_path(e2e):
    """Acceptance path: experiment auto-eval → compare run A vs run B with Δ.

    1. Create one eval suite via the REST API.
    2. Create two experiments with that suite in ``auto_eval``.
    3. Fire the scheduler hook directly (no real training subprocess) to
       enqueue the auto-eval runs.
    4. Stand in for the EvalDispatcher by manually claiming + executing
       each run with a different ``MockInferenceBackend`` (model A always
       answers correctly; model B gets one wrong).
    5. GET /evals/compare and verify: both runs visible, aggregate Δ
       reflects the regression, and the diverging sample is surfaced in
       the per-sample regression list.
    """
    client = TestClient(app)
    db = e2e["db"]
    dataset = _eval_dataset(e2e["tmp_path"])

    # (1) create the suite
    r = client.post(
        "/evals/suites",
        headers=HEADERS,
        json={
            "name": "capitals-suite",
            "dataset": str(dataset),
            "metrics": [{"kind": "exact_match"}],
            "inference_params": {"max_new_tokens": 16},
        },
    )
    assert r.status_code == 201, r.json()
    suite_id = r.json()["id"]

    # (2) create two experiments that auto-eval this suite
    async def make_experiment(name: str) -> str:
        async with db.connect() as conn:
            return await repository.create_experiment(
                conn,
                ExperimentSpec(
                    name=name, model="m", dataset=["d"], auto_eval=[suite_id]
                ),
            )

    exp_a = _run(make_experiment("model-a"))
    exp_b = _run(make_experiment("model-b"))

    # (3) fire the scheduler hook (the auto_eval trigger) directly
    sched = Scheduler(db, GpuPool([]))
    _run(sched._enqueue_auto_evals(exp_a))
    _run(sched._enqueue_auto_evals(exp_b))

    # Each experiment should now have exactly one queued eval run.
    runs_a = client.get(
        f"/evals/runs?experiment_id={exp_a}", headers=HEADERS
    ).json()
    runs_b = client.get(
        f"/evals/runs?experiment_id={exp_b}", headers=HEADERS
    ).json()
    assert len(runs_a) == 1 and runs_a[0]["status"] == "queued"
    assert len(runs_b) == 1 and runs_b[0]["status"] == "queued"
    rid_a, rid_b = runs_a[0]["id"], runs_b[0]["id"]

    # (4) drive both runs to completion via the EvalDispatcher with
    # different backends.
    def factory_for(model: str):
        # Closure captures the suite id implicitly via the run object.
        if model == "model-a":
            responses = {
                "capital of France": "Paris",
                "capital of Germany": "Berlin",
                "capital of Italy": "Rome",
            }
        else:
            responses = {
                "capital of France": "Paris",
                "capital of Germany": "Munich",  # regression
                "capital of Italy": "Rome",
            }
        return MockInferenceBackend(responses_by_key=responses)

    def dispatcher_factory(run, _suite):
        return factory_for(run.model_ref)

    dispatcher = EvalDispatcher(
        db,
        GpuPool([]),
        backend_factory=dispatcher_factory,
        gpus_per_run=0,
        poll_interval_sec=0.05,
    )

    async def drain():
        await dispatcher.start()
        try:
            for _ in range(200):
                async with db.connect() as conn:
                    a = await repository.get_eval_run(conn, rid_a)
                    b = await repository.get_eval_run(conn, rid_b)
                if (
                    a.status == EvalRunStatus.COMPLETED
                    and b.status == EvalRunStatus.COMPLETED
                ):
                    return
                await asyncio.sleep(0.05)
            raise AssertionError("dispatcher did not complete both runs in time")
        finally:
            await dispatcher.stop()

    _run(drain())

    # Sanity: aggregates landed.
    a_body = client.get(f"/evals/runs/{rid_a}", headers=HEADERS).json()
    b_body = client.get(f"/evals/runs/{rid_b}", headers=HEADERS).json()
    assert a_body["aggregate"]["exact_match"]["mean"] == 1.0
    assert b_body["aggregate"]["exact_match"]["mean"] == pytest.approx(2 / 3)

    # (5) compare
    cmp_resp = client.get(
        f"/evals/compare?run_ids={rid_a},{rid_b}", headers=HEADERS
    )
    assert cmp_resp.status_code == 200, cmp_resp.json()
    cmp = cmp_resp.json()
    assert cmp["suite_id"] == suite_id
    assert {r["id"] for r in cmp["runs"]} == {rid_a, rid_b}
    delta = cmp["aggregate_delta"]["exact_match"]
    assert delta[rid_a] == 1.0
    assert delta[rid_b] == pytest.approx(2 / 3)

    # The Berlin/Munich sample is index 1 in the dataset. It should be
    # the only sample in the regression list (A and C both scored 1.0
    # on both runs).
    regression_indices = [s["sample_index"] for s in cmp["regressions"]]
    assert 1 in regression_indices
    assert 0 not in regression_indices
    assert 2 not in regression_indices

    # The per-run breakdown for that sample must contain both predictions.
    sample = next(s for s in cmp["regressions"] if s["sample_index"] == 1)
    assert sample["per_run"][rid_a]["prediction"] == "Berlin"
    assert sample["per_run"][rid_b]["prediction"] == "Munich"
    assert sample["per_run"][rid_a]["scores"]["exact_match"] == 1.0
    assert sample["per_run"][rid_b]["scores"]["exact_match"] == 0.0
