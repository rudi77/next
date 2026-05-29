"""Tests for Phase 17: watches (interval + metric_threshold)."""

import asyncio

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from trainpipe.api.deps import (
    get_db,
    get_gpu_pool,
    get_pipeline_manager,
    get_scheduler,
    get_study_manager,
)
from trainpipe.api.main import app
from trainpipe.api.schemas import (
    EvalRunStatus,
    ExperimentSpec,
    InferenceParams,
    MetricAggregate,
    MetricConfig,
    PipelineConfig,
    StageSpec,
)
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.pipelines.manager import PipelineManager
from trainpipe.scheduler.gpu_pool import GpuPool
from trainpipe.watches.manager import WatchManager

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


@pytest_asyncio.fixture
async def state(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    monkeypatch.setattr("trainpipe.settings.settings.poll_interval_sec", 0.01)
    db = Database(tmp_path / "test.sqlite3")
    await db.init()
    pmanager = PipelineManager(db)
    wmanager = WatchManager(db, pmanager, poll_interval_sec=0.05)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: _NoopScheduler()
    app.dependency_overrides[get_gpu_pool] = lambda: GpuPool([])
    app.dependency_overrides[get_study_manager] = lambda: _StubStudyManager()
    app.dependency_overrides[get_pipeline_manager] = lambda: pmanager
    try:
        yield {"db": db, "pmanager": pmanager, "wmanager": wmanager}
    finally:
        try:
            await wmanager.stop()
            await pmanager.stop_all()
        except Exception:
            pass
        app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _pipeline_cfg() -> PipelineConfig:
    return PipelineConfig(
        name="auto-retrain",
        stages=[
            StageSpec(
                name="train",
                base_spec=ExperimentSpec(model="m", dataset=["/x"]),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Create / list / enable / disable
# ---------------------------------------------------------------------------


async def test_create_interval_watch(state, client):
    r = client.post(
        "/watches",
        headers=HEADERS,
        json={
            "name": "daily",
            "kind": "interval",
            "interval_seconds": 86400,
            "pipeline_config": _pipeline_cfg().model_dump(),
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "daily"
    assert body["interval_seconds"] == 86400
    assert body["enabled"] is True


async def test_create_interval_requires_field(state, client):
    r = client.post(
        "/watches",
        headers=HEADERS,
        json={
            "name": "x",
            "kind": "interval",
            "pipeline_config": _pipeline_cfg().model_dump(),
        },
    )
    assert r.status_code == 422
    assert "interval_seconds" in r.json()["detail"]["detail"]


async def test_create_threshold_requires_fields(state, client):
    r = client.post(
        "/watches",
        headers=HEADERS,
        json={
            "name": "drift",
            "kind": "metric_threshold",
            "suite_id": "abc",
            "pipeline_config": _pipeline_cfg().model_dump(),
        },
    )
    assert r.status_code == 422
    body = r.json()["detail"]
    assert "metric_name" in body["fields"] or "threshold" in body["fields"]


async def test_enable_disable(state, client):
    r = client.post(
        "/watches",
        headers=HEADERS,
        json={
            "name": "x",
            "kind": "interval",
            "interval_seconds": 60,
            "pipeline_config": _pipeline_cfg().model_dump(),
        },
    )
    wid = r.json()["id"]
    assert client.post(f"/watches/{wid}/disable", headers=HEADERS).json() == {"enabled": False}
    assert client.post(f"/watches/{wid}/enable", headers=HEADERS).json() == {"enabled": True}


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


async def test_interval_fires_when_due(state):
    """An interval watch with no last_fired_at must fire immediately."""
    async with state["db"].connect() as conn:
        wid = await repository.create_watch(
            conn,
            name="immediately",
            kind="interval",
            pipeline_config=_pipeline_cfg(),
            interval_seconds=60,
        )
    fired: list[str] = []

    async def fake_create_and_start(name, cfg):
        fired.append(name)
        return "fake-pipeline-id"

    state["wmanager"].pipelines.create_and_start = fake_create_and_start  # type: ignore[assignment]
    await state["wmanager"]._tick()
    assert fired == ["watch:immediately"]

    async with state["db"].connect() as conn:
        watch = await repository.get_watch(conn, wid)
    assert watch.last_fired_at is not None
    assert watch.last_fired_pipeline_id == "fake-pipeline-id"


async def test_interval_does_not_refire_immediately(state):
    """A watch that just fired shouldn't re-fire on the next tick."""
    async with state["db"].connect() as conn:
        await repository.create_watch(
            conn,
            name="cooldown",
            kind="interval",
            pipeline_config=_pipeline_cfg(),
            interval_seconds=60,
        )

    fire_count = 0

    async def counting_fire(name, cfg):
        nonlocal fire_count
        fire_count += 1
        return f"p{fire_count}"

    state["wmanager"].pipelines.create_and_start = counting_fire  # type: ignore[assignment]
    await state["wmanager"]._tick()
    await state["wmanager"]._tick()
    assert fire_count == 1


async def test_metric_threshold_fires_below(state):
    async with state["db"].connect() as conn:
        # Set up a suite + a completed eval run with mean=0.5.
        suite_id = await repository.create_eval_suite(
            conn,
            name="suite-drift",
            description=None,
            dataset_path="/tmp/eval.jsonl",
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        # Create + finalize an eval run with a low score.
        run_id = await repository.create_eval_run(
            conn,
            suite_id=suite_id,
            experiment_id=None,
            model_ref="invoice@production",
            triggered_by="auto",
        )
        await repository.finalize_eval_run(
            conn,
            run_id,
            status=EvalRunStatus.COMPLETED,
            aggregate={"exact_match": MetricAggregate(mean=0.5, std=0.0, count=10)},
            sample_count=10,
        )
        await repository.create_watch(
            conn,
            name="drift-detector",
            kind="metric_threshold",
            pipeline_config=_pipeline_cfg(),
            suite_id=suite_id,
            metric_name="exact_match",
            threshold=0.8,
        )

    fired: list[str] = []

    async def fake_create(name, cfg):
        fired.append(name)
        return "p1"

    state["wmanager"].pipelines.create_and_start = fake_create  # type: ignore[assignment]
    await state["wmanager"]._tick()
    assert fired == ["watch:drift-detector"]


async def test_metric_threshold_does_not_fire_when_above(state):
    async with state["db"].connect() as conn:
        suite_id = await repository.create_eval_suite(
            conn,
            name="suite-ok",
            description=None,
            dataset_path="/tmp/eval.jsonl",
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        run_id = await repository.create_eval_run(
            conn,
            suite_id=suite_id,
            experiment_id=None,
            model_ref="x",
            triggered_by="auto",
        )
        await repository.finalize_eval_run(
            conn,
            run_id,
            status=EvalRunStatus.COMPLETED,
            aggregate={"exact_match": MetricAggregate(mean=0.95, std=0.0, count=10)},
            sample_count=10,
        )
        await repository.create_watch(
            conn,
            name="no-drift",
            kind="metric_threshold",
            pipeline_config=_pipeline_cfg(),
            suite_id=suite_id,
            metric_name="exact_match",
            threshold=0.8,
        )

    fired = []

    async def fake_create(name, cfg):
        fired.append(name)
        return "p1"

    state["wmanager"].pipelines.create_and_start = fake_create  # type: ignore[assignment]
    await state["wmanager"]._tick()
    assert fired == []


async def test_failure_counter_increments_and_resets_on_success(state):
    """Each failed fire bumps consecutive_failures; a successful fire
    resets it back to 0."""
    async with state["db"].connect() as conn:
        wid = await repository.create_watch(
            conn,
            name="flaky",
            kind="interval",
            pipeline_config=_pipeline_cfg(),
            interval_seconds=60,
        )

    # First two ticks: pipeline creation raises.
    boom_count = 0

    async def boom(name, cfg):
        nonlocal boom_count
        boom_count += 1
        raise ValueError("malformed pipeline config")

    state["wmanager"].pipelines.create_and_start = boom  # type: ignore[assignment]
    await state["wmanager"]._tick()

    async with state["db"].connect() as conn:
        w = await repository.get_watch(conn, wid)
    assert w.consecutive_failures == 1
    assert "ValueError" in (w.last_error or "")
    assert w.enabled is True

    # Now flip to success — counter resets.
    async def ok(name, cfg):
        return "pipeline-id"

    state["wmanager"].pipelines.create_and_start = ok  # type: ignore[assignment]
    await state["wmanager"]._tick()
    async with state["db"].connect() as conn:
        w = await repository.get_watch(conn, wid)
    assert w.consecutive_failures == 0
    assert w.last_error is None


async def test_auto_disable_after_threshold(state):
    """After failure_disable_threshold consecutive failures, the watch
    must flip enabled=False so it stops getting polled."""
    state["wmanager"].failure_disable_threshold = 3

    async with state["db"].connect() as conn:
        wid = await repository.create_watch(
            conn,
            name="broken",
            kind="interval",
            pipeline_config=_pipeline_cfg(),
            interval_seconds=60,
        )

    async def boom(name, cfg):
        raise ValueError("DAG cycle")

    state["wmanager"].pipelines.create_and_start = boom  # type: ignore[assignment]

    # Three ticks → three failures → auto-disabled. Need to refresh
    # ``last_fired_at`` reasoning: since the watch never successfully
    # fired, the "should_fire" check always returns True for an interval
    # watch with no prior fire.
    for _ in range(3):
        await state["wmanager"]._tick()

    async with state["db"].connect() as conn:
        w = await repository.get_watch(conn, wid)
    assert w.consecutive_failures == 3
    assert w.enabled is False
    assert "DAG cycle" in (w.last_error or "")

    # Disabled watches must not be touched by subsequent ticks.
    call_count = 0

    async def counting(name, cfg):
        nonlocal call_count
        call_count += 1
        raise ValueError("still broken")

    state["wmanager"].pipelines.create_and_start = counting  # type: ignore[assignment]
    await state["wmanager"]._tick()
    assert call_count == 0


async def test_disabled_watch_not_polled(state):
    async with state["db"].connect() as conn:
        wid = await repository.create_watch(
            conn,
            name="off",
            kind="interval",
            pipeline_config=_pipeline_cfg(),
            interval_seconds=60,
        )
        await repository.set_watch_enabled(conn, wid, False)

    fired = []

    async def fake_create(name, cfg):
        fired.append(name)
        return "p"

    state["wmanager"].pipelines.create_and_start = fake_create  # type: ignore[assignment]
    await state["wmanager"]._tick()
    assert fired == []
