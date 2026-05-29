"""Tests for Phase 12: multi-stage pipelines."""

import asyncio

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from trainpipe.api.deps import (
    get_db,
    get_gpu_pool,
    get_inference_service,
    get_pipeline_manager,
    get_scheduler,
    get_study_manager,
)
from trainpipe.api.main import app
from trainpipe.api.schemas import (
    ExperimentSpec,
    PipelineConfig,
    PipelineStatus,
    StageSpec,
    StageStatus,
)
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.pipelines.manager import PipelineManager, _validate_dag
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


class _StubInferenceService:
    pass


@pytest_asyncio.fixture
async def state(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    monkeypatch.setattr("trainpipe.settings.settings.poll_interval_sec", 0.01)
    db = Database(tmp_path / "test.sqlite3")
    await db.init()
    manager = PipelineManager(db)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: _NoopScheduler()
    app.dependency_overrides[get_gpu_pool] = lambda: GpuPool([])
    app.dependency_overrides[get_study_manager] = lambda: _StubStudyManager()
    app.dependency_overrides[get_inference_service] = lambda: _StubInferenceService()
    app.dependency_overrides[get_pipeline_manager] = lambda: manager
    try:
        yield {"db": db, "manager": manager, "tmp": tmp_path}
    finally:
        try:
            await manager.stop_all()
        except Exception:
            pass
        app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _simple_stage(name: str, depends_on=None, input_from=None) -> StageSpec:
    return StageSpec(
        name=name,
        base_spec=ExperimentSpec(
            model="qwen/Qwen2-0.5B", dataset=["/tmp/ds.jsonl"]
        ),
        depends_on=depends_on or [],
        input_from_stage=input_from,
    )


# ---------------------------------------------------------------------------
# DAG validation
# ---------------------------------------------------------------------------


def test_duplicate_stage_name_rejected():
    cfg = PipelineConfig(
        name="p",
        stages=[_simple_stage("a"), _simple_stage("a")],
    )
    with pytest.raises(ValueError, match="duplicate"):
        _validate_dag(cfg)


def test_dangling_depends_on_rejected():
    cfg = PipelineConfig(
        name="p",
        stages=[_simple_stage("a", depends_on=["ghost"])],
    )
    with pytest.raises(ValueError, match="unknown stage"):
        _validate_dag(cfg)


def test_cycle_rejected():
    cfg = PipelineConfig(
        name="p",
        stages=[
            _simple_stage("a", depends_on=["b"]),
            _simple_stage("b", depends_on=["a"]),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        _validate_dag(cfg)


def test_input_from_stage_dangling_rejected():
    cfg = PipelineConfig(
        name="p",
        stages=[
            _simple_stage("a", input_from="ghost"),
        ],
    )
    with pytest.raises(ValueError, match="input_from_stage"):
        _validate_dag(cfg)


def test_valid_linear_dag_accepted():
    cfg = PipelineConfig(
        name="p",
        stages=[
            _simple_stage("a"),
            _simple_stage("b", depends_on=["a"]),
            _simple_stage("c", depends_on=["b"], input_from="b"),
        ],
    )
    _validate_dag(cfg)  # no raise


async def test_atomic_enqueue_stage_with_experiment(db):
    """``enqueue_stage_with_experiment`` must persist the experiment
    *and* flip the stage to QUEUED with the new experiment_id in one
    transaction — no orphan rows."""
    cfg = PipelineConfig(
        name="atomic",
        stages=[
            StageSpec(
                name="solo",
                base_spec=ExperimentSpec(model="m", dataset=["/x"]),
            )
        ],
    )
    async with db.connect() as conn:
        pipeline_id = await repository.create_pipeline(
            conn, name=cfg.name, config=cfg
        )
        exp_id = await repository.enqueue_stage_with_experiment(
            conn,
            pipeline_id=pipeline_id,
            stage_name="solo",
            spec=cfg.stages[0].base_spec,
            output_dir="/tmp/out",
        )
        pipeline = await repository.get_pipeline(conn, pipeline_id)
        exp = await repository.get_experiment(conn, exp_id)
    stage = next(s for s in pipeline.stages if s.stage_name == "solo")
    assert stage.experiment_id == exp_id
    assert stage.status == StageStatus.QUEUED
    assert stage.output_dir == "/tmp/out"
    assert exp is not None
    assert exp.status.value == "queued"


async def test_atomic_enqueue_rolls_back_on_failure(db):
    """If anything inside the transaction raises after the INSERT but
    before COMMIT, the experiment row must be rolled back too — no
    orphan rows."""
    spec = ExperimentSpec(model="m", dataset=["/x"])
    async with db.connect() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM experiments")
        before = (await cur.fetchone())[0]

        # Wrap conn.execute so it raises on the pipeline_stages UPDATE.
        original_execute = conn.execute

        async def failing_execute(sql, *args, **kwargs):
            if sql.lstrip().upper().startswith("UPDATE PIPELINE_STAGES"):
                raise RuntimeError("simulated stage UPDATE failure")
            return await original_execute(sql, *args, **kwargs)

        conn.execute = failing_execute  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="simulated stage"):
                await repository.enqueue_stage_with_experiment(
                    conn,
                    pipeline_id="nope",
                    stage_name="nope",
                    spec=spec,
                    output_dir=None,
                )
        finally:
            conn.execute = original_execute  # type: ignore[method-assign]

        cur = await conn.execute("SELECT COUNT(*) FROM experiments")
        after = (await cur.fetchone())[0]
    assert after == before, "experiment row leaked after rolled-back transaction"


# ---------------------------------------------------------------------------
# REST + driver smoke
# ---------------------------------------------------------------------------


async def test_create_pipeline_stores_record(state, client):
    cfg = PipelineConfig(
        name="cpt_sft_dpo",
        stages=[_simple_stage("cpt"), _simple_stage("sft", depends_on=["cpt"])],
    )
    r = client.post(
        "/pipelines",
        headers=HEADERS,
        json=cfg.model_dump(),
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "cpt_sft_dpo"
    assert len(body["stages"]) == 2
    assert body["stages"][0]["stage_name"] == "cpt"


async def test_create_pipeline_422_on_bad_dag(state, client):
    cfg = PipelineConfig(
        name="bad",
        stages=[
            _simple_stage("a", depends_on=["b"]),
            _simple_stage("b", depends_on=["a"]),
        ],
    )
    r = client.post(
        "/pipelines", headers=HEADERS, json=cfg.model_dump()
    )
    assert r.status_code == 422


async def test_pipeline_driver_advances_via_experiment_completion(state):
    """End-to-end: driver enqueues stage A, we complete the experiment
    out-of-band, driver should pick up the completion and enqueue B."""
    cfg = PipelineConfig(
        name="seq",
        stages=[
            _simple_stage("a"),
            _simple_stage("b", depends_on=["a"]),
        ],
    )
    pipeline_id = await state["manager"].create_and_start(cfg.name, cfg)

    # Wait until stage 'a' is enqueued.
    for _ in range(200):
        async with state["db"].connect() as conn:
            p = await repository.get_pipeline(conn, pipeline_id)
            stage_a = next(s for s in p.stages if s.stage_name == "a")
            if stage_a.experiment_id is not None:
                break
        await asyncio.sleep(0.02)
    assert stage_a.experiment_id is not None

    # Force complete A, then wait for B to be enqueued.
    async with state["db"].connect() as conn:
        await conn.execute(
            "UPDATE experiments SET status='completed' WHERE id=?",
            (stage_a.experiment_id,),
        )
        await conn.commit()
    for _ in range(200):
        async with state["db"].connect() as conn:
            p2 = await repository.get_pipeline(conn, pipeline_id)
            stage_b = next(s for s in p2.stages if s.stage_name == "b")
            if stage_b.experiment_id is not None:
                break
        await asyncio.sleep(0.02)
    assert stage_b.experiment_id is not None


async def test_pipeline_failure_skips_downstream(state):
    cfg = PipelineConfig(
        name="fail",
        stages=[
            _simple_stage("a"),
            _simple_stage("b", depends_on=["a"]),
        ],
    )
    pipeline_id = await state["manager"].create_and_start(cfg.name, cfg)

    for _ in range(200):
        async with state["db"].connect() as conn:
            p = await repository.get_pipeline(conn, pipeline_id)
            stage_a = next(s for s in p.stages if s.stage_name == "a")
            if stage_a.experiment_id is not None:
                await conn.execute(
                    "UPDATE experiments SET status='failed', "
                    "error='boom' WHERE id=?",
                    (stage_a.experiment_id,),
                )
                await conn.commit()
                break
        await asyncio.sleep(0.02)

    p_final = None
    for _ in range(200):
        async with state["db"].connect() as conn:
            p2 = await repository.get_pipeline(conn, pipeline_id)
        if p2.status in (PipelineStatus.FAILED, PipelineStatus.COMPLETED):
            p_final = p2
            break
        await asyncio.sleep(0.02)
    assert p_final is not None
    assert p_final.status == PipelineStatus.FAILED
    stage_b = next(s for s in p_final.stages if s.stage_name == "b")
    assert stage_b.status == StageStatus.SKIPPED


async def test_input_from_stage_rewrites_model(state):
    cfg = PipelineConfig(
        name="chain",
        stages=[
            _simple_stage("a"),
            _simple_stage("b", depends_on=["a"], input_from="a"),
        ],
    )
    pipeline_id = await state["manager"].create_and_start(cfg.name, cfg)
    a_out: str | None = None
    for _ in range(200):
        async with state["db"].connect() as conn:
            p = await repository.get_pipeline(conn, pipeline_id)
            a = next(s for s in p.stages if s.stage_name == "a")
            if a.experiment_id is not None:
                await conn.execute(
                    "UPDATE experiments SET status='completed' WHERE id=?",
                    (a.experiment_id,),
                )
                await conn.commit()
                a_out = a.output_dir
                break
        await asyncio.sleep(0.02)
    assert a_out is not None

    for _ in range(200):
        async with state["db"].connect() as conn:
            p2 = await repository.get_pipeline(conn, pipeline_id)
            b = next(s for s in p2.stages if s.stage_name == "b")
            if b.experiment_id:
                exp_b = await repository.get_experiment(conn, b.experiment_id)
                assert exp_b.spec.model == a_out
                return
        await asyncio.sleep(0.02)
    raise AssertionError("stage b never enqueued")


async def test_pipeline_cancel(state, client):
    cfg = PipelineConfig(name="cancel-me", stages=[_simple_stage("a")])
    pipeline_id = await state["manager"].create_and_start(cfg.name, cfg)
    r = client.post(f"/pipelines/{pipeline_id}/cancel", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
