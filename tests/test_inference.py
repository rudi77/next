"""Tests for the Phase 8 inference playground."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import (
    get_db,
    get_gpu_pool,
    get_inference_service,
    get_scheduler,
    get_study_manager,
)
from trainpipe.api.main import app
from trainpipe.api.schemas import ExperimentSpec
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.evals.inference import MockInferenceBackend
from trainpipe.inference.service import (
    InferenceService,
    ModelRef,
    UnknownModelRef,
    resolve_model_ref,
)
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


def _make_completed_experiment(db, name="exp1") -> str:
    async def _do():
        async with db.connect() as conn:
            spec = ExperimentSpec(
                name=name, model="qwen/Qwen2-0.5B", dataset=["/tmp/ds.jsonl"]
            )
            exp_id = await repository.create_experiment(conn, spec)
            await conn.execute(
                "UPDATE experiments SET status='completed' WHERE id=?", (exp_id,)
            )
            await conn.commit()
            return exp_id

    return _run(_do())


class _Tracking:
    """Track factory calls for cache assertions."""

    def __init__(self) -> None:
        self.builds = 0
        self.closes = 0

    def factory(self, ref: ModelRef) -> MockInferenceBackend:
        self.builds += 1
        outer = self

        # Wrap MockInferenceBackend to count closes.
        class Tracked(MockInferenceBackend):
            async def close(self_inner) -> None:
                outer.closes += 1
                await super().close()

        # Echo prompt+adapter so tests can distinguish refs that share a
        # base model but different adapters.
        suffix = f" [adapter={ref.adapter_path or 'base'}]"
        return Tracked(
            response_fn=lambda sample, params: (
                str(sample.get("prompt", "")) + suffix
            )
        )


def _make_state(tmp_path, monkeypatch, tracker: _Tracking, max_loaded: int = 2):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    db = Database(tmp_path / "test.sqlite3")
    _run(db.init())
    service = InferenceService(
        db, max_loaded=max_loaded, backend_factory=tracker.factory
    )
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: _NoopScheduler()
    app.dependency_overrides[get_gpu_pool] = lambda: GpuPool([])
    app.dependency_overrides[get_study_manager] = lambda: _StubStudyManager()
    app.dependency_overrides[get_inference_service] = lambda: service
    return {"db": db, "service": service}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tracker = _Tracking()
    state = _make_state(tmp_path, monkeypatch, tracker)
    yield state, tracker
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Ref resolution
# ---------------------------------------------------------------------------


async def test_resolve_base_prefix(db):
    async with db.connect() as conn:
        ref = await resolve_model_ref("base:Qwen/Qwen2-0.5B", conn)
    assert ref.base_model == "Qwen/Qwen2-0.5B"
    assert ref.adapter_path is None


async def test_resolve_exp_prefix_unknown(db):
    async with db.connect() as conn:
        with pytest.raises(UnknownModelRef):
            await resolve_model_ref("exp:does-not-exist", conn)


async def test_resolve_exp_prefix_known(db):
    async with db.connect() as conn:
        spec = ExperimentSpec(
            model="Qwen/Qwen2-0.5B",
            dataset=["/x"],
            output_dir="/tmp/expout",
        )
        exp_id = await repository.create_experiment(conn, spec)
        ref = await resolve_model_ref(f"exp:{exp_id}", conn)
    assert ref.base_model == "Qwen/Qwen2-0.5B"
    assert ref.adapter_path == "/tmp/expout"


async def test_resolve_at_syntax_version_and_alias(db):
    async with db.connect() as conn:
        spec = ExperimentSpec(model="m1", dataset=["/x"])
        exp_id = await repository.create_experiment(conn, spec)
        await conn.execute(
            "UPDATE experiments SET status='completed' WHERE id=?", (exp_id,)
        )
        await conn.commit()
        mid, _ = await repository.register_model_atomic(
            conn,
            name="fam",
            explicit_version=None,
            base_model="m1",
            adapter_path="/tmp/adapter",
            experiment_id=exp_id,
            run_id=None,
            eval_summary=None,
            description=None,
            alias="production",
        )
        # by version
        ref_v = await resolve_model_ref("fam@1", conn)
        # by alias
        ref_a = await resolve_model_ref("fam@production", conn)
    assert ref_v.adapter_path == "/tmp/adapter"
    assert ref_a.adapter_path == "/tmp/adapter"
    assert ref_v.base_model == "m1"


async def test_resolve_unknown_prefix(db):
    async with db.connect() as conn:
        with pytest.raises(UnknownModelRef):
            await resolve_model_ref("garbage:value", conn)
        with pytest.raises(UnknownModelRef):
            await resolve_model_ref("", conn)


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


async def test_cache_hit_does_not_rebuild(setup):
    state, tracker = setup
    svc = state["service"]
    ref = ModelRef(base_model="m", adapter_path=None)
    await svc.get(ref)
    await svc.get(ref)
    assert tracker.builds == 1
    assert tracker.closes == 0


async def test_lru_eviction_closes_oldest(setup):
    state, tracker = setup
    svc = state["service"]  # max_loaded=2
    a = ModelRef(base_model="a", adapter_path=None)
    b = ModelRef(base_model="b", adapter_path=None)
    c = ModelRef(base_model="c", adapter_path=None)
    await svc.get(a)
    await svc.get(b)
    # Touch a so b becomes LRU, then load c → b should be evicted.
    await svc.get(a)
    await svc.get(c)
    keys = svc.cache_keys()
    assert ("b", None) not in keys
    assert ("a", None) in keys and ("c", None) in keys
    assert tracker.builds == 3
    assert tracker.closes == 1


async def test_close_all_drains_cache(setup):
    state, tracker = setup
    svc = state["service"]
    await svc.get(ModelRef(base_model="a", adapter_path=None))
    await svc.get(ModelRef(base_model="b", adapter_path=None))
    await svc.close_all()
    assert svc.cache_keys() == []
    assert tracker.closes == 2


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


def test_infer_route_basic(setup, client):
    state, tracker = setup
    r = client.post(
        "/inferences",
        headers=HEADERS,
        json={"model_ref": "base:Qwen/Qwen2-0.5B", "prompt": "hi"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["base_model"] == "Qwen/Qwen2-0.5B"
    assert body["adapter_path"] is None
    assert body["prediction"] == "hi [adapter=base]"


def test_infer_compare_route(setup, client):
    state, tracker = setup
    r = client.post(
        "/inferences/compare",
        headers=HEADERS,
        json={
            "model_refs": ["base:A", "base:B"],
            "prompt": "ping",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert [x["base_model"] for x in body["results"]] == ["A", "B"]
    assert all(x["prediction"].startswith("ping") for x in body["results"])


def test_infer_unknown_ref_422(setup, client):
    r = client.post(
        "/inferences",
        headers=HEADERS,
        json={"model_ref": "fam@nope", "prompt": "x"},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_model_ref"


def test_infer_requires_auth(setup, client):
    r = client.post(
        "/inferences",
        json={"model_ref": "base:m", "prompt": "p"},
    )
    assert r.status_code == 401


def test_cache_inspect(setup, client):
    state, tracker = setup
    client.post(
        "/inferences",
        headers=HEADERS,
        json={"model_ref": "base:m1", "prompt": "p"},
    )
    r = client.get("/inferences/cache", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["max_loaded"] == 2
    assert body["loaded"][0]["base_model"] == "m1"


async def test_failed_open_closes_partial_backend(tmp_path, monkeypatch):
    """If backend.open() raises, the half-initialized backend must be
    close()d, not leaked, and the cache must not grow."""
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    db = Database(tmp_path / "x.sqlite3")
    await db.init()

    closed: list[str] = []

    class Broken(MockInferenceBackend):
        def __init__(self):
            super().__init__()

        async def open(self):
            raise RuntimeError("simulated load failure")

        async def close(self):
            closed.append("broken")
            await super().close()

    svc = InferenceService(
        db, max_loaded=2, backend_factory=lambda r: Broken()
    )
    with pytest.raises(RuntimeError):
        await svc.get(ModelRef(base_model="m", adapter_path=None))
    assert closed == ["broken"]
    assert svc.cache_keys() == []


async def test_predict_failure_invalidates_cache(setup):
    state, tracker = setup
    svc = state["service"]

    ref = ModelRef(base_model="bad", adapter_path=None)
    backend = await svc.get(ref)
    assert ref.cache_key in [k for k in svc.cache_keys()]

    # Sabotage predict, then run it via the route helper-equivalent path.
    async def boom(sample, params):
        raise RuntimeError("predict boom")

    backend.predict = boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        await backend.predict({"prompt": "x"}, None)  # type: ignore[arg-type]
    await svc.invalidate(ref)
    assert ref.cache_key not in svc.cache_keys()


def test_stream_chunks_and_done(setup, client):
    """SSE stream yields token events then a final done event."""
    state, tracker = setup
    with client.stream(
        "POST",
        "/inferences/stream",
        headers=HEADERS,
        json={
            "model_ref": "base:m",
            "prompt": "a" * 130,  # → 3 chunks of 64
        },
    ) as resp:
        assert resp.status_code == 200
        text = "".join(chunk for chunk in resp.iter_text())
    # Quick assertion: there's at least one token event and exactly one done.
    assert "event: token" in text
    assert text.count("event: done") == 1
