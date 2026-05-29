"""Tests for Phase 11: active learning runner + REST endpoints."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from trainpipe.active_learning.runner import (
    DoublePassScorer,
    LengthZScoreScorer,
    _char_diff_distance,
    iter_jsonl_samples,
    run_active_learning,
)
from trainpipe.api.deps import (
    get_db,
    get_gpu_pool,
    get_inference_service,
    get_scheduler,
    get_study_manager,
)
from trainpipe.api.main import app
from trainpipe.api.schemas import InferenceParams
from trainpipe.core.db import Database
from trainpipe.evals.inference import MockInferenceBackend
from trainpipe.inference.service import InferenceService
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


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_char_diff_distance_identical_is_zero():
    assert _char_diff_distance("hello world", "hello world") == 0.0


def test_char_diff_distance_disjoint_is_one():
    assert _char_diff_distance("aaaa", "zzzz") == 1.0


def test_char_diff_distance_both_empty_is_zero():
    assert _char_diff_distance("", "") == 0.0


def test_char_diff_distance_one_empty_is_one():
    assert _char_diff_distance("", "non-empty") == 1.0


def test_iter_jsonl_skips_malformed(tmp_path):
    p = tmp_path / "noisy.jsonl"
    p.write_text(
        '{"prompt":"a"}\n'
        'not-json garbage\n'
        '{"prompt":"b"}\n',
        encoding="utf-8",
    )
    out = list(iter_jsonl_samples(p))
    assert [r["prompt"] for _, r in out] == ["a", "b"]


def test_iter_jsonl_respects_limit(tmp_path):
    p = tmp_path / "all.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(10)) + "\n",
        encoding="utf-8",
    )
    assert len(list(iter_jsonl_samples(p, limit=3))) == 3


# ---------------------------------------------------------------------------
# Scorer behavior
# ---------------------------------------------------------------------------


async def test_double_pass_scorer_calls_backend_twice():
    """DoublePassScorer must produce one ScoredSample with two backend
    calls; identical canned responses → zero uncertainty."""
    backend = MockInferenceBackend(default_response="same answer")
    await backend.open()
    scorer = DoublePassScorer()
    out = await scorer.score(
        backend, 0, {"prompt": "x"}, InferenceParams()
    )
    assert out.prediction == "same answer"
    assert out.uncertainty == 0.0
    assert len(backend.predict_calls) == 2
    await backend.close()


async def test_double_pass_scorer_high_uncertainty_on_diff(monkeypatch):
    """Two different responses should yield uncertainty > 0."""
    responses = iter(["aaa", "zzz"])

    def fn(sample, params):
        return next(responses)

    backend = MockInferenceBackend(response_fn=fn)
    await backend.open()
    out = await DoublePassScorer().score(
        backend, 0, {"prompt": "x"}, InferenceParams()
    )
    assert out.uncertainty > 0.5
    await backend.close()


async def test_length_zscore_finalize_normalizes():
    backend = MockInferenceBackend(response_fn=lambda s, p: "x" * len(s.get("prompt", "")))
    await backend.open()
    scorer = LengthZScoreScorer()
    samples = [
        await scorer.score(backend, i, {"prompt": "x" * (i + 1)}, InferenceParams())
        for i in range(5)
    ]
    normalized = await scorer.finalize(samples)
    # After z-score normalization, the values should sum to 0 only in
    # expectation; here just verify they're abs'd (non-negative) and
    # the extremes have higher score than the median.
    assert all(s.uncertainty >= 0 for s in normalized)
    assert normalized[0].uncertainty > normalized[2].uncertainty


# ---------------------------------------------------------------------------
# End-to-end: run_active_learning + repository round-trip
# ---------------------------------------------------------------------------


async def test_run_active_learning_ranks_by_uncertainty(tmp_path):
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"prompt": f"q{i}", "kind": "stable" if i < 3 else "noisy"})
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )

    # Force noisy samples to return wildly different second-pass outputs.
    call_counter = {"n": 0}

    def fn(sample, params):
        call_counter["n"] += 1
        if sample.get("kind") == "noisy":
            return f"noisy-{call_counter['n']}"
        return "stable"

    backend = MockInferenceBackend(response_fn=fn)
    await backend.open()
    result = await run_active_learning(
        backend=backend,
        dataset_path=p,
        scorer=DoublePassScorer(),
        top_n=3,
    )
    assert result.scored_count == 6
    assert result.queued_count == 3
    # The top-3 (highest uncertainty) should be the noisy ones.
    assert all(
        s.sample.get("kind") == "noisy" for s in result.top_items
    )
    await backend.close()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


class _Tracking:
    def __init__(self) -> None:
        self.builds = 0

    def factory(self, ref):
        self.builds += 1
        # Distinct second-pass outputs → uncertainty > 0 for every sample.
        counter = {"i": 0}

        def fn(sample, params):
            counter["i"] += 1
            return f"resp-{counter['i']}-{sample.get('prompt', '')}"

        return MockInferenceBackend(response_fn=fn)


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    db = Database(tmp_path / "test.sqlite3")
    _run(db.init())
    tracker = _Tracking()
    service = InferenceService(db, max_loaded=2, backend_factory=tracker.factory)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: _NoopScheduler()
    app.dependency_overrides[get_gpu_pool] = lambda: GpuPool([])
    app.dependency_overrides[get_study_manager] = lambda: _StubStudyManager()
    app.dependency_overrides[get_inference_service] = lambda: service
    yield {"db": db, "service": service, "tmp": tmp_path, "tracker": tracker}
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _write_unlabeled(tmp_path, n=5) -> str:
    p = tmp_path / "unlabeled.jsonl"
    p.write_text(
        "\n".join(json.dumps({"prompt": f"q{i}"}) for i in range(n)) + "\n",
        encoding="utf-8",
    )
    return str(p)


def test_run_end_to_end(state, client):
    ds_path = _write_unlabeled(state["tmp"], n=5)
    r = client.post(
        "/active-learning/runs",
        headers=HEADERS,
        json={
            "model_ref": "base:Qwen/Qwen2-0.5B",
            "dataset": ds_path,
            "top_n": 3,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["scored_count"] == 5
    assert body["queued_count"] == 3

    queue = client.get(
        f"/active-learning/runs/{body['id']}/queue", headers=HEADERS
    )
    items = queue.json()
    assert len(items) == 3
    # Sorted by uncertainty descending.
    assert items[0]["uncertainty"] >= items[-1]["uncertainty"]


def test_run_422_on_bad_model_ref(state, client):
    ds_path = _write_unlabeled(state["tmp"])
    r = client.post(
        "/active-learning/runs",
        headers=HEADERS,
        json={
            "model_ref": "garbage:no",
            "dataset": ds_path,
            "top_n": 5,
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unknown_model_ref"


def test_run_422_on_missing_dataset_file(state, client):
    r = client.post(
        "/active-learning/runs",
        headers=HEADERS,
        json={
            "model_ref": "base:m",
            "dataset": "/does/not/exist.jsonl",
            "top_n": 5,
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "dataset_file_missing"


def test_mark_annotated(state, client):
    ds_path = _write_unlabeled(state["tmp"], n=3)
    r = client.post(
        "/active-learning/runs",
        headers=HEADERS,
        json={"model_ref": "base:m", "dataset": ds_path, "top_n": 3},
    )
    run_id = r.json()["id"]
    items = client.get(
        f"/active-learning/runs/{run_id}/queue", headers=HEADERS
    ).json()
    first = items[0]
    rm = client.post(
        f"/active-learning/runs/{run_id}/queue/{first['id']}/annotated",
        headers=HEADERS,
    )
    assert rm.json()["updated"]

    # only_unannotated filter respects the flag.
    remaining = client.get(
        f"/active-learning/runs/{run_id}/queue",
        params={"only_unannotated": "true"},
        headers=HEADERS,
    ).json()
    assert all(it["id"] != first["id"] for it in remaining)


def test_routes_require_auth(state, client):
    r = client.get("/active-learning/runs")
    assert r.status_code == 401


def test_run_falls_back_to_failed_on_scorer_crash(state, client, monkeypatch):
    """If the inference backend raises during ``get``, the run should
    finalize as FAILED rather than 500-ing the route."""
    ds_path = _write_unlabeled(state["tmp"])

    async def broken_get(ref):
        raise RuntimeError("simulated load failure")

    state["service"].get = broken_get  # type: ignore[assignment]

    r = client.post(
        "/active-learning/runs",
        headers=HEADERS,
        json={"model_ref": "base:m", "dataset": ds_path, "top_n": 3},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "failed"
    assert "simulated load failure" in (body["error"] or "")
