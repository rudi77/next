"""Tests for Phase 22: agentic data acquisition.

Covers the pure phase functions (intake, synthesize, curate), the
driver/manager state machine end-to-end (including the awaiting_input
pause/resume), and the REST surface.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from trainpipe.acquisition import runner
from trainpipe.acquisition.driver import AcquisitionDriver
from trainpipe.acquisition.manager import AcquisitionManager
from trainpipe.api.deps import get_acquisition_manager, get_db
from trainpipe.api.main import app
from trainpipe.api.schemas import AcquisitionSpec, AcquisitionStatus
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.synth.runner import MockProvider, SynthAborted

HEADERS = {"X-API-Key": "test-key"}


def _run(coro):
    return asyncio.run(coro)


def _json_provider(payload: dict) -> MockProvider:
    """A mock provider that always replies with a fixed JSON object."""
    blob = json.dumps(payload)
    return MockProvider(transform=lambda _prompt: blob)


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_extract_json_object_from_prose():
    assert runner._extract_json_object('here: {"a": 1} done') == {"a": 1}
    assert runner._extract_json_object("no json here") is None
    assert runner._extract_json_object("") is None


def test_intake_parses_json_spec():
    provider = _json_provider(
        {
            "domain": "accounting",
            "locales": ["de-DE", "de-AT"],
            "target_capabilities": ["book invoices"],
            "out_of_scope": ["legal tax advice"],
            "format": "sft",
            "open_questions": [],
        }
    )
    spec = runner.intake_spec(
        provider, model="m", brief="accountant LLM for DACH", target_count=7
    )
    assert spec.domain == "accounting"
    assert spec.locales == ["de-DE", "de-AT"]
    # Request's target_count always wins over whatever the model says.
    assert spec.target_count == 7
    assert spec.open_questions == []


def test_intake_falls_back_on_unparseable_reply():
    # Default MockProvider returns prose, not JSON → deterministic fallback.
    spec = runner.intake_spec(
        MockProvider(), model="m", brief="train a poetry bot", target_count=5
    )
    assert spec.target_count == 5
    assert spec.open_questions == []
    assert "poetry" in spec.domain


def test_intake_surfaces_open_questions():
    provider = _json_provider(
        {"domain": "x", "open_questions": ["which locale?", "which format?"]}
    )
    spec = runner.intake_spec(provider, model="m", brief="b", target_count=3)
    assert spec.open_questions == ["which locale?", "which format?"]


def test_synthesize_parses_and_counts():
    provider = _json_provider({"prompt": "p", "completion": "c"})
    spec = AcquisitionSpec(domain="d", target_count=4)
    recs = runner.synthesize_records(provider, model="m", spec=spec)
    assert len(recs) == 4
    assert all(r == {"prompt": "p", "completion": "c"} for r in recs)


def test_synthesize_falls_back_to_deterministic_records():
    spec = AcquisitionSpec(domain="d", target_count=3)
    recs = runner.synthesize_records(MockProvider(), model="m", spec=spec)
    assert len(recs) == 3
    # Fallback records are distinct per index (so curate keeps them all).
    assert len({r["prompt"] for r in recs}) == 3


def test_synthesize_aborts_on_persistent_provider_failure():
    class _Boom(MockProvider):
        def generate(self, prompt, *, model, max_tokens):
            raise RuntimeError("provider down")

    spec = AcquisitionSpec(domain="d", target_count=50)
    with pytest.raises(SynthAborted):
        runner.synthesize_records(
            _Boom(), model="m", spec=spec, max_consecutive_failures=3
        )


def test_synthesize_respects_should_stop():
    spec = AcquisitionSpec(domain="d", target_count=100)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2  # stop after a couple of records

    recs = runner.synthesize_records(
        MockProvider(), model="m", spec=spec, should_stop=stop
    )
    assert len(recs) < 100


def test_curate_dedups_exact_pairs():
    records = [
        {"prompt": "a", "completion": "1"},
        {"prompt": "a", "completion": "1"},
        {"prompt": "b", "completion": "2"},
    ]
    curated, dropped = runner.curate(records)
    assert dropped == 1
    assert curated == [
        {"prompt": "a", "completion": "1"},
        {"prompt": "b", "completion": "2"},
    ]


# ---------------------------------------------------------------------------
# Driver / manager state machine
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    instance = Database(tmp_path / "test.sqlite3")
    _run(instance.init())
    return instance


async def test_driver_completes_and_registers_dataset(db):
    run_id = await _create_run(db, target_count=5)
    driver = AcquisitionDriver(run_id, db)
    await driver._run()

    async with db.connect() as conn:
        run = await repository.get_acquisition_run(conn, run_id)
    assert run.status == AcquisitionStatus.COMPLETED
    assert run.dataset_id is not None
    assert run.raw_count == 5
    assert run.final_count == 5

    # The registered dataset exists and is non-empty.
    async with db.connect() as conn:
        ds = await repository.get_dataset(conn, run.dataset_id)
    assert ds is not None
    assert ds.line_count == 5


async def test_driver_parks_then_resumes_on_answers(db, monkeypatch):
    # Intake yields open questions → run parks in awaiting_input.
    provider = _json_provider({"domain": "x", "open_questions": ["locale?"]})
    monkeypatch.setattr(
        "trainpipe.acquisition.driver.make_provider", lambda _name: provider
    )
    run_id = await _create_run(db, target_count=3)
    await AcquisitionDriver(run_id, db)._run()

    async with db.connect() as conn:
        run = await repository.get_acquisition_run(conn, run_id)
    assert run.status == AcquisitionStatus.AWAITING_INPUT
    assert run.spec is not None and run.spec.open_questions == ["locale?"]
    assert run.dataset_id is None

    # Operator answers → manager flips to RUNNING; a fresh driver resumes
    # past intake (spec already on file) and completes.
    async with db.connect() as conn:
        await repository.update_acquisition_run(
            conn,
            run_id,
            answers={"locale?": "de-DE"},
            status=AcquisitionStatus.RUNNING,
        )
    await AcquisitionDriver(run_id, db)._run()
    async with db.connect() as conn:
        run = await repository.get_acquisition_run(conn, run_id)
    assert run.status == AcquisitionStatus.COMPLETED
    assert run.dataset_id is not None


async def test_manager_answer_rejects_non_parked_run(db):
    mgr = AcquisitionManager(db)
    run_id = await _create_run(db, target_count=2)
    # Freshly created run is 'queued', not awaiting_input.
    assert await mgr.answer(run_id, {"q": "a"}) is None


async def test_manager_cancel_marks_cancelled(db):
    mgr = AcquisitionManager(db)
    run_id = await _create_run(db, target_count=2)
    cancelled = await mgr.cancel(run_id)
    assert cancelled is not None
    assert cancelled.status == AcquisitionStatus.CANCELLED


# ---------------------------------------------------------------------------
# REST surface
# ---------------------------------------------------------------------------


class _StubManager:
    """Records calls without spawning background tasks (no event loop in
    the sync TestClient)."""

    def __init__(self, db):
        self.db = db

    async def create_and_start(self, **kwargs):
        async with self.db.connect() as conn:
            run_id = await repository.create_acquisition_run(conn, **kwargs)
            return await repository.get_acquisition_run(conn, run_id)

    async def answer(self, run_id, answers):
        return None

    async def cancel(self, run_id):
        async with self.db.connect() as conn:
            run = await repository.get_acquisition_run(conn, run_id)
            if run is None:
                return None
            await repository.update_acquisition_run(
                conn, run_id, status=AcquisitionStatus.CANCELLED
            )
            return await repository.get_acquisition_run(conn, run_id)


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    db = Database(tmp_path / "test.sqlite3")
    _run(db.init())
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_acquisition_manager] = lambda: _StubManager(db)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_create_and_get(api):
    resp = api.post(
        "/acquisitions",
        headers=HEADERS,
        json={"name": "dach-accounting", "brief": "accountant LLM for DACH"},
    )
    assert resp.status_code == 201, resp.text
    run = resp.json()
    assert run["status"] == "queued"
    assert run["name"] == "dach-accounting"

    got = api.get(f"/acquisitions/{run['id']}", headers=HEADERS)
    assert got.status_code == 200
    assert got.json()["brief"] == "accountant LLM for DACH"

    listed = api.get("/acquisitions", headers=HEADERS)
    assert listed.status_code == 200
    assert any(r["id"] == run["id"] for r in listed.json())


def test_api_requires_auth(api):
    assert api.post("/acquisitions", json={"name": "x", "brief": "y"}).status_code == 401


def test_api_get_missing_is_404(api):
    assert api.get("/acquisitions/nope", headers=HEADERS).status_code == 404


def test_api_answers_409_when_not_parked(api):
    resp = api.post(
        "/acquisitions",
        headers=HEADERS,
        json={"name": "n", "brief": "b"},
    )
    run_id = resp.json()["id"]
    patched = api.patch(
        f"/acquisitions/{run_id}/answers",
        headers=HEADERS,
        json={"answers": {"q": "a"}},
    )
    assert patched.status_code == 409


def test_api_cancel(api):
    resp = api.post("/acquisitions", headers=HEADERS, json={"name": "n", "brief": "b"})
    run_id = resp.json()["id"]
    cancelled = api.post(f"/acquisitions/{run_id}/cancel", headers=HEADERS)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_run(db, *, target_count: int) -> str:
    async with db.connect() as conn:
        return await repository.create_acquisition_run(
            conn,
            name="t",
            brief="build a tiny set",
            provider="mock",
            model="mock",
            target_count=target_count,
        )
