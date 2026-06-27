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
    curated, stats = runner.curate(records)
    assert stats.dropped == 1
    assert stats.redaction == {}  # no PII in these records
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


# ---------------------------------------------------------------------------
# Stage 3 — web research / acquisition
# ---------------------------------------------------------------------------

from trainpipe.acquisition import web  # noqa: E402
from trainpipe.acquisition.runner import (  # noqa: E402
    SourceEval,
    acquire_records,
    research_sources,
)


def test_simple_extractor_strips_scripts_and_tags():
    html = "<html><body><script>var x=1;</script><h1>Titel</h1><p>Ein  Satz.</p></body></html>"
    assert web.SimpleExtractor().extract(html, "u") == "Titel Ein Satz."


# Permissive injectables so the gate runs without network (no DNS, no robots).
def _open_gate(**kw):
    return web.make_fetch_gate(
        robots_fetcher=kw.pop("robots_fetcher", lambda _u: None),
        url_safety=lambda _u: True,
        **kw,
    )


def test_fetch_gate_blocks_on_robots():
    # robots.txt that disallows everything for our agent. license_status still
    # carries the real license verdict (not a robots sentinel).
    gate = _open_gate(robots_fetcher=lambda _u: "User-agent: *\nDisallow: /")
    d = gate("https://de.wikipedia.org/wiki/X")
    assert d.allowed is False
    assert d.license_status == "open"


def test_fetch_gate_blocks_unsafe_url():
    gate = web.make_fetch_gate(
        robots_fetcher=lambda _u: None, url_safety=lambda _u: False
    )
    assert gate("https://example.com/x").allowed is False


def test_fetch_gate_license_open_vs_unknown_and_strict():
    allow = _open_gate()
    assert allow("https://de.wikipedia.org/wiki/X").license_status == "open"
    unknown = allow("https://example.com/x")
    assert unknown.license_status == "unknown" and unknown.allowed is True
    strict = _open_gate(strict_license=True)
    assert strict("https://example.com/x").allowed is False


def test_license_status_anchored_to_host_suffix():
    # A substring match would wrongly call this "open"; the anchored check
    # must classify it "unknown".
    assert web._license_status("https://evil-wikipedia.org.attacker.net/x") == "unknown"
    assert web._license_status("https://de.wikipedia.org/x") == "open"


def test_mock_search_provider_caps_results():
    hits = [web.SearchHit(url=f"https://h{i}.test") for i in range(10)]
    got = web.MockSearchProvider(hits).search("q", max_results=3)
    assert [h.url for h in got] == ["https://h0.test", "https://h1.test", "https://h2.test"]


def test_research_sources_dedups_and_caps():
    spec = AcquisitionSpec(domain="accounting", target_capabilities=["a", "b"], target_count=1)
    # Same URL returned for both capability queries → deduped to one source.
    provider = web.MockSearchProvider(
        [web.SearchHit(url="https://x.test"), web.SearchHit(url="https://y.test")]
    )
    gate = _open_gate()
    sources = research_sources(provider, gate, spec, max_sources=5)
    urls = [s.url for s in sources]
    assert urls == ["https://x.test", "https://y.test"]  # deduped across queries
    assert all(s.allowed for s in sources)


def test_acquire_records_only_uses_allowed_and_reports_used():
    spec = AcquisitionSpec(domain="d", target_count=1)
    sources = [
        SourceEval("https://ok.test", "ok", "t", "open", allowed=True),
        SourceEval("https://blocked.test", "no", "t", "blocked_robots", allowed=False),
        SourceEval("https://empty.test", "e", "t", "unknown", allowed=True),
    ]

    def fetch_text(url):
        if url == "https://ok.test":
            return "some page text about the domain"
        return None  # empty.test fetches nothing → skipped

    records = acquire_records(
        MockProvider(),
        model="m",
        sources=sources,
        spec=spec,
        fetch_text=fetch_text,
        records_per_source=2,
    )
    # blocked never fetched, empty skipped → only ok.test marked used.
    assert [s.url for s in sources if s.used] == ["https://ok.test"]
    assert len(records) == 2


async def test_driver_web_path_records_sources_and_real_records(db, monkeypatch):
    hits = [web.SearchHit(url=f"https://src{i}.test", title=f"s{i}") for i in range(3)]
    monkeypatch.setattr(
        "trainpipe.acquisition.driver.make_search_provider",
        lambda _name: web.MockSearchProvider(hits),
    )
    monkeypatch.setattr(
        "trainpipe.acquisition.driver.make_fetch_gate",
        lambda **_k: web.make_fetch_gate(
            robots_fetcher=lambda _u: None, url_safety=lambda _u: True
        ),
    )
    monkeypatch.setattr(
        "trainpipe.acquisition.driver.make_text_fetcher",
        lambda _ext: (lambda _url: "page text about the domain"),
    )

    async with db.connect() as conn:
        run_id = await repository.create_acquisition_run(
            conn,
            name="web",
            brief="accounting helper",
            provider="mock",
            model="mock",
            target_count=4,
            search_provider="mock",
            max_sources=3,
        )
    await AcquisitionDriver(run_id, db)._run()

    async with db.connect() as conn:
        run = await repository.get_acquisition_run(conn, run_id)
        sources = await repository.list_acquisition_sources(conn, run_id)
    assert run.status == AcquisitionStatus.COMPLETED
    assert run.dataset_id is not None
    # 3 sources recorded, all allowed+fetched → used.
    assert len(sources) == 3
    assert all(s.used for s in sources)
    # raw_count includes real records on top of the 4 synthesized ones.
    assert run.raw_count > 4


# ---------------------------------------------------------------------------
# Stage 4 — hardening (mandatory redaction, cost budget, strict license)
# ---------------------------------------------------------------------------


def test_curate_redacts_pii():
    records = [
        {"prompt": "mail me at john@example.com", "completion": "ok"},
        {"prompt": "clean", "completion": "also clean"},
    ]
    curated, stats = runner.curate(records)
    assert stats.dropped == 0
    assert stats.redaction.get("email") == 1
    assert "john@example.com" not in curated[0]["prompt"]
    assert "[REDACTED_EMAIL]" in curated[0]["prompt"]


def test_curate_redacts_nested_pii():
    # Chat-format records nest PII inside a list of message dicts; the shared
    # recursive walker must reach it (a top-level-only walk would miss it).
    records = [{"messages": [{"role": "user", "content": "ping a@b.com"}]}]
    curated, stats = runner.curate(records)
    assert stats.redaction.get("email") == 1
    assert "a@b.com" not in curated[0]["messages"][0]["content"]


async def test_driver_persists_redaction_counts(db, monkeypatch):
    # Force the synthesizer to emit a record containing an email.
    provider = _json_provider({"prompt": "reach me: a@b.com", "completion": "fine"})
    monkeypatch.setattr(
        "trainpipe.acquisition.driver.make_provider", lambda _name: provider
    )
    run_id = await _create_run(db, target_count=3)
    await AcquisitionDriver(run_id, db)._run()
    async with db.connect() as conn:
        run = await repository.get_acquisition_run(conn, run_id)
    assert run.status == AcquisitionStatus.COMPLETED
    assert run.redaction and run.redaction.get("email", 0) >= 1


async def test_cost_budget_caps_llm_calls(db, monkeypatch):
    # Count how many times the provider is asked to generate.
    calls = {"n": 0}

    class _Counting(MockProvider):
        def generate(self, prompt, *, model, max_tokens):
            calls["n"] += 1
            return super().generate(prompt, model=model, max_tokens=max_tokens)

    monkeypatch.setattr(
        "trainpipe.acquisition.driver.make_provider", lambda _name: _Counting()
    )
    async with db.connect() as conn:
        run_id = await repository.create_acquisition_run(
            conn,
            name="budget",
            brief="x",
            provider="mock",
            model="mock",
            target_count=100,
            max_llm_calls=5,
        )
    await AcquisitionDriver(run_id, db)._run()
    # Synthesize stops once the budget is hit; intake (1 call) + at most the
    # budget for synthesis.
    assert calls["n"] <= 6
    async with db.connect() as conn:
        run = await repository.get_acquisition_run(conn, run_id)
    assert run.raw_count <= 5


def test_strict_license_skips_unknown_sources():
    spec = AcquisitionSpec(domain="d", target_capabilities=["x"], target_count=1)
    provider = web.MockSearchProvider([web.SearchHit(url="https://example.com/a")])
    gate = web.make_fetch_gate(
        strict_license=True, robots_fetcher=lambda _u: None, url_safety=lambda _u: True
    )
    sources = research_sources(provider, gate, spec, max_sources=5)
    assert len(sources) == 1
    assert sources[0].allowed is False  # unknown license, strict mode → blocked
