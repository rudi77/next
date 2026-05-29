"""Tests for the Label Studio bridge (Phase 10)."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import get_db, get_gpu_pool, get_scheduler, get_study_manager
from trainpipe.api.main import app
from trainpipe.core.db import Database
from trainpipe.integrations import labelstudio as ls
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


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr("trainpipe.settings.settings.api_key", "test-key")
    monkeypatch.setattr("trainpipe.settings.settings.data_dir", tmp_path)
    db = Database(tmp_path / "test.sqlite3")
    _run(db.init())
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_scheduler] = lambda: _NoopScheduler()
    app.dependency_overrides[get_gpu_pool] = lambda: GpuPool([])
    app.dependency_overrides[get_study_manager] = lambda: _StubStudyManager()
    yield {"db": db, "tmp": tmp_path}
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------


def test_detect_conversation_default():
    tasks = [{"data": {"text": "hi"}, "annotations": [{"result": []}]}]
    assert ls.detect_import_kind(tasks) == "conversation"


def test_detect_doc_layout():
    tasks = [
        {
            "data": {"image": "img.png"},
            "annotations": [
                {"result": [{"type": "rectanglelabels", "value": {}}]}
            ],
        }
    ]
    assert ls.detect_import_kind(tasks) == "doc_layout"


def test_detect_text_ner():
    tasks = [
        {
            "data": {"text": "Acme Corp lives in Vienna."},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "labels",
                            "value": {"start": 0, "end": 4, "labels": ["ORG"]},
                        }
                    ]
                }
            ],
        }
    ]
    assert ls.detect_import_kind(tasks) == "text_ner"


def test_map_conversation_textarea_response():
    tasks = [
        {
            "data": {"prompt": "Translate hello"},
            "annotations": [
                {
                    "result": [
                        {"value": {"text": ["Hallo"]}}
                    ]
                }
            ],
        }
    ]
    out = ls.map_tasks_to_jsonl(tasks, "conversation")
    assert out == [
        {
            "messages": [
                {"role": "user", "content": "Translate hello"},
                {"role": "assistant", "content": "Hallo"},
            ]
        }
    ]


def test_map_conversation_skips_empty_response():
    tasks = [
        {"data": {"prompt": "x"}, "annotations": [{"result": []}]}
    ]
    assert ls.map_tasks_to_jsonl(tasks, "conversation") == []


def test_map_conversation_skips_cancelled():
    tasks = [
        {
            "data": {"prompt": "x"},
            "annotations": [
                {"was_cancelled": True, "result": [{"value": {"text": "y"}}]}
            ],
        }
    ]
    assert ls.map_tasks_to_jsonl(tasks, "conversation") == []


def test_map_text_ner_extracts_spans():
    tasks = [
        {
            "data": {"text": "Acme is in Vienna."},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "labels",
                            "value": {
                                "start": 0, "end": 4, "labels": ["ORG"]
                            },
                        },
                        {
                            "type": "labels",
                            "value": {
                                "start": 11, "end": 17, "labels": ["LOC"]
                            },
                        },
                    ]
                }
            ],
        }
    ]
    out = ls.map_tasks_to_jsonl(tasks, "text_ner")
    assert out[0]["text"] == "Acme is in Vienna."
    assert len(out[0]["entities"]) == 2
    assert out[0]["entities"][0] == {"start": 0, "end": 4, "label": "ORG"}


def test_map_doc_layout_converts_percent_to_pixels():
    tasks = [
        {
            "data": {"image": "doc-001.png"},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "original_width": 1000,
                            "original_height": 500,
                            "value": {
                                "x": 10, "y": 20, "width": 30, "height": 40,
                                "rectanglelabels": ["header"],
                            },
                        }
                    ]
                }
            ],
        }
    ]
    out = ls.map_tasks_to_jsonl(tasks, "doc_layout")
    rec = out[0]
    assert rec["images"] == ["doc-001.png"]
    box = rec["gold_boxes"][0]
    assert box["label"] == "header"
    # (x, y) -> (100, 100); width 30% of 1000 = 300; height 40% of 500 = 200
    assert box["box"] == [100.0, 100.0, 400.0, 300.0]


def test_map_doc_layout_skips_task_without_image():
    tasks = [
        {
            "data": {},
            "annotations": [
                {"result": [{"type": "rectanglelabels", "value": {}}]}
            ],
        }
    ]
    # Bad task is logged + skipped, not raised.
    assert ls.map_tasks_to_jsonl(tasks, "doc_layout") == []


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_fetch_paginates_until_short_page():
    calls = []

    def transport(method, path, **kwargs):
        calls.append((method, path, kwargs))
        page = kwargs.get("params", {}).get("page", 1)
        page_size = kwargs.get("params", {}).get("page_size")
        if page == 1:
            return [{"id": i, "annotations": []} for i in range(page_size)]
        if page == 2:
            return [{"id": 999, "annotations": []}]  # short page
        return []

    tasks = ls.fetch_completed_tasks(transport, 42, page_size=200)
    assert len(tasks) == 201
    assert len(calls) == 2  # stops after the short page


def test_fetch_respects_max_tasks():
    def transport(method, path, **kwargs):
        ps = kwargs.get("params", {}).get("page_size")
        return [{"id": i} for i in range(ps)]

    tasks = ls.fetch_completed_tasks(transport, 1, page_size=100, max_tasks=10)
    assert len(tasks) == 10


def test_fetch_passes_since_param():
    seen = []

    def transport(method, path, **kwargs):
        seen.append(kwargs["params"])
        return []

    ls.fetch_completed_tasks(transport, 1, since_iso="2026-05-01T00:00:00Z")
    assert seen[0]["completed_at__gte"] == "2026-05-01T00:00:00Z"


# ---------------------------------------------------------------------------
# REST integration via injected transport
# ---------------------------------------------------------------------------


def test_route_imports_and_registers_dataset(state, client, monkeypatch):
    """End-to-end: POST /datasets/from-labelstudio creates a dataset whose
    file contains the mapped JSONL."""
    fake_tasks = [
        {
            "data": {"prompt": f"q{i}"},
            "annotations": [
                {"result": [{"value": {"text": f"a{i}"}}]}
            ],
        }
        for i in range(3)
    ]

    def fake_import_project(**kwargs):
        return "conversation", [
            r
            for r in ls.map_tasks_to_jsonl(fake_tasks, "conversation")
        ]

    monkeypatch.setattr(
        "trainpipe.integrations.labelstudio.import_project",
        fake_import_project,
    )

    r = client.post(
        "/datasets/from-labelstudio",
        headers=HEADERS,
        json={
            "base_url": "http://ls.local",
            "token": "fake-token",
            "project_id": 42,
            "name": "ls-import-test",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["line_count"] == 3
    assert body["format"] == "jsonl"
    assert "Label Studio project 42" in (body["description"] or "")

    # File on disk has 3 JSONL lines with the mapped messages.
    with open(body["path"], encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 3
    assert lines[0]["messages"][0]["content"] == "q0"


def test_route_422_on_empty_import(state, client, monkeypatch):
    monkeypatch.setattr(
        "trainpipe.integrations.labelstudio.import_project",
        lambda **kw: ("conversation", []),
    )
    r = client.post(
        "/datasets/from-labelstudio",
        headers=HEADERS,
        json={
            "base_url": "http://ls.local",
            "token": "t",
            "project_id": 1,
            "name": "empty",
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "no_records"


def test_route_502_on_ls_error(state, client, monkeypatch):
    def boom(**kw):
        raise ls.LabelStudioError("ls down")

    monkeypatch.setattr(
        "trainpipe.integrations.labelstudio.import_project", boom
    )
    r = client.post(
        "/datasets/from-labelstudio",
        headers=HEADERS,
        json={
            "base_url": "http://x",
            "token": "t",
            "project_id": 1,
            "name": "x",
        },
    )
    assert r.status_code == 502
    assert r.json()["detail"]["error"] == "labelstudio_error"


def test_route_dedupe_by_sha(state, client, monkeypatch):
    """Re-importing the same content returns the existing dataset record."""
    monkeypatch.setattr(
        "trainpipe.integrations.labelstudio.import_project",
        lambda **kw: (
            "conversation",
            [
                {
                    "messages": [
                        {"role": "user", "content": "x"},
                        {"role": "assistant", "content": "y"},
                    ]
                }
            ],
        ),
    )
    payload = {
        "base_url": "http://x",
        "token": "t",
        "project_id": 1,
        "name": "imp",
    }
    r1 = client.post("/datasets/from-labelstudio", headers=HEADERS, json=payload)
    r2 = client.post("/datasets/from-labelstudio", headers=HEADERS, json=payload)
    assert r1.status_code == 201
    # Dedup: second returns the existing record (HTTP 201 still since the
    # route is declared status_code=201, but the body is the same).
    assert r2.json()["id"] == r1.json()["id"]


def test_ssrf_blocks_localhost():
    with pytest.raises(ls.LabelStudioError, match="blocked address"):
        ls._validate_base_url("http://127.0.0.1:8080")


def test_ssrf_blocks_imds():
    with pytest.raises(ls.LabelStudioError):
        ls._validate_base_url("http://metadata.google.internal")
    # AWS IMDS by IP
    with pytest.raises(ls.LabelStudioError, match="blocked address"):
        ls._validate_base_url("http://169.254.169.254/latest/meta-data/")


def test_ssrf_blocks_private_ip():
    with pytest.raises(ls.LabelStudioError, match="blocked address"):
        ls._validate_base_url("http://10.0.0.5/")


def test_ssrf_rejects_non_http_scheme():
    with pytest.raises(ls.LabelStudioError, match="unsupported scheme"):
        ls._validate_base_url("file:///etc/passwd")
    with pytest.raises(ls.LabelStudioError, match="unsupported scheme"):
        ls._validate_base_url("gopher://ls.example.com/")


def test_validate_strips_credentials():
    # We can't easily call it without a public DNS lookup, so just test
    # the post-canonicalization step via strip_url_credentials directly.
    cleaned = ls.strip_url_credentials("https://user:s3cret@ls.example.com:8443/v1")
    assert "s3cret" not in cleaned
    assert "user" not in cleaned
    assert cleaned == "https://ls.example.com:8443/v1"


def test_token_redacted_from_error_body():
    """If LS bounces the Authorization header into a body, we must redact
    before surfacing it to the API caller."""
    text = '{"detail":"unauthorized; got Token leaky-token-abc"}'
    sanitized = ls._sanitize_error_body(text, "leaky-token-abc")
    assert "leaky-token-abc" not in sanitized
    assert "<redacted>" in sanitized


def test_route_provenance_strips_credentials(state, client, monkeypatch):
    """A base_url with userinfo must not land in the dataset description."""
    monkeypatch.setattr(
        "trainpipe.integrations.labelstudio.import_project",
        lambda **kw: (
            "conversation",
            [
                {
                    "messages": [
                        {"role": "user", "content": "x"},
                        {"role": "assistant", "content": "y"},
                    ]
                }
            ],
        ),
    )
    r = client.post(
        "/datasets/from-labelstudio",
        headers=HEADERS,
        json={
            "base_url": "https://user:s3cret@ls.example.com",
            "token": "t",
            "project_id": 1,
            "name": "creds",
        },
    )
    assert r.status_code == 201
    assert "s3cret" not in r.json()["description"]


def test_route_requires_auth(state, client):
    r = client.post(
        "/datasets/from-labelstudio",
        json={"base_url": "x", "token": "t", "project_id": 1, "name": "n"},
    )
    assert r.status_code == 401
