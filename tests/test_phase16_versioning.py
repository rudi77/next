"""Tests for Phase 16: dataset versioning + split + mix + @vN refs."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import get_db, get_gpu_pool, get_scheduler, get_study_manager
from trainpipe.api.main import app
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.scheduler.gpu_pool import GpuPool
from trainpipe.training.dataset_refs import (
    MalformedDatasetRef,
    parse_ref,
    parse_ref_with_version,
    resolve_single,
)

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


def _upload_jsonl(client, lines):
    content = "\n".join(json.dumps({"i": i, "txt": l}) for i, l in enumerate(lines)) + "\n"
    r = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": ("ds.jsonl", content.encode("utf-8"), "application/x-ndjson")},
        data={"name": "src"},
    )
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# Ref syntax: ds:<id>@vN#K
# ---------------------------------------------------------------------------


def test_parse_ref_with_version_basic():
    assert parse_ref_with_version("ds:abc123") == ("abc123", None, "")
    assert parse_ref_with_version("ds:abc123#500") == ("abc123", None, "#500")
    assert parse_ref_with_version("ds:abc123@v2") == ("abc123", 2, "")
    assert parse_ref_with_version("ds:abc123@v2#500") == ("abc123", 2, "#500")


def test_parse_ref_with_version_non_ref():
    assert parse_ref_with_version("/some/path.jsonl") is None
    assert parse_ref_with_version("hf/dataset") is None


def test_parse_ref_back_compat():
    """The old parse_ref must still work — Phase 6/etc. callers
    depend on it."""
    assert parse_ref("ds:deadbeef") == ("deadbeef", "")
    assert parse_ref("ds:deadbeef@v2#100") == ("deadbeef", "#100")


# ---------------------------------------------------------------------------
# Version mismatch in resolution
# ---------------------------------------------------------------------------


async def test_resolve_version_match(db):
    async with db.connect() as conn:
        ds_id = await repository.create_dataset(
            conn,
            name="x",
            path="/tmp/x.jsonl",
            fmt="jsonl",
            size_bytes=1,
            sha256="a" * 64,
            line_count=1,
            version=3,
        )
        path = await resolve_single(f"ds:{ds_id}@v3", conn)
    assert path == "/tmp/x.jsonl"


async def test_resolve_version_mismatch_raises(db):
    async with db.connect() as conn:
        ds_id = await repository.create_dataset(
            conn,
            name="x",
            path="/tmp/x.jsonl",
            fmt="jsonl",
            size_bytes=1,
            sha256="a" * 64,
            line_count=1,
            version=1,
        )
        with pytest.raises(MalformedDatasetRef, match="registered version"):
            await resolve_single(f"ds:{ds_id}@v99", conn)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def test_split_90_10(state, client):
    src_id = _upload_jsonl(client, [f"row {i}" for i in range(10)])
    r = client.post(
        f"/datasets/{src_id}/split",
        headers=HEADERS,
        json={"ratio": "90:10", "seed": 42},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    train = body["train"]
    val = body["val"]
    assert train["line_count"] == 9
    assert val["line_count"] == 1
    # Both derived from the source.
    assert train["derived_from"] == src_id
    assert val["derived_from"] == src_id
    # Versioned one above the source.
    assert train["version"] == 2
    assert val["version"] == 2


def test_split_deterministic_with_same_seed(state, client):
    src_id = _upload_jsonl(client, [f"row {i}" for i in range(20)])
    r1 = client.post(
        f"/datasets/{src_id}/split",
        headers=HEADERS,
        json={"ratio": "80:20", "seed": 7, "train_name": "t1", "val_name": "v1"},
    )
    r2 = client.post(
        f"/datasets/{src_id}/split",
        headers=HEADERS,
        json={"ratio": "80:20", "seed": 7, "train_name": "t2", "val_name": "v2"},
    )
    # Same content → same sha → same dataset id returned by the dedup branch.
    assert r1.json()["train"]["sha256"] == r2.json()["train"]["sha256"]
    assert r1.json()["val"]["sha256"] == r2.json()["val"]["sha256"]


def test_split_bad_ratio(state, client):
    src_id = _upload_jsonl(client, ["a", "b", "c"])
    r = client.post(
        f"/datasets/{src_id}/split",
        headers=HEADERS,
        json={"ratio": "70:20"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "bad_ratio"


def test_split_non_jsonl_rejected(state, client):
    async def _make_csv():
        async with state["db"].connect() as conn:
            return await repository.create_dataset(
                conn,
                name="csv",
                path="/tmp/x.csv",
                fmt="csv",
                size_bytes=1,
                sha256="b" * 64,
                line_count=1,
            )

    ds_id = _run(_make_csv())
    r = client.post(
        f"/datasets/{ds_id}/split", headers=HEADERS, json={"ratio": "90:10"}
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unsupported_format"


# ---------------------------------------------------------------------------
# Mix
# ---------------------------------------------------------------------------


def test_mix_combines_two_sources(state, client):
    a_id = _upload_jsonl(client, [f"a{i}" for i in range(5)])
    b_id = _upload_jsonl(client, [f"b{i}" for i in range(5)])
    r = client.post(
        "/datasets/mixes",
        headers=HEADERS,
        json={
            "name": "mixed",
            "sources": [
                {"dataset_id": a_id, "weight": 0.7},
                {"dataset_id": b_id, "weight": 0.3},
            ],
            "target_count": 50,
            "seed": 0,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["line_count"] == 50
    assert "mix of" in body["description"]


def test_mix_target_count_defaults_to_sum(state, client):
    a_id = _upload_jsonl(client, [f"a{i}" for i in range(3)])
    b_id = _upload_jsonl(client, [f"b{i}" for i in range(4)])
    r = client.post(
        "/datasets/mixes",
        headers=HEADERS,
        json={
            "name": "default-target",
            "sources": [
                {"dataset_id": a_id, "weight": 1},
                {"dataset_id": b_id, "weight": 1},
            ],
            "seed": 0,
        },
    )
    assert r.status_code == 201
    assert r.json()["line_count"] == 7


def test_mix_unknown_source_422(state, client):
    a_id = _upload_jsonl(client, ["x"])
    r = client.post(
        "/datasets/mixes",
        headers=HEADERS,
        json={
            "name": "n",
            "sources": [
                {"dataset_id": a_id, "weight": 1.0},
                {"dataset_id": "deadbeef", "weight": 1.0},
            ],
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unknown_dataset"


def test_mix_records_all_parents_in_lineage(state, client):
    """The mix endpoint must record an N:M lineage row per source —
    not just collapse to derived_from=first-parent. Otherwise the GDPR
    recursive query misses the other sources."""
    a_id = _upload_jsonl(client, [f"a{i}" for i in range(3)])
    b_id = _upload_jsonl(client, [f"b{i}" for i in range(3)])
    c_id = _upload_jsonl(client, [f"c{i}" for i in range(3)])
    r = client.post(
        "/datasets/mixes",
        headers=HEADERS,
        json={
            "name": "tri-mix",
            "sources": [
                {"dataset_id": a_id, "weight": 1},
                {"dataset_id": b_id, "weight": 1},
                {"dataset_id": c_id, "weight": 1},
            ],
            "target_count": 9,
        },
    )
    assert r.status_code == 201
    mix_id = r.json()["id"]

    async def _check():
        async with state["db"].connect() as conn:
            ancestors = await repository.dataset_ancestors(conn, mix_id)
        return ancestors

    ancestors = _run(_check())
    assert ancestors == {a_id, b_id, c_id}


def test_split_records_lineage(state, client):
    src_id = _upload_jsonl(client, [f"row {i}" for i in range(10)])
    r = client.post(
        f"/datasets/{src_id}/split",
        headers=HEADERS,
        json={"ratio": "80:20", "seed": 1},
    )
    train_id = r.json()["train"]["id"]
    val_id = r.json()["val"]["id"]

    async def _ancestors_of(did):
        async with state["db"].connect() as conn:
            return await repository.dataset_ancestors(conn, did)

    assert _run(_ancestors_of(train_id)) == {src_id}
    assert _run(_ancestors_of(val_id)) == {src_id}


def test_dataset_descendants_walk(state, client):
    """parent → split-train → mix(split-train + other_source). The
    descendants of the original parent should include train, val, AND
    the mix that consumed train."""
    parent_id = _upload_jsonl(client, [f"p{i}" for i in range(10)])
    other_id = _upload_jsonl(client, [f"o{i}" for i in range(3)])

    sp = client.post(
        f"/datasets/{parent_id}/split",
        headers=HEADERS,
        json={"ratio": "70:30", "seed": 0},
    ).json()
    train_id = sp["train"]["id"]
    val_id = sp["val"]["id"]

    mix = client.post(
        "/datasets/mixes",
        headers=HEADERS,
        json={
            "name": "downstream-mix",
            "sources": [
                {"dataset_id": train_id, "weight": 1},
                {"dataset_id": other_id, "weight": 1},
            ],
            "target_count": 5,
        },
    ).json()
    mix_id = mix["id"]

    async def _desc():
        async with state["db"].connect() as conn:
            return await repository.dataset_descendants(conn, parent_id)

    descendants = _run(_desc())
    assert train_id in descendants
    assert val_id in descendants
    assert mix_id in descendants


def test_mix_requires_minimum_two_sources(state, client):
    a_id = _upload_jsonl(client, ["x"])
    r = client.post(
        "/datasets/mixes",
        headers=HEADERS,
        json={
            "name": "n",
            "sources": [{"dataset_id": a_id, "weight": 1}],
        },
    )
    # Pydantic min_length=2 validation kicks in.
    assert r.status_code == 422
