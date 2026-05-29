"""Tests for Phase 15: PII redaction + model lineage audit."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from trainpipe.api.deps import get_db, get_gpu_pool, get_scheduler, get_study_manager
from trainpipe.api.main import app
from trainpipe.api.schemas import ExperimentSpec
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.redaction.redactor import (
    redact_jsonl,
    redact_text,
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
# Redactor unit tests
# ---------------------------------------------------------------------------


def test_redact_email():
    out, counts = redact_text("contact me at jane@example.com please")
    assert "[REDACTED_EMAIL]" in out
    assert "jane@example.com" not in out
    assert counts["email"] == 1


def test_redact_iban_only_valid_checksum():
    """Random AT-style strings without a valid mod-97 should be left alone."""
    # Valid IBAN (DE89370400440532013000 is a textbook example).
    valid = "transfer to DE89370400440532013000 today"
    out, counts = redact_text(valid)
    assert "[REDACTED_IBAN]" in out
    assert counts["iban"] == 1

    # Invalid checksum — should NOT redact.
    bogus = "ID code AB12CDEFGHIJKLMNOPQR"
    out2, counts2 = redact_text(bogus)
    assert "[REDACTED_IBAN]" not in out2
    assert counts2["iban"] == 0


def test_redact_phone_rejects_all_same_digits():
    out, counts = redact_text("call 1111111111")
    assert "[REDACTED_PHONE]" not in out
    assert counts["phone"] == 0

    out2, counts2 = redact_text("call +43 660 1234567 please")
    assert "[REDACTED_PHONE]" in out2
    assert counts2["phone"] == 1


def test_redact_credit_card():
    out, counts = redact_text("card 4111-1111-1111-1111 expires soon")
    assert "[REDACTED_CC]" in out
    assert counts["credit_card"] == 1


def test_redact_disable_entity_type():
    out, _ = redact_text(
        "call jane@example.com or +43 660 1234567",
        entities=["email"],
    )
    assert "[REDACTED_EMAIL]" in out
    assert "+43 660 1234567" in out  # phone NOT redacted


def test_redact_jsonl_round_trip(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "email me at jane@example.com"},
                {"role": "assistant", "content": "noted"},
            ],
            "meta": {"safe": "stays"},
        }) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    rows, counts = redact_jsonl(str(src), str(out))
    assert rows == 1
    assert counts["email"] == 1
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert "jane@example.com" not in json.dumps(parsed)
    assert parsed["meta"]["safe"] == "stays"  # untouched


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


def _upload_jsonl(client, content: str, name: str = "with-pii") -> str:
    r = client.post(
        "/datasets",
        headers=HEADERS,
        files={"file": (f"{name}.jsonl", content.encode("utf-8"), "application/x-ndjson")},
        data={"name": name},
    )
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_redact_endpoint_creates_new_dataset(state, client):
    src_id = _upload_jsonl(
        client,
        json.dumps({"prompt": "mail to jane@example.com"}) + "\n",
    )
    r = client.post(
        f"/datasets/{src_id}/redact",
        headers=HEADERS,
        json={},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] != src_id
    assert f"ds:{src_id}" in body["description"]
    # The redacted file has no email.
    with open(body["path"], encoding="utf-8") as f:
        content = f.read()
    assert "jane@example.com" not in content
    assert "REDACTED_EMAIL" in content


def test_redact_rejects_non_jsonl(state, client):
    # Use the repository to create a parquet stub (no actual file content needed
    # for the route's format check).
    async def _make():
        async with state["db"].connect() as conn:
            return await repository.create_dataset(
                conn,
                name="csv",
                path="/tmp/x.csv",
                fmt="csv",
                size_bytes=1,
                sha256="x" * 64,
                line_count=1,
            )

    ds_id = _run(_make())
    r = client.post(
        f"/datasets/{ds_id}/redact", headers=HEADERS, json={}
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "unsupported_format"


# ---------------------------------------------------------------------------
# Model lineage
# ---------------------------------------------------------------------------


async def test_register_model_records_lineage(db, tmp_path):
    """Registering an experiment whose dataset is in the registry must
    populate model_lineage automatically."""
    ds_file = tmp_path / "real.jsonl"
    ds_file.write_text('{"x":1}\n', encoding="utf-8")
    async with db.connect() as conn:
        ds_id = await repository.create_dataset(
            conn,
            name="ds1",
            path=str(ds_file),
            fmt="jsonl",
            size_bytes=10,
            sha256="a" * 64,
            line_count=1,
        )
        spec = ExperimentSpec(
            model="m", dataset=[str(ds_file)]
        )
        exp_id = await repository.create_experiment(conn, spec)
        await conn.execute(
            "UPDATE experiments SET status='completed' WHERE id=?", (exp_id,)
        )
        await conn.commit()

    # Hand-roll the lineage flow (it lives in routes/models.py; this test
    # reuses the repository pieces directly for unit-level verification).
    async with db.connect() as conn:
        model_id, _ = await repository.register_model_atomic(
            conn,
            name="fam",
            explicit_version=None,
            base_model="m",
            adapter_path=None,
            experiment_id=exp_id,
            run_id=None,
            eval_summary=None,
            description=None,
            alias=None,
        )
        paths = [str(ds_file)]
        path_map = await repository.datasets_by_paths(conn, paths)
        await repository.record_model_lineage(
            conn, model_id, list(path_map.values())
        )
        models_for_ds = await repository.models_using_dataset(conn, ds_id)
        datasets_for_model = await repository.datasets_used_by_model(
            conn, model_id
        )
    assert model_id in models_for_ds
    assert ds_id in datasets_for_model


def test_models_using_dataset_recursive_finds_mix_descendants(state, client):
    """The proper GDPR query: model trained on mix(parent, other) must
    show up when querying for ``parent``'s downstream models."""
    # Upload two source datasets.
    parent_id = _upload_jsonl(
        client,
        json.dumps({"x": 1}) + "\n",
        name="gdpr-parent",
    )
    other_id = _upload_jsonl(
        client,
        json.dumps({"y": 2}) + "\n",
        name="gdpr-other",
    )

    # Mix them — the mix consumes both.
    mix = client.post(
        "/datasets/mixes",
        headers=HEADERS,
        json={
            "name": "mix-for-gdpr",
            "sources": [
                {"dataset_id": parent_id, "weight": 1},
                {"dataset_id": other_id, "weight": 1},
            ],
            "target_count": 2,
        },
    ).json()
    mix_id = mix["id"]

    # Create a completed experiment that trained on the *mix*, then
    # register it as a model. The direct query for ``parent_id`` should
    # NOT find this model — but the recursive query SHOULD.
    async def _make_exp_on_mix():
        async with state["db"].connect() as conn:
            mix_rec = await repository.get_dataset(conn, mix_id)
            spec = ExperimentSpec(model="m", dataset=[mix_rec.path])
            exp_id = await repository.create_experiment(conn, spec)
            await conn.execute(
                "UPDATE experiments SET status='completed' WHERE id=?",
                (exp_id,),
            )
            await conn.commit()
            return exp_id

    exp_id = _run(_make_exp_on_mix())
    r = client.post(
        "/models",
        headers=HEADERS,
        json={"name": "fam", "experiment_id": exp_id},
    )
    assert r.status_code == 201
    model_id = r.json()["id"]

    # Direct query for the parent: no models trained on it directly.
    direct = client.get(
        f"/datasets/{parent_id}/models", headers=HEADERS
    ).json()
    assert model_id not in direct["model_ids"]

    # Recursive query: model_id IS found via parent → mix → model.
    recursive = client.get(
        f"/datasets/{parent_id}/models?recursive=true", headers=HEADERS
    ).json()
    assert model_id in recursive["model_ids"]


def test_models_using_dataset_endpoint(state, client):
    """The GET /datasets/{id}/models endpoint must surface lineage."""

    # Upload a dataset.
    ds_id = _upload_jsonl(client, '{"x":1}\n', name="lineage")

    async def _make_completed_exp_and_register():
        async with state["db"].connect() as conn:
            ds = await repository.get_dataset(conn, ds_id)
            spec = ExperimentSpec(model="m", dataset=[ds.path])
            exp_id = await repository.create_experiment(conn, spec)
            await conn.execute(
                "UPDATE experiments SET status='completed' WHERE id=?",
                (exp_id,),
            )
            await conn.commit()
            return exp_id

    exp_id = _run(_make_completed_exp_and_register())

    r = client.post(
        "/models",
        headers=HEADERS,
        json={"name": "fam", "experiment_id": exp_id},
    )
    assert r.status_code == 201, r.text
    model_id = r.json()["id"]

    audit = client.get(f"/datasets/{ds_id}/models", headers=HEADERS)
    assert audit.status_code == 200
    assert model_id in audit.json()["model_ids"]
