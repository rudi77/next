import asyncio

import pytest

from trainpipe.api.schemas import ExperimentSpec
from trainpipe.core import repository
from trainpipe.core.db import Database
from trainpipe.training.dataset_refs import (
    UnknownDatasetRef,
    is_ref,
    parse_ref,
    resolve_spec,
)


def test_parse_ref_plain():
    assert parse_ref("ds:abc123") == ("abc123", "")


def test_parse_ref_with_suffix():
    assert parse_ref("ds:abc123#500") == ("abc123", "#500")


def test_parse_ref_non_ref():
    assert parse_ref("AI-ModelScope/alpaca") is None
    assert parse_ref("/abs/path.jsonl") is None
    assert parse_ref("./rel.jsonl") is None


def test_parse_ref_rejects_non_hex():
    assert parse_ref("ds:not-hex!") is None


def test_is_ref():
    assert is_ref("ds:abc123") is True
    assert is_ref("ds:abc123#10") is True
    assert is_ref("/data/x.jsonl") is False


def _spec(ds, val=None):
    return ExperimentSpec(model="m", dataset=ds, val_dataset=val or [])


async def _setup_db(tmp_path, ds_id, ds_path):
    db = Database(tmp_path / "t.sqlite3")
    await db.init()
    async with db.connect() as conn:
        await repository.create_dataset(
            conn,
            name="alpaca",
            path=str(ds_path),
            fmt="jsonl",
            size_bytes=100,
            sha256="deadbeef",
            line_count=10,
            dataset_id=ds_id,
        )
    return db


def test_resolve_spec_replaces_ref(tmp_path):
    ds_id = "abc123ff"
    ds_path = tmp_path / "alpaca.jsonl"
    ds_path.write_text("{}\n", encoding="utf-8")

    async def go():
        db = await _setup_db(tmp_path, ds_id, ds_path)
        async with db.connect() as conn:
            spec = _spec(["ds:" + ds_id])
            resolved = await resolve_spec(spec, conn)
        return resolved

    resolved = asyncio.run(go())
    assert resolved.dataset == [str(ds_path)]


def test_resolve_spec_preserves_subsample_suffix(tmp_path):
    ds_id = "abc123ff"
    ds_path = tmp_path / "alpaca.jsonl"
    ds_path.write_text("{}\n", encoding="utf-8")

    async def go():
        db = await _setup_db(tmp_path, ds_id, ds_path)
        async with db.connect() as conn:
            return await resolve_spec(_spec(["ds:" + ds_id + "#500"]), conn)

    resolved = asyncio.run(go())
    assert resolved.dataset == [f"{ds_path}#500"]


def test_resolve_spec_passes_through_remote_and_local(tmp_path):
    async def go():
        db = Database(tmp_path / "t.sqlite3")
        await db.init()
        async with db.connect() as conn:
            return await resolve_spec(
                _spec(["AI-ModelScope/alpaca", "/abs/local.jsonl"]), conn
            )

    resolved = asyncio.run(go())
    assert resolved.dataset == ["AI-ModelScope/alpaca", "/abs/local.jsonl"]


def test_resolve_spec_unknown_ref_raises(tmp_path):
    async def go():
        db = Database(tmp_path / "t.sqlite3")
        await db.init()
        async with db.connect() as conn:
            await resolve_spec(_spec(["ds:deadbeef"]), conn)

    with pytest.raises(UnknownDatasetRef) as exc:
        asyncio.run(go())
    assert exc.value.ref_id == "deadbeef"


def test_resolve_spec_resolves_val_dataset_too(tmp_path):
    ds_id = "abc123ff"
    ds_path = tmp_path / "val.jsonl"
    ds_path.write_text("{}\n", encoding="utf-8")

    async def go():
        db = await _setup_db(tmp_path, ds_id, ds_path)
        async with db.connect() as conn:
            return await resolve_spec(
                _spec(["AI-ModelScope/x"], val=["ds:" + ds_id]), conn
            )

    resolved = asyncio.run(go())
    assert resolved.dataset == ["AI-ModelScope/x"]
    assert resolved.val_dataset == [str(ds_path)]
