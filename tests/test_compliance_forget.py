"""Tests for the GDPR "forget user Y" scan + CLI (Phase 15 follow-up)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from trainpipe.api.schemas import ExperimentSpec
from trainpipe.compliance.forget import (
    ForgetReport,
    _build_matcher,
    scan_datasets_for_term,
)
from trainpipe.core import repository


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def test_matcher_substring_case_insensitive_by_default():
    m = _build_matcher("jane@x.com", is_regex=False, case_sensitive=False)
    assert m("contact JANE@X.COM today") is True
    assert m("contact mike@y.com today") is False


def test_matcher_substring_case_sensitive():
    m = _build_matcher("Jane", is_regex=False, case_sensitive=True)
    assert m("Jane Doe") is True
    assert m("jane doe") is False


def test_matcher_regex():
    m = _build_matcher(
        r"AT\d{2}[A-Z0-9]+", is_regex=True, case_sensitive=False
    )
    assert m("IBAN AT123456X") is True
    assert m("not a match") is False


def test_matcher_invalid_regex_raises():
    with pytest.raises(ValueError, match="invalid regex"):
        _build_matcher("[unclosed", is_regex=True, case_sensitive=False)


# ---------------------------------------------------------------------------
# scan_datasets_for_term
# ---------------------------------------------------------------------------


async def test_scan_finds_hits_and_skips_clean(db, tmp_path):
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    hit_file = ds_dir / "with-pii.jsonl"
    hit_file.write_text(
        '{"prompt":"mail jane@example.com"}\n'
        '{"prompt":"unrelated"}\n'
        '{"prompt":"call jane@example.com again"}\n',
        encoding="utf-8",
    )
    clean_file = ds_dir / "clean.jsonl"
    clean_file.write_text(
        '{"prompt":"hi"}\n',
        encoding="utf-8",
    )

    async with db.connect() as conn:
        hit_id = await repository.create_dataset(
            conn,
            name="with-pii",
            path=str(hit_file),
            fmt="jsonl",
            size_bytes=hit_file.stat().st_size,
            sha256="a" * 64,
            line_count=3,
        )
        await repository.create_dataset(
            conn,
            name="clean",
            path=str(clean_file),
            fmt="jsonl",
            size_bytes=clean_file.stat().st_size,
            sha256="b" * 64,
            line_count=1,
        )
        report = await scan_datasets_for_term(conn, "jane@example.com")

    assert report.scanned_datasets == 2
    assert len(report.hits) == 1
    hit = report.hits[0]
    assert hit.dataset_id == hit_id
    assert hit.hit_count == 2
    assert hit.sample_line_numbers == [1, 3]


async def test_scan_skips_non_jsonl_formats(db, tmp_path):
    """Parquet etc. are listed in ``skipped_datasets`` not hits."""
    csv_file = tmp_path / "x.csv"
    csv_file.write_text("a,b\n1,2\n", encoding="utf-8")
    async with db.connect() as conn:
        ds_id = await repository.create_dataset(
            conn,
            name="csv",
            path=str(csv_file),
            fmt="csv",
            size_bytes=10,
            sha256="c" * 64,
            line_count=1,
        )
        report = await scan_datasets_for_term(conn, "anything")
    assert any(ds_id in s for s in report.skipped_datasets)
    assert report.scanned_datasets == 0


async def test_scan_skips_missing_file(db, tmp_path):
    """A registered row pointing at a deleted file shouldn't crash —
    just skip with a note."""
    async with db.connect() as conn:
        ds_id = await repository.create_dataset(
            conn,
            name="gone",
            path=str(tmp_path / "does-not-exist.jsonl"),
            fmt="jsonl",
            size_bytes=10,
            sha256="d" * 64,
            line_count=1,
        )
        report = await scan_datasets_for_term(conn, "any")
    assert any(ds_id in s and "missing" in s for s in report.skipped_datasets)


async def test_scan_resolves_recursive_lineage(db, tmp_path):
    """Critical GDPR requirement: a model trained on a MIX of (parent +
    other) must show up in the report when ``parent`` has hits."""
    parent_file = tmp_path / "parent.jsonl"
    parent_file.write_text(
        '{"prompt":"jane@example.com here"}\n',
        encoding="utf-8",
    )
    mix_file = tmp_path / "mix.jsonl"
    mix_file.write_text(
        '{"prompt":"derived row 1"}\n'
        '{"prompt":"derived row 2"}\n',
        encoding="utf-8",
    )
    async with db.connect() as conn:
        parent_id = await repository.create_dataset(
            conn,
            name="parent",
            path=str(parent_file),
            fmt="jsonl",
            size_bytes=parent_file.stat().st_size,
            sha256="p" * 64,
            line_count=1,
        )
        mix_id = await repository.create_dataset(
            conn,
            name="mix",
            path=str(mix_file),
            fmt="jsonl",
            size_bytes=mix_file.stat().st_size,
            sha256="m" * 64,
            line_count=2,
            derived_from=parent_id,
        )
        await repository.record_dataset_lineage(
            conn, mix_id, [parent_id], role="mix-of"
        )

        spec = ExperimentSpec(model="qwen/x", dataset=[str(mix_file)])
        exp_id = await repository.create_experiment(conn, spec)
        await conn.execute(
            "UPDATE experiments SET status='completed' WHERE id=?",
            (exp_id,),
        )
        await conn.commit()

        # Register a model on the mix and record the direct lineage row.
        model_id, _ = await repository.register_model_atomic(
            conn,
            name="famZ",
            explicit_version=None,
            base_model="qwen/x",
            adapter_path=None,
            experiment_id=exp_id,
            run_id=None,
            eval_summary=None,
            description=None,
            alias=None,
        )
        await repository.record_model_lineage(conn, model_id, [mix_id])

        report = await scan_datasets_for_term(conn, "jane@example.com")

    # The hit is on parent, but the impacted model trained on mix.
    assert any(h.dataset_id == parent_id for h in report.hits)
    impacted_ids = [m.model_id for m in report.impacted_models]
    assert model_id in impacted_ids


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


def test_report_to_dict_is_json_serializable():
    r = ForgetReport(
        term="x", is_regex=False, case_sensitive=False, scanned_datasets=0
    )
    # Should not raise.
    body = json.dumps(r.to_dict())
    parsed = json.loads(body)
    assert parsed["term"] == "x"
    assert parsed["hits"] == []
    assert parsed["impacted_models"] == []


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_help_runs():
    """``trainpipe-forget --help`` must exit 0 — proves the entry point
    is wired and argparse builds."""
    result = subprocess.run(
        [sys.executable, "-m", "trainpipe.compliance.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "trainpipe-forget" in result.stdout
    assert "regex" in result.stdout


def test_cli_missing_db_exits_2(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trainpipe.compliance.cli",
            "--db",
            str(tmp_path / "no-such.sqlite3"),
            "anything",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr


def test_cli_no_hits_exits_zero(tmp_path):
    """A scan that finds nothing exits 0 (clean compliance check)."""
    from trainpipe.core.db import Database

    db_path = tmp_path / "test.sqlite3"

    import asyncio
    async def _init():
        db = Database(db_path)
        await db.init()

    asyncio.run(_init())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trainpipe.compliance.cli",
            "--db",
            str(db_path),
            "no-such-term",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Dataset hits (0)" in result.stdout


def test_cli_writes_json_when_output_given(tmp_path):
    """``--output report.json`` writes a JSON file matching the dict."""
    from trainpipe.core.db import Database

    db_path = tmp_path / "test.sqlite3"

    import asyncio
    async def _init():
        db = Database(db_path)
        await db.init()

    asyncio.run(_init())
    out = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trainpipe.compliance.cli",
            "--db",
            str(db_path),
            "--output",
            str(out),
            "anything",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["term"] == "anything"
    assert isinstance(body["hits"], list)
