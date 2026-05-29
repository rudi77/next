from trainpipe.api.schemas import (
    EvalRunStatus,
    ExperimentSpec,
    InferenceParams,
    MetricAggregate,
    MetricConfig,
)
from trainpipe.core import repository


def _metric(kind: str = "exact_match", **cfg) -> MetricConfig:
    return MetricConfig(kind=kind, config=cfg)


async def _make_experiment(db, name: str = "e") -> str:
    async with db.connect() as conn:
        return await repository.create_experiment(
            conn, ExperimentSpec(name=name, model="m", dataset=["d"]),
        )


async def _make_suite(
    db,
    name: str = "default",
    dataset_path: str = "/tmp/eval.jsonl",
    metrics: list[MetricConfig] | None = None,
) -> str:
    async with db.connect() as conn:
        return await repository.create_eval_suite(
            conn,
            name=name,
            description=None,
            dataset_path=dataset_path,
            metrics=metrics or [_metric()],
            inference_params=InferenceParams(),
        )


async def test_create_then_get_eval_suite(db):
    suite_id = await _make_suite(db, name="invoice-extract")
    async with db.connect() as conn:
        suite = await repository.get_eval_suite(conn, suite_id)
    assert suite is not None
    assert suite.name == "invoice-extract"
    assert suite.dataset_path == "/tmp/eval.jsonl"
    assert suite.metrics[0].kind == "exact_match"
    assert suite.inference_params.max_new_tokens == 512


async def test_eval_suite_name_unique(db):
    await _make_suite(db, name="dupes")
    import sqlite3

    import pytest

    with pytest.raises((sqlite3.IntegrityError, Exception)) as excinfo:
        await _make_suite(db, name="dupes", dataset_path="/tmp/other.jsonl")
    assert "UNIQUE" in str(excinfo.value) or "unique" in str(excinfo.value).lower()


async def test_get_eval_suite_by_name(db):
    await _make_suite(db, name="by-name-test")
    async with db.connect() as conn:
        suite = await repository.get_eval_suite_by_name(conn, "by-name-test")
    assert suite is not None
    assert suite.name == "by-name-test"


async def test_list_eval_suites(db):
    await _make_suite(db, name="a")
    await _make_suite(db, name="b")
    async with db.connect() as conn:
        suites = await repository.list_eval_suites(conn)
    names = {s.name for s in suites}
    assert names == {"a", "b"}


async def test_delete_eval_suite(db):
    sid = await _make_suite(db, name="ephemeral")
    async with db.connect() as conn:
        deleted = await repository.delete_eval_suite(conn, sid)
        gone = await repository.get_eval_suite(conn, sid)
    assert deleted is True
    assert gone is None


async def test_create_then_get_eval_run(db):
    sid = await _make_suite(db, name="for-run")
    exp_id = await _make_experiment(db, name="exp-for-run")
    async with db.connect() as conn:
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=exp_id,
            model_ref=exp_id,
            triggered_by="manual",
        )
        run = await repository.get_eval_run(conn, run_id)
    assert run is not None
    assert run.status == EvalRunStatus.QUEUED
    assert run.suite_id == sid
    assert run.experiment_id == exp_id
    assert run.triggered_by == "manual"


async def test_claim_eval_run_atomic(db):
    sid = await _make_suite(db, name="for-claim")
    async with db.connect() as conn:
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="base",
            triggered_by="manual",
        )
        first = await repository.claim_eval_run(conn, run_id)
        second = await repository.claim_eval_run(conn, run_id)
        run = await repository.get_eval_run(conn, run_id)
    assert first is True
    assert second is False
    assert run.status == EvalRunStatus.RUNNING
    assert run.started_at is not None


async def test_finalize_eval_run_persists_aggregate(db):
    sid = await _make_suite(db, name="for-finalize")
    async with db.connect() as conn:
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="m1",
            triggered_by="auto",
        )
        await repository.claim_eval_run(conn, run_id)
        await repository.finalize_eval_run(
            conn,
            run_id,
            status=EvalRunStatus.COMPLETED,
            aggregate={
                "exact_match": MetricAggregate(mean=0.87, std=0.12, count=100),
            },
            sample_count=100,
        )
        run = await repository.get_eval_run(conn, run_id)
    assert run.status == EvalRunStatus.COMPLETED
    assert run.aggregate is not None
    assert run.aggregate["exact_match"].mean == 0.87
    assert run.sample_count == 100
    assert run.finished_at is not None


async def test_list_eval_runs_filters(db):
    sid_a = await _make_suite(db, name="suite-a")
    sid_b = await _make_suite(db, name="suite-b")
    e1 = await _make_experiment(db, name="e1")
    e2 = await _make_experiment(db, name="e2")
    async with db.connect() as conn:
        a1 = await repository.create_eval_run(
            conn,
            suite_id=sid_a,
            experiment_id=e1,
            model_ref=e1,
            triggered_by="manual",
        )
        await repository.create_eval_run(
            conn,
            suite_id=sid_b,
            experiment_id=e2,
            model_ref=e2,
            triggered_by="manual",
        )
        by_suite_a = await repository.list_eval_runs(conn, suite_id=sid_a)
        by_exp = await repository.list_eval_runs(conn, experiment_id=e1)
    assert {r.id for r in by_suite_a} == {a1}
    assert {r.id for r in by_exp} == {a1}


async def test_add_and_list_eval_results(db):
    sid = await _make_suite(db, name="for-results")
    async with db.connect() as conn:
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="m",
            triggered_by="manual",
        )
        for i in range(3):
            await repository.add_eval_result(
                conn,
                run_id=run_id,
                sample_index=i,
                input={"q": f"question {i}"},
                prediction=f"answer {i}",
                gold={"a": f"answer {i}"},
                scores={"exact_match": 1.0 if i % 2 == 0 else 0.0},
            )
        results = await repository.list_eval_results(conn, run_id)
        count = await repository.count_eval_results(conn, run_id)
    assert count == 3
    assert [r.sample_index for r in results] == [0, 1, 2]
    assert results[0].scores == {"exact_match": 1.0}
    assert results[1].scores == {"exact_match": 0.0}


async def test_active_eval_runs_for_suite(db):
    sid = await _make_suite(db, name="for-active")
    async with db.connect() as conn:
        r_queued = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="m1",
            triggered_by="manual",
        )
        r_done = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="m2",
            triggered_by="manual",
        )
        await repository.finalize_eval_run(
            conn, r_done, status=EvalRunStatus.COMPLETED, sample_count=0,
        )
        active = await repository.active_eval_runs_for_suite(conn, sid)
    assert r_queued in active
    assert r_done not in active


async def test_request_cancel_eval_run(db):
    sid = await _make_suite(db, name="for-cancel")
    async with db.connect() as conn:
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="m",
            triggered_by="manual",
        )
        cancelled = await repository.request_cancel_eval_run(conn, run_id)
        run = await repository.get_eval_run(conn, run_id)
    assert cancelled == "cancelled"
    assert run.status == EvalRunStatus.CANCELLED

    async with db.connect() as conn:
        missing = await repository.request_cancel_eval_run(conn, "deadbeef")
    assert missing == "not_found"
