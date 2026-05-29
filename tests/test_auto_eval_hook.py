"""Tests for the after-training auto_eval hook.

The hook lives in ``Scheduler._enqueue_auto_evals`` and is called from
``_monitor`` after a successful training run. We exercise the method
directly to avoid spinning up a real subprocess.
"""

from trainpipe.api.schemas import (
    EvalRunStatus,
    ExperimentSpec,
    InferenceParams,
    MetricConfig,
)
from trainpipe.core import repository
from trainpipe.scheduler.gpu_pool import GpuPool
from trainpipe.scheduler.loop import Scheduler


async def _make_suite(db, name: str) -> str:
    async with db.connect() as conn:
        return await repository.create_eval_suite(
            conn,
            name=name,
            description=None,
            dataset_path="/tmp/x.jsonl",
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )


async def _make_experiment_with_auto_eval(
    db, *, suite_ids: list[str], name: str = "auto-exp"
) -> str:
    spec = ExperimentSpec(
        name=name,
        model="m",
        dataset=["d"],
        auto_eval=suite_ids,
    )
    async with db.connect() as conn:
        return await repository.create_experiment(conn, spec)


async def test_enqueue_auto_evals_creates_one_run_per_suite(db):
    s1 = await _make_suite(db, "suite-1")
    s2 = await _make_suite(db, "suite-2")
    exp_id = await _make_experiment_with_auto_eval(
        db, suite_ids=[s1, s2], name="parent-exp"
    )

    sched = Scheduler(db, GpuPool([]))
    await sched._enqueue_auto_evals(exp_id)

    async with db.connect() as conn:
        runs = await repository.list_eval_runs(conn, experiment_id=exp_id)
    assert len(runs) == 2
    assert all(r.triggered_by == "auto" for r in runs)
    assert all(r.status == EvalRunStatus.QUEUED for r in runs)
    assert {r.suite_id for r in runs} == {s1, s2}
    assert all(r.model_ref == "parent-exp" for r in runs)


async def test_enqueue_auto_evals_skips_unknown_suite(db, caplog):
    real = await _make_suite(db, "real-suite")
    exp_id = await _make_experiment_with_auto_eval(
        db, suite_ids=[real, "nonexistent-id"]
    )

    sched = Scheduler(db, GpuPool([]))
    with caplog.at_level("WARNING"):
        await sched._enqueue_auto_evals(exp_id)

    async with db.connect() as conn:
        runs = await repository.list_eval_runs(conn, experiment_id=exp_id)
    # Only the real suite enqueued; the unknown one is just a warning.
    assert len(runs) == 1
    assert runs[0].suite_id == real
    assert any("auto_eval skipped" in rec.message for rec in caplog.records)


async def test_enqueue_auto_evals_noop_when_list_empty(db):
    exp_id = await _make_experiment_with_auto_eval(db, suite_ids=[])
    sched = Scheduler(db, GpuPool([]))
    await sched._enqueue_auto_evals(exp_id)

    async with db.connect() as conn:
        runs = await repository.list_eval_runs(conn, experiment_id=exp_id)
    assert runs == []


async def test_enqueue_auto_evals_falls_back_to_experiment_id_when_unnamed(db):
    suite = await _make_suite(db, "name-fallback")
    async with db.connect() as conn:
        exp_id = await repository.create_experiment(
            conn,
            ExperimentSpec(model="m", dataset=["d"], auto_eval=[suite]),
        )

    sched = Scheduler(db, GpuPool([]))
    await sched._enqueue_auto_evals(exp_id)

    async with db.connect() as conn:
        runs = await repository.list_eval_runs(conn, experiment_id=exp_id)
    assert len(runs) == 1
    assert runs[0].model_ref == exp_id
