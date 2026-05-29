from trainpipe.api.schemas import ExperimentSpec, ExperimentStatus
from trainpipe.core import repository


def _spec(**over) -> ExperimentSpec:
    base = dict(model="m1", dataset=["d1"])
    base.update(over)
    return ExperimentSpec(**base)


async def test_create_then_get_experiment(db):
    spec = _spec(name="run-a", gpu_count=2, priority=5)
    async with db.connect() as conn:
        exp_id = await repository.create_experiment(conn, spec)
        rec = await repository.get_experiment(conn, exp_id)
    assert rec is not None
    assert rec.spec == spec
    assert rec.status == ExperimentStatus.QUEUED
    assert rec.priority == 5
    assert rec.gpu_ids is None
    assert rec.started_at is None


async def test_list_experiments_filter_by_status(db):
    async with db.connect() as conn:
        a = await repository.create_experiment(conn, _spec(name="a"))
        b = await repository.create_experiment(conn, _spec(name="b"))
        # Mark b as running to test the filter
        await conn.execute(
            "UPDATE experiments SET status = 'running' WHERE id = ?", (b,)
        )
        await conn.commit()
        queued = await repository.list_experiments(conn, status=ExperimentStatus.QUEUED)
        running = await repository.list_experiments(conn, status=ExperimentStatus.RUNNING)
    assert {r.id for r in queued} == {a}
    assert {r.id for r in running} == {b}


async def test_request_cancel_on_queued_marks_cancelled(db):
    async with db.connect() as conn:
        exp_id = await repository.create_experiment(conn, _spec())
        result = await repository.request_cancel(conn, exp_id)
        rec = await repository.get_experiment(conn, exp_id)
    assert result == "cancelled"
    assert rec.status == ExperimentStatus.CANCELLED
    assert rec.finished_at is not None


async def test_request_cancel_on_running_signals_caller(db):
    async with db.connect() as conn:
        exp_id = await repository.create_experiment(conn, _spec())
        await conn.execute(
            "UPDATE experiments SET status = 'running' WHERE id = ?", (exp_id,)
        )
        await conn.commit()
        result = await repository.request_cancel(conn, exp_id)
    assert result == "running"


async def test_request_cancel_on_missing(db):
    async with db.connect() as conn:
        result = await repository.request_cancel(conn, "deadbeef")
    assert result == "not_found"
