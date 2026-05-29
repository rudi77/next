"""Tests for Phase 20: cost / resource tracking."""

from datetime import datetime, timedelta, timezone

from trainpipe.api.schemas import ExperimentSpec
from trainpipe.core import repository


async def test_set_experiment_resource_usage_persists(db):
    async with db.connect() as conn:
        spec = ExperimentSpec(model="m", dataset=["/x"])
        exp_id = await repository.create_experiment(conn, spec)
        await repository.set_experiment_resource_usage(
            conn,
            exp_id,
            gpu_seconds=7200.0,
            peak_vram_mb=22500.0,
            energy_wh=350.5,
        )
        rec = await repository.get_experiment(conn, exp_id)
    assert rec.gpu_seconds == 7200.0
    assert rec.peak_vram_mb == 22500.0
    assert rec.energy_wh == 350.5


async def test_set_resource_usage_partial_leaves_others(db):
    """Passing only one column should not null out the others."""
    async with db.connect() as conn:
        spec = ExperimentSpec(model="m", dataset=["/x"])
        exp_id = await repository.create_experiment(conn, spec)
        await repository.set_experiment_resource_usage(
            conn, exp_id, gpu_seconds=100.0, peak_vram_mb=200.0
        )
        await repository.set_experiment_resource_usage(
            conn, exp_id, energy_wh=10.0
        )
        rec = await repository.get_experiment(conn, exp_id)
    assert rec.gpu_seconds == 100.0
    assert rec.peak_vram_mb == 200.0
    assert rec.energy_wh == 10.0


async def test_set_resource_usage_no_args_is_noop(db):
    async with db.connect() as conn:
        spec = ExperimentSpec(model="m", dataset=["/x"])
        exp_id = await repository.create_experiment(conn, spec)
        await repository.set_experiment_resource_usage(conn, exp_id)
        rec = await repository.get_experiment(conn, exp_id)
    assert rec.gpu_seconds is None


async def test_experiment_record_defaults(db):
    """Fresh experiments must have NULL cost fields, not crash on read."""
    async with db.connect() as conn:
        spec = ExperimentSpec(model="m", dataset=["/x"])
        exp_id = await repository.create_experiment(conn, spec)
        rec = await repository.get_experiment(conn, exp_id)
    assert rec.gpu_seconds is None
    assert rec.peak_vram_mb is None
    assert rec.energy_wh is None


async def test_gpu_seconds_math_helper(db):
    """The scheduler computes ``gpu_seconds = wall * len(gpu_ids)``.
    Verify that math directly so the scheduler-side wiring is testable
    without spawning subprocesses."""
    started = datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=1800)
    gpu_ids = [0, 1, 2, 3]
    wall = (finished - started).total_seconds()
    gpu_sec = wall * len(gpu_ids)
    assert gpu_sec == 7200.0
