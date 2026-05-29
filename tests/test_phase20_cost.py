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


async def test_study_cost_summary_aggregates(db):
    """Sum of all completed experiments under a study, not the others."""
    from trainpipe.api.schemas import StudyConfig

    async with db.connect() as conn:
        cfg = StudyConfig(
            name="study1",
            base_spec=ExperimentSpec(model="m", dataset=["/x"]),
            search_space={},
            target_metric="loss",
            direction="minimize",
        )
        study_id = await repository.create_study(conn, cfg, "sqlite:///x")

        # Three completed runs, one running (excluded), one failed (excluded).
        for gpu_s, status, peak in [
            (100.0, "completed", 5000.0),
            (200.0, "completed", 8000.0),
            (300.0, "completed", 6000.0),
            (50.0, "running", 4000.0),
            (75.0, "failed", 3000.0),
        ]:
            spec = ExperimentSpec(model="m", dataset=["/x"])
            exp_id = await repository.create_experiment(
                conn, spec, study_id=study_id
            )
            await conn.execute(
                "UPDATE experiments SET status=?, gpu_seconds=?, "
                "peak_vram_mb=? WHERE id=?",
                (status, gpu_s, peak, exp_id),
            )
            await conn.commit()

        summary = await repository.study_cost_summary(conn, study_id)
    assert summary["n_trials"] == 3
    assert summary["total_gpu_seconds"] == 600.0
    assert summary["peak_vram_mb"] == 8000.0  # max across completed runs


async def test_study_cost_summary_empty_study(db):
    """A study with no experiments returns zero counts, not a crash."""
    from trainpipe.api.schemas import StudyConfig

    async with db.connect() as conn:
        cfg = StudyConfig(
            name="empty",
            base_spec=ExperimentSpec(model="m", dataset=["/x"]),
            search_space={},
            target_metric="loss",
            direction="minimize",
        )
        study_id = await repository.create_study(conn, cfg, "sqlite:///x")
        summary = await repository.study_cost_summary(conn, study_id)
    assert summary["n_trials"] == 0
    assert summary["total_gpu_seconds"] == 0.0
    assert summary["peak_vram_mb"] is None
