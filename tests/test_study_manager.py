import asyncio

from trainpipe.api.schemas import ExperimentSpec, SearchSpaceEntry, StudyConfig
from trainpipe.autoresearch import manager as manager_module
from trainpipe.autoresearch.manager import StudyManager
from trainpipe.core import repository
from trainpipe.core.db import Database


class _StubDriver:
    instances: list["_StubDriver"] = []

    def __init__(self, study_id, config, storage, db):
        self.study_id = study_id
        self.config = config
        self.storage = storage
        self.db = db
        self.started = False
        self.stopped = False
        _StubDriver.instances.append(self)

    def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


def _make_config() -> StudyConfig:
    return StudyConfig(
        name="sweep-test",
        base_spec=ExperimentSpec(model="m", dataset=["d"]),
        search_space={
            "hyperparameters.learning_rate": SearchSpaceEntry(
                kind="loguniform", low=1e-5, high=1e-2
            ),
        },
        target_metric="eval/loss",
        n_trials=5,
    )


def test_create_and_start_persists_row_and_starts_driver(monkeypatch, tmp_path):
    _StubDriver.instances.clear()
    monkeypatch.setattr(manager_module, "StudyDriver", _StubDriver)
    monkeypatch.setattr(
        "trainpipe.autoresearch.manager.settings.data_dir", tmp_path
    )

    async def go():
        db = Database(tmp_path / "trainpipe.sqlite3")
        await db.init()
        m = StudyManager(db)
        sid = await m.create_and_start(_make_config())
        async with db.connect() as conn:
            rec = await repository.get_study(conn, sid)
        return sid, rec, m

    sid, rec, m = asyncio.run(go())

    assert sid
    assert rec is not None
    assert rec.name == "sweep-test"
    assert rec.optuna_storage.startswith("sqlite:///")
    assert sid in rec.optuna_storage
    assert (tmp_path / "studies").exists()
    assert len(_StubDriver.instances) == 1
    driver = _StubDriver.instances[0]
    assert driver.started
    assert driver.study_id == sid


def test_cancel_unknown_returns_false(monkeypatch, tmp_path):
    _StubDriver.instances.clear()
    monkeypatch.setattr(manager_module, "StudyDriver", _StubDriver)
    monkeypatch.setattr(
        "trainpipe.autoresearch.manager.settings.data_dir", tmp_path
    )

    async def go():
        db = Database(tmp_path / "t.sqlite3")
        await db.init()
        m = StudyManager(db)
        return await m.cancel("never-existed")

    assert asyncio.run(go()) is False


def test_cancel_stops_driver_and_marks_cancelled(monkeypatch, tmp_path):
    _StubDriver.instances.clear()
    monkeypatch.setattr(manager_module, "StudyDriver", _StubDriver)
    monkeypatch.setattr(
        "trainpipe.autoresearch.manager.settings.data_dir", tmp_path
    )

    async def go():
        db = Database(tmp_path / "t.sqlite3")
        await db.init()
        m = StudyManager(db)
        sid = await m.create_and_start(_make_config())
        ok = await m.cancel(sid)
        async with db.connect() as conn:
            rec = await repository.get_study(conn, sid)
        return ok, rec

    ok, rec = asyncio.run(go())
    assert ok is True
    assert rec.status.value == "cancelled"
    assert _StubDriver.instances[0].stopped is True
