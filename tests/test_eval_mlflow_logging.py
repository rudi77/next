"""Tests for the MLflow publishing path in EvalDriver.

We don't talk to a real MLflow server. The integration goes through
``trainpipe.evals.runner._log_eval_to_mlflow``; tests monkeypatch it to
record the call and verify the driver invokes it with the right args
under the right conditions, without breaking on MLflow exceptions.
"""

import json

import pytest_asyncio

from trainpipe.api.schemas import (
    EvalRunStatus,
    ExperimentSpec,
    InferenceParams,
    MetricAggregate,
    MetricConfig,
)
from trainpipe.core import repository
from trainpipe.evals import runner as runner_mod
from trainpipe.evals.inference import MockInferenceBackend
from trainpipe.evals.runner import EvalDriver, _log_eval_to_mlflow, _mlflow_key


def test_mlflow_key_normalizes_unsafe_chars():
    assert _mlflow_key("my suite") == "my_suite"
    assert _mlflow_key("OK_chars.are/kept-1") == "OK_chars.are/kept-1"
    assert _mlflow_key("with:colons!") == "with_colons_"


@pytest_asyncio.fixture
async def setup(db, tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "x1", "gold": "right"}),
                json.dumps({"prompt": "x2", "gold": "right"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    async with db.connect() as conn:
        suite_id = await repository.create_eval_suite(
            conn,
            name="ml flow suite",  # space → should get normalized
            description=None,
            dataset_path=str(dataset),
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        exp_id = await repository.create_experiment(
            conn, ExperimentSpec(name="parent", model="m", dataset=["d"]),
        )
        # Pretend the training scheduler already created an MLflow run.
        await conn.execute(
            "UPDATE experiments SET mlflow_run_id = ?, mlflow_experiment_id = ? "
            "WHERE id = ?",
            ("mlflow-run-abc", "exp-1", exp_id),
        )
        await conn.commit()
        run_id = await repository.create_eval_run(
            conn,
            suite_id=suite_id,
            experiment_id=exp_id,
            model_ref=exp_id,
            triggered_by="auto",
        )
        await repository.claim_eval_run(conn, run_id)
        run = await repository.get_eval_run(conn, run_id)
        suite = await repository.get_eval_suite(conn, suite_id)
    return run, suite, exp_id


async def test_mlflow_logging_called_with_expected_args(db, setup, monkeypatch):
    run, suite, _exp_id = setup
    calls: list[dict] = []

    def fake_log(mlflow_run_id, suite_name, eval_run_id, aggregate):
        calls.append(
            {
                "mlflow_run_id": mlflow_run_id,
                "suite_name": suite_name,
                "eval_run_id": eval_run_id,
                "aggregate_keys": sorted(aggregate.keys()),
                "exact_match_mean": aggregate["exact_match"].mean,
            }
        )

    monkeypatch.setattr(runner_mod, "_log_eval_to_mlflow", fake_log)

    backend = MockInferenceBackend(default_response="right")
    driver = EvalDriver(db=db, run=run, suite=suite, backend=backend)
    await driver.execute()

    assert len(calls) == 1
    call = calls[0]
    assert call["mlflow_run_id"] == "mlflow-run-abc"
    assert call["suite_name"] == "ml flow suite"  # raw — _mlflow_key applied later
    assert call["eval_run_id"] == run.id
    assert call["aggregate_keys"] == ["exact_match"]
    assert call["exact_match_mean"] == 1.0


async def test_mlflow_logging_skipped_when_no_experiment(db, tmp_path, monkeypatch):
    dataset = tmp_path / "x.jsonl"
    dataset.write_text(json.dumps({"prompt": "p", "gold": "y"}) + "\n")
    async with db.connect() as conn:
        sid = await repository.create_eval_suite(
            conn,
            name="no-exp",
            description=None,
            dataset_path=str(dataset),
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        rid = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="x",
            triggered_by="manual",
        )
        await repository.claim_eval_run(conn, rid)
        run = await repository.get_eval_run(conn, rid)
        suite = await repository.get_eval_suite(conn, sid)

    called = []
    monkeypatch.setattr(
        runner_mod, "_log_eval_to_mlflow", lambda *a, **k: called.append(a),
    )

    driver = EvalDriver(
        db=db, run=run, suite=suite, backend=MockInferenceBackend(default_response="y"),
    )
    await driver.execute()
    assert called == []


async def test_mlflow_logging_skipped_when_experiment_lacks_run_id(
    db, tmp_path, monkeypatch
):
    dataset = tmp_path / "x.jsonl"
    dataset.write_text(json.dumps({"prompt": "p", "gold": "y"}) + "\n")
    async with db.connect() as conn:
        sid = await repository.create_eval_suite(
            conn,
            name="no-mlflow",
            description=None,
            dataset_path=str(dataset),
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        exp_id = await repository.create_experiment(
            conn, ExperimentSpec(model="m", dataset=["d"]),
        )
        # NB: don't set mlflow_run_id — simulating an experiment that
        # finished before MLflow was wired up (or one that failed to log).
        rid = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=exp_id,
            model_ref="x",
            triggered_by="auto",
        )
        await repository.claim_eval_run(conn, rid)
        run = await repository.get_eval_run(conn, rid)
        suite = await repository.get_eval_suite(conn, sid)

    called = []
    monkeypatch.setattr(
        runner_mod, "_log_eval_to_mlflow", lambda *a, **k: called.append(a),
    )

    driver = EvalDriver(
        db=db, run=run, suite=suite, backend=MockInferenceBackend(default_response="y"),
    )
    await driver.execute()
    assert called == []


async def test_mlflow_exception_does_not_fail_eval(db, setup, monkeypatch, caplog):
    run, suite, _exp_id = setup

    def explode(*_args, **_kwargs):
        raise RuntimeError("mlflow connection refused")

    monkeypatch.setattr(runner_mod, "_log_eval_to_mlflow", explode)

    with caplog.at_level("WARNING"):
        await EvalDriver(
            db=db, run=run, suite=suite,
            backend=MockInferenceBackend(default_response="right"),
        ).execute()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run.id)
    # Run should still be COMPLETED — MLflow failure must not poison it.
    assert finished.status == EvalRunStatus.COMPLETED
    assert finished.aggregate["exact_match"].mean == 1.0
    assert any("mlflow publish failed" in r.message for r in caplog.records)


async def test_mlflow_not_called_on_failed_run(db, tmp_path, monkeypatch):
    """If the eval fails (no aggregates ever computed), we don't push."""
    async with db.connect() as conn:
        sid = await repository.create_eval_suite(
            conn,
            name="will-fail",
            description=None,
            dataset_path="/nonexistent/x.jsonl",
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        exp_id = await repository.create_experiment(
            conn, ExperimentSpec(model="m", dataset=["d"]),
        )
        await conn.execute(
            "UPDATE experiments SET mlflow_run_id = ? WHERE id = ?",
            ("mlflow-run-fail", exp_id),
        )
        await conn.commit()
        rid = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=exp_id,
            model_ref=exp_id,
            triggered_by="auto",
        )
        await repository.claim_eval_run(conn, rid)
        run = await repository.get_eval_run(conn, rid)
        suite = await repository.get_eval_suite(conn, sid)

    called = []
    monkeypatch.setattr(
        runner_mod, "_log_eval_to_mlflow", lambda *a, **k: called.append(a),
    )

    driver = EvalDriver(
        db=db, run=run, suite=suite, backend=MockInferenceBackend(),
    )
    await driver.execute()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run.id)
    assert finished.status == EvalRunStatus.FAILED
    assert called == []


def test_log_eval_to_mlflow_makes_expected_calls(monkeypatch):
    """Verify the sync helper itself: which client methods it calls,
    with which key shapes. We swap mlflow.tracking.MlflowClient for a
    recording double."""
    import sys
    import types

    recorded: dict[str, list] = {"metric": [], "tag": []}

    class FakeClient:
        def log_metric(self, run_id, key, value):
            recorded["metric"].append((run_id, key, value))

        def set_tag(self, run_id, key, value):
            recorded["tag"].append((run_id, key, value))

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda _uri: None  # type: ignore[attr-defined]
    fake_tracking = types.ModuleType("mlflow.tracking")
    fake_tracking.MlflowClient = FakeClient  # type: ignore[attr-defined]
    fake_mlflow.tracking = fake_tracking  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

    _log_eval_to_mlflow(
        "ml-run-1",
        "my suite",  # space — should be normalized to "my_suite"
        "eval-id-7",
        {
            "exact_match": MetricAggregate(mean=0.83, std=0.12, count=100),
            "rouge_l": MetricAggregate(mean=0.45, std=None, count=100),
        },
    )

    metric_keys = sorted(k for _, k, _ in recorded["metric"])
    # mean + count for each (no std for rouge_l since std=None)
    assert "eval.my_suite.exact_match" in metric_keys
    assert "eval.my_suite.exact_match.std" in metric_keys
    assert "eval.my_suite.exact_match.count" in metric_keys
    assert "eval.my_suite.rouge_l" in metric_keys
    assert "eval.my_suite.rouge_l.count" in metric_keys
    assert "eval.my_suite.rouge_l.std" not in metric_keys

    tag_keys = sorted(k for _, k, _ in recorded["tag"])
    assert tag_keys == [
        "trainpipe.eval.my_suite",
        "trainpipe.eval.my_suite.completed",
    ]
    # eval_run_id tag value is the eval run id
    tag_id_value = next(v for _, k, v in recorded["tag"] if k == "trainpipe.eval.my_suite")
    assert tag_id_value == "eval-id-7"
