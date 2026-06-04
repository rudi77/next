"""Unit + integration tests for the eval runner pipeline.

The MockInferenceBackend keeps these tests free of model weights and GPUs.
The real :class:`TransformersInferenceBackend` is exercised only on the
deploy host (separate manual smoke).
"""

import asyncio
import json

import pytest
import pytest_asyncio

from trainpipe.api.schemas import (
    EvalRunStatus,
    ExperimentSpec,
    InferenceParams,
    MetricConfig,
)
from trainpipe.core import repository
from trainpipe.evals.dispatcher import EvalDispatcher
from trainpipe.evals.inference import (
    MockInferenceBackend,
    default_prompt_extractor,
)
from trainpipe.evals.runner import (
    DatasetReadError,
    EvalDriver,
    _instantiate_metrics,
    _load_samples,
)
from trainpipe.scheduler.gpu_pool import GpuPool

# ---------------------------------------------------------------------------
# default_prompt_extractor
# ---------------------------------------------------------------------------


def test_prompt_extractor_messages_uses_last_user():
    sample = {
        "messages": [
            {"role": "user", "content": "first user"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "second user"},
        ]
    }
    assert default_prompt_extractor(sample) == "second user"


def test_prompt_extractor_falls_back_to_prompt():
    assert default_prompt_extractor({"prompt": "p"}) == "p"


def test_prompt_extractor_falls_back_to_query_then_input():
    assert default_prompt_extractor({"query": "q"}) == "q"
    assert default_prompt_extractor({"input": "i"}) == "i"


def test_prompt_extractor_raises_on_missing():
    with pytest.raises(ValueError):
        default_prompt_extractor({"only": "noise"})


# ---------------------------------------------------------------------------
# dataset loading
# ---------------------------------------------------------------------------


def test_load_samples_jsonl(tmp_path):
    f = tmp_path / "evals.jsonl"
    f.write_text(
        '{"q": "a", "gold": "1"}\n{"q": "b", "gold": "2"}\n', encoding="utf-8"
    )
    samples = _load_samples(f, limit=None)
    assert len(samples) == 2
    assert samples[0]["q"] == "a"


def test_load_samples_respects_limit(tmp_path):
    f = tmp_path / "evals.jsonl"
    lines = "\n".join(
        json.dumps({"i": i, "gold": str(i)}) for i in range(10)
    )
    f.write_text(lines + "\n", encoding="utf-8")
    samples = _load_samples(f, limit=3)
    assert [s["i"] for s in samples] == [0, 1, 2]


def test_load_samples_json_list(tmp_path):
    f = tmp_path / "evals.json"
    f.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    samples = _load_samples(f, limit=None)
    assert samples == [{"a": 1}, {"a": 2}]


def test_load_samples_csv(tmp_path):
    f = tmp_path / "evals.csv"
    f.write_text("q,gold\nfoo,1\nbar,2\n", encoding="utf-8")
    samples = _load_samples(f, limit=None)
    assert samples[0] == {"q": "foo", "gold": "1"}


def test_load_samples_unknown_extension(tmp_path):
    f = tmp_path / "evals.xyz"
    f.write_text("anything", encoding="utf-8")
    with pytest.raises(DatasetReadError):
        _load_samples(f, limit=None)


def test_load_samples_broken_jsonl(tmp_path):
    f = tmp_path / "broken.jsonl"
    f.write_text('{"ok": 1}\nthis is not json\n', encoding="utf-8")
    with pytest.raises(DatasetReadError):
        _load_samples(f, limit=None)


def test_load_samples_rejects_parquet(tmp_path):
    # Parquet is a valid *upload* format but the eval runner does in-process
    # streaming reads and deliberately does not support it (spec: eval
    # datasets are jsonl/json/csv/tsv only). The suffix is rejected before any
    # read, so the file contents don't matter.
    f = tmp_path / "evals.parquet"
    f.write_text("not really parquet", encoding="utf-8")
    with pytest.raises(DatasetReadError, match="parquet"):
        _load_samples(f, limit=None)


# ---------------------------------------------------------------------------
# metric instantiation
# ---------------------------------------------------------------------------


def test_instantiate_metrics_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown metric kind"):
        _instantiate_metrics([MetricConfig(kind="never-existed")])


def test_instantiate_metrics_duplicate_name_raises():
    cfgs = [
        MetricConfig(kind="exact_match"),
        MetricConfig(kind="exact_match"),  # same default name → collision
    ]
    with pytest.raises(ValueError, match="duplicate metric name"):
        _instantiate_metrics(cfgs)


def test_instantiate_metrics_same_kind_different_name_ok():
    cfgs = [
        MetricConfig(kind="exact_match", name="em_strict"),
        MetricConfig(
            kind="exact_match", name="em_loose", config={"strip_punctuation": True}
        ),
    ]
    metrics = _instantiate_metrics(cfgs)
    assert [n for n, _ in metrics] == ["em_strict", "em_loose"]


# ---------------------------------------------------------------------------
# EvalDriver end-to-end
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def eval_setup(db, tmp_path):
    """Build a suite + run + 3-sample dataset, ready for EvalDriver."""
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "capital of France", "gold": "Paris"}),
                json.dumps({"prompt": "capital of Germany", "gold": "Berlin"}),
                json.dumps({"prompt": "capital of Italy", "gold": "Rome"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    async with db.connect() as conn:
        suite_id = await repository.create_eval_suite(
            conn,
            name="capitals",
            description=None,
            dataset_path=str(dataset),
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        exp_id = await repository.create_experiment(
            conn, ExperimentSpec(model="m", dataset=["d"]),
        )
        run_id = await repository.create_eval_run(
            conn,
            suite_id=suite_id,
            experiment_id=exp_id,
            model_ref=exp_id,
            triggered_by="manual",
        )
        await repository.claim_eval_run(conn, run_id)
        run = await repository.get_eval_run(conn, run_id)
        suite = await repository.get_eval_suite(conn, suite_id)
    return run, suite, dataset


async def test_eval_driver_all_correct(db, eval_setup):
    run, suite, _ = eval_setup
    backend = MockInferenceBackend(
        responses_by_key={
            "capital of France": "Paris",
            "capital of Germany": "Berlin",
            "capital of Italy": "Rome",
        }
    )
    driver = EvalDriver(db=db, run=run, suite=suite, backend=backend)
    await driver.execute()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run.id)
        results = await repository.list_eval_results(conn, run.id)
    assert finished.status == EvalRunStatus.COMPLETED
    assert finished.sample_count == 3
    assert finished.aggregate["exact_match"].mean == 1.0
    assert len(results) == 3
    assert all(r.scores["exact_match"] == 1.0 for r in results)
    assert backend._opened
    assert backend._closed


async def test_eval_driver_mixed_correctness(db, eval_setup):
    run, suite, _ = eval_setup
    backend = MockInferenceBackend(
        responses_by_key={
            "capital of France": "Paris",
            "capital of Germany": "Munich",  # wrong
            "capital of Italy": "Rome",
        }
    )
    driver = EvalDriver(db=db, run=run, suite=suite, backend=backend)
    await driver.execute()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run.id)
        results = await repository.list_eval_results(conn, run.id)
    assert finished.aggregate["exact_match"].mean == pytest.approx(2 / 3)
    by_idx = {r.sample_index: r for r in results}
    assert by_idx[0].scores["exact_match"] == 1.0
    assert by_idx[1].scores["exact_match"] == 0.0
    assert by_idx[2].scores["exact_match"] == 1.0


async def test_eval_driver_predict_failure_persisted(db, eval_setup):
    run, suite, _ = eval_setup

    def boom(sample, _params):
        if sample["prompt"] == "capital of Germany":
            raise RuntimeError("simulated model crash")
        return sample["gold"]

    backend = MockInferenceBackend(response_fn=boom)
    driver = EvalDriver(db=db, run=run, suite=suite, backend=backend)
    await driver.execute()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run.id)
        results = await repository.list_eval_results(conn, run.id)
    # Run still COMPLETED — per-sample failure shouldn't kill the whole run.
    assert finished.status == EvalRunStatus.COMPLETED
    failed_row = next(r for r in results if r.sample_index == 1)
    assert failed_row.prediction == ""
    assert "simulated model crash" in (failed_row.error or "")
    assert failed_row.scores["exact_match"] == 0.0


async def test_eval_driver_missing_dataset_fails_run(db):
    async with db.connect() as conn:
        sid = await repository.create_eval_suite(
            conn,
            name="no-data",
            description=None,
            dataset_path="/nonexistent/eval.jsonl",
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="x",
            triggered_by="manual",
        )
        await repository.claim_eval_run(conn, run_id)
        run = await repository.get_eval_run(conn, run_id)
        suite = await repository.get_eval_suite(conn, sid)

    backend = MockInferenceBackend()
    driver = EvalDriver(db=db, run=run, suite=suite, backend=backend)
    await driver.execute()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run.id)
    assert finished.status == EvalRunStatus.FAILED
    assert "failed to open dataset" in (finished.error or "")


async def test_eval_driver_unknown_metric_fails_run(db, tmp_path):
    dataset = tmp_path / "x.jsonl"
    dataset.write_text(json.dumps({"prompt": "x", "gold": "y"}) + "\n")
    async with db.connect() as conn:
        sid = await repository.create_eval_suite(
            conn,
            name="bad-metric",
            description=None,
            dataset_path=str(dataset),
            metrics=[MetricConfig(kind="never_existed_metric")],
            inference_params=InferenceParams(),
        )
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="x",
            triggered_by="manual",
        )
        await repository.claim_eval_run(conn, run_id)
        run = await repository.get_eval_run(conn, run_id)
        suite = await repository.get_eval_suite(conn, sid)

    driver = EvalDriver(
        db=db, run=run, suite=suite, backend=MockInferenceBackend(),
    )
    await driver.execute()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run.id)
    assert finished.status == EvalRunStatus.FAILED
    assert "unknown metric kind" in (finished.error or "")


async def test_eval_driver_sample_limit_applied(db, tmp_path):
    dataset = tmp_path / "ten.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"prompt": f"q{i}", "gold": str(i)}) for i in range(10))
        + "\n"
    )
    async with db.connect() as conn:
        sid = await repository.create_eval_suite(
            conn,
            name="limited",
            description=None,
            dataset_path=str(dataset),
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(sample_limit=4),
        )
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="x",
            triggered_by="manual",
        )
        await repository.claim_eval_run(conn, run_id)
        run = await repository.get_eval_run(conn, run_id)
        suite = await repository.get_eval_suite(conn, sid)

    backend = MockInferenceBackend(default_response="0")  # always says "0"
    driver = EvalDriver(db=db, run=run, suite=suite, backend=backend)
    await driver.execute()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run.id)
    assert finished.sample_count == 4
    # Only the first sample's gold is "0" → 1/4 = 0.25
    assert finished.aggregate["exact_match"].mean == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# EvalDispatcher
# ---------------------------------------------------------------------------


async def test_dispatcher_drains_queue(db, tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"prompt": "x", "gold": "y"}) + "\n")
    async with db.connect() as conn:
        sid = await repository.create_eval_suite(
            conn,
            name="d1",
            description=None,
            dataset_path=str(dataset),
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="x",
            triggered_by="manual",
        )

    def factory(_run, _suite):
        return MockInferenceBackend(default_response="y")

    dispatcher = EvalDispatcher(
        db,
        GpuPool([]),
        backend_factory=factory,
        gpus_per_run=0,
        poll_interval_sec=0.05,
    )
    await dispatcher.start()
    try:
        for _ in range(100):
            async with db.connect() as conn:
                r = await repository.get_eval_run(conn, run_id)
            if r.status == EvalRunStatus.COMPLETED:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("dispatcher never completed the queued run")
    finally:
        await dispatcher.stop()

    async with db.connect() as conn:
        finished = await repository.get_eval_run(conn, run_id)
    assert finished.status == EvalRunStatus.COMPLETED
    assert finished.aggregate["exact_match"].mean == 1.0


async def test_dispatcher_requeues_running_on_start(db):
    """Crash recovery: a 'running' eval_run at startup is requeued."""
    async with db.connect() as conn:
        sid = await repository.create_eval_suite(
            conn,
            name="for-recovery",
            description=None,
            dataset_path="/tmp/nothing.jsonl",
            metrics=[MetricConfig(kind="exact_match")],
            inference_params=InferenceParams(),
        )
        run_id = await repository.create_eval_run(
            conn,
            suite_id=sid,
            experiment_id=None,
            model_ref="x",
            triggered_by="manual",
        )
        await repository.claim_eval_run(conn, run_id)

    # Build the dispatcher but don't start its loop — invoke recovery directly.
    async with db.connect() as conn:
        touched = await repository.recover_eval_runs(conn)
        recovered = await repository.get_eval_run(conn, run_id)
    assert touched >= 1
    assert recovered.status == EvalRunStatus.QUEUED
