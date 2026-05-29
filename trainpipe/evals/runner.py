"""EvalDriver — runs one eval to completion.

Lifecycle (called by the dispatcher after a successful ``claim_eval_run``):

1. Load the suite's dataset (JSONL/JSON/CSV/TSV/Parquet) up to
   ``sample_limit`` samples.
2. Instantiate each :class:`Metric` from the suite's :class:`MetricConfig`
   list, validating their configs.
3. ``await backend.open()`` — loads model + adapter.
4. For each sample: predict → score with every metric → persist one
   ``eval_results`` row.
5. Aggregate per-metric statistics, finalize the ``eval_runs`` row.
6. Push the aggregates as metrics + tags to the originating experiment's
   MLflow run (best-effort; never fails the eval).
7. ``await backend.close()``.

Errors at any stage flip the run to FAILED with a descriptive ``error``
field. Per-sample errors don't kill the run; they're persisted on the
result row and the metric contributes 0.0 for that sample.
"""

import asyncio
import csv
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..api.schemas import (
    EvalRun,
    EvalRunStatus,
    EvalSuite,
    MetricAggregate,
    MetricConfig,
)
from ..core import repository
from ..core.db import Database
from ..settings import settings
from .inference import InferenceBackend
from .metrics import Metric, UnknownMetricKind, get_metric_class

logger = logging.getLogger(__name__)


_MLFLOW_KEY_SAFE = re.compile(r"[^A-Za-z0-9_./-]")


def _mlflow_key(name: str) -> str:
    """MLflow restricts metric / tag keys to ``[A-Za-z0-9_./-]``.

    We replace anything else with ``_`` so suite/metric names with spaces
    or other oddities don't break the log call.
    """
    return _MLFLOW_KEY_SAFE.sub("_", name)


class DatasetReadError(RuntimeError):
    """Raised when the suite's dataset can't be loaded."""


def _load_samples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read up to ``limit`` records from ``path``. Format inferred from suffix.

    Returns a list of dicts. Lines that fail to parse raise
    :class:`DatasetReadError` with the offending line number so the user
    can fix the file rather than silently dropping samples.
    """
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("jsonl", "ndjson"):
        return list(_take(_iter_jsonl(path), limit))
    if suffix == "json":
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise DatasetReadError(f"failed to load json dataset {path}: {e}") from None
        if not isinstance(data, list):
            raise DatasetReadError("json dataset must be a list of records")
        return list(_take((d for d in data if isinstance(d, dict)), limit))
    if suffix in ("csv", "tsv"):
        delim = "," if suffix == "csv" else "\t"
        return list(_take(_iter_delimited(path, delim), limit))
    raise DatasetReadError(
        f"unsupported eval-dataset extension '.{suffix}' (need jsonl/json/csv/tsv)"
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        f = path.open("r", encoding="utf-8")
    except OSError as e:
        raise DatasetReadError(f"failed to open dataset {path}: {e}") from None
    with f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetReadError(
                    f"jsonl line {lineno} not valid JSON: {e}"
                ) from None
            if not isinstance(obj, dict):
                raise DatasetReadError(
                    f"jsonl line {lineno} must decode to an object, got {type(obj).__name__}"
                )
            yield obj


def _iter_delimited(path: Path, delim: str) -> Iterable[dict[str, Any]]:
    try:
        f = path.open("r", encoding="utf-8", newline="")
    except OSError as e:
        raise DatasetReadError(f"failed to open dataset {path}: {e}") from None
    with f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            yield dict(row)


def _take(it: Iterable[dict[str, Any]], limit: int | None) -> Iterable[dict[str, Any]]:
    if limit is None:
        yield from it
        return
    for i, item in enumerate(it):
        if i >= limit:
            return
        yield item


def _instantiate_metrics(configs: list[MetricConfig]) -> list[tuple[str, Metric]]:
    """Resolve each MetricConfig to a (display_name, Metric instance) pair.

    Raises ValueError if a kind is unknown or a config is invalid.
    """
    seen: set[str] = set()
    out: list[tuple[str, Metric]] = []
    for cfg in configs:
        try:
            cls = get_metric_class(cfg.kind)
        except UnknownMetricKind as e:
            raise ValueError(str(e)) from None
        instance = cls(cfg.config)
        name = cfg.metric_name
        if name in seen:
            raise ValueError(
                f"duplicate metric name '{name}' in eval suite — set MetricConfig.name "
                "to disambiguate when using the same kind twice"
            )
        seen.add(name)
        out.append((name, instance))
    return out


class EvalDriver:
    """Runs one queued (already-claimed) eval to completion."""

    def __init__(
        self,
        *,
        db: Database,
        run: EvalRun,
        suite: EvalSuite,
        backend: InferenceBackend,
    ) -> None:
        self.db = db
        self.run = run
        self.suite = suite
        self.backend = backend

    async def execute(self) -> None:
        try:
            metrics = _instantiate_metrics(self.suite.metrics)
        except ValueError as e:
            await self._fail(f"metric setup failed: {e}")
            return

        try:
            samples = _load_samples(
                Path(self.suite.dataset_path),
                self.suite.inference_params.sample_limit,
            )
        except DatasetReadError as e:
            await self._fail(str(e))
            return

        if not samples:
            await self._fail("eval dataset is empty")
            return

        per_metric_scores: dict[str, list[float]] = {n: [] for n, _ in metrics}

        try:
            await self.backend.open()
        except Exception as e:
            logger.exception("backend.open failed for eval run=%s", self.run.id)
            await self._fail(f"inference backend failed to open: {e}")
            return

        try:
            for idx, sample in enumerate(samples):
                prediction, predict_err = await self._predict_one(sample)
                scores, sample_err = self._score_one(
                    prediction, sample, metrics, per_metric_scores
                )
                gold = sample.get("gold") if isinstance(sample.get("gold"), dict) else None
                async with self.db.connect() as conn:
                    await repository.add_eval_result(
                        conn,
                        run_id=self.run.id,
                        sample_index=idx,
                        input=sample,
                        prediction=prediction,
                        gold=gold,
                        scores=scores,
                        error=predict_err or sample_err,
                    )

            aggregate = self._aggregate(metrics, per_metric_scores)
            async with self.db.connect() as conn:
                await repository.finalize_eval_run(
                    conn,
                    self.run.id,
                    status=EvalRunStatus.COMPLETED,
                    aggregate=aggregate,
                    sample_count=len(samples),
                )
            await self._publish_aggregates_to_mlflow(aggregate)
        except Exception as e:
            logger.exception("eval run=%s crashed mid-execution", self.run.id)
            await self._fail(f"unexpected runner error: {e}")
        finally:
            try:
                await self.backend.close()
            except Exception:
                logger.exception(
                    "backend.close raised for eval run=%s (ignored)", self.run.id
                )

    async def _predict_one(
        self, sample: dict[str, Any]
    ) -> tuple[str, str | None]:
        try:
            return await self.backend.predict(sample, self.suite.inference_params), None
        except Exception as e:
            logger.warning(
                "predict failed for run=%s sample=%s: %s",
                self.run.id,
                sample.get("id", "?"),
                e,
            )
            return "", f"predict failed: {e}"

    @staticmethod
    def _score_one(
        prediction: str,
        sample: dict[str, Any],
        metrics: list[tuple[str, Metric]],
        per_metric_scores: dict[str, list[float]],
    ) -> tuple[dict[str, float], str | None]:
        scores: dict[str, float] = {}
        errs: list[str] = []
        for name, metric in metrics:
            try:
                value = float(metric.score(prediction, sample))
            except Exception as e:
                logger.warning("metric %s raised on sample: %s", name, e)
                value = 0.0
                errs.append(f"{name}: {e}")
            scores[name] = value
            per_metric_scores[name].append(value)
        return scores, "; ".join(errs) if errs else None

    @staticmethod
    def _aggregate(
        metrics: list[tuple[str, Metric]],
        per_metric_scores: dict[str, list[float]],
    ) -> dict[str, MetricAggregate]:
        return {
            name: metric.aggregate(per_metric_scores[name]) for name, metric in metrics
        }

    async def _fail(self, error: str) -> None:
        async with self.db.connect() as conn:
            await repository.finalize_eval_run(
                conn,
                self.run.id,
                status=EvalRunStatus.FAILED,
                error=error,
            )

    async def _publish_aggregates_to_mlflow(
        self, aggregate: dict[str, MetricAggregate]
    ) -> None:
        """Push per-metric mean scores to the originating experiment's MLflow run.

        Best-effort: if the eval wasn't triggered by an experiment, or the
        experiment never produced an MLflow run, or the MLflow server is
        unreachable, the eval still finalizes normally — we just log a
        warning. The metrics surface in the MLflow UI as ``eval.<suite>.<metric>``
        so users can sort / filter their experiment table by them.
        """
        if not self.run.experiment_id:
            return
        async with self.db.connect() as conn:
            exp = await repository.get_experiment(conn, self.run.experiment_id)
        if exp is None or not exp.mlflow_run_id:
            return
        try:
            await asyncio.to_thread(
                _log_eval_to_mlflow,
                exp.mlflow_run_id,
                self.suite.name,
                self.run.id,
                aggregate,
            )
        except Exception:
            logger.warning(
                "mlflow publish failed for eval run=%s (run still finalized)",
                self.run.id,
                exc_info=True,
            )


def _log_eval_to_mlflow(
    mlflow_run_id: str,
    suite_name: str,
    eval_run_id: str,
    aggregate: dict[str, MetricAggregate],
) -> None:
    """Synchronous MLflow client call. Imported lazily to keep the cold path light.

    For each metric ``m`` we log:

    * metric ``eval.<suite>.<m>``        — the mean (sortable in MLflow UI)
    * metric ``eval.<suite>.<m>.std``    — standard deviation
    * metric ``eval.<suite>.<m>.count``  — sample count
    * tag ``trainpipe.eval.<suite>``     — set to the eval_run_id so users
      can drill back from MLflow to the eval run
    * tag ``trainpipe.eval.<suite>.completed`` — ``"true"`` so MLflow
      search like ``tags."trainpipe.eval.foo.completed" = "true"`` works
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    safe_suite = _mlflow_key(suite_name)
    for metric_name, agg in aggregate.items():
        safe_metric = _mlflow_key(metric_name)
        base = f"eval.{safe_suite}.{safe_metric}"
        client.log_metric(mlflow_run_id, base, float(agg.mean))
        if agg.std is not None:
            client.log_metric(mlflow_run_id, f"{base}.std", float(agg.std))
        client.log_metric(mlflow_run_id, f"{base}.count", float(agg.count))
    client.set_tag(
        mlflow_run_id, f"trainpipe.eval.{safe_suite}", eval_run_id,
    )
    client.set_tag(
        mlflow_run_id, f"trainpipe.eval.{safe_suite}.completed", "true",
    )
