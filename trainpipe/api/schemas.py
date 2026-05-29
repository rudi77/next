from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SFTType = Literal["lora", "full", "qlora", "longlora", "adalora", "ia3"]


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StudyStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TrainingHyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_length: int | None = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    save_steps: int = 500
    eval_steps: int = 150
    logging_steps: int = 5
    seed: int = 42

    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])


class MultimodalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_factor: int = 8
    max_pixels: int = 602112


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    model: str
    model_type: str | None = None
    sft_type: SFTType = "lora"

    # NB: not enforcing min_length=1 here. Historical rows in the DB may
    # have been written with an empty list (pre-validation), and re-reading
    # them would 500 the list endpoints. Empty-dataset *submits* are blocked
    # in the routes via api.validation.enforce_dataset_not_empty.
    dataset: list[str]
    val_dataset: list[str] = Field(default_factory=list)

    gpu_count: int = Field(1, ge=1, le=8)
    priority: int = 0

    hyperparameters: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    multimodal: MultimodalSettings | None = None

    extra_args: dict[str, Any] = Field(default_factory=dict)

    output_dir: str | None = None

    # Suite IDs to evaluate against after the training run completes.
    # The scheduler enqueues one EvalRun per suite on status=completed.
    auto_eval: list[str] = Field(default_factory=list)


class ExperimentRecord(BaseModel):
    id: str
    spec: ExperimentSpec
    status: ExperimentStatus
    priority: int
    study_id: str | None = None
    trial_number: int | None = None
    gpu_ids: list[int] | None = None
    mlflow_run_id: str | None = None
    mlflow_experiment_id: str | None = None
    log_path: str | None = None
    error: str | None = None
    created_at: datetime
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_heartbeat_at: datetime | None = None


class SearchSpaceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["categorical", "uniform", "loguniform", "int"]
    choices: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    step: float | None = None


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_spec: ExperimentSpec
    search_space: dict[str, SearchSpaceEntry]
    target_metric: str
    direction: Literal["minimize", "maximize"] = "minimize"
    n_trials: int | None = Field(None, ge=1)
    max_concurrent: int = Field(4, ge=1, le=64)
    sampler: Literal["tpe", "random", "cmaes"] = "tpe"


class StudyRecord(BaseModel):
    id: str
    name: str
    config: StudyConfig
    status: StudyStatus
    optuna_storage: str
    n_trials_target: int | None = None
    n_trials_completed: int = 0
    best_value: float | None = None
    best_trial_id: str | None = None
    created_at: datetime
    updated_at: datetime


DatasetFormat = Literal["jsonl", "json", "csv", "tsv", "parquet"]


class Dataset(BaseModel):
    id: str
    name: str
    path: str
    format: DatasetFormat
    line_count: int | None = None
    size_bytes: int
    sha256: str
    description: str | None = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Eval framework (Phase 6)
# ---------------------------------------------------------------------------


class EvalRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


EvalTriggeredBy = Literal["manual", "auto", "study"]


class MetricConfig(BaseModel):
    """One metric to compute on each sample in an eval suite.

    ``kind`` is the registry name (resolved at runtime via the metric plugin
    scan). ``name`` lets the same metric appear twice with different configs
    (e.g. two ``field_level_f1`` blocks scoring different field subsets);
    defaults to ``kind`` if omitted. ``config`` is metric-specific and
    validated by the metric implementation, not here.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., min_length=1)
    name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def metric_name(self) -> str:
        return self.name or self.kind


class InferenceParams(BaseModel):
    """Generation + sampling parameters for the eval runner."""

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(512, ge=1, le=32768)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    sample_limit: int | None = Field(None, ge=1)
    batch_size: int = Field(1, ge=1, le=64)


class EvalSuiteSpec(BaseModel):
    """User-supplied data when creating an eval suite via POST /evals/suites."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str | None = None
    dataset: str = Field(..., min_length=1)
    metrics: list[MetricConfig] = Field(..., min_length=1)
    inference_params: InferenceParams = Field(default_factory=InferenceParams)


class EvalSuite(BaseModel):
    """Persisted eval suite (dataset already resolved to a real path)."""

    id: str
    name: str
    description: str | None = None
    dataset_path: str
    metrics: list[MetricConfig]
    inference_params: InferenceParams
    created_at: datetime


class EvalRunRequest(BaseModel):
    """User-supplied data when triggering an eval run via POST /evals/runs.

    Phase 6 supports only ``experiment_id`` as the model target — the runner
    resolves it to the experiment's adapter output dir. Phase 7's model
    registry will extend this with named-model targets.
    """

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    experiment_id: str
    triggered_by: EvalTriggeredBy = "manual"


class MetricAggregate(BaseModel):
    """Aggregate statistics for one metric over an eval run."""

    model_config = ConfigDict(extra="forbid")

    mean: float
    std: float | None = None
    count: int
    extras: dict[str, Any] = Field(default_factory=dict)


class EvalRun(BaseModel):
    id: str
    suite_id: str
    experiment_id: str | None = None
    model_ref: str
    status: EvalRunStatus
    gpu_ids: list[int] | None = None
    log_path: str | None = None
    error: str | None = None
    aggregate: dict[str, MetricAggregate] | None = None
    sample_count: int | None = None
    triggered_by: EvalTriggeredBy
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class EvalResult(BaseModel):
    id: int
    run_id: str
    sample_index: int
    input: dict[str, Any]
    prediction: str
    gold: dict[str, Any] | None = None
    scores: dict[str, float]
    error: str | None = None
    created_at: datetime


class EvalComparisonSample(BaseModel):
    """One sample's prediction + per-run scores across N runs."""

    model_config = ConfigDict(extra="forbid")

    sample_index: int
    input: dict[str, Any]
    gold: dict[str, Any] | None = None
    per_run: dict[str, dict[str, Any]]  # run_id -> {prediction, scores, error}


class EvalComparison(BaseModel):
    """N-way comparison of eval runs against the same suite."""

    suite_id: str
    runs: list[EvalRun]
    aggregate_delta: dict[str, dict[str, float]]  # metric_name -> {run_id -> mean}
    regressions: list[EvalComparisonSample]  # samples where any run scored lower
