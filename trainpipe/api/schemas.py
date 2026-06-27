from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SFTType = Literal["lora", "full", "qlora", "longlora", "adalora", "ia3"]

# Phase 13 — high-level training mode. SFT is the default
# instruction-tuning path; ``pt`` is (continued) pretraining via
# ``swift pt`` on raw-text corpora; the *PO family runs via
# ``swift rlhf`` with the matching ``--rlhf_type``.
TrainKind = Literal["sft", "pt", "dpo", "kto", "ppo", "grpo"]

# Train kinds that run through ``swift rlhf``.
RLHF_KINDS = frozenset({"dpo", "kto", "ppo", "grpo"})
# RL kinds that optimise against a reward signal (vs. fixed preference
# pairs): they need a reward model and/or reward functions to even start.
RL_REWARD_KINDS = frozenset({"ppo", "grpo"})


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


class RLHFHyperparameters(BaseModel):
    """RL / preference-tuning knobs for the ``swift rlhf`` family (Phase 13).

    Only consumed when ``train_kind`` is one of dpo/kto/ppo/grpo. Each field
    maps to an ms-swift v4 ``swift rlhf`` flag; ``None`` means "leave the
    trainer default". The cross-field rules (e.g. GRPO/PPO need a reward
    signal) are enforced by :class:`ExperimentSpec`'s validator so a bad
    spec is rejected at submit time instead of failing minutes into a run.
    """

    model_config = ConfigDict(extra="forbid")

    # KL-regularisation coefficient. DPO/KTO/GRPO use it; PPO ignores it.
    beta: float | None = Field(None, ge=0.0)
    # Reward-model id/path for the RL kinds (--reward_model). Required for
    # PPO; one of reward_model / reward_funcs is required for GRPO.
    reward_model: str | None = None
    # GRPO built-in reward functions (e.g. ["accuracy", "format"]) →
    # --reward_funcs. GRPO-only.
    reward_funcs: list[str] = Field(default_factory=list)
    # GRPO group size: completions sampled per prompt (--num_generations).
    # GRPO-only; must be >= 2 for a meaningful group advantage.
    num_generations: int | None = Field(None, ge=2)
    # Rollout completion length for GRPO/PPO (--max_completion_length).
    max_completion_length: int | None = Field(None, ge=1)
    # Sampling temperature for rollouts (--temperature). GRPO/PPO.
    temperature: float | None = Field(None, ge=0.0, le=2.0)


class MultimodalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_factor: int = 8
    max_pixels: int = 602112


class DistributedConfig(BaseModel):
    """Distributed training settings (Phase 18).

    ``deepspeed_zero_stage``: 0 → off; 1/2/3 → enable ZeRO at that stage
    via ``--deepspeed_zero<N>`` to ms-swift. Stage 3 is most aggressive
    (partitions parameters + grads + optimizer states across GPUs).

    ``num_nodes`` > 1 + ``host_list``: switches the launcher to
    ``torchrun`` with the appropriate ``--nnodes`` / ``--node_rank``;
    actual multi-host orchestration (SSH spawn on every host) lives at
    the operator level today — the spec records intent, the scheduler
    surfaces it as an env variable so the operator's launcher can pick
    it up. Full Kubernetes-managed distribution is out of scope.
    """

    model_config = ConfigDict(extra="forbid")

    deepspeed_zero_stage: int = Field(0, ge=0, le=3)
    num_nodes: int = Field(1, ge=1, le=64)
    # SSH hosts in torchrun order. Empty for single-host runs.
    host_list: list[str] = Field(default_factory=list)
    # Master address for multi-node; required when num_nodes > 1.
    master_addr: str | None = None
    master_port: int = Field(29500, ge=1024, le=65535)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    model: str
    model_type: str | None = None
    sft_type: SFTType = "lora"
    # Phase 13. Default ``sft`` keeps old specs working unchanged.
    train_kind: TrainKind = "sft"

    # NB: not enforcing min_length=1 here. Historical rows in the DB may
    # have been written with an empty list (pre-validation), and re-reading
    # them would 500 the list endpoints. Empty-dataset *submits* are blocked
    # in the routes via api.validation.enforce_dataset_not_empty.
    dataset: list[str]
    val_dataset: list[str] = Field(default_factory=list)

    gpu_count: int = Field(1, ge=1, le=8)
    priority: int = 0

    hyperparameters: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    # RL / preference knobs; only valid when train_kind is in RLHF_KINDS.
    rlhf: RLHFHyperparameters | None = None
    multimodal: MultimodalSettings | None = None
    distributed: DistributedConfig | None = None
    # Phase 21: domain vocab extension. Each entry is added as a special
    # token via ms-swift's ``--special_tokens`` flag; the model's
    # embedding layer is resized accordingly.
    extra_tokens: list[str] = Field(
        default_factory=list,
        max_length=10000,
        description=(
            "Additional special tokens for the tokenizer (e.g. "
            '["[INV_HEAD]", "[INV_FOOT]"]). Resizes the embedding layer.'
        ),
    )

    extra_args: dict[str, Any] = Field(default_factory=dict)

    output_dir: str | None = None

    # Suite IDs to evaluate against after the training run completes.
    # The scheduler enqueues one EvalRun per suite on status=completed.
    auto_eval: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_rlhf(self) -> "ExperimentSpec":
        """Keep RL config and train_kind consistent, and require a reward
        signal for the RL kinds so PPO/GRPO can't be queued in a state that
        only fails once the trainer is already spinning."""
        rlhf = self.rlhf
        if self.train_kind not in RLHF_KINDS:
            if rlhf is not None:
                kinds = ", ".join(sorted(RLHF_KINDS))
                raise ValueError(
                    "rlhf settings are only valid for preference/RL train_kinds "
                    f"({kinds}), not {self.train_kind}"
                )
            return self

        # GRPO-only knobs must not leak onto other kinds.
        if rlhf is not None and self.train_kind != "grpo":
            if rlhf.reward_funcs:
                raise ValueError("rlhf.reward_funcs is only valid for train_kind=grpo")
            if rlhf.num_generations is not None:
                raise ValueError("rlhf.num_generations is only valid for train_kind=grpo")

        # PPO/GRPO optimise against a reward and can't start without one.
        if self.train_kind in RL_REWARD_KINDS:
            has_reward_model = rlhf is not None and rlhf.reward_model is not None
            has_reward_funcs = rlhf is not None and bool(rlhf.reward_funcs)
            if self.train_kind == "ppo" and not has_reward_model:
                raise ValueError("train_kind=ppo requires rlhf.reward_model")
            if self.train_kind == "grpo" and not (has_reward_model or has_reward_funcs):
                raise ValueError(
                    "train_kind=grpo requires rlhf.reward_model or rlhf.reward_funcs"
                )
        return self


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
    # Phase 20:
    gpu_seconds: float | None = None
    peak_vram_mb: float | None = None
    energy_wh: float | None = None


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
    # Phase 9 multimodal:
    media_kinds: list[str] = Field(default_factory=list)
    image_root: str | None = None
    # Phase 16 versioning + derivation:
    version: int = 1
    derived_from: str | None = None


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


# ---------------------------------------------------------------------------
# Model registry (Phase 7)
# ---------------------------------------------------------------------------


class ModelRegisterRequest(BaseModel):
    """Register an experiment's completed run as a named, versioned model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    experiment_id: str = Field(..., min_length=1)
    description: str | None = None
    # Optional explicit version; auto-incremented within ``name`` if omitted.
    version: int | None = Field(None, ge=1)
    # Optional alias to set after register (e.g. "staging" / "production").
    alias: str | None = None


class RegisteredModel(BaseModel):
    """A persisted named model version pointing at an experiment's adapter dir."""

    id: str
    name: str
    version: int
    run_id: str | None = None
    experiment_id: str | None = None
    base_model: str
    adapter_path: str | None = None
    eval_summary: dict[str, Any] | None = None
    description: str | None = None
    created_at: datetime
    aliases: list[str] = Field(default_factory=list)


class ModelAlias(BaseModel):
    name: str
    alias: str
    model_id: str
    updated_at: datetime


# ---------------------------------------------------------------------------
# Active learning (Phase 11)
# ---------------------------------------------------------------------------


class ALRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActiveLearningRunRequest(BaseModel):
    """Submit an active-learning pass.

    The runner loads the model behind ``model_ref`` (see Phase 8 ref
    syntax), reads ``dataset_path`` (a ``ds:<id>`` ref or path), scores
    every sample with the configured ``UncertaintyScorer``, and surfaces
    the top ``top_n`` as an annotation queue.
    """

    model_config = ConfigDict(extra="forbid")

    model_ref: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1, description="ds:<id> or path")
    top_n: int = Field(50, ge=1, le=10000)
    sample_limit: int | None = Field(None, ge=1)
    # "double_pass": two T=0.7 samples + diff
    # "length_zscore": deviation from mean response length
    scorer: Literal["double_pass", "length_zscore"] = "double_pass"


class ActiveLearningRun(BaseModel):
    id: str
    model_ref: str
    dataset_path: str
    top_n: int
    sample_limit: int | None = None
    status: ALRunStatus
    error: str | None = None
    scored_count: int | None = None
    queued_count: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AnnotationQueueItem(BaseModel):
    id: int
    run_id: str
    sample_index: int
    input: dict[str, Any]
    prediction: str
    uncertainty: float
    annotated: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# Multi-stage pipelines (Phase 12)
# ---------------------------------------------------------------------------


class PipelineStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class StageSpec(BaseModel):
    """One stage of a pipeline.

    ``base_spec`` is a full :class:`ExperimentSpec`. ``input_from_stage``
    references a sibling stage by name; the driver will rewrite
    ``base_spec.model`` to point at that stage's adapter dir before
    enqueuing. ``depends_on`` is the strict-ordering DAG edge.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=64)
    base_spec: ExperimentSpec
    depends_on: list[str] = Field(default_factory=list)
    # Take the adapter dir from this stage and feed it as ``--model`` /
    # ``--adapter_name_or_path`` for this one. Optional; if omitted the
    # stage uses ``base_spec.model`` as-is.
    input_from_stage: str | None = None


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    stages: list[StageSpec] = Field(..., min_length=1, max_length=16)


class PipelineStage(BaseModel):
    stage_name: str
    stage_index: int
    depends_on: list[str] = Field(default_factory=list)
    experiment_id: str | None = None
    status: StageStatus
    output_dir: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Pipeline(BaseModel):
    id: str
    name: str
    status: PipelineStatus
    config: PipelineConfig
    stages: list[PipelineStage]
    error: str | None = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Watches (Phase 17)
# ---------------------------------------------------------------------------


WatchKind = Literal["interval", "metric_threshold"]


class WatchCreateRequest(BaseModel):
    """Create a watch that fires a pipeline on schedule or on drift."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    kind: WatchKind
    pipeline_config: PipelineConfig

    # Required when kind=interval:
    interval_seconds: int | None = Field(None, ge=60, le=86400 * 30)

    # Required when kind=metric_threshold:
    model_name: str | None = None
    suite_id: str | None = None
    metric_name: str | None = None
    threshold: float | None = None


class Watch(BaseModel):
    id: str
    name: str
    kind: WatchKind
    enabled: bool
    interval_seconds: int | None = None
    model_name: str | None = None
    suite_id: str | None = None
    metric_name: str | None = None
    threshold: float | None = None
    pipeline_config: PipelineConfig
    last_fired_at: datetime | None = None
    last_fired_pipeline_id: str | None = None
    created_at: datetime
    # Auto-disable bookkeeping (migration v12). Reset to 0 on a
    # successful fire; bumped on each fire attempt that raised. Once
    # it reaches WatchManager.failure_disable_threshold the watch
    # flips ``enabled = False`` automatically and the operator has
    # to /enable it after fixing the config.
    consecutive_failures: int = 0
    last_error: str | None = None
