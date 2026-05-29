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

    dataset: list[str] = Field(..., min_length=1)
    val_dataset: list[str] = Field(default_factory=list)

    gpu_count: int = Field(1, ge=1, le=8)
    priority: int = 0

    hyperparameters: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    multimodal: MultimodalSettings | None = None

    extra_args: dict[str, Any] = Field(default_factory=dict)

    output_dir: str | None = None


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
