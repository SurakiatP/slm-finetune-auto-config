"""Request and response schemas for the product API."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SeedDataSource:
    """Where user-provided seed examples are stored after upload."""

    type: str
    path: str | None = None
    records: list[dict[str, Any]] | None = None
    format: str = "json"
    text_field: str = "text"
    label_field: str = "label"


@dataclass(frozen=True)
class TrainingPreference:
    """User-facing training preference."""

    backend: str = "unsloth"
    method: str = "qlora"


@dataclass(frozen=True)
class AutoConfigPreference:
    """Auto hyperparameter tuning preference."""

    controller: str = "oumi_style"
    tuning_budget: str = "small_budget"
    max_trials: int = 6
    primary_objective: str = "task_metric"


@dataclass(frozen=True)
class FinetuneRequest:
    """User-facing fine-tuning request."""

    request_id: str
    task_type: str
    task_description: str
    synthetic_target_count: int
    slm_model: str
    config_mode: str
    seed_data: SeedDataSource
    training: TrainingPreference = field(default_factory=TrainingPreference)
    auto_config: AutoConfigPreference | None = field(default_factory=AutoConfigPreference)
    manual_config: dict[str, Any] | None = None
    outputs: dict[str, Any] = field(default_factory=lambda: {"run_root": "runs"})
