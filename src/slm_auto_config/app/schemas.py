"""Request schemas for the Phase 1-3 classification pipeline."""

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
class FinetuneRequest:
    """User request fields needed by the retained Phase 1-3 flow."""

    request_id: str
    task_type: str
    task_description: str
    synthetic_target_count: int
    seed_data: SeedDataSource
    slm_model: str | None = None
    outputs: dict[str, Any] = field(default_factory=lambda: {"run_root": "runs"})
