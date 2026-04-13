"""Build a dry-run Phase 1-3 pipeline decision plan from a user request."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_auto_config.app.schemas import FinetuneRequest, SeedDataSource


SUPPORTED_TASK_TYPES = {
    "classification",
    "ner",
    "qa",
    "extraction",
    "ranking",
    "function_calling",
}


@dataclass(frozen=True)
class PipelineDecisionPlan:
    """A dry-run description of the retained Phase 1-3 branches."""

    request_id: str
    task_type: str
    task_description: str
    synthetic_target_count: int
    seed_data_contract: dict[str, Any]
    sdg_stage: dict[str, Any]
    dataset_stage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_user_request(path: str | Path) -> FinetuneRequest:
    """Load a simulated user request from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FinetuneRequest(
        request_id=payload["request_id"],
        task_type=payload["task_type"],
        task_description=payload["task_description"],
        synthetic_target_count=payload["synthetic_target_count"],
        seed_data=SeedDataSource(**payload["seed_data"]),
        slm_model=payload.get("slm_model"),
        outputs=payload.get("outputs", {"run_root": "runs"}),
    )


def build_decision_plan(request: FinetuneRequest) -> PipelineDecisionPlan:
    """Validate a request and choose the pipeline branches for a dry run."""
    _validate_request(request)

    run_id = _slugify(request.request_id)
    run_root = _join_path(request.outputs.get("run_root", "runs"), run_id)
    task_name = request.task_type

    return PipelineDecisionPlan(
        request_id=request.request_id,
        task_type=task_name,
        task_description=request.task_description,
        synthetic_target_count=request.synthetic_target_count,
        seed_data_contract={
            "source_type": request.seed_data.type,
            "path": request.seed_data.path,
            "format": request.seed_data.format,
            "text_field": request.seed_data.text_field,
            "label_field": request.seed_data.label_field,
            "expected_record_shape": _expected_seed_shape(task_name),
        },
        sdg_stage={
            "router": "sdg_router",
            "selected_pipeline": f"{task_name}_sdg",
            "implementation": "slm_auto_config.synthetic.classification_sdg"
            if task_name == "classification"
            else "pending",
            "output_path": _join_path(run_root, "sdg", "synthetic_data.json"),
            "report_path": _join_path(run_root, "sdg", "sdg_report.json"),
        },
        dataset_stage={
            "strategy": "stratified_split" if task_name == "classification" else "task_adapter_split",
            "train_path": _join_path(run_root, "input", "train.jsonl"),
            "validation_path": _join_path(run_root, "input", "validation.jsonl"),
            "test_path": _join_path(run_root, "input", "test.jsonl"),
            "record_paths": {
                "train": _join_path(run_root, "input", "train.records.jsonl"),
                "validation": _join_path(run_root, "input", "validation.records.jsonl"),
                "test": _join_path(run_root, "input", "test.records.jsonl"),
            },
            "prompt_template_path": _join_path(
                run_root,
                "input",
                "classification_prompt_template.json",
            ),
            "few_shot_examples_path": _join_path(
                run_root,
                "input",
                "classification_few_shot_examples.json",
            ),
        },
    )


def _validate_request(request: FinetuneRequest) -> None:
    if request.task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {request.task_type}")
    if not request.task_description.strip():
        raise ValueError("task_description is required.")
    if request.synthetic_target_count <= 0:
        raise ValueError("synthetic_target_count must be greater than zero.")
    if request.seed_data.type == "file" and not request.seed_data.path:
        raise ValueError("seed_data.path is required when seed_data.type is file.")
    if request.seed_data.type == "inline" and not request.seed_data.records:
        raise ValueError("seed_data.records is required when seed_data.type is inline.")


def _expected_seed_shape(task_type: str) -> dict[str, str]:
    if task_type == "classification":
        return {"text": "str", "label": "str"}
    return {"task_specific_fields": "pending"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "request"


def _join_path(*parts: str) -> str:
    return Path(*parts).as_posix()
