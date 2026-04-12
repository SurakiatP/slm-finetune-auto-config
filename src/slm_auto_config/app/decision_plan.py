"""Build a dry-run pipeline decision plan from a user request."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_auto_config.app.schemas import (
    AutoConfigPreference,
    FinetuneRequest,
    SeedDataSource,
    TrainingPreference,
)


SUPPORTED_TASK_TYPES = {
    "classification",
    "ner",
    "qa",
    "extraction",
    "ranking",
    "function_calling",
}
SUPPORTED_CONFIG_MODES = {"manual", "auto"}


@dataclass(frozen=True)
class PipelineDecisionPlan:
    """A dry-run description of which pipeline branches will be used."""

    request_id: str
    task_type: str
    task_description: str
    synthetic_target_count: int
    seed_data_contract: dict[str, Any]
    sdg_stage: dict[str, Any]
    dataset_stage: dict[str, Any]
    config_stage: dict[str, Any]
    training_stage: dict[str, Any]
    evaluation_stage: dict[str, Any]
    inference_stage: dict[str, Any]
    export_stage: dict[str, Any]

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
        slm_model=payload["slm_model"],
        config_mode=payload["config_mode"],
        seed_data=SeedDataSource(**payload["seed_data"]),
        training=TrainingPreference(**payload.get("training", {})),
        auto_config=AutoConfigPreference(**payload["auto_config"])
        if payload.get("auto_config")
        else None,
        manual_config=payload.get("manual_config"),
        outputs=payload.get("outputs", {"run_root": "runs"}),
    )


def build_decision_plan(request: FinetuneRequest) -> PipelineDecisionPlan:
    """Validate a request and choose the pipeline branches for a dry run."""
    _validate_request(request)

    run_id = _slugify(request.request_id)
    run_root = _join_path(request.outputs.get("run_root", "runs"), run_id)
    task_name = request.task_type

    config_stage = _build_config_stage(request, run_root)
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
        },
        config_stage=config_stage,
        training_stage={
            "status": "future_phase",
            "backend": request.training.backend,
            "method": request.training.method,
            "model_id": request.slm_model,
            "runner": "pending",
            "artifact_dir": _join_path(run_root, "artifacts", "adapter"),
            "metrics_path": _join_path(run_root, "metrics", "train_metrics.json"),
        },
        evaluation_stage={
            "metrics": _metrics_for_task(task_name),
            "primary_metric": _primary_metric_for_task(task_name),
            "metrics_path": _join_path(run_root, "metrics", "eval_metrics.json"),
            "comparison_path": _join_path(run_root, "metrics", "comparison.json"),
            "auto_loop_policy": "send metrics back to auto config while trial budget remains"
            if request.config_mode == "auto"
            else "no auto loop",
        },
        inference_stage={
            "purpose": "fine_tuned_model_smoke_test",
            "output_path": _join_path(run_root, "inference", "smoke_test_predictions.jsonl"),
        },
        export_stage={
            "default_export": "peft_adapter",
            "output_dir": _join_path(run_root, "exports"),
            "include": ["adapter", "tokenizer", "train_config", "eval_config", "metrics", "report"],
        },
    )


def _build_config_stage(request: FinetuneRequest, run_root: str) -> dict[str, Any]:
    if request.config_mode == "manual":
        return {
            "mode": "manual",
            "router": "config_mode_router",
            "config_path": _join_path(run_root, "configs", "manual_train.yaml"),
            "manual_config": request.manual_config or {},
        }

    auto_config = request.auto_config or AutoConfigPreference()
    return {
        "mode": "auto",
        "router": "config_mode_router",
        "controller": auto_config.controller,
        "tuning_budget": auto_config.tuning_budget,
        "max_trials": auto_config.max_trials,
        "primary_objective": auto_config.primary_objective,
        "search_space_path": _join_path(run_root, "configs", "search_space.yaml"),
        "trial_config_dir": _join_path(run_root, "configs", "trials"),
        "best_config_path": _join_path(run_root, "configs", "best_train.yaml"),
    }


def _validate_request(request: FinetuneRequest) -> None:
    if request.task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {request.task_type}")
    if request.config_mode not in SUPPORTED_CONFIG_MODES:
        raise ValueError(f"Unsupported config_mode: {request.config_mode}")
    if not request.task_description.strip():
        raise ValueError("task_description is required.")
    if request.synthetic_target_count <= 0:
        raise ValueError("synthetic_target_count must be greater than zero.")
    if request.seed_data.type == "file" and not request.seed_data.path:
        raise ValueError("seed_data.path is required when seed_data.type is file.")
    if request.seed_data.type == "inline" and not request.seed_data.records:
        raise ValueError("seed_data.records is required when seed_data.type is inline.")
    if request.config_mode == "manual" and request.manual_config is None:
        raise ValueError("manual_config is required when config_mode is manual.")


def _expected_seed_shape(task_type: str) -> dict[str, str]:
    if task_type == "classification":
        return {"text": "str", "label": "str"}
    return {"task_specific_fields": "pending"}


def _primary_metric_for_task(task_type: str) -> str:
    return {
        "classification": "macro_f1",
        "ner": "entity_f1",
        "qa": "token_f1",
        "extraction": "schema_match_rate",
        "ranking": "ndcg",
        "function_calling": "tool_call_success_rate",
    }[task_type]


def _metrics_for_task(task_type: str) -> list[str]:
    return {
        "classification": ["accuracy", "macro_f1", "per_class_precision_recall_f1", "confusion_matrix"],
        "ner": ["entity_precision", "entity_recall", "entity_f1", "exact_span_match"],
        "qa": ["exact_match", "token_f1", "optional_llm_judge"],
        "extraction": ["valid_json_rate", "schema_match_rate", "field_f1", "exact_record_match"],
        "ranking": ["ndcg", "map", "mrr", "pairwise_accuracy"],
        "function_calling": [
            "function_name_accuracy",
            "json_validity",
            "argument_schema_validity",
            "argument_match",
            "tool_call_success_rate",
        ],
    }[task_type]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "request"


def _join_path(*parts: str) -> str:
    return Path(*parts).as_posix()
