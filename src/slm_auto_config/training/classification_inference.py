"""Classification inference smoke test for a trained Unsloth adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_auto_config.training.classification import ClassificationTrainConfig


@dataclass(frozen=True)
class ClassificationInferenceConfig:
    """Config for a small post-training inference check."""

    test_path: str
    adapter_path: str
    prompt_template_path: str
    predictions_path: str
    report_path: str
    sample_count: int = 10
    max_new_tokens: int = 32


def build_default_inference_config(
    train_config: ClassificationTrainConfig,
    *,
    sample_count: int = 10,
) -> ClassificationInferenceConfig:
    """Build the default inference-smoke config from the Phase 4 train config."""
    train_path = Path(train_config.train_path)
    run_dir = train_path.parent.parent
    return ClassificationInferenceConfig(
        test_path=(train_path.parent / "test.jsonl").as_posix(),
        adapter_path=train_config.output_dir,
        prompt_template_path=train_config.prompt_template_path,
        predictions_path=(run_dir / "predictions" / "inference_smoke.jsonl").as_posix(),
        report_path=(run_dir / "metrics" / "inference_smoke_report.json").as_posix(),
        sample_count=sample_count,
    )


def dry_run_inference_config(config: ClassificationInferenceConfig) -> dict[str, Any]:
    """Validate inference inputs without importing GPU libraries."""
    _validate_inference_files(config, require_adapter=False)
    labels = load_allowed_labels(config.prompt_template_path)
    samples = load_inference_samples(config.test_path, sample_count=config.sample_count)
    return {
        "status": "pass",
        "test_path": config.test_path,
        "adapter_path": config.adapter_path,
        "predictions_path": config.predictions_path,
        "report_path": config.report_path,
        "sample_count": len(samples),
        "allowed_labels": labels,
        "first_expected_label": samples[0].get("expected_label") if samples else None,
    }


def run_unsloth_classification_inference_smoke(
    train_config: ClassificationTrainConfig,
    inference_config: ClassificationInferenceConfig,
) -> dict[str, Any]:
    """Run a small inference check against the trained adapter/model output."""
    _validate_inference_files(inference_config, require_adapter=True)

    from unsloth import FastLanguageModel
    import torch

    labels = load_allowed_labels(inference_config.prompt_template_path)
    samples = load_inference_samples(
        inference_config.test_path,
        sample_count=inference_config.sample_count,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=inference_config.adapter_path,
        max_seq_length=train_config.max_seq_length,
        dtype=None,
        load_in_4bit=train_config.precision == "4bit",
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions_path = Path(inference_config.predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as output_file:
        for index, sample in enumerate(samples):
            prompt = _messages_to_generation_prompt(tokenizer, sample["messages"])
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=inference_config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            prediction = normalize_label_prediction(raw_prediction, labels)
            shape_check = inspect_prediction_shape(raw_prediction, labels)
            record = {
                "index": index,
                "expected_label": sample["expected_label"],
                "prediction": prediction,
                "raw_prediction": raw_prediction.strip(),
                "is_valid_label": prediction in labels,
                "is_correct": prediction == sample["expected_label"],
                **shape_check,
            }
            results.append(record)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = build_inference_report(results, labels)
    report_path = Path(inference_config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "pass",
        "adapter_path": inference_config.adapter_path,
        "predictions_path": predictions_path.as_posix(),
        "report_path": report_path.as_posix(),
        "report": report,
    }


def load_allowed_labels(prompt_template_path: str | Path) -> list[str]:
    """Load allowed labels from the Phase 1-3 prompt template artifact."""
    payload = json.loads(Path(prompt_template_path).read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"No labels found in prompt template: {prompt_template_path}")
    return [str(label) for label in labels]


def load_inference_samples(path: str | Path, *, sample_count: int) -> list[dict[str, Any]]:
    """Load test records and remove the assistant answer before generation."""
    samples: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if len(messages) != 3:
                raise ValueError(f"Expected 3 SFT messages in: {path}")
            samples.append(
                {
                    "messages": messages[:2],
                    "expected_label": messages[2]["content"],
                }
            )
            if len(samples) >= sample_count:
                break
    if not samples:
        raise ValueError(f"No inference samples found in: {path}")
    return samples


def normalize_label_prediction(raw_prediction: str, labels: list[str]) -> str:
    """Convert a generated answer into a single label when possible."""
    cleaned = raw_prediction.strip().strip("`\"' ")
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    first_line = first_line.strip("`\"' ")
    if first_line in labels:
        return first_line
    for label in sorted(labels, key=len, reverse=True):
        if label and label in cleaned:
            return label
    return first_line


def inspect_prediction_shape(raw_prediction: str, labels: list[str]) -> dict[str, Any]:
    """Check whether the raw output is a clean label-only classification answer."""
    cleaned = raw_prediction.strip().strip("`\"' ")
    nonempty_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    first_line = nonempty_lines[0].strip("`\"' ") if nonempty_lines else ""
    thinking_markers = [
        "<think",
        "</think",
        "reasoning",
        "analysis:",
        "explanation:",
        "เหตุผล",
        "คำอธิบาย",
    ]
    lower_cleaned = cleaned.lower()
    has_thinking_trace = any(marker in lower_cleaned for marker in thinking_markers)
    is_single_label_output = len(nonempty_lines) == 1 and first_line in labels
    return {
        "is_single_label_output": is_single_label_output,
        "has_no_explanation": is_single_label_output,
        "has_no_thinking_trace": not has_thinking_trace,
        "raw_nonempty_line_count": len(nonempty_lines),
    }


def build_inference_report(results: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    """Build a compact smoke-test report."""
    total = len(results)
    valid_count = sum(1 for result in results if result["is_valid_label"])
    correct_count = sum(1 for result in results if result["is_correct"])
    single_label_count = sum(1 for result in results if result.get("is_single_label_output"))
    no_explanation_count = sum(1 for result in results if result.get("has_no_explanation"))
    no_thinking_count = sum(1 for result in results if result.get("has_no_thinking_trace"))
    status = "pass"
    if not total or valid_count != total:
        status = "review"
    if single_label_count != total or no_explanation_count != total or no_thinking_count != total:
        status = "review"
    return {
        "status": status,
        "total": total,
        "valid_label_count": valid_count,
        "valid_label_rate": valid_count / total if total else 0.0,
        "single_label_output_count": single_label_count,
        "single_label_output_rate": single_label_count / total if total else 0.0,
        "no_explanation_count": no_explanation_count,
        "no_explanation_rate": no_explanation_count / total if total else 0.0,
        "no_thinking_trace_count": no_thinking_count,
        "no_thinking_trace_rate": no_thinking_count / total if total else 0.0,
        "correct_count": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "allowed_labels": labels,
    }


def _messages_to_generation_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _validate_inference_files(
    config: ClassificationInferenceConfig,
    *,
    require_adapter: bool,
) -> None:
    for path in [config.test_path, config.prompt_template_path]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if require_adapter and not Path(config.adapter_path).exists():
        raise FileNotFoundError(config.adapter_path)
    if config.sample_count <= 0:
        raise ValueError("sample_count must be greater than 0.")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0.")
