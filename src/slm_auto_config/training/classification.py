"""Classification-only Unsloth training config and runner.

This module intentionally stays small. It is the Phase 4 bridge from the
Phase 1-3 SFT JSONL files to a single-GPU Unsloth training run.
"""

from __future__ import annotations

import json
import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_CHAT_TEMPLATES = {
    "qwen": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "llama": {
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "gemma": {
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
    },
}


@dataclass(frozen=True)
class ResponseMaskConfig:
    """Markers used by Unsloth to train only on assistant responses."""

    instruction_part: str
    response_part: str


@dataclass(frozen=True)
class LoraConfig:
    """LoRA adapter parameters for a small first training run."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass(frozen=True)
class ClassificationTrainingArguments:
    """TrainingArguments values for a conservative Vast.ai smoke run."""

    num_train_epochs: float = 1.0
    max_steps: int = 10
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    logging_steps: int = 1
    eval_steps: int = 5
    save_steps: int = 10
    save_total_limit: int = 2
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_8bit"
    seed: int = 42


@dataclass(frozen=True)
class ClassificationTrainConfig:
    """Config file consumed by the classification Unsloth trainer."""

    schema_version: str = "phase4.classification.unsloth.v1"
    task_type: str = "classification"
    backend: str = "unsloth"
    model_id: str = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
    precision: str = "4bit"
    max_seq_length: int = 1024
    train_path: str = "runs/demo-thai-classification-auto-001/input/train.jsonl"
    validation_path: str = "runs/demo-thai-classification-auto-001/input/validation.jsonl"
    output_dir: str = "runs/demo-thai-classification-auto-001/model"
    metrics_path: str = "runs/demo-thai-classification-auto-001/metrics/train_metrics.json"
    prompt_template_path: str = (
        "runs/demo-thai-classification-auto-001/input/classification_prompt_template.json"
    )
    response_only_training: bool = True
    response_mask: ResponseMaskConfig = field(
        default_factory=lambda: ResponseMaskConfig(**SUPPORTED_CHAT_TEMPLATES["qwen"])
    )
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: ClassificationTrainingArguments = field(
        default_factory=ClassificationTrainingArguments
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config."""
        return asdict(self)


def response_mask_for_chat_template(chat_template: str) -> ResponseMaskConfig:
    """Return response-mask markers for a known chat template family."""
    try:
        return ResponseMaskConfig(**SUPPORTED_CHAT_TEMPLATES[chat_template])
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_CHAT_TEMPLATES))
        raise ValueError(f"Unsupported chat template: {chat_template}. Use one of: {supported}") from exc


def build_default_classification_train_config(
    *,
    run_dir: str | Path,
    model_id: str = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    chat_template: str = "qwen",
    smoke_run: bool = True,
) -> ClassificationTrainConfig:
    """Build the first small Phase 4 config for a classification run."""
    run_root = Path(run_dir)
    max_steps = 10 if smoke_run else -1
    epochs = 1.0 if smoke_run else 2.0
    return ClassificationTrainConfig(
        model_id=model_id,
        train_path=(run_root / "input" / "train.jsonl").as_posix(),
        validation_path=(run_root / "input" / "validation.jsonl").as_posix(),
        output_dir=(run_root / "model").as_posix(),
        metrics_path=(run_root / "metrics" / "train_metrics.json").as_posix(),
        prompt_template_path=(run_root / "input" / "classification_prompt_template.json").as_posix(),
        response_mask=response_mask_for_chat_template(chat_template),
        training=ClassificationTrainingArguments(
            max_steps=max_steps,
            num_train_epochs=epochs,
        ),
    )


def load_train_config(path: str | Path) -> ClassificationTrainConfig:
    """Load a JSON train config."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClassificationTrainConfig(
        schema_version=payload.get("schema_version", "phase4.classification.unsloth.v1"),
        task_type=payload.get("task_type", "classification"),
        backend=payload.get("backend", "unsloth"),
        model_id=payload["model_id"],
        precision=payload.get("precision", "4bit"),
        max_seq_length=int(payload.get("max_seq_length", 1024)),
        train_path=payload["train_path"],
        validation_path=payload["validation_path"],
        output_dir=payload["output_dir"],
        metrics_path=payload["metrics_path"],
        prompt_template_path=payload.get("prompt_template_path", ""),
        response_only_training=bool(payload.get("response_only_training", True)),
        response_mask=ResponseMaskConfig(**payload["response_mask"]),
        lora=LoraConfig(**payload.get("lora", {})),
        training=ClassificationTrainingArguments(**payload.get("training", {})),
    )


def write_train_config(config: ClassificationTrainConfig, path: str | Path) -> None:
    """Write a train config as UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def dry_run_train_config(config: ClassificationTrainConfig) -> dict[str, Any]:
    """Validate files and record shape without importing GPU training libraries."""
    _validate_common_config(config)
    train_sample = _load_first_jsonl_record(config.train_path)
    validation_sample = _load_first_jsonl_record(config.validation_path)
    _validate_sft_record(train_sample, "train")
    _validate_sft_record(validation_sample, "validation")

    return {
        "status": "pass",
        "schema_version": config.schema_version,
        "backend": config.backend,
        "model_id": config.model_id,
        "precision": config.precision,
        "train_path": config.train_path,
        "validation_path": config.validation_path,
        "output_dir": config.output_dir,
        "metrics_path": config.metrics_path,
        "response_only_training": config.response_only_training,
        "response_mask": asdict(config.response_mask),
        "train_sample_roles": [message["role"] for message in train_sample["messages"]],
        "validation_sample_roles": [
            message["role"] for message in validation_sample["messages"]
        ],
        "train_sample_label": train_sample.get("metadata", {}).get("label"),
    }


def run_unsloth_classification_training(config: ClassificationTrainConfig) -> dict[str, Any]:
    """Run real Unsloth SFT training. Use this on a CUDA machine, such as Vast.ai."""
    _validate_common_config(config)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.precision == "4bit",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.training.seed,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": config.train_path,
            "validation": config.validation_path,
        },
    )
    dataset = dataset.map(
        lambda row: {"text": _messages_to_training_text(tokenizer, row["messages"])},
        remove_columns=dataset["train"].column_names,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        optim=config.training.optim,
        seed=config.training.seed,
        report_to="none",
    )
    trainer = _build_sft_trainer(
        SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        training_args=training_args,
        max_seq_length=config.max_seq_length,
    )
    if config.response_only_training:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=config.response_mask.instruction_part,
            response_part=config.response_mask.response_part,
        )

    train_result = trainer.train()
    trainer.save_model(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())

    metrics = dict(train_result.metrics)
    metrics_path = Path(config.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "trained",
        "output_dir": output_dir.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "metrics": metrics,
    }


def _build_sft_trainer(
    sft_trainer_class: Any,
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    training_args: Any,
    max_seq_length: int,
) -> Any:
    """Build TRL SFTTrainer across older and newer constructor signatures."""
    signature = inspect.signature(sft_trainer_class)
    parameters = signature.parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
    }
    if "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in parameters:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in parameters:
        kwargs["max_seq_length"] = max_seq_length
    return sft_trainer_class(**kwargs)


def _messages_to_training_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def _validate_common_config(config: ClassificationTrainConfig) -> None:
    if config.task_type != "classification":
        raise ValueError("Phase 4 runner currently supports task_type='classification' only.")
    if config.backend != "unsloth":
        raise ValueError("Phase 4 runner currently supports backend='unsloth' only.")
    if config.precision not in {"4bit", "16bit"}:
        raise ValueError("precision must be either '4bit' or '16bit'.")
    for path in [config.train_path, config.validation_path]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if config.prompt_template_path and not Path(config.prompt_template_path).is_file():
        raise FileNotFoundError(config.prompt_template_path)
    if config.response_only_training:
        if not config.response_mask.instruction_part or not config.response_mask.response_part:
            raise ValueError("response_mask is required when response_only_training=true.")


def _load_first_jsonl_record(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"JSONL file has no records: {path}")


def _validate_sft_record(record: dict[str, Any], split_name: str) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{split_name} sample must contain exactly 3 messages.")
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"{split_name} sample roles must be system,user,assistant. Got: {roles}")
    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{split_name} sample has an empty message content.")
