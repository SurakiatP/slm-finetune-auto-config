"""Classification dataset build utilities."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from slm_auto_config.datasets.classification_prompt import (
    ClassificationPromptTemplate,
    to_classification_sft_record,
)


TextLabelRecord = dict[str, Any]


def load_text_label_json(path: str | Path) -> list[TextLabelRecord]:
    """Load JSON records that use the classification `text`/`label` contract."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Classification dataset must be a JSON array.")
    validate_text_label_records(payload)
    return payload


def validate_text_label_records(records: list[TextLabelRecord]) -> None:
    """Validate classification records without assuming a domain."""
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Classification record at index {index} must be an object.")
        for field_name in ("text", "label"):
            value = record.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Classification record at index {index} requires non-empty field: {field_name}"
                )


def deduplicate_text_label_records(records: list[TextLabelRecord]) -> list[TextLabelRecord]:
    """Deduplicate classification records by normalized text and label."""
    deduped: list[TextLabelRecord] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (_normalize_text(record["text"]), record["label"].strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"text": record["text"].strip(), "label": record["label"].strip()})
    return deduped


def stratified_split_text_label_records(
    records: list[TextLabelRecord],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[TextLabelRecord]]:
    """Split records while preserving label distribution as much as possible."""
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Invalid split ratios.")

    grouped: dict[str, list[TextLabelRecord]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label_records in grouped.values():
        label_records = list(label_records)
        rng.shuffle(label_records)
        count = len(label_records)
        train_count = int(count * train_ratio)
        validation_count = int(count * validation_ratio)
        if count >= 3:
            train_count = max(1, train_count)
            validation_count = max(1, validation_count)
            if train_count + validation_count >= count:
                train_count = count - 2
                validation_count = 1

        validation_end = train_count + validation_count
        splits["train"].extend(label_records[:train_count])
        splits["validation"].extend(label_records[train_count:validation_end])
        splits["test"].extend(label_records[validation_end:])

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_classification_split_and_sft(
    *,
    seed_data_path: str | Path,
    synthetic_data_path: str | Path,
    output_dir: str | Path,
    task_description: str,
    split_seed: int = 42,
    prompt_template: ClassificationPromptTemplate | None = None,
    few_shot_per_label: int = 0,
) -> dict[str, Any]:
    """Combine seed and synthetic data, then write record and SFT split files."""
    seed_records = load_text_label_json(seed_data_path)
    synthetic_records = load_text_label_json(synthetic_data_path)
    combined_records = deduplicate_text_label_records(seed_records + synthetic_records)
    labels = sorted({record["label"] for record in combined_records})
    splits = stratified_split_text_label_records(combined_records, seed=split_seed)
    few_shot_examples = _build_few_shot_examples(
        splits["train"],
        per_label=few_shot_per_label,
        seed=split_seed,
    )

    output_root = Path(output_dir)
    record_paths = {
        split_name: output_root / f"{split_name}.records.jsonl" for split_name in splits
    }
    sft_paths = {split_name: output_root / f"{split_name}.jsonl" for split_name in splits}
    template = prompt_template or ClassificationPromptTemplate()
    prompt_template_path = output_root / "classification_prompt_template.json"
    few_shot_examples_path = output_root / "classification_few_shot_examples.json"

    for split_name, split_records in splits.items():
        write_jsonl(record_paths[split_name], split_records)
        write_jsonl(
            sft_paths[split_name],
            [
                to_classification_sft_record(
                    record,
                    task_description=task_description,
                    labels=labels,
                    prompt_template=template,
                    few_shot_examples=few_shot_examples,
                )
                for record in split_records
            ],
        )
    prompt_template_path.write_text(
        json.dumps(
            {
                "template": template.to_dict(),
                "task_description": task_description,
                "labels": labels,
                "few_shot_per_label": few_shot_per_label,
                "few_shot_source_split": "train",
                "few_shot_examples": few_shot_examples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    few_shot_examples_path.write_text(
        json.dumps(few_shot_examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "seed_count": len(seed_records),
        "synthetic_count": len(synthetic_records),
        "deduplicated_count": len(combined_records),
        "labels": labels,
        "split_counts": {split_name: len(split_records) for split_name, split_records in splits.items()},
        "prompt_template": template.template_id,
        "prompt_template_path": prompt_template_path.as_posix(),
        "few_shot_per_label": few_shot_per_label,
        "few_shot_count": len(few_shot_examples),
        "few_shot_examples_path": few_shot_examples_path.as_posix(),
        "record_paths": {split_name: path.as_posix() for split_name, path in record_paths.items()},
        "sft_paths": {split_name: path.as_posix() for split_name, path in sft_paths.items()},
    }


def _build_few_shot_examples(
    records: list[TextLabelRecord],
    *,
    per_label: int,
    seed: int,
) -> list[TextLabelRecord]:
    if per_label <= 0:
        return []
    grouped: dict[str, list[TextLabelRecord]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)
    rng = random.Random(seed)
    examples: list[TextLabelRecord] = []
    for label in sorted(grouped):
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        examples.extend(candidates[:per_label])
    return examples


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9ก-๙]", "", str(text)).lower()
