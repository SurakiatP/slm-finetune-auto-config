"""Dataset validation helpers."""

from slm_auto_config.datasets.canonical import (
    SUPPORTED_SOURCES,
    SUPPORTED_SPLITS,
    SUPPORTED_TASK_TYPES,
    CanonicalRecord,
)


def validate_canonical_record(record: CanonicalRecord) -> None:
    """Validate shared canonical fields."""
    if not record.id or not record.id.strip():
        raise ValueError("CanonicalRecord.id is required.")
    if record.task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {record.task_type!r}.")
    if not record.task_description or not record.task_description.strip():
        raise ValueError("CanonicalRecord.task_description is required.")
    if record.input is None:
        raise ValueError("CanonicalRecord.input is required.")
    if record.expected_output is None:
        raise ValueError("CanonicalRecord.expected_output is required.")
    if record.source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source: {record.source!r}.")
    if record.split is not None and record.split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split: {record.split!r}.")
    if record.quality_score is not None and not 0.0 <= record.quality_score <= 1.0:
        raise ValueError("CanonicalRecord.quality_score must be between 0.0 and 1.0.")


def validate_canonical_records(records: list[CanonicalRecord]) -> None:
    """Validate a collection of canonical records and reject duplicate ids."""
    seen_ids: set[str] = set()
    for record in records:
        validate_canonical_record(record)
        if record.id in seen_ids:
            raise ValueError(f"Duplicate canonical record id: {record.id}")
        seen_ids.add(record.id)
