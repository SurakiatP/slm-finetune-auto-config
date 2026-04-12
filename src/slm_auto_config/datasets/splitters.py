"""Dataset split helpers."""

from slm_auto_config.datasets.canonical import CanonicalRecord


def split_records(
    records: list[CanonicalRecord],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> dict[str, list[CanonicalRecord]]:
    """Split records into train, validation, and test buckets."""
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Invalid split ratios.")

    train_end = int(len(records) * train_ratio)
    validation_end = train_end + int(len(records) * validation_ratio)
    return {
        "train": records[:train_end],
        "validation": records[train_end:validation_end],
        "test": records[validation_end:],
    }
