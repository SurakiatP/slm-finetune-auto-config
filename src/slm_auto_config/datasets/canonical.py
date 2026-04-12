"""Canonical dataset schema shared by all task types.

Every task-specific dataset should be normalized into this schema before SDG,
training, evaluation, or dashboard reporting.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TaskType = Literal[
    "classification",
    "ner",
    "qa",
    "extraction",
    "ranking",
    "function_calling",
]
SplitName = Literal["train", "validation", "test"]
SourceName = Literal["seed", "synthetic"]

SUPPORTED_TASK_TYPES: frozenset[str] = frozenset(
    {
        "classification",
        "ner",
        "qa",
        "extraction",
        "ranking",
        "function_calling",
    }
)
SUPPORTED_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})
SUPPORTED_SOURCES: frozenset[str] = frozenset({"seed", "synthetic"})


@dataclass(frozen=True)
class CanonicalRecord:
    """Task-agnostic training/evaluation record."""

    id: str
    task_type: TaskType
    task_description: str
    input: Any
    expected_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    source: SourceName = "seed"
    quality_score: float | None = None
    split: SplitName | None = None

    def __post_init__(self) -> None:
        """Validate shared schema invariants at construction time."""
        if not self.id or not self.id.strip():
            raise ValueError("CanonicalRecord.id is required.")
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {self.task_type!r}.")
        if not self.task_description or not self.task_description.strip():
            raise ValueError("CanonicalRecord.task_description is required.")
        if self.input is None:
            raise ValueError("CanonicalRecord.input is required.")
        if self.expected_output is None:
            raise ValueError("CanonicalRecord.expected_output is required.")
        if self.source not in SUPPORTED_SOURCES:
            raise ValueError(f"Unsupported source: {self.source!r}.")
        if self.split is not None and self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split: {self.split!r}.")
        if self.quality_score is not None and not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("CanonicalRecord.quality_score must be between 0.0 and 1.0.")

    def to_dict(self) -> dict[str, Any]:
        """Convert this record to a JSONL-friendly dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalRecord":
        """Create a canonical record from a dictionary."""
        required_fields = {
            "id",
            "task_type",
            "task_description",
            "input",
            "expected_output",
        }
        missing = required_fields.difference(payload)
        if missing:
            raise ValueError(f"Missing canonical fields: {sorted(missing)}")

        return cls(
            id=payload["id"],
            task_type=payload["task_type"],
            task_description=payload["task_description"],
            input=payload["input"],
            expected_output=payload["expected_output"],
            metadata=payload.get("metadata", {}),
            source=payload.get("source", "seed"),
            quality_score=payload.get("quality_score"),
            split=payload.get("split"),
        )


def make_seed_record(
    *,
    record_id: str,
    task_type: TaskType,
    task_description: str,
    input_value: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
) -> CanonicalRecord:
    """Create a canonical record from one user-provided seed example."""
    return CanonicalRecord(
        id=record_id,
        task_type=task_type,
        task_description=task_description,
        input=input_value,
        expected_output=expected_output,
        metadata=metadata or {},
        source="seed",
    )
