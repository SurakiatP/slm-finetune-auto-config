"""Tests for the canonical dataset schema."""

import unittest

from slm_auto_config.datasets.canonical import CanonicalRecord, make_seed_record
from slm_auto_config.datasets.validators import validate_canonical_records


class CanonicalDatasetTest(unittest.TestCase):
    def test_make_seed_record_sets_shared_fields(self) -> None:
        record = make_seed_record(
            record_id="seed-001",
            task_type="classification",
            task_description="Classify the customer ticket intent.",
            input_value="I need a refund.",
            expected_output={"label": "refund"},
            metadata={"language": "en"},
        )

        self.assertEqual(record.id, "seed-001")
        self.assertEqual(record.task_type, "classification")
        self.assertEqual(record.source, "seed")
        self.assertEqual(record.metadata, {"language": "en"})

    def test_canonical_record_round_trip_dict(self) -> None:
        record = CanonicalRecord(
            id="synthetic-001",
            task_type="extraction",
            task_description="Extract invoice fields.",
            input="Invoice total is 120 THB.",
            expected_output={"total": 120, "currency": "THB"},
            source="synthetic",
            quality_score=0.9,
            split="train",
        )

        self.assertEqual(CanonicalRecord.from_dict(record.to_dict()), record)

    def test_invalid_task_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported task_type"):
            CanonicalRecord(
                id="bad-001",
                task_type="summarization",
                task_description="Summarize text.",
                input="hello",
                expected_output="hello",
            )

    def test_quality_score_must_be_between_zero_and_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "quality_score"):
            CanonicalRecord(
                id="bad-002",
                task_type="classification",
                task_description="Classify text.",
                input="hello",
                expected_output="greeting",
                quality_score=1.5,
            )

    def test_validate_canonical_records_rejects_duplicate_ids(self) -> None:
        records = [
            make_seed_record(
                record_id="seed-001",
                task_type="classification",
                task_description="Classify text.",
                input_value="hello",
                expected_output="greeting",
            ),
            make_seed_record(
                record_id="seed-001",
                task_type="classification",
                task_description="Classify text.",
                input_value="bye",
                expected_output="farewell",
            ),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate canonical record id"):
            validate_canonical_records(records)


if __name__ == "__main__":
    unittest.main()
