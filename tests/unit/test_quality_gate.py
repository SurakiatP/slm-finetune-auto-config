"""Tests for synthetic data quality reports."""

import tempfile
import unittest
from pathlib import Path

from slm_auto_config.datasets.quality import (
    QualityGateConfig,
    build_classification_quality_report,
    write_classification_quality_report,
)


class QualityGateTest(unittest.TestCase):
    def test_flags_quality_issues_and_scores_report(self) -> None:
        records = [
            {"text": "alpha contract text", "label": "class_a"},
            {"text": "alpha contract text", "label": "class_a"},
            {"text": "Candidate output label example", "label": "unknown"},
            {"text": "[company_name] ______ placeholder text", "label": "class_b"},
        ]

        report = build_classification_quality_report(
            records,
            expected_target_count=4,
            config=QualityGateConfig(max_exact_duplicate_ratio=0.0),
        )

        self.assertEqual(report["status"], "warn")
        self.assertEqual(report["issue_counts"]["exact_duplicate"], 1)
        self.assertEqual(report["issue_counts"]["prompt_artifact"], 1)
        self.assertEqual(report["issue_counts"]["placeholder"], 1)
        self.assertIn("prompt_artifact", report["flagged_examples"])
        self.assertTrue(report["summary"]["topic_assumptions"]["topic_agnostic"])

    def test_domain_keyword_overlap_is_optional(self) -> None:
        records = [
            {"text": "this record mentions contract", "label": "unknown"},
            {"text": "general domain text", "label": "class_a"},
        ]

        topic_agnostic_report = build_classification_quality_report(records)
        domain_report = build_classification_quality_report(
            records,
            config=QualityGateConfig(label_keyword_map={"class_a": ["contract"]}),
        )

        self.assertEqual(topic_agnostic_report["issue_counts"]["unknown_in_scope_keyword"], 0)
        self.assertEqual(domain_report["issue_counts"]["unknown_in_scope_keyword"], 1)

    def test_writes_quality_report_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "synthetic.json"
            output_path = root / "sdg_report.json"
            input_path.write_text(
                '[{"text": "sample text", "label": "class_a"}]',
                encoding="utf-8",
            )

            report = write_classification_quality_report(
                input_path=input_path,
                output_path=output_path,
                expected_target_count=1,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(report["summary"]["record_count"], 1)


if __name__ == "__main__":
    unittest.main()
