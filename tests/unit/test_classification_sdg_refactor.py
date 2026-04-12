"""Tests for the refactored classification SDG pure helpers."""

import json
import unittest

from slm_auto_config.synthetic.classification_sdg import (
    JudgeOutput,
    build_label_plan,
    calculate_overall_judge_score,
    extract_multiple_outputs,
    normalize_text,
)


class ClassificationSDGRefactorTest(unittest.TestCase):
    def test_extract_multiple_outputs_from_json(self) -> None:
        payload = json.dumps({"results": [" one ", "two", ""]})

        self.assertEqual(extract_multiple_outputs(payload), ["one", "two"])

    def test_extract_multiple_outputs_from_fenced_json(self) -> None:
        payload = '```json\n{"results": ["one", "two"]}\n```'

        self.assertEqual(extract_multiple_outputs(payload), ["one", "two"])

    def test_normalize_text_keeps_thai_letters_and_digits(self) -> None:
        self.assertEqual(normalize_text(" A! ข้อความ 123 "), "aข้อความ123")

    def test_build_label_plan_matches_original_quota_policy(self) -> None:
        seed_data = [
            {"text": "a1", "label": "a"},
            {"text": "a2", "label": "a"},
            {"text": "b1", "label": "b"},
        ]

        plan = build_label_plan(seed_data, target_count=100)

        self.assertEqual(plan.seed_labels, ["a", "b"])
        self.assertEqual(plan.labels, ["a", "b", "unknown"])
        self.assertEqual(plan.target_unknown, 10)
        self.assertEqual(plan.target_per_seed_label, 45)
        self.assertEqual(plan.target_counts, {"a": 45, "b": 45, "unknown": 10})

    def test_build_label_plan_allocates_small_target_without_overshoot(self) -> None:
        seed_data = [
            {"text": "a", "label": "a"},
            {"text": "b", "label": "b"},
            {"text": "c", "label": "c"},
            {"text": "d", "label": "d"},
        ]

        plan = build_label_plan(seed_data, target_count=10)

        self.assertEqual(sum(plan.target_counts.values()), 10)
        self.assertEqual(plan.target_counts, {"a": 3, "b": 2, "c": 2, "d": 2, "unknown": 1})

    def test_calculate_overall_judge_score(self) -> None:
        score = calculate_overall_judge_score(
            JudgeOutput(fidelity=1.0, naturalness=0.5, utility=0.0, reasoning="ok")
        )

        self.assertEqual(score, 0.55)


if __name__ == "__main__":
    unittest.main()
