"""Tests for user request decision planning."""

import unittest

from slm_auto_config.app.decision_plan import build_decision_plan
from slm_auto_config.app.schemas import FinetuneRequest, SeedDataSource


class DecisionPlanTest(unittest.TestCase):
    def test_builds_classification_phase_1_3_plan(self) -> None:
        request = FinetuneRequest(
            request_id="demo-request",
            task_type="classification",
            task_description="Classify text.",
            synthetic_target_count=100,
            seed_data=SeedDataSource(
                type="file",
                path="data/raw/thai-text-classification-seed-data.json",
            ),
        )

        plan = build_decision_plan(request)

        self.assertEqual(plan.task_type, "classification")
        self.assertEqual(plan.sdg_stage["selected_pipeline"], "classification_sdg")
        self.assertEqual(plan.dataset_stage["strategy"], "stratified_split")
        self.assertTrue(plan.dataset_stage["train_path"].endswith("input/train.jsonl"))
        self.assertTrue(
            plan.dataset_stage["prompt_template_path"].endswith(
                "input/classification_prompt_template.json"
            )
        )

    def test_rejects_missing_inline_records(self) -> None:
        request = FinetuneRequest(
            request_id="demo-request",
            task_type="classification",
            task_description="Classify text.",
            synthetic_target_count=100,
            seed_data=SeedDataSource(type="inline", records=None),
        )

        with self.assertRaisesRegex(ValueError, "seed_data.records"):
            build_decision_plan(request)


if __name__ == "__main__":
    unittest.main()
