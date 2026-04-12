"""Tests for user request decision planning."""

import unittest

from slm_auto_config.app.decision_plan import build_decision_plan
from slm_auto_config.app.schemas import AutoConfigPreference, FinetuneRequest, SeedDataSource


class DecisionPlanTest(unittest.TestCase):
    def test_builds_classification_auto_plan(self) -> None:
        request = FinetuneRequest(
            request_id="demo-request",
            task_type="classification",
            task_description="Classify text.",
            synthetic_target_count=100,
            slm_model="Qwen/Qwen2.5-0.5B-Instruct",
            config_mode="auto",
            seed_data=SeedDataSource(
                type="file",
                path="data/raw/thai-text-classification-seed-data.json",
            ),
            auto_config=AutoConfigPreference(
                controller="oumi_style",
                tuning_budget="small_budget",
                max_trials=6,
                primary_objective="macro_f1",
            ),
        )

        plan = build_decision_plan(request)

        self.assertEqual(plan.task_type, "classification")
        self.assertEqual(plan.sdg_stage["selected_pipeline"], "classification_sdg")
        self.assertEqual(plan.config_stage["mode"], "auto")
        self.assertEqual(plan.training_stage["backend"], "unsloth")
        self.assertEqual(plan.evaluation_stage["primary_metric"], "macro_f1")

    def test_manual_config_is_required_for_manual_mode(self) -> None:
        request = FinetuneRequest(
            request_id="demo-request",
            task_type="classification",
            task_description="Classify text.",
            synthetic_target_count=100,
            slm_model="Qwen/Qwen2.5-0.5B-Instruct",
            config_mode="manual",
            seed_data=SeedDataSource(type="file", path="seed.json"),
            manual_config=None,
        )

        with self.assertRaisesRegex(ValueError, "manual_config"):
            build_decision_plan(request)


if __name__ == "__main__":
    unittest.main()
