"""Tests for classification phase 1-3 bridge."""

import unittest

from slm_auto_config.app.classification_pipeline import build_classification_phase_plan
from slm_auto_config.app.decision_plan import load_user_request
from slm_auto_config.app.schemas import FinetuneRequest, SeedDataSource


class ClassificationPipelineTest(unittest.TestCase):
    def test_builds_phase_plan_from_example_request(self) -> None:
        request = FinetuneRequest(
            request_id="unit-test-missing-synthetic",
            task_type="classification",
            task_description="Classify text.",
            synthetic_target_count=10,
            slm_model="Qwen/Qwen2.5-0.5B-Instruct",
            config_mode="auto",
            seed_data=SeedDataSource(
                type="file",
                path="data/raw/thai-text-classification-seed-data.json",
            ),
            outputs={"run_root": "runs"},
        )

        phase_plan = build_classification_phase_plan(request)

        self.assertEqual(phase_plan.sdg_config.target_count, 10)
        self.assertTrue(phase_plan.sdg_config.output_path.endswith("sdg/synthetic_data.json"))
        self.assertEqual(phase_plan.dataset_output_dir, "runs/unit-test-missing-synthetic/input")
        self.assertIn("runs/unit-test-missing-synthetic/sdg/synthetic_data.json", phase_plan.missing_inputs)


if __name__ == "__main__":
    unittest.main()
