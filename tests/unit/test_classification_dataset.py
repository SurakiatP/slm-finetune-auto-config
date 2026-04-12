"""Tests for classification dataset build and SFT formatting."""

import json
import tempfile
import unittest
from pathlib import Path

from slm_auto_config.datasets.classification import (
    build_classification_split_and_sft,
    to_classification_sft_record,
)


class ClassificationDatasetTest(unittest.TestCase):
    def test_builds_split_and_sft_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            seed_path = root / "seed.json"
            synthetic_path = root / "synthetic.json"
            seed_path.write_text(
                json.dumps(
                    [
                        {"text": "seed a1", "label": "a"},
                        {"text": "seed a2", "label": "a"},
                        {"text": "seed b1", "label": "b"},
                        {"text": "seed b2", "label": "b"},
                    ]
                ),
                encoding="utf-8",
            )
            synthetic_path.write_text(
                json.dumps(
                    [
                        {"text": "synthetic a1", "label": "a"},
                        {"text": "synthetic a2", "label": "a"},
                        {"text": "synthetic b1", "label": "b"},
                        {"text": "synthetic b2", "label": "b"},
                    ]
                ),
                encoding="utf-8",
            )

            summary = build_classification_split_and_sft(
                seed_data_path=seed_path,
                synthetic_data_path=synthetic_path,
                output_dir=root / "input",
                task_description="Classify text.",
            )

            train_path = Path(summary["sft_paths"]["train"])
            self.assertTrue(train_path.exists())
            first_record = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("messages", first_record)
            self.assertEqual(first_record["messages"][0]["role"], "system")
            self.assertIn("Valid labels:", first_record["messages"][0]["content"])
            self.assertIn("Return exactly one label.", first_record["messages"][0]["content"])
            self.assertEqual(first_record["messages"][1]["content"].splitlines()[0], "Text:")
            self.assertIn(first_record["messages"][2]["content"], {"a", "b"})
            self.assertEqual(first_record["metadata"]["prompt_template"], "classification.topic_agnostic.v1")
            self.assertTrue(Path(summary["prompt_template_path"]).exists())
            self.assertTrue(Path(summary["few_shot_examples_path"]).exists())

    def test_optional_few_shot_examples_are_inserted_into_system_prompt(self) -> None:
        record = {"text": "new text", "label": "a"}

        sft_record = to_classification_sft_record(
            record,
            task_description="Classify text.",
            labels=["a", "b"],
            few_shot_examples=[{"text": "example text", "label": "b"}],
        )

        system_prompt = sft_record["messages"][0]["content"]
        self.assertIn("Examples:", system_prompt)
        self.assertIn("Input: example text", system_prompt)
        self.assertIn("Category: b", system_prompt)
        self.assertEqual(sft_record["metadata"]["few_shot_count"], 1)


if __name__ == "__main__":
    unittest.main()
