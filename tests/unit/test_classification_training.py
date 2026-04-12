"""Tests for Phase 4 classification training config."""

import json
import tempfile
import unittest
from pathlib import Path

from slm_auto_config.training.classification import (
    build_default_classification_train_config,
    dry_run_train_config,
    response_mask_for_chat_template,
    write_train_config,
)


class ClassificationTrainingTest(unittest.TestCase):
    def test_builds_qwen_response_mask(self) -> None:
        mask = response_mask_for_chat_template("qwen")

        self.assertEqual(mask.instruction_part, "<|im_start|>user\n")
        self.assertEqual(mask.response_part, "<|im_start|>assistant\n")

    def test_dry_run_validates_sft_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            input_dir = run_dir / "input"
            input_dir.mkdir()
            sample = {
                "messages": [
                    {"role": "system", "content": "Classify text."},
                    {"role": "user", "content": "Text:\nhello"},
                    {"role": "assistant", "content": "greeting"},
                ],
                "metadata": {"label": "greeting"},
            }
            for split in ["train", "validation"]:
                (input_dir / f"{split}.jsonl").write_text(
                    json.dumps(sample, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            (input_dir / "classification_prompt_template.json").write_text(
                "{}",
                encoding="utf-8",
            )

            config = build_default_classification_train_config(run_dir=run_dir)
            config_path = run_dir / "train_config.json"
            write_train_config(config, config_path)

            result = dry_run_train_config(config)

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["train_sample_roles"], ["system", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
