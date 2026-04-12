"""Tests for Phase 4 classification training config."""

import json
import tempfile
import unittest
from pathlib import Path

from slm_auto_config.training.classification import (
    _build_sft_trainer,
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

    def test_build_sft_trainer_uses_legacy_tokenizer_argument(self) -> None:
        class LegacySFTTrainer:
            def __init__(
                self,
                *,
                model,
                tokenizer,
                train_dataset,
                eval_dataset,
                dataset_text_field,
                max_seq_length,
                args,
            ):
                self.kwargs = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "train_dataset": train_dataset,
                    "eval_dataset": eval_dataset,
                    "dataset_text_field": dataset_text_field,
                    "max_seq_length": max_seq_length,
                    "args": args,
                }

        trainer = _build_sft_trainer(
            LegacySFTTrainer,
            model="model",
            tokenizer="tokenizer",
            train_dataset="train",
            eval_dataset="validation",
            training_args="args",
            max_seq_length=1024,
        )

        self.assertEqual(trainer.kwargs["tokenizer"], "tokenizer")
        self.assertEqual(trainer.kwargs["dataset_text_field"], "text")
        self.assertEqual(trainer.kwargs["max_seq_length"], 1024)

    def test_build_sft_trainer_uses_new_processing_class_argument(self) -> None:
        class NewSFTTrainer:
            def __init__(
                self,
                *,
                model,
                processing_class,
                train_dataset,
                eval_dataset,
                args,
            ):
                self.kwargs = {
                    "model": model,
                    "processing_class": processing_class,
                    "train_dataset": train_dataset,
                    "eval_dataset": eval_dataset,
                    "args": args,
                }

        trainer = _build_sft_trainer(
            NewSFTTrainer,
            model="model",
            tokenizer="tokenizer",
            train_dataset="train",
            eval_dataset="validation",
            training_args="args",
            max_seq_length=1024,
        )

        self.assertEqual(trainer.kwargs["processing_class"], "tokenizer")
        self.assertNotIn("dataset_text_field", trainer.kwargs)
        self.assertNotIn("max_seq_length", trainer.kwargs)


if __name__ == "__main__":
    unittest.main()
