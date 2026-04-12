"""Tests for Phase 4 classification inference smoke helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from slm_auto_config.training.classification_inference import (
    ClassificationInferenceConfig,
    build_inference_report,
    inspect_prediction_shape,
    load_allowed_labels,
    load_inference_samples,
    normalize_label_prediction,
    run_unsloth_classification_inference_smoke,
)
from slm_auto_config.training.classification import build_default_classification_train_config


class ClassificationInferenceTest(unittest.TestCase):
    def test_load_allowed_labels_from_prompt_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "classification_prompt_template.json"
            prompt_path.write_text(
                json.dumps({"labels": ["unknown", "contract"]}),
                encoding="utf-8",
            )

            labels = load_allowed_labels(prompt_path)

            self.assertEqual(labels, ["unknown", "contract"])

    def test_load_inference_samples_removes_assistant_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.jsonl"
            record = {
                "messages": [
                    {"role": "system", "content": "Classify."},
                    {"role": "user", "content": "Text:\nhello"},
                    {"role": "assistant", "content": "contract"},
                ]
            }
            test_path.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            samples = load_inference_samples(test_path, sample_count=1)

            self.assertEqual(samples[0]["expected_label"], "contract")
            self.assertEqual(len(samples[0]["messages"]), 2)
            self.assertEqual(samples[0]["messages"][-1]["role"], "user")

    def test_normalize_label_prediction_prefers_exact_label(self) -> None:
        labels = ["unknown", "NDA"]

        prediction = normalize_label_prediction("NDA\n", labels)

        self.assertEqual(prediction, "NDA")

    def test_normalize_label_prediction_extracts_label_from_text(self) -> None:
        labels = ["unknown", "NDA"]

        prediction = normalize_label_prediction("Answer: NDA", labels)

        self.assertEqual(prediction, "NDA")

    def test_inspect_prediction_shape_accepts_label_only_output(self) -> None:
        result = inspect_prediction_shape("NDA\n", ["unknown", "NDA"])

        self.assertTrue(result["is_single_label_output"])
        self.assertTrue(result["has_no_explanation"])
        self.assertTrue(result["has_no_thinking_trace"])

    def test_inspect_prediction_shape_flags_explanation_and_thinking(self) -> None:
        result = inspect_prediction_shape("<think>because</think>\nAnswer: NDA", ["unknown", "NDA"])

        self.assertFalse(result["is_single_label_output"])
        self.assertFalse(result["has_no_explanation"])
        self.assertFalse(result["has_no_thinking_trace"])

    def test_build_inference_report(self) -> None:
        report = build_inference_report(
            [
                {
                    "is_valid_label": True,
                    "is_correct": True,
                    "is_single_label_output": True,
                    "has_no_explanation": True,
                    "has_no_thinking_trace": True,
                },
                {
                    "is_valid_label": True,
                    "is_correct": False,
                    "is_single_label_output": True,
                    "has_no_explanation": True,
                    "has_no_thinking_trace": True,
                },
            ],
            ["unknown", "NDA"],
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["valid_label_rate"], 1.0)
        self.assertEqual(report["single_label_output_rate"], 1.0)
        self.assertEqual(report["no_thinking_trace_rate"], 1.0)
        self.assertEqual(report["accuracy"], 0.5)

    def test_build_inference_report_flags_non_label_output_for_review(self) -> None:
        report = build_inference_report(
            [
                {
                    "is_valid_label": True,
                    "is_correct": True,
                    "is_single_label_output": False,
                    "has_no_explanation": False,
                    "has_no_thinking_trace": True,
                },
            ],
            ["unknown", "NDA"],
        )

        self.assertEqual(report["status"], "review")
        self.assertEqual(report["single_label_output_rate"], 0.0)
        self.assertEqual(report["no_explanation_rate"], 0.0)

    def test_inference_smoke_reports_missing_adapter_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            input_dir = run_dir / "input"
            input_dir.mkdir()
            sample = {
                "messages": [
                    {"role": "system", "content": "Classify."},
                    {"role": "user", "content": "Text:\nhello"},
                    {"role": "assistant", "content": "contract"},
                ]
            }
            (input_dir / "test.jsonl").write_text(
                json.dumps(sample, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            prompt_path = input_dir / "classification_prompt_template.json"
            prompt_path.write_text(
                json.dumps({"labels": ["unknown", "contract"]}),
                encoding="utf-8",
            )
            train_config = build_default_classification_train_config(run_dir=run_dir)
            inference_config = ClassificationInferenceConfig(
                test_path=(input_dir / "test.jsonl").as_posix(),
                adapter_path=(run_dir / "missing-model").as_posix(),
                prompt_template_path=prompt_path.as_posix(),
                predictions_path=(run_dir / "predictions.jsonl").as_posix(),
                report_path=(run_dir / "report.json").as_posix(),
            )

            with self.assertRaisesRegex(FileNotFoundError, "Adapter/model path not found"):
                run_unsloth_classification_inference_smoke(train_config, inference_config)


if __name__ == "__main__":
    unittest.main()
