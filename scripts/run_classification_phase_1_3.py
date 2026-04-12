"""Run or dry-run classification pipeline phases 1-3.

Default mode prints the plan only. Use `--run-sdg` to call the real SDG pipeline
and `--build-dataset` after synthetic data exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from slm_auto_config.app.classification_pipeline import (  # noqa: E402
    build_classification_dataset_from_phase_plan,
    build_classification_phase_plan,
)
from slm_auto_config.app.decision_plan import load_user_request  # noqa: E402
from slm_auto_config.datasets.quality import write_classification_quality_report  # noqa: E402
from slm_auto_config.synthetic.classification_sdg import run_classification_sdg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--request",
        default=str(PROJECT_ROOT / "examples" / "user_requests" / "classification_auto_request.json"),
    )
    parser.add_argument("--run-sdg", action="store_true")
    parser.add_argument("--quality-report", action="store_true")
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--few-shot-per-label", type=int, default=0)
    parser.add_argument("--sdg-target-count", type=int)
    parser.add_argument("--sdg-max-loops", type=int)
    parser.add_argument("--sdg-model")
    parser.add_argument("--rule-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--rule-max-new-tokens", type=int)
    parser.add_argument("--generator-max-new-tokens", type=int)
    parser.add_argument("--judge-max-new-tokens", type=int)
    parser.add_argument("--pipeline-batch-size", type=int)
    args = parser.parse_args()

    request = load_user_request(args.request)
    if args.sdg_target_count is not None:
        request = replace(request, synthetic_target_count=args.sdg_target_count)
    phase_plan = build_classification_phase_plan(request)
    sdg_config = phase_plan.sdg_config
    if args.sdg_max_loops is not None:
        sdg_config = replace(sdg_config, max_loops=args.sdg_max_loops)
    if args.sdg_model is not None:
        sdg_config = replace(sdg_config, model_name=args.sdg_model)
    if args.rule_model is not None:
        sdg_config = replace(sdg_config, rule_model_name=args.rule_model)
    if args.judge_model is not None:
        sdg_config = replace(sdg_config, judge_model_name=args.judge_model)
    if args.rule_max_new_tokens is not None:
        sdg_config = replace(sdg_config, rule_max_new_tokens=args.rule_max_new_tokens)
    if args.generator_max_new_tokens is not None:
        sdg_config = replace(sdg_config, generator_max_new_tokens=args.generator_max_new_tokens)
    if args.judge_max_new_tokens is not None:
        sdg_config = replace(sdg_config, judge_max_new_tokens=args.judge_max_new_tokens)
    if args.pipeline_batch_size is not None:
        sdg_config = replace(sdg_config, pipeline_batch_size=args.pipeline_batch_size)
    phase_plan = replace(phase_plan, sdg_config=sdg_config)
    print(json.dumps(phase_plan.to_dict(), ensure_ascii=False, indent=2))

    if args.run_sdg:
        run_classification_sdg(sdg_config)
        phase_plan = build_classification_phase_plan(request)

    if args.quality_report:
        quality_report_path = Path(phase_plan.decision_plan.sdg_stage["report_path"])
        report = write_classification_quality_report(
            input_path=phase_plan.decision_plan.sdg_stage["output_path"],
            output_path=quality_report_path,
            expected_target_count=sdg_config.target_count,
        )
        print(json.dumps({"quality_report": report}, ensure_ascii=False, indent=2))

    if args.build_dataset:
        dataset_summary = build_classification_dataset_from_phase_plan(
            phase_plan,
            few_shot_per_label=args.few_shot_per_label,
        )
        print(json.dumps({"dataset_summary": dataset_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
