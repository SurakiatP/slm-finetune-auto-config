"""Run a post-training classification inference smoke test with Unsloth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from slm_auto_config.training.classification import load_train_config  # noqa: E402
from slm_auto_config.training.classification_inference import (  # noqa: E402
    build_default_inference_config,
    dry_run_inference_config,
    run_unsloth_classification_inference_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="runs/demo-thai-classification-auto-001/train_config.json",
        help="Path to the Phase 4 training config JSON.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of test samples to run through inference.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_config = load_train_config(args.config)
    inference_config = build_default_inference_config(
        train_config,
        sample_count=args.sample_count,
    )
    if args.dry_run:
        result = dry_run_inference_config(inference_config)
    else:
        result = run_unsloth_classification_inference_smoke(train_config, inference_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
