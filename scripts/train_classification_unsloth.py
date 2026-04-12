"""Initialize, validate, or run Phase 4 classification training with Unsloth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from slm_auto_config.training.classification import (  # noqa: E402
    build_default_classification_train_config,
    dry_run_train_config,
    load_train_config,
    run_unsloth_classification_training,
    write_train_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="runs/demo-thai-classification-auto-001/train_config.json",
        help="Path to the Phase 4 training config JSON.",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/demo-thai-classification-auto-001",
        help="Run directory that contains the Phase 1-3 input folder.",
    )
    parser.add_argument("--model-id", default="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit")
    parser.add_argument("--chat-template", choices=["qwen", "llama", "gemma"], default="qwen")
    parser.add_argument("--init-config", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.init_config:
        config = build_default_classification_train_config(
            run_dir=args.run_dir,
            model_id=args.model_id,
            chat_template=args.chat_template,
            smoke_run=not args.full_run,
        )
        write_train_config(config, config_path)
        print(json.dumps({"config_path": config_path.as_posix(), "config": config.to_dict()}, ensure_ascii=False, indent=2))
        return

    config = load_train_config(config_path)
    if args.dry_run:
        print(json.dumps(dry_run_train_config(config), ensure_ascii=False, indent=2))
        return

    print(json.dumps(run_unsloth_classification_training(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
