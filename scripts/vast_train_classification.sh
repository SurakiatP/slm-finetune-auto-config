#!/usr/bin/env bash
set -o pipefail

RUN_DIR="${RUN_DIR:-runs/demo-thai-classification-auto-001}"
CONFIG_PATH="${CONFIG_PATH:-$RUN_DIR/train_config.json}"
LOG_DIR="$RUN_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODE="${1:---dry-run}"

mkdir -p "$LOG_DIR"
export PYTHONPATH="${PYTHONPATH:-src}"

if [[ "$MODE" == "--train" ]]; then
  LOG_PATH="$LOG_DIR/phase4_train_$TIMESTAMP.log"
  python -B scripts/train_classification_unsloth.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"
elif [[ "$MODE" == "--infer" ]]; then
  LOG_PATH="$LOG_DIR/phase4_infer_$TIMESTAMP.log"
  python -B scripts/infer_classification_unsloth.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"
elif [[ "$MODE" == "--infer-dry-run" ]]; then
  LOG_PATH="$LOG_DIR/phase4_infer_dry_run_$TIMESTAMP.log"
  python -B scripts/infer_classification_unsloth.py --config "$CONFIG_PATH" --dry-run 2>&1 | tee "$LOG_PATH"
else
  LOG_PATH="$LOG_DIR/phase4_dry_run_$TIMESTAMP.log"
  python -B scripts/train_classification_unsloth.py --config "$CONFIG_PATH" --dry-run 2>&1 | tee "$LOG_PATH"
fi

EXIT_CODE="${PIPESTATUS[0]}"
echo "Log saved to: $LOG_PATH"
exit "$EXIT_CODE"
