"""Simulate receiving a user request and print the pipeline decision plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from slm_auto_config.app.decision_plan import build_decision_plan, load_user_request  # noqa: E402


def main() -> None:
    request_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PROJECT_ROOT / "examples" / "user_requests" / "classification_auto_request.json"
    )
    request = load_user_request(request_path)
    plan = build_decision_plan(request)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
