"""Compare original and refactored classification SDG pure contracts.

This script intentionally avoids running the real LLM/Distilabel pipeline. It
compares pure helper behavior and the public SDG entrypoint signature so we can
check refactor safety before spending API/GPU budget.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from slm_auto_config.synthetic import classification_sdg as refactored  # noqa: E402


def get_original_entrypoint_parameters() -> list[str]:
    """Read the original function signature without importing heavy dependencies."""
    source_path = PROJECT_ROOT / "classification_sdg_edit_promptV2.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_classification_sdg_iterative":
            return [arg.arg for arg in node.args.args]
    raise RuntimeError("Original run_classification_sdg_iterative function was not found.")


def main() -> None:
    sample_generation = json.dumps(
        {
            "results": [
                "candidate text one",
                "candidate text two",
                "candidate text three",
                "candidate text four",
                "candidate text five",
            ]
        }
    )
    fenced_generation = f"```json\n{sample_generation}\n```"
    refactored_parameters = list(
        inspect.signature(refactored.run_classification_sdg_iterative).parameters
    )

    checks = {
        "normalize_text_contract": refactored.normalize_text(" A! ข้อความ 123 ") == "aข้อความ123",
        "extract_multiple_outputs_json_contract": refactored.extract_multiple_outputs(sample_generation)
        == [
            "candidate text one",
            "candidate text two",
            "candidate text three",
            "candidate text four",
            "candidate text five",
        ],
        "extract_multiple_outputs_fenced_json_contract": refactored.extract_multiple_outputs(fenced_generation)
        == [
            "candidate text one",
            "candidate text two",
            "candidate text three",
            "candidate text four",
            "candidate text five",
        ],
        "entrypoint_parameters_match_original": get_original_entrypoint_parameters()
        == refactored_parameters,
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Refactor comparison failed: {failed}")

    print("Refactor comparison passed for pure contracts.")


if __name__ == "__main__":
    main()
