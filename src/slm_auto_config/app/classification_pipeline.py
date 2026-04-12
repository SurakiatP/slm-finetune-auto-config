"""Phase 1-3 classification pipeline bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_auto_config.app.decision_plan import PipelineDecisionPlan, build_decision_plan
from slm_auto_config.app.schemas import FinetuneRequest
from slm_auto_config.datasets.classification import build_classification_split_and_sft
from slm_auto_config.synthetic.classification_sdg import ClassificationSDGConfig


@dataclass(frozen=True)
class ClassificationPhasePlan:
    """Dry-run plan for classification phases 1-3."""

    decision_plan: PipelineDecisionPlan
    sdg_config: ClassificationSDGConfig
    dataset_output_dir: str
    can_build_dataset_now: bool
    missing_inputs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_plan": self.decision_plan.to_dict(),
            "sdg_config": asdict(self.sdg_config),
            "dataset_output_dir": self.dataset_output_dir,
            "can_build_dataset_now": self.can_build_dataset_now,
            "missing_inputs": self.missing_inputs,
        }


def build_classification_phase_plan(request: FinetuneRequest) -> ClassificationPhasePlan:
    """Connect the user request to classification SDG and dataset-build stages."""
    if request.task_type != "classification":
        raise ValueError("This pipeline currently supports classification only.")
    if request.seed_data.type != "file" or not request.seed_data.path:
        raise ValueError("Classification SDG currently expects seed_data.type=file with a path.")

    decision_plan = build_decision_plan(request)
    synthetic_output_path = decision_plan.sdg_stage["output_path"]
    dataset_train_path = Path(decision_plan.dataset_stage["train_path"])
    dataset_output_dir = dataset_train_path.parent.as_posix()
    missing_inputs = []
    if not Path(request.seed_data.path).exists():
        missing_inputs.append(request.seed_data.path)
    if not Path(synthetic_output_path).exists():
        missing_inputs.append(synthetic_output_path)

    return ClassificationPhasePlan(
        decision_plan=decision_plan,
        sdg_config=ClassificationSDGConfig(
            seed_data_path=request.seed_data.path,
            output_path=synthetic_output_path,
            target_count=request.synthetic_target_count,
            task_description=request.task_description,
            log_path=(Path(synthetic_output_path).parents[1] / "logs" / "sdg_debug.log").as_posix(),
        ),
        dataset_output_dir=dataset_output_dir,
        can_build_dataset_now=len(missing_inputs) == 0,
        missing_inputs=missing_inputs,
    )


def build_classification_dataset_from_phase_plan(
    phase_plan: ClassificationPhasePlan,
    *,
    few_shot_per_label: int = 0,
) -> dict[str, Any]:
    """Build split and SFT files after SDG output exists."""
    if phase_plan.missing_inputs:
        raise FileNotFoundError(
            "Cannot build classification dataset yet. Missing: "
            + ", ".join(phase_plan.missing_inputs)
        )

    decision_plan = phase_plan.decision_plan
    return build_classification_split_and_sft(
        seed_data_path=phase_plan.sdg_config.seed_data_path,
        synthetic_data_path=phase_plan.sdg_config.output_path,
        output_dir=phase_plan.dataset_output_dir,
        task_description=decision_plan.task_description,
        few_shot_per_label=few_shot_per_label,
    )
