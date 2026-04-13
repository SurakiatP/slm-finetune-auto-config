"""Classification prompt and SFT message formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TextLabelRecord = dict[str, Any]


@dataclass(frozen=True)
class ClassificationPromptTemplate:
    """Topic-agnostic classification prompt template for SFT records."""

    template_id: str = "classification.topic_agnostic.v1"
    system_role: str = "system"
    user_role: str = "user"
    assistant_role: str = "assistant"
    system_intro: str = "You are a text classification model."
    rules: tuple[str, ...] = (
        "Return exactly one label.",
        "The label must be one of the valid labels.",
        "Do not explain your answer.",
    )
    task_prefix: str = "Task:"
    label_list_prefix: str = "Valid labels:"
    text_prefix: str = "Text:"
    few_shot_prefix: str = "Examples:"
    few_shot_input_prefix: str = "Input:"
    few_shot_label_prefix: str = "Category:"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable prompt policy for run artifacts."""
        return {
            "template_id": self.template_id,
            "roles": {
                "system": self.system_role,
                "user": self.user_role,
                "assistant": self.assistant_role,
            },
            "system_intro": self.system_intro,
            "rules": list(self.rules),
            "task_prefix": self.task_prefix,
            "label_list_prefix": self.label_list_prefix,
            "text_prefix": self.text_prefix,
            "few_shot_prefix": self.few_shot_prefix,
            "few_shot_input_prefix": self.few_shot_input_prefix,
            "few_shot_label_prefix": self.few_shot_label_prefix,
            "metadata": dict(self.metadata),
        }


def to_classification_sft_record(
    record: TextLabelRecord,
    *,
    task_description: str,
    labels: list[str],
    prompt_template: ClassificationPromptTemplate | None = None,
    few_shot_examples: list[TextLabelRecord] | None = None,
) -> dict[str, Any]:
    """Convert one text/label record to a topic-agnostic messages format."""
    template = prompt_template or ClassificationPromptTemplate()
    few_shot_examples = few_shot_examples or []
    system_prompt = build_classification_system_prompt(
        task_description=task_description,
        labels=labels,
        template=template,
        few_shot_examples=few_shot_examples,
    )
    return {
        "messages": [
            {"role": template.system_role, "content": system_prompt},
            {"role": template.user_role, "content": f"{template.text_prefix}\n{record['text']}"},
            {"role": template.assistant_role, "content": record["label"]},
        ],
        "metadata": {
            "task_type": "classification",
            "label": record["label"],
            "prompt_template": template.template_id,
            "few_shot_count": len(few_shot_examples),
            **template.metadata,
        },
    }


def build_classification_system_prompt(
    *,
    task_description: str,
    labels: list[str],
    template: ClassificationPromptTemplate,
    few_shot_examples: list[TextLabelRecord],
) -> str:
    """Build the system prompt shared by classification SFT records."""
    label_list = ", ".join(labels)
    sections = [
        template.system_intro,
        f"{template.task_prefix}\n{task_description}",
        f"{template.label_list_prefix}\n{label_list}",
        "Rules:\n" + "\n".join(f"- {rule}" for rule in template.rules),
    ]
    if few_shot_examples:
        sections.append(
            template.few_shot_prefix
            + "\n"
            + "\n\n".join(
                (
                    f"{template.few_shot_input_prefix} {example['text']}\n"
                    f"{template.few_shot_label_prefix} {example['label']}"
                )
                for example in few_shot_examples
            )
        )
    return "\n\n".join(sections)
