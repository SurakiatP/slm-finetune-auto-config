"""Synthetic data quality gate and report helpers."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median, quantiles
from typing import Any

from slm_auto_config.datasets.classification import load_text_label_json


THAI_CHAR_RE = re.compile(r"[\u0e00-\u0e7f]")
ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
PROMPT_ARTIFACT_RE = re.compile(r"\b(candidate|example|target|output|label)\b", re.IGNORECASE)
JSON_ARTIFACT_RE = re.compile(r"[{}]|```|\\n")
PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]|_{3,}")


DEFAULT_CLASSIFICATION_KEYWORDS: dict[str, list[str]] = {}


@dataclass(frozen=True)
class QualityGateConfig:
    """Thresholds for a lightweight, topic-agnostic pre-training quality gate.

    Defaults intentionally avoid domain assumptions. The gate does not know
    whether the seed data is about legal documents, medical text, food reviews,
    finance, Thai, English, or a mixed-language task. Domain-specific checks
    can be enabled later by passing `label_keyword_map` or stricter language
    thresholds from a task adapter or UI setting.
    """

    min_records: int = 1
    max_exact_duplicate_ratio: float = 0.01
    max_empty_ratio: float = 0.0
    max_english_heavy_ratio: float = 1.0
    max_prompt_artifact_ratio: float = 0.02
    max_placeholder_ratio: float = 0.20
    max_unknown_in_scope_keyword_ratio: float = 1.0
    min_label_count_ratio_to_expected: float = 0.70
    sample_limit_per_issue: int = 5
    label_keyword_map: dict[str, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_CLASSIFICATION_KEYWORDS)
    )


def build_classification_quality_report(
    records: list[dict[str, Any]],
    *,
    expected_target_count: int | None = None,
    config: QualityGateConfig | None = None,
) -> dict[str, Any]:
    """Build a deterministic quality report for classification `text`/`label` records."""
    gate_config = config or QualityGateConfig()
    record_count = len(records)
    labels = [str(record.get("label", "")).strip() for record in records]
    texts = [str(record.get("text", "")).strip() for record in records]
    normalized_texts = [_normalize_for_duplicate(text) for text in texts]
    label_counts = Counter(labels)
    exact_duplicate_count = record_count - len(set(normalized_texts))
    empty_count = sum(1 for text in texts if not text)
    length_values = [len(text) for text in texts]

    flagged_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    english_heavy_count = 0
    prompt_artifact_count = 0
    json_artifact_count = 0
    placeholder_count = 0
    unknown_in_scope_keyword_count = 0

    in_scope_keywords = _flatten_keywords(gate_config.label_keyword_map)
    for record in records:
        text = str(record.get("text", "")).strip()
        label = str(record.get("label", "")).strip()
        if _is_english_heavy(text):
            english_heavy_count += 1
            _add_flag(flagged_examples, "english_heavy", record, gate_config.sample_limit_per_issue)
        if PROMPT_ARTIFACT_RE.search(text):
            prompt_artifact_count += 1
            _add_flag(flagged_examples, "prompt_artifact", record, gate_config.sample_limit_per_issue)
        if JSON_ARTIFACT_RE.search(text):
            json_artifact_count += 1
            _add_flag(flagged_examples, "json_artifact", record, gate_config.sample_limit_per_issue)
        if PLACEHOLDER_RE.search(text):
            placeholder_count += 1
            _add_flag(flagged_examples, "placeholder", record, gate_config.sample_limit_per_issue)
        if in_scope_keywords and label == "unknown" and _contains_any_keyword(text, in_scope_keywords):
            unknown_in_scope_keyword_count += 1
            _add_flag(
                flagged_examples,
                "unknown_in_scope_keyword",
                record,
                gate_config.sample_limit_per_issue,
            )

    issue_counts = {
        "exact_duplicate": exact_duplicate_count,
        "empty_text": empty_count,
        "english_heavy": english_heavy_count,
        "prompt_artifact": prompt_artifact_count,
        "json_artifact": json_artifact_count,
        "placeholder": placeholder_count,
        "unknown_in_scope_keyword": unknown_in_scope_keyword_count,
    }
    ratios = {
        issue_name: _ratio(count, record_count) for issue_name, count in issue_counts.items()
    }
    gate_results = _evaluate_gate(
        record_count=record_count,
        label_counts=label_counts,
        ratios=ratios,
        expected_target_count=expected_target_count,
        config=gate_config,
    )
    score = _quality_score(gate_results, ratios, gate_config)
    return {
        "status": _overall_status(gate_results),
        "score": score,
        "summary": {
            "record_count": record_count,
            "expected_target_count": expected_target_count,
            "schema": "classification_text_label",
            "topic_assumptions": {
                "topic_agnostic": not bool(gate_config.label_keyword_map),
                "label_keyword_map_provided": bool(gate_config.label_keyword_map),
                "english_heavy_gate_enabled": gate_config.max_english_heavy_ratio < 1.0,
            },
            "label_counts": dict(label_counts),
            "length": _length_summary(length_values),
        },
        "issue_counts": issue_counts,
        "issue_ratios": ratios,
        "gate_config": asdict(gate_config),
        "gate_results": gate_results,
        "flagged_examples": dict(flagged_examples),
    }


def write_classification_quality_report(
    *,
    input_path: str | Path,
    output_path: str | Path,
    expected_target_count: int | None = None,
    config: QualityGateConfig | None = None,
) -> dict[str, Any]:
    """Load classification records, write a quality report, and return it."""
    records = load_text_label_json(input_path)
    report = build_classification_quality_report(
        records,
        expected_target_count=expected_target_count,
        config=config,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _evaluate_gate(
    *,
    record_count: int,
    label_counts: Counter[str],
    ratios: dict[str, float],
    expected_target_count: int | None,
    config: QualityGateConfig,
) -> dict[str, dict[str, Any]]:
    expected_min_count = config.min_records
    if expected_target_count is not None:
        expected_min_count = max(expected_min_count, int(expected_target_count * 0.95))

    known_label_counts = [count for label, count in label_counts.items() if label != "unknown"]
    min_label_count = min(known_label_counts) if known_label_counts else 0
    max_label_count = max(known_label_counts) if known_label_counts else 0
    label_balance_ratio = _ratio(min_label_count, max_label_count) if max_label_count else 1.0

    return {
        "record_count": {
            "status": "pass" if record_count >= expected_min_count else "fail",
            "value": record_count,
            "threshold": expected_min_count,
        },
        "exact_duplicate_ratio": _max_ratio_gate(
            ratios["exact_duplicate"], config.max_exact_duplicate_ratio
        ),
        "empty_ratio": _max_ratio_gate(ratios["empty_text"], config.max_empty_ratio),
        "english_heavy_ratio": _max_ratio_gate(
            ratios["english_heavy"], config.max_english_heavy_ratio
        ),
        "prompt_artifact_ratio": _max_ratio_gate(
            ratios["prompt_artifact"], config.max_prompt_artifact_ratio
        ),
        "placeholder_ratio": _max_ratio_gate(ratios["placeholder"], config.max_placeholder_ratio),
        "unknown_in_scope_keyword_ratio": _max_ratio_gate(
            ratios["unknown_in_scope_keyword"], config.max_unknown_in_scope_keyword_ratio
        ),
        "label_balance_ratio": {
            "status": "pass"
            if label_balance_ratio >= config.min_label_count_ratio_to_expected
            else "warn",
            "value": round(label_balance_ratio, 4),
            "threshold": config.min_label_count_ratio_to_expected,
        },
    }


def _max_ratio_gate(value: float, threshold: float) -> dict[str, Any]:
    return {
        "status": "pass" if value <= threshold else "warn",
        "value": round(value, 4),
        "threshold": threshold,
    }


def _quality_score(
    gate_results: dict[str, dict[str, Any]],
    ratios: dict[str, float],
    config: QualityGateConfig,
) -> int:
    penalty = 0
    for result in gate_results.values():
        if result["status"] == "warn":
            penalty += 8
        elif result["status"] == "fail":
            penalty += 25
    penalty += int(round(ratios["exact_duplicate"] * 200))
    penalty += int(round(ratios["prompt_artifact"] * 200))
    penalty += int(round(ratios["placeholder"] * 50))
    if config.max_english_heavy_ratio < 1.0:
        penalty += int(round(ratios["english_heavy"] * 100))
    if config.label_keyword_map:
        penalty += int(round(ratios["unknown_in_scope_keyword"] * 100))
    return max(0, 100 - penalty)


def _overall_status(gate_results: dict[str, dict[str, Any]]) -> str:
    statuses = {result["status"] for result in gate_results.values()}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _length_summary(length_values: list[int]) -> dict[str, float | int]:
    if not length_values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0}
    if len(length_values) < 4:
        value = int(median(length_values))
        return {
            "min": min(length_values),
            "p25": value,
            "median": value,
            "p75": value,
            "max": max(length_values),
        }
    quartiles = quantiles(length_values, n=4)
    return {
        "min": min(length_values),
        "p25": round(quartiles[0], 2),
        "median": round(median(length_values), 2),
        "p75": round(quartiles[2], 2),
        "max": max(length_values),
    }


def _is_english_heavy(text: str) -> bool:
    thai_count = len(THAI_CHAR_RE.findall(text))
    english_count = len(ASCII_ALPHA_RE.findall(text))
    return english_count > thai_count and english_count >= 20


def _flatten_keywords(label_keyword_map: dict[str, list[str]]) -> list[str]:
    return sorted(
        {keyword.lower() for keywords in label_keyword_map.values() for keyword in keywords}
    )


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in keywords)


def _add_flag(
    flagged_examples: dict[str, list[dict[str, str]]],
    issue_name: str,
    record: dict[str, Any],
    sample_limit: int,
) -> None:
    if len(flagged_examples[issue_name]) >= sample_limit:
        return
    flagged_examples[issue_name].append(
        {
            "label": str(record.get("label", "")),
            "text": str(record.get("text", ""))[:500],
        }
    )


def _ratio(value: int, total: int) -> float:
    return 0.0 if total == 0 else round(value / total, 4)


def _normalize_for_duplicate(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
