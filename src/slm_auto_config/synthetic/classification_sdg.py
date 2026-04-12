"""Refactored classification synthetic data generation pipeline.

This module keeps the same external contract as `classification_sdg_edit_promptV2.py`:
input seed records use `text` and `label`, and accepted synthetic records are
written as JSON records with `text` and `label`.
"""

import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:

    def Field(default: Any, **_: Any) -> Any:
        return default

    class BaseModel:
        """Small fallback used only when pydantic is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            for field_name in self.__annotations__:
                if field_name not in kwargs:
                    raise ValueError(f"Missing field: {field_name}")
                setattr(self, field_name, kwargs[field_name])

        @classmethod
        def model_validate_json(cls, payload: str):
            return cls(**json.loads(payload))

if TYPE_CHECKING:
    from openai import OpenAI


try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

logger = logging.getLogger(__name__)

try:
    from distilabel.steps import Step, StepInput
except ModuleNotFoundError:
    StepInput = list[dict[str, Any]]

    class Step:  # type: ignore[no-redef]
        """Fallback so pure helper tests can import without Distilabel installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

GENERATOR_SYSTEM_PROMPT = """[Role]
You are a Senior Synthetic Data Engineer specializing in generating high-quality datasets for Small Language Model (SLM) fine-tuning.

[Task]
Your mission is to produce highly specific, diverse, and accurate text classifications that match a target label perfectly.

[Context]
High data fidelity is critical. The generated text must exhibit the nuances, vocabulary, and structural variations characteristic of authentic human-written content. Ensure maximum variation across multiple generated samples.

[Few-shot Guideline]
When provided with seed examples, capture their stylistic essence (tone, length, vocabulary) while ensuring all your generated outputs are 100% unique from the seeds and entirely distinct from one another.

[Output Instructions]
1. Brainstorm and generate EXACTLY 5 highly diverse candidate texts.
2. Ensure each candidate has a significantly different tone, perspective, professional terminology, length, or structure from the others.
3. Output ONLY a valid JSON object matching this exact schema:
{
  "results": [
    "diverse candidate text 1...",
    "diverse candidate text 2...",
    "diverse candidate text 3...",
    "diverse candidate text 4...",
    "diverse candidate text 5..."
  ]
}
DO NOT include any markdown formatting, <think> tags, explanations, or conversational text. Output only the pure JSON string.
"""

UNIFIED_GENERATOR_TEMPLATE = """Task Description: {{ task_description }}
Target Category/Label: '{{ label }}'

[Constraint: Diversity/Difficulty]
Rule: {{ diversity_rule }}
Difficulty: {{ difficulty }}

[Style Reference: Seed Examples (DO NOT COPY)]
{% if examples %}
{% for ex in examples %}
- {{ ex }}
{% endfor %}
{% else %}
No specific examples provided. Please generate highly creative and original examples for the '{{ label }}' category that are distinct from the general task.
{% endif %}

Please provide EXACTLY 5 unique text examples that fulfill the requirements of the '{{ label }}' label.
CRITICAL DIVERSITY CHECK: Each of the 5 examples MUST be drastically different from the others. Do not just change a few words. Create completely different scenarios, writing styles, or contexts for each one of them while strictly adhering to the Constraint and Label.

Output ONLY a raw JSON object containing an array of 5 strings under the "results" key.
DO NOT include any introductions, output candidate numbers, brainstorming, or internal evaluation. If you include any text outside the JSON structure, the entire output will be rejected."""

JUDGE_TEMPLATE = """Evaluate the following generated text for a classification task.
Task Description: {{ task_description }}
Generated Text: {{ cleaned_text }}
Target Label: {{ label }}

# Definition of 'unknown' Label:
If the Target Label is 'unknown', it means the text should be irrelevant, out-of-scope, or nonsensical relative to the Task Description provided.

# Evaluation Metrics (0.0 to 1.0):
1. Fidelity: Does the core meaning of the text a perfect match for the category '{{ label }}'?
2. Naturalness: Is the review fluent and realistic? (For 'unknown', it should just be coherent text unless it's intended to be gibberish).
3. Utility: Is this a high-quality example for training a classifier?

Output ONLY a JSON object matching this schema:
{
  "fidelity": float,
  "naturalness": float,
  "utility": float,
  "reasoning": "string"
}
Do not include any other text. Output scores MUST be a FLOAT between 0.0 and 1.0 (e.g., 0.85). DO NOT use fractions (like 9/10), strings, or negative values. Be extremely strict."""


class JudgeOutput(BaseModel):
    fidelity: float = Field(..., ge=0, le=1, description="Score matching the target label")
    naturalness: float = Field(..., ge=0, le=1, description="Score for fluency and realism")
    utility: float = Field(..., ge=0, le=1, description="Score for training value/nuance")
    reasoning: str = Field(..., description="Explanation for the scores")


class SDGRules(BaseModel):
    diversity_rules: list[str] = Field(
        ..., description="List of at least 8 rules for diversity in generation"
    )
    unknown_diversity_rules: list[str] = Field(
        ..., description="List of at least 5 rules for 'unknown' class data generation"
    )


class GeneratorBatchOutput(BaseModel):
    results: list[str] = Field(..., description="List of generated diverse candidate strings")


@dataclass(frozen=True)
class ClassificationSDGConfig:
    seed_data_path: str
    output_path: str
    target_count: int
    task_description: str
    model_name: str = "qwen/qwen3-235b-a22b-2507"
    rule_model_name: str = "qwen/qwen3.6-plus"
    judge_model_name: str = "openai/gpt-4o-mini"
    max_loops: int = 20
    unknown_ratio: float = 0.10
    judge_threshold: float = 0.70
    semantic_similarity_threshold: float = 0.85
    minhash_threshold: float = 0.90
    minhash_num_perm: int = 128
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    log_path: str = "logs/sdg_debug.log"
    rule_max_new_tokens: int = 1024
    generator_max_new_tokens: int = 2048
    judge_max_new_tokens: int = 512
    pipeline_batch_size: int = 50


class CleanTextStep(Step):
    """Explode a generator response into one cleaned text row per candidate."""

    @property
    def inputs(self) -> list[str]:
        return ["generated_text"]

    @property
    def outputs(self) -> list[str]:
        return ["cleaned_text"]

    def process(self, inputs: StepInput):
        exploded_outputs = []
        for item in inputs:
            cleaned_list = extract_multiple_outputs(item.get("generated_text", ""))
            if not cleaned_list:
                new_item = item.copy()
                new_item["cleaned_text"] = ""
                exploded_outputs.append(new_item)
                continue
            for extracted_text in cleaned_list:
                new_item = item.copy()
                new_item["cleaned_text"] = extracted_text
                exploded_outputs.append(new_item)
        yield exploded_outputs


@dataclass(frozen=True)
class LabelPlan:
    seed_labels: list[str]
    labels: list[str]
    label_examples: dict[str, list[str]]
    target_counts: dict[str, int]
    target_unknown: int
    target_per_seed_label: int


def configure_logging(log_path: str = "logs/sdg_debug.log") -> None:
    """Configure file and console logging once."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def normalize_text(text: Any) -> str:
    """Normalize text for duplicate checks."""
    return re.sub(r"[^a-zA-Z0-9ก-๙]", "", str(text)).lower()


def extract_multiple_outputs(text: str) -> list[str]:
    """Parse the LLM JSON output to extract generated candidate strings."""
    if not text:
        return []

    text = re.sub(r"<(think|reasoning)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)

    try:
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = clean_text[start_idx : end_idx + 1]
            validated_data = GeneratorBatchOutput.model_validate_json(json_str)
            return [result.strip() for result in validated_data.results if result.strip()]
    except Exception as exc:
        logger.warning("Error parsing JSON from Generator: %s | Raw text: %s...", exc, text[:150])
        results = re.findall(r'"([^"]+)"', clean_text)
        filtered_results = [result for result in results if result != "results" and len(result.strip()) > 10]
        if filtered_results:
            return filtered_results
    return []


def load_seed_data(seed_data_path: str | Path) -> list[dict[str, Any]]:
    """Load seed examples from the original `text`/`label` JSON contract."""
    path = Path(seed_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed data file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Seed data must be a JSON array.")
    return payload


def build_label_plan(seed_data: list[dict[str, Any]], target_count: int, unknown_ratio: float = 0.10) -> LabelPlan:
    """Discover labels and compute per-label quotas using the original policy."""
    label_examples: dict[str, list[str]] = {}
    for item in seed_data:
        label = item.get("label")
        text = item.get("text")
        if label and text:
            label_examples.setdefault(label, []).append(text)

    seed_labels = list(label_examples.keys())
    if not seed_labels:
        raise ValueError("No valid labels found in seed data.")

    target_unknown = math.ceil(target_count * unknown_ratio)
    target_others = target_count - target_unknown
    target_per_seed_label = math.ceil(target_others / len(seed_labels))

    labels = seed_labels + ["unknown"]
    label_examples["unknown"] = []
    target_counts = _allocate_seed_label_targets(seed_labels, target_others)
    target_counts["unknown"] = target_unknown

    return LabelPlan(
        seed_labels=seed_labels,
        labels=labels,
        label_examples=label_examples,
        target_counts=target_counts,
        target_unknown=target_unknown,
        target_per_seed_label=target_per_seed_label,
    )


def _allocate_seed_label_targets(seed_labels: list[str], target_others: int) -> dict[str, int]:
    base_count = target_others // len(seed_labels)
    remainder = target_others % len(seed_labels)
    target_counts = {}
    for index, label in enumerate(seed_labels):
        target_counts[label] = base_count + (1 if index < remainder else 0)
    return target_counts


def build_minhash(text: str, *, num_perm: int = 128):
    """Build a MinHash for a text using the original 5-gram policy."""
    from datasketch import MinHash

    minhash = MinHash(num_perm=num_perm)
    clean_text = normalize_text(text)
    if len(clean_text) < 5:
        minhash.update(clean_text.encode("utf-8"))
    else:
        for index in range(len(clean_text) - 4):
            minhash.update(clean_text[index : index + 5].encode("utf-8"))
    return minhash


def generate_sdg_rules(
    client: "OpenAI",
    task_description: str,
    labels: list[str],
    model_name: str = "qwen/qwen3.6-plus",
    max_tokens: int = 1024,
) -> SDGRules:
    """Use a meta-prompt to generate SDG diversity rules."""
    prompt = f"""You are a Senior Data Engineer. Your task is to brainstorm "Diversity Rules" for a Synthetic Data Generation pipeline.
These rules will be used to guide an LLM to generate high-quality, diverse, and realistic training data for a classification task.

[Task Description]
{task_description}

[Target Labels]
{", ".join(labels)}

[Requirements]
1. Diversity Rules: These rules should encourage the generator to use different perspectives, professional jargon, tones, or focus on specific nuances of the task. (e.g., 'Write from the perspective of an expert', 'Focus on edge cases like X').
2. Unknown Rules: These rules should create 'out-of-distribution' data that is NOT related to the main task but might be common noise (e.g., 'General greetings', 'Unstructured chatter about weather', 'Technical manuals for unrelated items').

Please provide a JSON object following this schema:
{{
  "diversity_rules": ["rule1", "rule2", ...],
  "unknown_diversity_rules": ["rule1", "rule2", ...]
}}
Output ONLY the JSON object.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a Senior Data Engineer. Return ONLY JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=max_tokens,
    )
    raw_json = response.choices[0].message.content or "{}"
    clean_json = raw_json.strip()
    if clean_json.startswith("```"):
        clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
        clean_json = re.sub(r"\s*```$", "", clean_json)
    return SDGRules.model_validate_json(clean_json)


def make_pipeline_inputs(
    *,
    label_plan: LabelPlan,
    collected_counts: dict[str, int],
    diversity_rules: list[str],
    unknown_diversity_rules: list[str],
    task_description: str,
    difficulties: list[str],
) -> list[dict[str, Any]]:
    """Create Distilabel input rows for labels that still need more examples."""
    pipeline_inputs: list[dict[str, Any]] = []
    for label in label_plan.labels:
        needed = label_plan.target_counts[label] - collected_counts[label]
        if needed <= 0:
            continue

        generate_quota = max(1, math.ceil((needed / 4.0) * 1.5))
        for _ in range(generate_quota):
            if label == "unknown":
                examples: list[str] = []
                diversity_rule = random.choice(unknown_diversity_rules)
            else:
                examples = random.sample(label_plan.label_examples[label], min(3, len(label_plan.label_examples[label])))
                diversity_rule = random.choice(diversity_rules)
            pipeline_inputs.append(
                {
                    "task_description": task_description,
                    "label": label,
                    "examples": examples,
                    "difficulty": random.choice(difficulties),
                    "diversity_rule": diversity_rule,
                }
            )
    random.shuffle(pipeline_inputs)
    return pipeline_inputs


def calculate_overall_judge_score(judge_data: JudgeOutput) -> float:
    """Calculate the original weighted judge score."""
    return (judge_data.fidelity * 0.4) + (judge_data.naturalness * 0.3) + (judge_data.utility * 0.3)


def run_classification_sdg_iterative(
    seed_data_path: str,
    output_path: str,
    target_count: int,
    task_description: str,
    model_name: str = "qwen/qwen3-235b-a22b-2507",
    rule_model_name: str = "qwen/qwen3.6-plus",
    judge_model_name: str = "openai/gpt-4o-mini",
    max_loops: int = 20,
    rule_max_new_tokens: int = 1024,
    generator_max_new_tokens: int = 2048,
    judge_max_new_tokens: int = 512,
    pipeline_batch_size: int = 50,
) -> None:
    """Run classification SDG with the same callable signature as the original script."""
    config = ClassificationSDGConfig(
        seed_data_path=seed_data_path,
        output_path=output_path,
        target_count=target_count,
        task_description=task_description,
        model_name=model_name,
        rule_model_name=rule_model_name,
        judge_model_name=judge_model_name,
        max_loops=max_loops,
        rule_max_new_tokens=rule_max_new_tokens,
        generator_max_new_tokens=generator_max_new_tokens,
        judge_max_new_tokens=judge_max_new_tokens,
        pipeline_batch_size=pipeline_batch_size,
    )
    run_classification_sdg(config)


def run_classification_sdg(config: ClassificationSDGConfig) -> list[dict[str, str]]:
    """Run classification SDG and return accepted synthetic records."""
    from datasketch import MinHashLSH
    from distilabel.llms import OpenAILLM
    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts
    from distilabel.steps.tasks import TextGeneration
    from openai import OpenAI
    import faiss
    from sentence_transformers import SentenceTransformer

    configure_logging(config.log_path)
    logger.info("Loading seed data from %s...", config.seed_data_path)

    seed_data = load_seed_data(config.seed_data_path)
    label_plan = build_label_plan(seed_data, config.target_count, config.unknown_ratio)

    logger.info("Found seed labels: %s", label_plan.seed_labels)
    logger.info("Target count: %s", config.target_count)
    logger.info("Target per seed label: %s", label_plan.target_per_seed_label)
    logger.info("Target unknown count: %s", label_plan.target_unknown)

    collected_output: list[dict[str, str]] = []
    collected_counts = {label: 0 for label in label_plan.labels}

    global_lsh = MinHashLSH(threshold=config.minhash_threshold, num_perm=config.minhash_num_perm)
    lsh_counter = 0
    seed_texts = []
    for item in seed_data:
        text = item.get("text", "")
        if text:
            global_lsh.insert(f"seed_{lsh_counter}", build_minhash(text, num_perm=config.minhash_num_perm))
            lsh_counter += 1
            seed_texts.append(text)

    logger.info("Loading embedding model %s...", config.embedding_model_name)
    embedding_model = SentenceTransformer(config.embedding_model_name)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    if seed_texts:
        seed_embeddings = embedding_model.encode(seed_texts, normalize_embeddings=True)
        faiss_index.add(seed_embeddings)

    base_url = os.getenv("OPENROUTER_BASE_URL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        sdg_rules = generate_sdg_rules(
            client,
            config.task_description,
            label_plan.labels,
            model_name=config.rule_model_name,
            max_tokens=config.rule_max_new_tokens,
        )
        diversity_rules = sdg_rules.diversity_rules
        unknown_diversity_rules = sdg_rules.unknown_diversity_rules
    except Exception as exc:
        logger.warning("Meta-prompting failed, using fallback rules: %s", exc)
        diversity_rules = ["Focus on specific details.", "Use formal language.", "Consider different perspectives."]
        unknown_diversity_rules = ["Generic chatter.", "Unrelated news snippets."]

    difficulties = ["easy", "medium", "hard", "complex-structure"]
    loop_count = 1

    while sum(collected_counts.values()) < config.target_count and loop_count <= config.max_loops:
        logger.info("Starting SDG loop %s/%s", loop_count, config.max_loops)
        pipeline_inputs = make_pipeline_inputs(
            label_plan=label_plan,
            collected_counts=collected_counts,
            diversity_rules=diversity_rules,
            unknown_diversity_rules=unknown_diversity_rules,
            task_description=config.task_description,
            difficulties=difficulties,
        )

        with Pipeline(name=f"classification-sdg-loop-{loop_count}") as pipeline:
            loader = LoadDataFromDicts(data=pipeline_inputs, batch_size=config.pipeline_batch_size)
            generator = TextGeneration(
                name="generate_text",
                llm=OpenAILLM(
                    model=config.model_name,
                    base_url=base_url,
                    api_key=api_key,
                    generation_kwargs={
                        "max_new_tokens": config.generator_max_new_tokens,
                        "temperature": 0.7,
                        "response_format": {"type": "json_object"},
                    },
                ),
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                template=UNIFIED_GENERATOR_TEMPLATE,
                columns=["task_description", "label", "examples", "difficulty", "diversity_rule"],
                output_mappings={"generation": "generated_text"},
                input_batch_size=config.pipeline_batch_size,
            )
            cleaner = CleanTextStep(name="clean_text", input_batch_size=config.pipeline_batch_size)
            judge = TextGeneration(
                name="judge_text",
                llm=OpenAILLM(
                    model=config.judge_model_name,
                    base_url=base_url,
                    api_key=api_key,
                    generation_kwargs={
                        "max_new_tokens": config.judge_max_new_tokens,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                ),
                template=JUDGE_TEMPLATE,
                columns=["task_description", "label", "cleaned_text"],
                output_mappings={"generation": "judge_raw_output"},
                input_batch_size=config.pipeline_batch_size,
            )
            loader >> generator >> cleaner >> judge

        distiset = pipeline.run(use_cache=False)
        dataset = distiset["default"]["train"]

        all_texts = [row.get("cleaned_text", "") for row in dataset]
        valid_indices = [
            index for index, text in enumerate(all_texts) if text and text.strip() and text.lower() != "none"
        ]
        valid_texts = [all_texts[index] for index in valid_indices]
        vector_map = {}
        if valid_texts:
            batch_vectors = embedding_model.encode(valid_texts, normalize_embeddings=True, show_progress_bar=False)
            vector_map = {index: batch_vectors[position] for position, index in enumerate(valid_indices)}

        added_in_loop = 0
        for index, row in enumerate(dataset):
            label = row["label"]
            text = row.get("cleaned_text", "")
            if collected_counts[label] >= label_plan.target_counts[label]:
                continue
            if index not in vector_map:
                logger.warning("Generation failed or was empty for row %s label %s.", index, label)
                continue

            try:
                judge_data = JudgeOutput.model_validate_json(row.get("judge_raw_output", "{}"))
                overall_score = calculate_overall_judge_score(judge_data)
                if overall_score < config.judge_threshold:
                    logger.info("Discarded: score %.2f < %.2f", overall_score, config.judge_threshold)
                    continue
            except Exception as exc:
                logger.warning("Judge parsing/validation error, rejecting row: %s", exc)
                continue

            try:
                new_embedding = vector_map[index].reshape(1, -1)
                distances, _indices = faiss_index.search(new_embedding, 1)
                max_similarity = float(distances[0][0])
                if max_similarity >= config.semantic_similarity_threshold:
                    logger.info(
                        "Discarded: FAISS similarity %.2f >= %.2f",
                        max_similarity,
                        config.semantic_similarity_threshold,
                    )
                    continue
            except Exception as exc:
                logger.warning("FAISS similarity check error: %s", exc)

            global_lsh.insert(f"gen_{lsh_counter}", build_minhash(text, num_perm=config.minhash_num_perm))
            lsh_counter += 1
            faiss_index.add(new_embedding)
            collected_output.append({"text": text, "label": label})
            collected_counts[label] += 1
            added_in_loop += 1

            if sum(collected_counts.values()) >= config.target_count:
                break

        logger.info("Finished loop %s, accepted %s records.", loop_count, added_in_loop)
        loop_count += 1

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(collected_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved synthetic data to %s", output_path)
    return collected_output
