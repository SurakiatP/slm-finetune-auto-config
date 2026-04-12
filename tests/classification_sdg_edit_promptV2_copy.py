import json
import math
import random
import os
import re
import logging
from datetime import datetime
from dotenv import load_dotenv

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, Step, StepInput
from distilabel.steps.tasks import TextGeneration
from distilabel.llms import OpenAILLM
from openai import OpenAI
from typing import List
from pydantic import BaseModel, Field
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Pydantic Schema for Judge Output ---
class JudgeOutput(BaseModel):
    fidelity: float = Field(..., ge=0, le=1, description="Score matching the target label")
    naturalness: float = Field(..., ge=0, le=1, description="Score for fluency and realism")
    utility: float = Field(..., ge=0, le=1, description="Score for training value/nuance")
    reasoning: str = Field(..., description="Explanation for the scores")

# --- Meta-Prompting Pydantic Schema ---
class SDGRules(BaseModel):
    diversity_rules: List[str] = Field(..., description="List of at least 8 rules for diversity in generation")
    unknown_diversity_rules: List[str] = Field(..., description="List of at least 5 rules for 'unknown' class data generation")

# --- Generator Output Pydantic Schema ---
class GeneratorBatchOutput(BaseModel):
    results: List[str] = Field(..., description="List of generated diverse candidate strings")

# --- 0. Logging Configuration ---
log_filename = "runs/demo-thai-classification-copy-001/logs/sdg_debug.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # รับทุกระดับเข้า Logger

# File Handler (เก็บละเอียดถึง DEBUG)
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console Handler (แสดงแค่ INFO เพื่อความสะอาด)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- 1. Generator System Prompt (Advanced Architecture) ---
# โครงสร้างแบบ RTC-FO: Role, Task, Context, Few-shot (Guidelines), Output
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

# --- 2. Templates ---
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

def normalize_text(text):
    # ยุบตัวอักษรและตัดสัญลักษณ์พิเศษทิ้งเพื่อเทียบลองว่าประโยคเหมือนกันเป๊ะไหม
    return re.sub(r'[^a-zA-Z0-9ก-๙]', '', str(text)).lower()

def extract_multiple_outputs(text: str) -> List[str]:
    """
    Parse the LLM JSON output to extract the array of generated strings.
    """
    if not text:
        return []
    
    # 1. Strip reasoning blocks
    text = re.sub(r'<(think|reasoning)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Extract JSON payload
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)
        
    try:
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = clean_text[start_idx:end_idx+1]
            validated_data = GeneratorBatchOutput.model_validate_json(json_str)
            return [res.strip() for res in validated_data.results if res.strip()]
    except Exception as e:
        logger.warning(f"Error parsing JSON from Generator: {e} | Raw text: {text[:150]}...")
        # Fallback raw regex
        results = re.findall(r'"([^"]+)"', clean_text)
        filtered_results = [r for r in results if r != "results" and len(r.strip()) > 10]
        if filtered_results:
            return filtered_results
    return []

class CleanTextStep(Step):
    @property
    def inputs(self) -> List[str]:
        return ["generated_text"]

    @property
    def outputs(self) -> List[str]:
        return ["cleaned_text"]

    def process(self, inputs: StepInput):
        exploded_outputs = []
        for item in inputs:
            cleaned_list = extract_multiple_outputs(item.get("generated_text", ""))
            if not cleaned_list:
                # If extraction failed, yield empty string to be caught by downstream filter
                new_item = item.copy()
                new_item["cleaned_text"] = ""
                exploded_outputs.append(new_item)
                continue
                
            for extracted_text in cleaned_list:
                new_item = item.copy()
                new_item["cleaned_text"] = extracted_text
                exploded_outputs.append(new_item)
        yield exploded_outputs

def generate_sdg_rules(
    client: OpenAI, 
    task_description: str, 
    labels: List[str], 
    model_name: str = "qwen/qwen3.6-plus"
) -> SDGRules:
    """
    ใช้ Meta-Prompting ให้ LLM ช่วยคิดกฎความหลากหลาย (Diversity Rules) 
    ที่เหมาะสมกับงานโดยอัตโนมัติ
    """
    logger.info(f"🚀 กำลังใช้ Meta-Prompting เพื่อสร้างกฎสำหรับงาน: {task_description[:50]}...")
    print(f"DEBUG: Starting Meta-Prompting for labels: {labels}")
    
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

    print("DEBUG: Sending request to LLM for rules...")
    response = client.chat.completions.create(
        model=model_name, 
        messages=[{"role": "system", "content": "You are a Senior Data Engineer. Return ONLY JSON."}, 
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    print("DEBUG: Received response from LLM.")
    
    raw_json = response.choices[0].message.content
    
    # เอา Markdown Code Block ออกในกรณีที่ LLM แถมมาให้
    clean_json = raw_json.strip()
    if clean_json.startswith("```"):
        clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
        clean_json = re.sub(r"\s*```$", "", clean_json)

    print(f"DEBUG: Clean JSON: {clean_json[:100]}...")
    return SDGRules.model_validate_json(clean_json)

def run_classification_sdg_iterative(
    seed_data_path: str,
    output_path: str,
    target_count: int,
    task_description: str,
    model_name: str = "qwen/qwen3-235b-a22b-2507",
    rule_model_name: str = "qwen/qwen3.6-plus",
    judge_model_name: str = "openai/gpt-4o-mini",
    max_loops: int = 20
):
    """
    ฟังก์ชันแบบ Iteration Loop:
    จะทำการวนรันโควต้า สร้างไปเรื่อยๆ จนกว่าจะได้ชิ้นส่วนที่ผ่านการตรวจสอบ "ครบเป๊ะ 100%" ตามเป้า
    และใช้ Global Set ดักคำซ้ำเพื่อรับประกันว่า Dataset ห้ามมีคำซ้ำเด็ดขาด
    """
    logger.info(f"1. กำลังโหลด Seed Data จาก {seed_data_path}...")
    if not os.path.exists(seed_data_path):
        logger.error(f"Error: ไม่พบไฟล์เมล็ดพันธุ์ {seed_data_path}")
        return

    with open(seed_data_path, "r", encoding="utf-8") as f:
        seed_data = json.load(f)

    # --- ส่วนที่ 2: ตรวจจับ Label อัตโนมัติ ---
    label_examples = {}
    for item in seed_data:
        lbl = item.get("label")
        txt = item.get("text")
        if lbl and txt:
            if lbl not in label_examples:
                label_examples[lbl] = []
            label_examples[lbl].append(txt)

    unique_labels_seed = list(label_examples.keys())
    num_classes_seed = len(unique_labels_seed)
    
    if num_classes_seed == 0:
        print("Error: ไม่พบข้อมูล Label ที่ถูกต้องใน Seed Data")
        return

    # --- ส่วนที่ 2.5: คำนวณ Quota (10% unknown, 90% Others) ---
    target_unknown = math.ceil(target_count * 0.10)
    target_others = target_count - target_unknown
    target_per_class = math.ceil(target_others / num_classes_seed)
    
    # รวม "unknown" เข้าไปในลิสต์หลัก
    unique_labels = unique_labels_seed + ["unknown"]
    label_examples["unknown"] = [] # จองไว้เผื่อเคส unknown (ไม่มี seed)

    print(f"พบ Label จาก Seed {num_classes_seed} ประเภท: {unique_labels_seed}")
    print(f"เป้าหมาย: รวม {target_count} ชุด")
    print(f"  - คลาสปกติ (เฉลี่ย): {target_per_class} ชุด/คลาส")
    print(f"  - คลาส unknown: {target_unknown} ชุด (10%)")

    # --- ส่วนที่ 3: ตัวแปร Tracking ติดตามค่าข้ามลูป ---
    collected_output = [] # เก็บข้อมูลผลลัพธ์สุดท้าย
    # กำหนดเป้าหมายรายคลาสให้ชัดเจน
    target_counts_map = {lbl: target_per_class for lbl in unique_labels_seed}
    target_counts_map["unknown"] = target_unknown
    
    collected_counts = {lbl: 0 for lbl in unique_labels} 
    
    # Global Deduplication (ป้องกันบั๊ก ข้ามลูป และคุมคำคล้ายได้แม่นยำกว่า Python Set ปกติ)
    # ติดตั้ง MinHashLSH ของ datasketch แบบ Global
    from datasketch import MinHash, MinHashLSH
    
    threshold = 0.90 # ระดับความคล้าย 90% จะถูกมองว่าซ้ำ
    num_perm = 128
    global_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    lsh_counter = 0

    def compute_minhash(text: str) -> MinHash:
        # ใช้ 5-gram ระดับ Character เพื่อความละเอียดในการกวาดภาษาไทย/อังกฤษ
        m = MinHash(num_perm=num_perm)
        # เคลียร์ตัวอักษรขยะออกก่อนวิเคราะห์
        clean_text = re.sub(r'[^a-zA-Z0-9ก-๙]', '', str(text)).lower()
        if len(clean_text) < 5:
            m.update(clean_text.encode('utf-8'))
        else:
            for i in range(len(clean_text) - 4):
                ngram = clean_text[i:i+5]
                m.update(ngram.encode('utf-8'))
        return m

    # ดึง Seed data เข้าประวัติซ้ำด้วย เพื่อกัน LLM ลอกต้นฉบับ
    # โดยนำไปคำนวณ MinHash และจัดเก็บเข้า LSH Index
    seed_texts = []
    for item in seed_data:
        text = item.get("text", "")
        if text:
            m = compute_minhash(text)
            global_lsh.insert(f"seed_{lsh_counter}", m)
            lsh_counter += 1
            seed_texts.append(text)

    # --- ส่วนที่ 3.5: เตรียม FAISS สำหรับ Semantic Check ---
    logger.info("กำลังโหลด Embedding Model (paraphrase-multilingual-MiniLM-L12-v2)...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    
    # ใช้ IndexFlatIP สำหรับ Inner Product (ซึ่งคือ Cosine Similarity เมื่อ Normalize เวกเตอร์แล้ว)
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    
    if seed_texts:
        logger.info(f"กำลังทำการ Embedding Seed Data {len(seed_texts)} ชุด...")
        seed_embeddings = embedding_model.encode(seed_texts, normalize_embeddings=True)
        faiss_index.add(seed_embeddings)

    similarity_threshold = 0.85

    # รายการความหลากหลาย
    base_url = os.getenv("OPENROUTER_BASE_URL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # --- ส่วนที่ 2.8: Meta-Prompting เพื่อสร้างกฎ (Diversity Rules) ---
    try:
        sdg_rules = generate_sdg_rules(client, task_description, unique_labels, model_name=rule_model_name)
        diversity_rules = sdg_rules.diversity_rules
        unknown_diversity_rules = sdg_rules.unknown_diversity_rules
        logger.info("✅ สร้างกฎความหลากหลายเรียบร้อยแล้ว:")
        for i, rule in enumerate(diversity_rules[:3]):
            logger.info(f"   [D-{i+1}] {rule}")
        logger.info("   ...")
    except Exception as e:
        logger.warning(f"⚠️ Meta-Prompting ล้มเหลว (Fallback ไปใช้ค่าพื้นฐาน): {e}")
        # Fallback rules
        diversity_rules = ["Focus on specific details.", "Use formal language.", "Consider different perspectives."]
        unknown_diversity_rules = ["Generic chatter.", "Unrelated news snippets."]

    difficulties = ["easy", "medium", "hard", "complex-structure"]
    
    threshold = 0.7 # คะแนนขั้นต่ำที่ยอมรับได้
    
    loop_count = 1
    
    # --- ส่วนที่ 4: การรันแบบ While Loop จนกว่าจะได้ข้อมูลครบ ---
    while sum(collected_counts.values()) < target_count and loop_count <= max_loops:
        logger.info(f"\n{'='*40}")
        logger.info(f" 🔁 เริ่มทำงาน Loop ที่ {loop_count}/{max_loops}")
        logger.info(f"{'='*40}")
        logger.info(f"สถานะความคืบหน้า: เก็บข้อมูลได้แล้ว {sum(collected_counts.values())}/{target_count} ชุด")
        for lbl, count in collected_counts.items():
            logger.info(f"  - {lbl}: {count}/{target_counts_map[lbl]}")

        pipeline_inputs = []
        
        # จัดเตรียมท่อสร้างข้อมูล (เฉพาะคลาสที่ยังได้ไม่ครบเท่านั้น)
        for label in unique_labels:
            needed = target_counts_map[label] - collected_counts[label]
            if needed <= 0:
                continue # เติมเต็มโควต้าแล้ว ไม่ต้องแต่งเพิ่ม
                
            # สร้างตัวคูณเผื่อทิ้ง (Margin) ในแต่ละลูป. หาร 4 เพราะ 1 Request พ่น 4-5 ประโยค
            generate_quota = max(1, math.ceil((needed / 4.0) * 1.5))
            
            for _ in range(generate_quota):
                if label == "unknown":
                    examples = [] # unknown ไม่มีเมล็ดพันธุ์
                    div_rule = random.choice(unknown_diversity_rules)
                else:
                    examples = random.sample(label_examples[label], min(3, len(label_examples[label])))
                    div_rule = random.choice(diversity_rules)

                pipeline_inputs.append({
                    "task_description": task_description,
                    "label": label,
                    "examples": examples,
                    "difficulty": random.choice(difficulties),
                    "diversity_rule": div_rule 
                })

        # สลับคำสั่งเพื่อไม่ให้โมเดลจำรูปแบบติดกันรวดเดียว
        random.shuffle(pipeline_inputs)

        # สร้าง Pipeline
        print(f"\n[ Distilabel ] ดึงเข้าระบบเพื่อแต่งประโยคใหม่จำนวน {len(pipeline_inputs)} ชิ้น...")
        # หมายเหตุ: นำไลบรารี Minhash ออกเพราะเรามี Python Set ทำหน้าที่ตัดคำซ้ำให้แบบ 100% และแม่นยำกว่าแล้ว
        with Pipeline(name=f"classification-sdg-loop-{loop_count}") as pipeline:
            loader = LoadDataFromDicts(data=pipeline_inputs, batch_size=100)
            
            # ใช้ UNIFIED_TEMPLATE (สร้าง + เช็คตัวเอง)
            generator = TextGeneration(
                name="generate_text",
                llm=OpenAILLM(
                    model=model_name, 
                    base_url=base_url, 
                    api_key=api_key, 
                    generation_kwargs={"max_new_tokens": 4096, "temperature": 0.7, "response_format": {"type": "json_object"}}
                ),
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                template=UNIFIED_GENERATOR_TEMPLATE,
                columns=["task_description", "label", "examples", "difficulty", "diversity_rule"],
                output_mappings={"generation": "generated_text"},
                input_batch_size=100
            )
            
            cleaner = CleanTextStep(name="clean_text", input_batch_size=100)
            
            judge = TextGeneration(
                name="judge_text",
                llm=OpenAILLM(
                    model=judge_model_name, 
                    base_url=base_url, 
                    api_key=api_key, 
                    generation_kwargs={"temperature": 0.0, "response_format": {"type": "json_object"}}
                ),
                template=JUDGE_TEMPLATE,
                columns=["task_description", "label", "cleaned_text"],
                output_mappings={"generation": "judge_raw_output"},
                input_batch_size=100
            )
            
            loader >> generator >> cleaner >> judge

        distiset = pipeline.run(use_cache=False)
        ds = distiset["default"]["train"]
        
        # --- [Optimization] Batch Embedding ---
        logger.info(f"⚡ กำลังทำ Batch Embedding สำหรับข้อมูล {len(ds)} ชุด...")
        all_texts = [row.get("cleaned_text", "") for row in ds]
        valid_indices = [idx for idx, txt in enumerate(all_texts) if txt and txt.strip() and txt.lower() != "none"]
        valid_texts = [all_texts[idx] for idx in valid_indices]
        
        vector_map = {}
        if valid_texts:
            # คำนวณ Vector ทั้งหมดในคำสั่งเดียว (เร็วขึ้น 10-50 เท่า)
            batch_vectors = embedding_model.encode(valid_texts, normalize_embeddings=True, show_progress_bar=False)
            vector_map = {idx: batch_vectors[i] for i, idx in enumerate(valid_indices)}

        # --- กรองและเก็บผลลัพธ์รอบล่าสุด ---
        print("\n[ Filter ] กำลังเช็คภาพรวม แจกแจงคะแนนจากขบวนการ Pipeline และตรวจคำซ้ำ...")
        added_in_loop = 0
        
        for i, row in enumerate(ds):
            lbl = row["label"]
            text = row.get("cleaned_text", "")
            
            # ถ้าคลาสนี้โควต้าชิ้นงานเต็มแล้ว ไม่ต้องเก็บแล้วปล่อยผ่านได้เลย
            if collected_counts[lbl] >= target_counts_map[lbl]:
                continue
                
            if i not in vector_map:
                logger.warning(f"⚠️ Generation Failed or Empty for batch row {i} (Label: {lbl}).")
                continue

            judge_raw_output = row.get("judge_raw_output", "{}")
            
            try:
                # Use Pydantic to validate and parse pipeline's judge output
                judge_data = JudgeOutput.model_validate_json(judge_raw_output)
                
                # --- Python-side Scoring Calculation ---
                # Weight: Fidelity (0.4), Naturalness (0.3), Utility (0.3)
                overall_score = (judge_data.fidelity * 0.4) + \
                                (judge_data.naturalness * 0.3) + \
                                (judge_data.utility * 0.3)
                
                reason = judge_data.reasoning
                decision = "accept" if overall_score >= threshold else "reject"
                
                logger.debug(f"[DEBUG] Text: {text[:50]}...")
                logger.debug(f"[DEBUG] Judge Scores: F={judge_data.fidelity}, N={judge_data.naturalness}, U={judge_data.utility}")
                logger.debug(f"[DEBUG] Overall: {overall_score:.2f} | Reason: {reason}")
                
                if decision == "reject":
                    logger.info(f"❌ Discarded: Score {overall_score:.2f} < {threshold} (Reason: {reason})")
                    continue 
            except Exception as e:
                logger.warning(f"⚠️ Judge Parsing/Validation Error (fallback to reject): {e} | Raw JSON: {judge_raw_output}")
                continue
                
            # Filter 3: Semantic Similarity (FAISS)
            # ตรวจสอบความซ้ำซ้อนเชิงความหมาย (Semantic Redundancy)
            try:
                # ดึง Vector ที่คำนวณไว้แล้วมาใช้งานทันที (ไม่ต้อง encode ใหม่)
                new_embedding = vector_map[i].reshape(1, -1)
                
                # ค้นหา 1 อันที่ใกล้ที่สุดในฐานข้อมูล
                D, I = faiss_index.search(new_embedding, 1)
                max_sim = float(D[0][0])
                
                if max_sim >= similarity_threshold:
                    logger.info(f"❌ Discarded: Semantically redundant (FAISS Similarity {max_sim:.2f} >= {similarity_threshold})")
                    continue
            except Exception as e:
                logger.warning(f"⚠️ FAISS Similarity Check Error: {e}")

            # ถ้าผ่านรอดด่านทั้งหมด 100% ให้บันทึกลงกระเป๋าและอินเด็กซ์ได้!
            m = compute_minhash(text) # คำนวณ MinHash สำหรับด่านแรก (ถ้าจำเป็นต้องเก็บประวัติ)
            global_lsh.insert(f"gen_{lsh_counter}", m)
            lsh_counter += 1
            faiss_index.add(new_embedding)
            
            collected_output.append({
                "text": text,
                "label": lbl
            })
            collected_counts[lbl] += 1
            added_in_loop += 1
            
            # ชนเพดานเป้าหมายก็สั่งหยุดทันทีระหว่างเช็คทีละแถว
            if sum(collected_counts.values()) >= target_count:
                break
                
        print(f"❇️ สิ้นสุดลูปที่ {loop_count}: ของดีที่รอดและสามารถจัดเก็บได้จากลูปนี้คือ {added_in_loop} ชิ้น")
        loop_count += 1

    # --- สิ้นสุดการทำงาน ---
    if sum(collected_counts.values()) < target_count:
        logger.warning(f"\n⚠️ จบการทำงานเพราะถึง Limit สูงสุด ({max_loops} ลูป). สร้างได้ {sum(collected_counts.values())}/{target_count} ชุด")
        logger.info("คำแนะนำ: แปลว่า Task ยากไปจน Validator มองหาแต่ข้อผิดพลาดตลอดเวลา ลองเพิ่ม max_loops จ่ายแก้ดูครับ")
    else:
        logger.info(f"\n🎯 ยอดเยี่ยม! สร้างข้อมูลครบเป้าหมาย {target_count} ชุดพอดีเป๊ะ!")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collected_output, f, ensure_ascii=False, indent=2)
    logger.info(f"บันทึกข้อมูลเรียบร้อยที่: {output_path}")


# --- จุดรันสคริปต์ (Main Entry Point) ---
if __name__ == "__main__":
    run_classification_sdg_iterative(
        seed_data_path="data/raw/thai-text-classification-seed-data.json", # ไฟล์ต้นฉบับ
        output_path="runs/demo-thai-classification-copy-001/sdg/synthetic_data.json", # baseline copy output
        target_count=10, # small manual test run
        task_description="จำแนกเอกสารและสัญญาทางกฎหมายของไทยออกเป็นหมวดหมู่ เช่น สัญญาจ้างงาน สัญญาเช่า เอกสารจัดซื้อจัดจ้าง และสัญญารักษาความลับ (NDA)",
        model_name="qwen/qwen3-235b-a22b-2507", 
        rule_model_name="qwen/qwen3.6-plus",
        judge_model_name="openai/gpt-4o-mini", 
        max_loops=20
    )
