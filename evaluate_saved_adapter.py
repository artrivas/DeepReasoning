"""Fast paired greedy evaluation of an already-trained LoRA adapter."""

from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", ROOT / "artifacts"))
ADAPTER_DIR = Path(
    os.getenv(
        "ADAPTER_DIR",
        ARTIFACT_ROOT / "screen_len1024_r8_a16_lr5e5_e1_bs8" / "adapter",
    )
)
OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        ARTIFACT_ROOT / "confirm_len1024_r8_a16_lr5e5_e1_bs8_greedy100",
    )
)
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
SEED = int(os.getenv("SEED", "42"))
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "100"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_INSTRUCTION = """You are a meticulous reasoning tutor.
For every problem, answer using EXACTLY these tags in order:
<thinking>
Reason step by step. Show every intermediate calculation.
</thinking>
<reflection>
Re-check your reasoning. Look for arithmetic slips or wrong assumptions.
</reflection>
<answer>
Give only the final answer.
</answer>"""


def extract_tag(text: str, tag: str) -> str:
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def normalize_number(text: str) -> str:
    numbers = re.findall(
        r"-?\d+(?:\.\d+)?", str(text).replace(",", "").replace("$", "")
    )
    if not numbers:
        return ""
    value = float(numbers[-1])
    return str(int(value)) if value.is_integer() else str(value)


def features(text: str) -> dict:
    thinking = extract_tag(text, "thinking")
    reflection = extract_tag(text, "reflection")
    answer = extract_tag(text, "answer")
    return {
        "prediction": normalize_number(answer),
        "valid_format": bool(thinking and reflection and answer),
        "has_reflection": bool(reflection),
        "thinking_words": len(thinking.split()),
        "reflection_words": len(reflection.split()),
        "trace_words": len(text.split()),
    }


tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
model.config.use_cache = True

base_cache = ARTIFACT_ROOT / "shared_base_predictions.jsonl"
base_records = [
    json.loads(line)
    for line in base_cache.read_text(encoding="utf-8").splitlines()
    if line.strip()
][:N_EXAMPLES]
test_indices = [row["dataset_index"] for row in base_records]
gsm8k = load_dataset("openai/gsm8k", "main", split="test").select(test_indices)

prediction_path = OUTPUT_DIR / "adapter_predictions.jsonl"
records = []
if prediction_path.exists():
    records = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

for i in range(len(records), len(gsm8k)):
    example = gsm8k[i]
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": example["question"]},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    torch.manual_seed(SEED + i)
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - started
    trace = tokenizer.decode(
        output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    gold = normalize_number(example["answer"].split("####")[-1])
    parsed = features(trace)
    record = {
        "dataset_index": test_indices[i],
        "question": example["question"],
        "gold_answer": gold,
        "trace": trace,
        **parsed,
        "correct": parsed["prediction"] == gold,
        "latency_seconds": latency,
    }
    records.append(record)
    with prediction_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{i + 1}/{len(gsm8k)} correct={record['correct']} latency={latency:.1f}s")

base_correct = np.array([row["greedy_correct"] for row in base_records], dtype=float)
adapter_correct = np.array([row["correct"] for row in records], dtype=float)
rng = np.random.default_rng(SEED)
samples = rng.integers(0, len(records), size=(10_000, len(records)))
differences = (adapter_correct[samples] - base_correct[samples]).mean(axis=1)

summary = {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "adapter_dir": str(ADAPTER_DIR),
    "n_examples": len(records),
    "max_new_tokens": MAX_NEW_TOKENS,
    "base_accuracy": float(base_correct.mean()),
    "adapter_accuracy": float(adapter_correct.mean()),
    "adapter_minus_base": float((adapter_correct - base_correct).mean()),
    "bootstrap_95_ci": [
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    ],
    "adapter_wins": int(np.sum((adapter_correct == 1) & (base_correct == 0))),
    "base_wins": int(np.sum((adapter_correct == 0) & (base_correct == 1))),
    "ties": int(np.sum(adapter_correct == base_correct)),
    "valid_format_rate": float(np.mean([row["valid_format"] for row in records])),
    "reflection_rate": float(np.mean([row["has_reflection"] for row in records])),
    "mean_trace_words": float(np.mean([row["trace_words"] for row in records])),
    "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in records])),
}
(OUTPUT_DIR / "evaluation_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
(OUTPUT_DIR / "COMPLETED").write_text(
    datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
