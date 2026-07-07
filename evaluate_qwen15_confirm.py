
from __future__ import annotations

import json, os, random, re, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", ROOT / "artifacts_qwen15b"))
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", ARTIFACT_ROOT / "qwen15_len1024_r8_a16_lr5e5_e3_es" / "adapter"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ARTIFACT_ROOT / "confirm_qwen15_len1024_r8_a16_lr5e5_e3_es_greedy100"))
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "768"))
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
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", "").replace("$", ""))
    if not numbers:
        return ""
    value = float(numbers[-1])
    return str(int(value)) if value.is_integer() else str(value)

def features(text: str) -> dict:
    thinking = extract_tag(text, "thinking")
    reflection = extract_tag(text, "reflection")
    answer = extract_tag(text, "answer")
    strict_prediction = normalize_number(answer)
    loose_prediction = strict_prediction or normalize_number(text)
    return {
        "strict_prediction": strict_prediction,
        "loose_prediction": loose_prediction,
        "valid_format": bool(thinking and reflection and answer),
        "has_reflection": bool(reflection),
        "thinking_words": len(thinking.split()),
        "reflection_words": len(reflection.split()),
        "trace_words": len(text.split()),
    }

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def summarize_pair(base_records: list[dict], adapter_records: list[dict], key: str) -> dict:
    base = np.array([row[key] for row in base_records], dtype=float)
    adapter = np.array([row[key] for row in adapter_records], dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(base), size=(10_000, len(base)))
    diffs = (adapter[samples] - base[samples]).mean(axis=1)
    return {
        "base_accuracy": float(base.mean()),
        "adapter_accuracy": float(adapter.mean()),
        "adapter_minus_base": float((adapter - base).mean()),
        "bootstrap_95_ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "adapter_wins": int(np.sum((adapter == 1) & (base == 0))),
        "base_wins": int(np.sum((adapter == 0) & (base == 1))),
        "ties": int(np.sum(adapter == base)),
    }

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval(); model.config.use_cache = True

gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
test_indices = sorted(random.Random(SEED).sample(range(len(gsm8k_test)), min(N_EXAMPLES, len(gsm8k_test))))
(OUTPUT_DIR / "gsm8k_test_indices.json").write_text(json.dumps(test_indices, indent=2), encoding="utf-8")
gsm8k_subset = gsm8k_test.select(test_indices)

@torch.inference_mode()
def generate(question: str, seed_offset: int) -> tuple[str, float]:
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": question}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    torch.manual_seed(SEED + seed_offset); torch.cuda.manual_seed_all(SEED + seed_offset)
    started = time.time()
    out = model.generate(**encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - started
    trace = tokenizer.decode(out[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    return trace, elapsed

def evaluate_mode(mode: str, adapter_enabled: bool) -> list[dict]:
    path = OUTPUT_DIR / f"{mode}_predictions.jsonl"
    records = load_jsonl(path)
    ctx = torch.no_grad() if adapter_enabled else model.disable_adapter()
    with ctx:
        for i in range(len(records), len(gsm8k_subset)):
            ex = gsm8k_subset[i]
            gold = normalize_number(ex["answer"].split("####")[-1])
            trace, latency = generate(ex["question"], seed_offset=(0 if adapter_enabled else 10000) + i)
            parsed = features(trace)
            record = {
                "mode": mode,
                "dataset_index": test_indices[i],
                "question": ex["question"],
                "gold_answer": gold,
                "trace": trace,
                **parsed,
                "strict_correct": parsed["strict_prediction"] == gold,
                "loose_correct": parsed["loose_prediction"] == gold,
                "latency_seconds": latency,
            }
            records.append(record); append_jsonl(path, record)
            print(f"{mode} {i+1}/{len(gsm8k_subset)} strict={record['strict_correct']} loose={record['loose_correct']} fmt={record['valid_format']} latency={latency:.1f}s", flush=True)
    return records

base_records = evaluate_mode("base", adapter_enabled=False)
adapter_records = evaluate_mode("adapter", adapter_enabled=True)
summary = {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "adapter_dir": str(ADAPTER_DIR),
    "n_examples": len(adapter_records),
    "max_new_tokens": MAX_NEW_TOKENS,
    "strict_tag_answer": summarize_pair(base_records, adapter_records, "strict_correct"),
    "loose_numeric_answer": summarize_pair(base_records, adapter_records, "loose_correct"),
    "base_valid_format_rate": float(np.mean([r["valid_format"] for r in base_records])),
    "adapter_valid_format_rate": float(np.mean([r["valid_format"] for r in adapter_records])),
    "base_reflection_rate": float(np.mean([r["has_reflection"] for r in base_records])),
    "adapter_reflection_rate": float(np.mean([r["has_reflection"] for r in adapter_records])),
    "base_mean_latency_seconds": float(np.mean([r["latency_seconds"] for r in base_records])),
    "adapter_mean_latency_seconds": float(np.mean([r["latency_seconds"] for r in adapter_records])),
}
(OUTPUT_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUTPUT_DIR / "COMPLETED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
