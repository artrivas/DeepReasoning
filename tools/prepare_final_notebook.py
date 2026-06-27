#!/usr/bin/env python3
"""Prepare the DeepReasoning final notebook and pre-cleaned SFT file.

This script keeps the existing project structure but replaces the Kaggle-bound
notebook cells with a portable Lightning AI / Colab / local Jupyter workflow.
It also creates a structurally cleaned SFT JSONL from the existing 2k file.
The notebook itself revalidates the clean file against GSM8K train gold labels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_SFT = ROOT / "sft_reasoning_2k.jsonl"
CLEAN_SFT = ROOT / "sft_reasoning_2k_clean.jsonl"
NOTEBOOK = ROOT / "deepreasoning.ipynb"

ORDERED_TAG_RE = re.compile(
    r"^\s*<thinking>(?P<thinking>[\s\S]*?)</thinking>\s*"
    r"<reflection>(?P<reflection>[\s\S]*?)</reflection>\s*"
    r"<answer>(?P<answer>[\s\S]*?)</answer>\s*$",
    re.IGNORECASE,
)


def structurally_clean_sft() -> dict[str, int]:
    raw_rows = []
    with RAW_SFT.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_rows.append(json.loads(line))

    kept = []
    valid_tag_rows = 0
    nonempty_rows = 0
    for row in raw_rows:
        trace = row.get("trace", "")
        match = ORDERED_TAG_RE.match(trace)
        if not match:
            continue
        valid_tag_rows += 1
        if not all(match.group(name).strip() for name in ("thinking", "reflection", "answer")):
            continue
        nonempty_rows += 1
        kept.append(row)

    with CLEAN_SFT.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "raw_rows": len(raw_rows),
        "valid_tag_rows": valid_tag_rows,
        "nonempty_rows": nonempty_rows,
        "final_kept_structural": len(kept),
        "dropped_structural": len(raw_rows) - len(kept),
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def build_notebook() -> dict:
    cells = [
        md(
            """
# DeepReasoning Final Pipeline

This notebook implements the final GSM8K-only DeepReasoning project:

1. Teacher distillation with structured `<thinking>`, `<reflection>`, `<answer>` traces.
2. SFT + QLoRA fine-tuning of `Qwen/Qwen2.5-3B-Instruct`.
3. Test-time compute with self-consistency / majority voting.
4. A 50-case held-out evaluation with exact accuracy, win rate, and blind pairwise LLM-as-a-Judge.

The notebook is portable across Lightning AI Notebook, Google Colab, and local Jupyter. It intentionally avoids Kaggle paths and Kaggle secrets.
"""
        ),
        md(
            """
## 0. Setup

Run this cell once per fresh runtime. Set `INSTALL_DEPS = False` if your environment already has the packages.
"""
        ),
        code(
            """
INSTALL_DEPS = True

if INSTALL_DEPS:
    import subprocess
    import sys

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "transformers>=4.45",
        "peft>=0.13",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "google-genai",
        "pandas",
        "matplotlib",
        "tqdm",
    ])
"""
        ),
        code(
            """
import gc
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
"""
        ),
        md(
            """
## 1. Portable Runtime Configuration

The same notebook should run in Lightning AI, Colab, or a local clone. All paths are derived from `PROJECT_DIR`.
"""
        ),
        code(
            """
def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def detect_project_dir() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd / "DeepReasoning",
        Path("/content/DeepReasoning"),
        Path("/teamspace/studios/this_studio/DeepReasoning"),
        Path("/teamspace/studios/this_studio"),
    ]
    for candidate in candidates:
        if (candidate / "deepreasoning.ipynb").exists() and (candidate / "sft_reasoning_2k.jsonl").exists():
            return candidate.resolve()
    return cwd


PROJECT_DIR = detect_project_dir()
DATA_DIR = PROJECT_DIR
OUT_DIR = PROJECT_DIR / "outputs"
ADAPTER_DIR = OUT_DIR / "qwen-reasoning-lora-final"
RESULTS_DIR = OUT_DIR / "results"

for path in (OUT_DIR, ADAPTER_DIR, RESULTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

RAW_SFT_PATH = DATA_DIR / "sft_reasoning_2k.jsonl"
CLEAN_SFT_PATH = DATA_DIR / "sft_reasoning_2k_clean.jsonl"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 1024
SEED = 42
EVAL_CASES = 50

random.seed(SEED)
torch.manual_seed(SEED)

print("PROJECT_DIR:", PROJECT_DIR)
print("RAW_SFT_PATH:", RAW_SFT_PATH)
print("CLEAN_SFT_PATH:", CLEAN_SFT_PATH)
print("ADAPTER_DIR:", ADAPTER_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
        ),
        code(
            """
def get_gemini_api_key(required: bool = False) -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key and in_colab():
        try:
            from google.colab import userdata
            key = userdata.get("GEMINI_API_KEY")
        except Exception:
            key = None

    if not key:
        msg = (
            "GEMINI_API_KEY is not set. Distillation and LLM-as-a-Judge cells "
            "need it. Set os.environ['GEMINI_API_KEY'] in Lightning/local, or "
            "add GEMINI_API_KEY in Colab Secrets."
        )
        if required:
            raise RuntimeError(msg)
        print(msg)
    return key


_ = get_gemini_api_key(required=False)
"""
        ),
        md(
            """
## 2. Phase 1 - GSM8K Teacher Distillation

The final pipeline uses GSM8K only. MBPP is intentionally omitted from the final run to keep the domain and metrics clean.
"""
        ),
        code(
            """
gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")

print("GSM8K train:", len(gsm8k_train), gsm8k_train.column_names)
print("GSM8K test :", len(gsm8k_test), gsm8k_test.column_names)
"""
        ),
        code(
            """
SYSTEM_INSTRUCTION = '''You are a meticulous reasoning tutor.
For every problem, you MUST answer using EXACTLY these three tags, in this order:

<thinking>
Reason step by step. Show every intermediate calculation.
</thinking>
<reflection>
Re-check your own reasoning above. Look for arithmetic slips or wrong assumptions.
If you find a mistake, correct it here explicitly.
</reflection>
<answer>
The final answer only. For math: just the final number.
</answer>

Do not write anything outside these tags.'''

ORDERED_TAG_RE = re.compile(
    r"^\\s*<thinking>(?P<thinking>[\\s\\S]*?)</thinking>\\s*"
    r"<reflection>(?P<reflection>[\\s\\S]*?)</reflection>\\s*"
    r"<answer>(?P<answer>[\\s\\S]*?)</answer>\\s*$",
    re.IGNORECASE,
)


def extract_ordered_sections(text: str) -> dict[str, str] | None:
    match = ORDERED_TAG_RE.match(text or "")
    if not match:
        return None
    return {name: match.group(name).strip() for name in ("thinking", "reflection", "answer")}


def extract_tag(text: str, tag: str) -> str:
    match = re.search(fr"<{tag}>([\\s\\S]*?)</{tag}>", text or "", re.IGNORECASE)
    return match.group(1).strip() if match else ""
"""
        ),
        code(
            """
def normalize_numeric_text(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = re.sub(r"\\s+", " ", text)
    return text


def decimal_to_key(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def parse_decimal_from_text(text: str) -> str | None:
    clean = normalize_numeric_text(text)
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", clean)
    if not numbers:
        return None
    try:
        return decimal_to_key(Decimal(numbers[-1]))
    except InvalidOperation:
        return None


def answer_key_from_trace(trace: str) -> str | None:
    answer = extract_tag(trace, "answer")
    if not answer:
        answer = trace
    numeric = parse_decimal_from_text(answer)
    if numeric is not None:
        return numeric
    fallback = normalize_numeric_text(answer).lower()
    return fallback or None


def gsm8k_gold_key(answer_field: str) -> str | None:
    final = str(answer_field).split("####")[-1]
    return parse_decimal_from_text(final)


def answer_is_correct(pred_key: str | None, gold_key: str | None) -> bool:
    return pred_key is not None and gold_key is not None and pred_key == gold_key
"""
        ),
        md(
            """
### Optional teacher generation

The repo already includes distilled data. This cell is kept for reproducibility and for extending the dataset. It does not run unless you call `generate_distilled_rows(...)`.
"""
        ),
        code(
            """
def build_gemini_client(required: bool = True):
    from google import genai
    api_key = get_gemini_api_key(required=required)
    return genai.Client(api_key=api_key)


def ask_teacher(question: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7, max_retries: int = 5) -> str:
    from google.genai import types
    client = build_gemini_client(required=True)
    cfg = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=temperature)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_name, contents=question, config=cfg)
            return response.text or ""
        except Exception as exc:
            wait = min(60, 8 * (attempt + 1))
            print(f"teacher retry {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}")
            time.sleep(wait)
    raise RuntimeError("Teacher generation failed after retries.")


def generate_distilled_rows(dataset, output_path: Path, target_kept: int = 100, sleep: float = 2.0) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["idx"])

    kept_rows = []
    indices = [i for i in range(len(dataset)) if i not in done]
    random.shuffle(indices)

    with output_path.open("a", encoding="utf-8") as f:
        for idx in indices:
            if len(done) >= target_kept:
                break
            ex = dataset[idx]
            trace = ask_teacher(ex["question"])
            sections = extract_ordered_sections(trace)
            pred = answer_key_from_trace(trace)
            gold = gsm8k_gold_key(ex["answer"])
            if sections and all(sections.values()) and answer_is_correct(pred, gold):
                row = {"idx": idx, "question": ex["question"], "trace": trace}
                f.write(json.dumps(row, ensure_ascii=False) + "\\n")
                f.flush()
                kept_rows.append(row)
                done.add(idx)
            time.sleep(sleep)

    return kept_rows
"""
        ),
        md(
            """
### Clean and validate the 2k distilled dataset

This cell reads `sft_reasoning_2k.jsonl`, validates ordered tags, checks GSM8K train gold answers by `idx`, applies the tokenizer length guard, and saves `sft_reasoning_2k_clean.jsonl`.
"""
        ),
        code(
            """
from transformers import AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")


def prompt_has_trainable_tokens(tokenizer, question: str, trace: str, max_length: int) -> bool:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": question},
        {"role": "assistant", "content": trace},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return len(prompt_ids) < len(full_ids)


raw_rows = load_jsonl(RAW_SFT_PATH)
gold_by_idx = {i: gsm8k_gold_key(ex["answer"]) for i, ex in enumerate(gsm8k_train)}
clean_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

valid_tag_rows = []
nonempty_rows = []
gold_matching_rows = []
final_rows = []
drop_reasons = Counter()
seen_idx = set()

for row in raw_rows:
    idx = row.get("idx")
    trace = row.get("trace", "")
    sections = extract_ordered_sections(trace)
    if not sections:
        drop_reasons["invalid_ordered_tags"] += 1
        continue
    valid_tag_rows.append(row)

    if not all(sections.values()):
        drop_reasons["empty_section"] += 1
        continue
    nonempty_rows.append(row)

    if idx in seen_idx:
        drop_reasons["duplicate_idx"] += 1
        continue
    seen_idx.add(idx)

    gold = gold_by_idx.get(idx)
    pred = answer_key_from_trace(trace)
    if not answer_is_correct(pred, gold):
        drop_reasons["gold_mismatch"] += 1
        continue
    gold_matching_rows.append(row)

    if not prompt_has_trainable_tokens(clean_tokenizer, row["question"], trace, MAX_SEQ_LENGTH):
        drop_reasons["no_trainable_tokens_after_truncation"] += 1
        continue

    final_rows.append(row)

save_jsonl(final_rows, CLEAN_SFT_PATH)

dataset_audit = pd.DataFrame([
    {"metric": "raw rows", "value": len(raw_rows)},
    {"metric": "valid ordered tag rows", "value": len(valid_tag_rows)},
    {"metric": "non-empty section rows", "value": len(nonempty_rows)},
    {"metric": "gold-matching rows", "value": len(gold_matching_rows)},
    {"metric": "final kept rows", "value": len(final_rows)},
    {"metric": "dropped rows", "value": len(raw_rows) - len(final_rows)},
])
drop_audit = pd.DataFrame([{"reason": k, "count": v} for k, v in drop_reasons.items()])

dataset_audit.to_csv(RESULTS_DIR / "dataset_audit.csv", index=False)
drop_audit.to_csv(RESULTS_DIR / "dataset_drop_reasons.csv", index=False)

display(dataset_audit)
display(drop_audit)
print("Saved:", CLEAN_SFT_PATH)
assert len(final_rows) >= 500, f"Cleaned dataset is suspiciously small: {len(final_rows)}"
"""
        ),
        md(
            """
## 3. Phase 2 - SFT + QLoRA Fine-Tuning

This phase trains only low-rank adapters on top of the 4-bit base model. The base model weights remain frozen.
"""
        ),
        code(
            """
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


rows = load_jsonl(CLEAN_SFT_PATH)
print("Clean SFT rows:", len(rows))
assert len(rows) >= 500, f"Training dataset is suspiciously small ({len(rows)} rows). Did cleaning fail?"

BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
COMPUTE_DTYPE = torch.bfloat16 if BF16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

LORA_R = 16
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable percent: {100 * trainable_params / total_params:.4f}%")
"""
        ),
        code(
            """
def tokenize_example(ex: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": ex["trace"]},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    assert full_ids[: len(prompt_ids)] == prompt_ids, "Prompt is not a prefix of the full chat template."
    assert len(prompt_ids) < len(full_ids), "No assistant tokens left after truncation."

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
        "trainable_label_count": sum(label != -100 for label in labels),
    }


tokenized = Dataset.from_list(rows).map(tokenize_example, remove_columns=["idx", "question", "trace"])
print(tokenized)
print("Min trainable labels:", min(tokenized["trainable_label_count"]))
print("Median trainable labels:", int(pd.Series(tokenized["trainable_label_count"]).median()))

tokenized = tokenized.remove_columns(["trainable_label_count"])
split = tokenized.train_test_split(test_size=0.05, seed=SEED)
print(split)
assert len(split["train"]) >= 500, f"Train split too small: {len(split['train'])}"
"""
        ),
        code(
            """
debug_ex = split["train"][0]
ignored = [tok for tok, label in zip(debug_ex["input_ids"], debug_ex["labels"]) if label == -100]
trained = [tok for tok, label in zip(debug_ex["input_ids"], debug_ex["labels"]) if label != -100]

print("Ignored prompt tokens:", len(ignored))
print("Trainable assistant tokens:", len(trained))
print("\\n=== MASKED PROMPT ===")
print(tokenizer.decode(ignored)[:1000])
print("\\n=== TRAINED ASSISTANT TRACE ===")
print(tokenizer.decode(trained)[:2000])
"""
        ),
        code(
            """
collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

training_args = TrainingArguments(
    output_dir=str(OUT_DIR / "qwen-reasoning-lora-checkpoints"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=torch.cuda.is_available() and not BF16,
    bf16=BF16,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=collator,
)

train_output = trainer.train()
print(train_output)
"""
        ),
        code(
            """
import matplotlib.pyplot as plt

hist = trainer.state.log_history
with (RESULTS_DIR / "trainer_log_history.json").open("w", encoding="utf-8") as f:
    json.dump(hist, f, indent=2)

train_loss = [(h["step"], h["loss"]) for h in hist if "loss" in h]
eval_loss = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]

plt.figure(figsize=(8, 4))
if train_loss:
    plt.plot(*zip(*train_loss), label="train")
if eval_loss:
    plt.plot(*zip(*eval_loss), "o-", label="eval")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("QLoRA Training Loss - Qwen2.5-3B")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_loss_curve.png", dpi=160)
plt.show()

trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Saved LoRA adapters to:", ADAPTER_DIR)
"""
        ),
        md(
            """
## 4. Phase 3 - Inference and Self-Consistency

This section implements robust answer extraction, base/fine-tuned generation, and majority voting over multiple sampled reasoning paths.
"""
        ),
        code(
            """
@torch.no_grad()
def generate_n_paths(
    question: str,
    model,
    tokenizer,
    N: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
) -> tuple[list[str], list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    do_sample = temperature is not None and temperature > 0
    previous_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        num_return_sequences=N,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = previous_cache

    prompt_len = inputs["input_ids"].shape[1]
    paths = [tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]
    token_counts = [int(out[prompt_len:].shape[0]) for out in outputs]
    return paths, token_counts


def majority_vote(paths: list[str]) -> dict:
    extracted = [answer_key_from_trace(path) for path in paths]
    valid = [ans for ans in extracted if ans is not None]
    if not valid:
        return {
            "answer": None,
            "vote_count": 0,
            "tie": False,
            "valid_answer_rate": 0.0,
            "all_answers": extracted,
            "winning_path": None,
        }

    counts = Counter(valid)
    max_votes = max(counts.values())
    tied_answers = {ans for ans, count in counts.items() if count == max_votes}
    winner = next(ans for ans in extracted if ans in tied_answers)
    winning_path = next((path for path, ans in zip(paths, extracted) if ans == winner), None)

    return {
        "answer": winner,
        "vote_count": max_votes,
        "tie": len(tied_answers) > 1,
        "valid_answer_rate": len(valid) / len(paths),
        "all_answers": extracted,
        "winning_path": winning_path,
    }
"""
        ),
        code(
            """
def unload_model(model_obj=None):
    if model_obj is not None:
        del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_for_eval():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base.eval()
    return base, tok


def load_finetuned_for_eval(adapter_dir: Path = ADAPTER_DIR):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    ft = PeftModel.from_pretrained(base, adapter_dir)
    ft.eval()
    return ft, tok


# Fixed inference smoke test. The old notebook failed because q was undefined.
q = gsm8k_test[0]["question"]
print("Smoke-test question:", q)
try:
    sample_paths, sample_token_counts = generate_n_paths(q, model, tokenizer, N=1, max_new_tokens=256)
    print(sample_paths[0])
    print("Extracted answer:", answer_key_from_trace(sample_paths[0]))
except NameError:
    print("Train/load the model first, then rerun this smoke test.")
"""
        ),
        md(
            """
## 5. Final 50-Case Evaluation

Evaluation uses only held-out GSM8K test examples. Distillation and SFT use GSM8K train indices, so this avoids leakage.
"""
        ),
        code(
            """
def select_eval_cases(dataset, n: int = EVAL_CASES, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n)
    cases = []
    for case_id, idx in enumerate(indices):
        ex = dataset[idx]
        cases.append({
            "case_id": case_id,
            "dataset_idx": idx,
            "question": ex["question"],
            "gold_answer": gsm8k_gold_key(ex["answer"]),
            "raw_gold": ex["answer"],
        })
    return cases


eval_cases = select_eval_cases(gsm8k_test, n=EVAL_CASES, seed=SEED)
pd.DataFrame(eval_cases).head()
"""
        ),
        code(
            """
def evaluate_model(
    model_label: str,
    model_obj,
    tok,
    cases: list[dict],
    n_values: list[int],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
) -> tuple[pd.DataFrame, list[dict]]:
    all_records = []
    metrics = []

    for N in n_values:
        correct = 0
        valid_rates = []
        ties = 0
        latencies = []
        token_totals = []

        for case in tqdm(cases, desc=f"{model_label} N={N}"):
            start = time.perf_counter()
            paths, token_counts = generate_n_paths(
                case["question"],
                model_obj,
                tok,
                N=N,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            latency = time.perf_counter() - start
            vote = majority_vote(paths)
            is_correct = answer_is_correct(vote["answer"], case["gold_answer"])

            correct += int(is_correct)
            valid_rates.append(vote["valid_answer_rate"])
            ties += int(vote["tie"])
            latencies.append(latency)
            token_totals.append(sum(token_counts))

            all_records.append({
                "model": model_label,
                "N": N,
                "case_id": case["case_id"],
                "dataset_idx": case["dataset_idx"],
                "question": case["question"],
                "gold_answer": case["gold_answer"],
                "pred_answer": vote["answer"],
                "correct": is_correct,
                "vote_count": vote["vote_count"],
                "tie": vote["tie"],
                "valid_answer_rate": vote["valid_answer_rate"],
                "latency_sec": latency,
                "generated_tokens_total": sum(token_counts),
                "all_answers": vote["all_answers"],
                "paths": paths,
                "winning_path": vote["winning_path"],
            })

        metrics.append({
            "model": model_label,
            "N": N,
            "cases": len(cases),
            "exact_accuracy": correct / len(cases),
            "valid_answer_rate": sum(valid_rates) / len(valid_rates),
            "tie_rate": ties / len(cases),
            "avg_latency_sec": sum(latencies) / len(latencies),
            "avg_generated_tokens": sum(token_totals) / len(token_totals),
        })

    return pd.DataFrame(metrics), all_records


def save_eval_outputs(metrics_df: pd.DataFrame, records: list[dict], prefix: str) -> None:
    metrics_df.to_csv(RESULTS_DIR / f"{prefix}_metrics.csv", index=False)
    with (RESULTS_DIR / f"{prefix}_records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\\n")
"""
        ),
        code(
            """
# Base model evaluation. Required: N=1. Optional self-consistency for base can be enabled if runtime allows.
RUN_BASE_SC = False
BASE_N_VALUES = [1, 5, 7] if RUN_BASE_SC else [1]

base_model, base_tokenizer = load_base_for_eval()
base_metrics, base_records = evaluate_model("base", base_model, base_tokenizer, eval_cases, BASE_N_VALUES)
save_eval_outputs(base_metrics, base_records, "base")
display(base_metrics)
unload_model(base_model)
"""
        ),
        code(
            """
# Fine-tuned adapter evaluation. Required N sweep for test-time compute.
FT_N_VALUES = [1, 3, 5, 7]

ft_model, ft_tokenizer = load_finetuned_for_eval(ADAPTER_DIR)
ft_metrics, ft_records = evaluate_model("fine_tuned", ft_model, ft_tokenizer, eval_cases, FT_N_VALUES)
save_eval_outputs(ft_metrics, ft_records, "fine_tuned")
display(ft_metrics)
unload_model(ft_model)
"""
        ),
        code(
            """
all_metrics = pd.concat([base_metrics, ft_metrics], ignore_index=True)
all_metrics.to_csv(RESULTS_DIR / "final_evaluation_metrics.csv", index=False)
display(all_metrics)

plt.figure(figsize=(7, 4))
for model_label, group in all_metrics.groupby("model"):
    group = group.sort_values("N")
    plt.plot(group["N"], group["exact_accuracy"] * 100, marker="o", label=model_label)
plt.xlabel("N reasoning paths")
plt.ylabel("Exact numeric accuracy (%)")
plt.title("Accuracy vs N - Self-Consistency on GSM8K Test")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_vs_n.png", dpi=160)
plt.show()
"""
        ),
        code(
            """
def records_by_case(records: list[dict], model_label: str, N: int) -> dict[int, dict]:
    return {r["case_id"]: r for r in records if r["model"] == model_label and r["N"] == N}


base_n1 = records_by_case(base_records, "base", 1)
ft_n1 = records_by_case(ft_records, "fine_tuned", 1)

ft_wins_exact = 0
base_wins_exact = 0
ties_exact = 0
for case in eval_cases:
    b = base_n1[case["case_id"]]["correct"]
    f = ft_n1[case["case_id"]]["correct"]
    if f and not b:
        ft_wins_exact += 1
    elif b and not f:
        base_wins_exact += 1
    else:
        ties_exact += 1

exact_win_table = pd.DataFrame([{
    "comparison": "fine_tuned_N1_vs_base_N1_exact",
    "fine_tuned_wins": ft_wins_exact,
    "base_wins": base_wins_exact,
    "ties": ties_exact,
    "fine_tuned_win_rate_excluding_ties": ft_wins_exact / max(1, ft_wins_exact + base_wins_exact),
    "fine_tuned_win_rate_including_ties_as_half": (ft_wins_exact + 0.5 * ties_exact) / len(eval_cases),
}])
exact_win_table.to_csv(RESULTS_DIR / "exact_win_rate.csv", index=False)
display(exact_win_table)
"""
        ),
        md(
            """
## 6. Blind Pairwise LLM-as-a-Judge

The judge compares base and fine-tuned outputs without knowing which is which. Gold answers are included so correctness is grounded and long wrong reasoning is not rewarded.
"""
        ),
        code(
            """
JUDGE_MODEL = "gemini-2.5-flash"

PAIRWISE_JUDGE_PROMPT = '''You are judging two anonymous model answers to a GSM8K math problem.
You must return JSON only, with no markdown or extra text.

Problem:
{question}

Gold final answer:
{gold_answer}

Answer A:
{answer_a}

Answer B:
{answer_b}

Rules:
- Correctness against the gold final answer matters most.
- Reasoning rigor should reward concise, valid steps, not verbosity.
- Penalize long but wrong reasoning.
- A generic "I checked and it is correct" reflection counts only as "checks_only".
- Use "detects_and_fixes_error" only when reflection identifies or fixes a meaningful issue.
- Do not assume A or B is the fine-tuned model.

Return exactly this JSON schema:
{{
  "case_id": {case_id},
  "gold_answer": "{gold_answer}",
  "A": {{
    "final_correct": true,
    "cot_rigor": 0,
    "self_correction": "none|checks_only|detects_and_fixes_error|incorrect_correction",
    "format_valid": true
  }},
  "B": {{
    "final_correct": true,
    "cot_rigor": 0,
    "self_correction": "none|checks_only|detects_and_fixes_error|incorrect_correction",
    "format_valid": true
  }},
  "winner": "A|B|tie",
  "reason": "short explanation"
}}'''


def parse_json_response(text: str) -> dict:
    cleaned = (text or "").strip()
    cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
    return json.loads(cleaned)


def judge_pair(client, case: dict, answer_a: str, answer_b: str) -> dict:
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        case_id=case["case_id"],
        question=case["question"],
        gold_answer=case["gold_answer"],
        answer_a=answer_a,
        answer_b=answer_b,
    )
    response = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    try:
        return parse_json_response(response.text)
    except Exception:
        repair_prompt = (
            "Repair the following text into valid JSON matching the requested schema. "
            "Return JSON only.\\n\\n" + (response.text or "")
        )
        repaired = client.models.generate_content(model=JUDGE_MODEL, contents=repair_prompt)
        return parse_json_response(repaired.text)
"""
        ),
        code(
            """
def run_pairwise_judge(base_records: list[dict], ft_records: list[dict], cases: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    from google import genai

    api_key = get_gemini_api_key(required=True)
    client = genai.Client(api_key=api_key)
    rng = random.Random(seed)
    base_by_case = records_by_case(base_records, "base", 1)
    ft_by_case = records_by_case(ft_records, "fine_tuned", 1)

    judged = []
    failures = []
    for case in tqdm(cases, desc="LLM-as-a-Judge"):
        base_answer = base_by_case[case["case_id"]]["winning_path"] or ""
        ft_answer = ft_by_case[case["case_id"]]["winning_path"] or ""
        ft_is_a = rng.random() < 0.5
        answer_a = ft_answer if ft_is_a else base_answer
        answer_b = base_answer if ft_is_a else ft_answer
        mapping = {"A": "fine_tuned" if ft_is_a else "base", "B": "base" if ft_is_a else "fine_tuned"}

        try:
            result = judge_pair(client, case, answer_a, answer_b)
            result["mapping"] = mapping
            result["ft_is_a"] = ft_is_a
            judged.append(result)
        except Exception as exc:
            failures.append({"case_id": case["case_id"], "error": repr(exc)})

    with (RESULTS_DIR / "judge_pairwise_results.jsonl").open("w", encoding="utf-8") as f:
        for row in judged:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")
    with (RESULTS_DIR / "judge_failures.json").open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    return judged, failures


judge_results, judge_failures = run_pairwise_judge(base_records, ft_records, eval_cases)
print("Judge successes:", len(judge_results))
print("Judge failures:", len(judge_failures))
"""
        ),
        code(
            """
def summarize_judge_results(judge_results: list[dict], judge_failures: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wins = Counter()
    rigor = defaultdict(list)
    correction = defaultdict(Counter)

    for row in judge_results:
        winner = row.get("winner", "tie")
        mapping = row["mapping"]
        if winner in ("A", "B"):
            wins[mapping[winner]] += 1
        else:
            wins["tie"] += 1

        for side in ("A", "B"):
            model_name = mapping[side]
            side_data = row.get(side, {})
            if isinstance(side_data.get("cot_rigor"), (int, float)):
                rigor[model_name].append(side_data["cot_rigor"])
            correction[model_name][side_data.get("self_correction", "missing")] += 1

    win_df = pd.DataFrame([{
        "base_wins": wins["base"],
        "fine_tuned_wins": wins["fine_tuned"],
        "ties": wins["tie"],
        "judge_failures": len(judge_failures),
        "fine_tuned_win_rate_excluding_ties": wins["fine_tuned"] / max(1, wins["fine_tuned"] + wins["base"]),
        "fine_tuned_win_rate_including_ties_as_half": (wins["fine_tuned"] + 0.5 * wins["tie"]) / max(1, len(judge_results)),
    }])

    rigor_df = pd.DataFrame([
        {"model": model_name, "avg_cot_rigor": sum(values) / len(values), "n": len(values)}
        for model_name, values in rigor.items()
        if values
    ])

    corr_rows = []
    for model_name, counts in correction.items():
        total = sum(counts.values())
        for category, count in counts.items():
            corr_rows.append({"model": model_name, "self_correction": category, "count": count, "rate": count / max(1, total)})
    correction_df = pd.DataFrame(corr_rows)

    win_df.to_csv(RESULTS_DIR / "judge_win_rate.csv", index=False)
    rigor_df.to_csv(RESULTS_DIR / "judge_cot_rigor.csv", index=False)
    correction_df.to_csv(RESULTS_DIR / "judge_self_correction_distribution.csv", index=False)
    return win_df, rigor_df, correction_df


judge_win_df, judge_rigor_df, judge_correction_df = summarize_judge_results(judge_results, judge_failures)
display(judge_win_df)
display(judge_rigor_df)
display(judge_correction_df)
"""
        ),
        md(
            """
## 7. Qualitative Analysis

Use these helpers after evaluation to show failure cases and successful cases in the final report.
"""
        ),
        code(
            """
def show_failure_cases(records: list[dict], limit: int = 3):
    failures = [r for r in records if not r["correct"]]
    for r in failures[:limit]:
        print("=" * 100)
        print(f"{r['model']} N={r['N']} case={r['case_id']} gold={r['gold_answer']} pred={r['pred_answer']}")
        print(r["question"])
        print("\\nWinning path:\\n", (r["winning_path"] or "")[:3000])


def show_self_consistency_rescues(ft_records: list[dict], low_n: int = 1, high_n: int = 5, limit: int = 3):
    low = records_by_case(ft_records, "fine_tuned", low_n)
    high = records_by_case(ft_records, "fine_tuned", high_n)
    shown = 0
    for case_id, high_record in high.items():
        if shown >= limit:
            break
        low_record = low.get(case_id)
        if low_record and (not low_record["correct"]) and high_record["correct"]:
            print("=" * 100)
            print(f"case={case_id} gold={high_record['gold_answer']} N={low_n} pred={low_record['pred_answer']} N={high_n} pred={high_record['pred_answer']}")
            print(high_record["question"])
            print("All N answers:", high_record["all_answers"])
            shown += 1


# Examples to run after evaluation:
# show_failure_cases(ft_records)
# show_self_consistency_rescues(ft_records, low_n=1, high_n=5)
"""
        ),
        md(
            """
## 8. Final Interpretation Template

This project is not just a fixed training script. The contribution is an experimental validation of a deep-reasoning recipe under small-model constraints:

- Distillation: a frontier teacher produces structured reasoning traces, and verifier-based rejection sampling keeps only GSM8K-correct examples.
- Alignment: QLoRA trains low-rank adapters so the 3B student imitates the structured reasoning format without updating the frozen base model.
- Test-time compute: multiple sampled reasoning paths are aggregated by semantic/numeric majority vote.
- Evaluation: held-out GSM8K test cases measure exact numeric accuracy, formatting reliability, latency/cost, and blind pairwise judge preferences.

When writing the final report, include:

1. Dataset audit table.
2. Training loss curve.
3. Accuracy vs N plot.
4. Base vs fine-tuned exact win-rate table.
5. LLM-as-a-Judge win rate, CoT rigor, and self-correction distribution.
6. Two or three failure cases and one or two successful self-consistency examples.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "nbconvert_exporter": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    audit = structurally_clean_sft()
    nb = build_notebook()
    with NOTEBOOK.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("Wrote", NOTEBOOK)
    print("Wrote", CLEAN_SFT)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
