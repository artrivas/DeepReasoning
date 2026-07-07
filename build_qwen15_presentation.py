
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts_qwen15b"
PLOTS = ART / "presentation_plots"
PLOTS.mkdir(exist_ok=True)
RUNS = [
    "qwen15_len1024_r8_a16_lr5e5_e3_es",
    "qwen15_len1024_r8_a16_lr2e5_e3_es",
    "qwen15_len1024_r16_a32_lr5e5_e3_es",
]

def load(path):
    return json.loads(Path(path).read_text())

def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def extract_tag(text, tag):
    m = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def normalize_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", "").replace("$", ""))
    if not nums:
        return ""
    val = float(nums[-1])
    return str(int(val)) if val.is_integer() else str(val)

def loose_from_trace(trace):
    strict = normalize_number(extract_tag(trace, "answer"))
    return strict or normalize_number(trace)

rows = []
for run in RUNS:
    rd = ART / run
    cfg = load(rd / "config.json")
    train = load(rd / "training_summary.json")
    ev = load(rd / "evaluation_summary.json")
    params = load(rd / "parameter_stats.json")
    toks = load(rd / "tokenization_stats.json")
    adapter_records = load_jsonl(rd / "raw" / "adapter_predictions.jsonl")
    loose_correct = []
    for rec in adapter_records:
        pred = loose_from_trace(rec.get("greedy_trace", rec.get("trace", "")))
        loose_correct.append(pred == rec["gold_answer"])
    rows.append({
        "run": run,
        "rank": cfg["lora_r"],
        "alpha": cfg["lora_alpha"],
        "lr": cfg["learning_rate"],
        "epochs": cfg["num_epochs"],
        "train_batch": cfg["train_batch_size"],
        "grad_accum": cfg["gradient_accumulation_steps"],
        "effective_batch": cfg["train_batch_size"] * cfg["gradient_accumulation_steps"],
        "trainable_params_m": params["trainable_parameters"] / 1e6,
        "truncation_rate": toks["train"]["truncated_fraction"],
        "eval_loss": train["eval_loss"],
        "best_eval_loss": train["best_metric"],
        "train_minutes": train["train_wall_time_seconds"] / 60,
        "max_gpu_gib": train["max_gpu_memory_allocated_gib"],
        "strict_accuracy_30": ev["adapter"]["greedy_accuracy"],
        "loose_accuracy_30": sum(loose_correct) / len(loose_correct),
        "format_rate_30": ev["adapter"]["valid_format_rate"],
        "reflection_rate_30": ev["adapter"]["reflection_rate"],
        "mean_latency_s_30": ev["adapter"]["mean_greedy_latency_seconds"],
    })

df = pd.DataFrame(rows).sort_values("strict_accuracy_30", ascending=False)
df.to_csv(ART / "qwen15b_screen_summary.csv", index=False)
confirm = load(ART / "confirm_qwen15_len1024_r8_a16_lr5e5_e3_es_greedy100" / "evaluation_summary.json")

plt.figure(figsize=(9, 4.8))
labels = [r.replace("qwen15_", "").replace("_e3_es", "") for r in df["run"]]
x = range(len(df))
plt.bar([i - 0.18 for i in x], 100 * df["strict_accuracy_30"], width=0.36, label="strict tagged")
plt.bar([i + 0.18 for i in x], 100 * df["loose_accuracy_30"], width=0.36, label="loose numeric")
plt.xticks(list(x), labels, rotation=20, ha="right")
plt.ylabel("Accuracy on 30-question screen (%)")
plt.title("Qwen2.5-1.5B screen accuracy")
plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "screen_accuracy.png", dpi=180)

plt.figure(figsize=(7, 4.8))
plt.scatter(df["eval_loss"], 100 * df["strict_accuracy_30"], s=90)
for _, r in df.iterrows():
    plt.annotate(f"r{int(r['rank'])}, lr={r['lr']:.0e}", (r["eval_loss"], 100*r["strict_accuracy_30"]), xytext=(5,5), textcoords="offset points")
plt.xlabel("Validation loss")
plt.ylabel("Strict screen accuracy (%)")
plt.title("Validation loss did not perfectly rank answer quality")
plt.tight_layout(); plt.savefig(PLOTS / "loss_vs_accuracy.png", dpi=180)

plt.figure(figsize=(7, 4.5))
metrics = ["strict_tag_answer", "loose_numeric_answer"]
metric_labels = ["Strict tagged", "Loose numeric"]
base_vals = [100*confirm[m]["base_accuracy"] for m in metrics]
adapt_vals = [100*confirm[m]["adapter_accuracy"] for m in metrics]
x = range(len(metrics))
plt.bar([i - 0.18 for i in x], base_vals, width=0.36, label="base")
plt.bar([i + 0.18 for i in x], adapt_vals, width=0.36, label="adapter")
plt.xticks(list(x), metric_labels)
plt.ylabel("Accuracy on 100-question confirmation (%)")
plt.title("Leader confirmation: format success vs loose-answer tradeoff")
plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "confirm_accuracy.png", dpi=180)

plt.figure(figsize=(7, 4.5))
plt.bar(["base", "adapter"], [100*confirm["base_valid_format_rate"], 100*confirm["adapter_valid_format_rate"]])
plt.ylabel("Valid required format (%)")
plt.title("100-question format compliance")
plt.tight_layout(); plt.savefig(PLOTS / "confirm_format_rate.png", dpi=180)

plt.figure(figsize=(9, 5))
for run in RUNS:
    hist = pd.read_json(ART / run / "raw" / "trainer_log_history.json")
    evh = hist.dropna(subset=["eval_loss"]) if "eval_loss" in hist else pd.DataFrame()
    if not evh.empty:
        plt.plot(evh["step"], evh["eval_loss"], marker="o", label=run.replace("qwen15_", ""))
plt.xlabel("Optimizer step"); plt.ylabel("Validation loss"); plt.title("Validation curves")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(PLOTS / "validation_curves.png", dpi=180)

fmt = lambda v: f"{100*v:.1f}%"
leader = df.iloc[0]
report = f"""# Qwen2.5-1.5B DeepReasoning ablation report

## One-slide conclusion

We repeated the QLoRA DeepReasoning pipeline with `Qwen/Qwen2.5-1.5B-Instruct` because the 3B model was less stressed by the dataset. The best adapter was `{leader['run']}`.

| Metric | Base | Adapter | Interpretation |
|---|---:|---:|---|
| Strict tagged answer, 100 questions | {fmt(confirm['strict_tag_answer']['base_accuracy'])} | {fmt(confirm['strict_tag_answer']['adapter_accuracy'])} | Adapter wins because the required format matters. |
| Loose numeric answer, 100 questions | {fmt(confirm['loose_numeric_answer']['base_accuracy'])} | {fmt(confirm['loose_numeric_answer']['adapter_accuracy'])} | Base remains slightly better at raw math answers. |
| Valid required format | {fmt(confirm['base_valid_format_rate'])} | {fmt(confirm['adapter_valid_format_rate'])} | LoRA mainly teaches format/instruction following. |

## Why this pipeline

- Same fixed training split and GSM8K evaluation indices across runs.
- Separate `artifacts_qwen15b` root so no cached 3B base generations contaminate 1.5B results.
- 30-question greedy screens for quick hyperparameter comparison.
- 100-question no-retraining confirmation only for the best screened adapter.
- Strict and loose metrics are both reported because format compliance and final-answer accuracy answer different questions.

## Runtime decisions

- Batch 32 / grad accum 1 OOMed near full VRAM.
- Batch 16 / grad accum 2 also OOMed during the fp32 full-vocabulary loss.
- Batch 8 / grad accum 4 was stable and kept the same effective batch size of 32.
- All stable runs used 3 epochs with early stopping enabled. In practice, validation loss kept improving, so all three ran the full 3 epochs.

## 30-question screen

{df.to_markdown(index=False, floatfmt='.4f')}

The screen selected `{leader['run']}` because it had the best strict accuracy ({fmt(leader['strict_accuracy_30'])}) and best format rate ({fmt(leader['format_rate_30'])}). The r=16 run had lower validation loss but worse answer accuracy, which is why we did not rely on validation loss alone.

## Final interpretation

The best defensible configuration is `{leader['run']}` if the assignment rewards structured reasoning traces. If the only objective is raw GSM8K numeric accuracy, the base 1.5B model is still slightly better. The experiment therefore demonstrates a format-following gain with a raw-answer tradeoff.
"""
(ART / "QWEN15B_PRESENTATION_REPORT.md").write_text(report, encoding="utf-8")

setup_code = """from pathlib import Path
import json
import pandas as pd
from IPython.display import Markdown, Image, display

ART = Path('artifacts_qwen15b')
df = pd.read_csv(ART / 'qwen15b_screen_summary.csv')
confirm = json.loads((ART / 'confirm_qwen15_len1024_r8_a16_lr5e5_e3_es_greedy100' / 'evaluation_summary.json').read_text())
report_md = (ART / 'QWEN15B_PRESENTATION_REPORT.md').read_text()

def pct(x):
    return f'{100*x:.1f}%'

print('Loaded artifacts from', ART)
print('Screened runs:', len(df))
print('Confirmation examples:', confirm['n_examples'])"""

training_config_code = """# This cell documents the actual training configuration.
# It is not executed during presentation; the executed results are loaded from artifacts.
RUN_FULL_TRAINING = False

if RUN_FULL_TRAINING:
    MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
    DATA_PATH = 'data/sft_reasoning_2k.jsonl'
    ARTIFACT_ROOT = 'artifacts_qwen15b'

    MAX_LEN = 1024
    LORA_R = 8
    LORA_ALPHA = 16
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3

    TRAIN_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    EFFECTIVE_BATCH_SIZE = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    EVAL_BATCH_SIZE = 8
    GRADIENT_CHECKPOINTING = False

    EVAL_PROBLEMS = 30
    MAX_NEW_TOKENS = 768
    RUN_SELF_CONSISTENCY = False"""

lora_training_code = """# Core QLoRA training cell used by the experiment.
# Kept behind RUN_FULL_TRAINING so this presentation notebook opens quickly.
if RUN_FULL_TRAINING:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        device_map={'': 0},
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir=f'{ARTIFACT_ROOT}/{RUN_NAME}/checkpoints',
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        optim='paged_adamw_8bit',
        bf16=True,
        tf32=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        logging_steps=1,
        eval_strategy='steps',
        eval_steps=20,
        save_strategy='steps',
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                             label_pad_token_id=-100,
                                             pad_to_multiple_of=8),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3,
                                         early_stopping_threshold=5e-4)],
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)"""

eval_code = """# Evaluation logic used for both the 30-question screen and 100-question confirmation.
# Strict accuracy requires the model to place the final answer inside <answer> tags.
# Loose accuracy extracts the last number from the whole trace, even if tags are missing.
import re

SYSTEM_INSTRUCTION = \"\"\"You are a meticulous reasoning tutor.
For every problem, answer using EXACTLY these tags in order:
<thinking>
Reason step by step. Show every intermediate calculation.
</thinking>
<reflection>
Re-check your reasoning. Look for arithmetic slips or wrong assumptions.
</reflection>
<answer>
Give only the final answer.
</answer>\"\"\"

def extract_tag(text: str, tag: str) -> str:
    match = re.search(fr'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ''

def normalize_number(text: str) -> str:
    numbers = re.findall(r'-?\\d+(?:\\.\\d+)?', str(text).replace(',', '').replace('$', ''))
    if not numbers:
        return ''
    value = float(numbers[-1])
    return str(int(value)) if value.is_integer() else str(value)

def score_trace(trace: str, gold: str) -> dict:
    thinking = extract_tag(trace, 'thinking')
    reflection = extract_tag(trace, 'reflection')
    answer = extract_tag(trace, 'answer')
    strict_prediction = normalize_number(answer)
    loose_prediction = strict_prediction or normalize_number(trace)
    return {
        'valid_format': bool(thinking and reflection and answer),
        'has_reflection': bool(reflection),
        'strict_correct': strict_prediction == gold,
        'loose_correct': loose_prediction == gold,
    }"""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Qwen2.5-1.5B DeepReasoning ablation\n\nThis executed notebook is the presentation artifact. It shows the training/evaluation code path, but loads saved cloud artifacts so the notebook is fast and reproducible during class."),
    nbf.v4.new_code_cell(setup_code),
    nbf.v4.new_markdown_cell("## 1. Research question and conclusion\n\nWe switched from Qwen2.5-3B to Qwen2.5-1.5B because the 3B model was strong enough that the dataset was less revealing. The adapter learns the required reasoning format, but does not improve loose final-answer accuracy over the base model."),
    nbf.v4.new_code_cell("display(Markdown(report_md))"),
    nbf.v4.new_markdown_cell("## 2. Training configuration we actually used\n\nThe next cells show the training setup. `RUN_FULL_TRAINING` is deliberately `False`: the heavy training already ran on the RTX PRO 6000 and the notebook loads those saved artifacts."),
    nbf.v4.new_code_cell(training_config_code),
    nbf.v4.new_markdown_cell("## 3. Core QLoRA training code\n\nThis is the implementation: 4-bit NF4 loading, LoRA target modules, Trainer settings, early stopping, and adapter saving."),
    nbf.v4.new_code_cell(lora_training_code),
    nbf.v4.new_markdown_cell("## 4. Evaluation code and why there are two metrics\n\nStrict accuracy asks whether the model obeyed the required format. Loose accuracy asks whether the final numeric answer can be recovered anywhere in the trace."),
    nbf.v4.new_code_cell(eval_code),
    nbf.v4.new_markdown_cell("## 5. Hyperparameter grid and saved training artifacts\n\nThis table is loaded from artifacts, not recomputed. It is the evidence used to select the leader."),
    nbf.v4.new_code_cell("cols = ['run', 'rank', 'alpha', 'lr', 'train_batch', 'grad_accum', 'effective_batch', 'eval_loss', 'strict_accuracy_30', 'loose_accuracy_30', 'format_rate_30', 'train_minutes', 'max_gpu_gib']\ndisplay(df[cols])"),
    nbf.v4.new_markdown_cell("## 6. Charts: screen accuracy and loss/accuracy relationship\n\nThe r=16 adapter had the best validation loss but worse strict accuracy. That is why the final choice uses held-out generation behavior, not validation loss alone."),
    nbf.v4.new_code_cell("display(Image(filename=str(ART / 'presentation_plots' / 'screen_accuracy.png')))\ndisplay(Image(filename=str(ART / 'presentation_plots' / 'loss_vs_accuracy.png')))"),
    nbf.v4.new_markdown_cell("## 7. Training stability and runtime choices\n\nBatch 32 and batch 16 OOMed; batch 8 with grad accumulation 4 was stable and preserved effective batch size 32."),
    nbf.v4.new_code_cell("display(Image(filename=str(ART / 'presentation_plots' / 'validation_curves.png')))\nfor oom_dir in sorted(ART.glob('oom_*')):\n    record = oom_dir / 'OOM_RECORD.md'\n    if record.exists():\n        print(f'--- {oom_dir.name} ---')\n        print(record.read_text())"),
    nbf.v4.new_markdown_cell("## 8. 100-question confirmation\n\nAfter the 30-question screen, only the leader was confirmed on 100 questions without retraining."),
    nbf.v4.new_code_cell("display(Image(filename=str(ART / 'presentation_plots' / 'confirm_accuracy.png')))\ndisplay(Image(filename=str(ART / 'presentation_plots' / 'confirm_format_rate.png')))\nprint(json.dumps(confirm, indent=2))"),
    nbf.v4.new_markdown_cell("## 9. Final takeaway\n\nUse `qwen15_len1024_r8_a16_lr5e5_e3_es` if structured reasoning traces are required. If the only objective is loose GSM8K numeric answer accuracy, the base 1.5B model is still slightly better. The experiment demonstrates a format-following gain with a raw-answer tradeoff."),
]
nbf.write(nb, ROOT / "qwen15b_presentation.ipynb")
print(report)
