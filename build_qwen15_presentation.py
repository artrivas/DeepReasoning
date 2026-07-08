
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts_qwen15b"

report = (ART / "QWEN15B_PRESENTATION_REPORT.md").read_text(encoding="utf-8")

setup_code = r"""
from pathlib import Path
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

ART = Path('artifacts_qwen15b')
RUNS = [
    'qwen15_len1024_r8_a16_lr5e5_e3_es',
    'qwen15_len1024_r8_a16_lr2e5_e3_es',
    'qwen15_len1024_r16_a32_lr5e5_e3_es',
]
LEADER_RUN = 'qwen15_len1024_r8_a16_lr5e5_e3_es'
CONFIRM_DIR = ART / 'confirm_qwen15_len1024_r8_a16_lr5e5_e3_es_greedy100'

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def extract_tag(text, tag):
    match = re.search(fr'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ''

def normalize_number(text):
    numbers = re.findall(r'-?\d+(?:\.\d+)?', str(text).replace(',', '').replace('$', ''))
    if not numbers:
        return ''
    value = float(numbers[-1])
    return str(int(value)) if value.is_integer() else str(value)

def score_trace(trace, gold):
    thinking = extract_tag(trace, 'thinking')
    reflection = extract_tag(trace, 'reflection')
    answer = extract_tag(trace, 'answer')
    strict_prediction = normalize_number(answer)
    loose_prediction = strict_prediction or normalize_number(trace)
    return {
        'strict_prediction': strict_prediction,
        'loose_prediction': loose_prediction,
        'valid_format': bool(thinking and reflection and answer),
        'has_reflection': bool(reflection),
        'strict_correct': strict_prediction == gold,
        'loose_correct': loose_prediction == gold,
    }

confirm_summary = load_json(CONFIRM_DIR / 'evaluation_summary.json')
print('Loaded artifact root:', ART)
print('Runs:', len(RUNS))
print('100-question confirmation examples:', confirm_summary['n_examples'])
"""

training_config_code = r"""
# This cell documents the actual training configuration.
# It is safe during presentation because RUN_FULL_TRAINING is False.
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

    # Batch 32 and batch 16 OOMed; batch 8 kept memory stable.
    TRAIN_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    EFFECTIVE_BATCH_SIZE = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    EVAL_BATCH_SIZE = 8
    GRADIENT_CHECKPOINTING = False

    EVAL_STEPS = 20
    SAVE_STEPS = 20
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 5e-4
    MAX_NEW_TOKENS = 768
"""

training_code = r"""
# Core QLoRA training code used by the experiment.
# This mirrors the cloud run but is guarded to avoid accidental retraining.
if RUN_FULL_TRAINING:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        DataCollatorForSeq2Seq, EarlyStoppingCallback,
        Trainer, TrainingArguments,
    )

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
        eval_steps=EVAL_STEPS,
        save_strategy='steps',
        save_steps=SAVE_STEPS,
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
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        )],
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
"""

scoring_code = r"""
# Evaluation/scoring code used in the screen and confirmation.
# Strict accuracy requires the final answer inside <answer>; loose accuracy
# extracts the last number from the whole trace even if tags are missing.
SYSTEM_INSTRUCTION = '''You are a meticulous reasoning tutor.
For every problem, answer using EXACTLY these tags in order:
<thinking>
Reason step by step. Show every intermediate calculation.
</thinking>
<reflection>
Re-check your reasoning. Look for arithmetic slips or wrong assumptions.
</reflection>
<answer>
Give only the final answer.
</answer>'''

print(score_trace('<thinking>x</thinking><reflection>ok</reflection><answer>42</answer>', '42'))
"""

screen_recompute_code = r"""
# Recompute the screen table from raw artifacts and raw prediction JSONLs.
# This is not just displaying a pre-made image: we parse predictions again here.
rows = []
for run in RUNS:
    run_dir = ART / run
    cfg = load_json(run_dir / 'config.json')
    train = load_json(run_dir / 'training_summary.json')
    params = load_json(run_dir / 'parameter_stats.json')
    tokens = load_json(run_dir / 'tokenization_stats.json')
    predictions = load_jsonl(run_dir / 'raw' / 'adapter_predictions.jsonl')

    strict_scores = []
    loose_scores = []
    format_scores = []
    for row in predictions:
        trace = row.get('greedy_trace', row.get('trace', ''))
        scored = score_trace(trace, row['gold_answer'])
        strict_scores.append(scored['strict_correct'])
        loose_scores.append(scored['loose_correct'])
        format_scores.append(scored['valid_format'])

    rows.append({
        'run': run,
        'rank': cfg['lora_r'],
        'alpha': cfg['lora_alpha'],
        'lr': cfg['learning_rate'],
        'train_batch': cfg['train_batch_size'],
        'grad_accum': cfg['gradient_accumulation_steps'],
        'effective_batch': cfg['train_batch_size'] * cfg['gradient_accumulation_steps'],
        'trainable_params_m': params['trainable_parameters'] / 1e6,
        'truncation_rate': tokens['train']['truncated_fraction'],
        'eval_loss': train['eval_loss'],
        'train_minutes': train['train_wall_time_seconds'] / 60,
        'max_gpu_gib': train['max_gpu_memory_allocated_gib'],
        'strict_accuracy_30': np.mean(strict_scores),
        'loose_accuracy_30': np.mean(loose_scores),
        'format_rate_30': np.mean(format_scores),
    })

screen_df = pd.DataFrame(rows).sort_values('strict_accuracy_30', ascending=False)
display(screen_df)
leader = screen_df.iloc[0]
print('Selected leader:', leader['run'])
"""

screen_plot_code = r"""
# Generate charts live from screen_df.
labels = [name.replace('qwen15_', '').replace('_e3_es', '') for name in screen_df['run']]
x = np.arange(len(screen_df))

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.bar(x - 0.2, 100 * screen_df['strict_accuracy_30'], width=0.4, label='strict tagged')
ax.bar(x + 0.2, 100 * screen_df['loose_accuracy_30'], width=0.4, label='loose numeric')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel('Accuracy on 30-question screen (%)')
ax.set_title('Qwen2.5-1.5B screen accuracy generated from raw predictions')
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.scatter(screen_df['eval_loss'], 100 * screen_df['strict_accuracy_30'], s=100)
for _, row in screen_df.iterrows():
    ax.annotate(f"r{int(row['rank'])}, lr={row['lr']:.0e}",
                (row['eval_loss'], 100 * row['strict_accuracy_30']),
                xytext=(6, 5), textcoords='offset points')
ax.set_xlabel('Validation loss')
ax.set_ylabel('Strict screen accuracy (%)')
ax.set_title('Why validation loss alone was not enough')
plt.tight_layout()
plt.show()
"""

training_curves_code = r"""
# Generate validation curves live from Trainer log history.
fig, ax = plt.subplots(figsize=(10, 5))
for run in RUNS:
    history = pd.read_json(ART / run / 'raw' / 'trainer_log_history.json')
    eval_history = history.dropna(subset=['eval_loss'])
    ax.plot(eval_history['step'], eval_history['eval_loss'], marker='o', label=run.replace('qwen15_', ''))
ax.set_xlabel('Optimizer step')
ax.set_ylabel('Validation loss')
ax.set_title('Validation loss during training')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

print('OOM evidence from failed speed attempts:')
for oom_dir in sorted(ART.glob('oom_*')):
    record = oom_dir / 'OOM_RECORD.md'
    if record.exists():
        print(f'\n--- {oom_dir.name} ---')
        print(record.read_text())
"""

confirm_recompute_code = r"""
# Recompute the 100-question confirmation from raw base/adapter prediction JSONLs.
base_records = load_jsonl(CONFIRM_DIR / 'base_predictions.jsonl')
adapter_records = load_jsonl(CONFIRM_DIR / 'adapter_predictions.jsonl')

summary_rows = []
for mode, records in [('base', base_records), ('adapter', adapter_records)]:
    strict = []
    loose = []
    fmt = []
    reflection = []
    for row in records:
        scored = score_trace(row['trace'], row['gold_answer'])
        strict.append(scored['strict_correct'])
        loose.append(scored['loose_correct'])
        fmt.append(scored['valid_format'])
        reflection.append(scored['has_reflection'])
    summary_rows.append({
        'mode': mode,
        'strict_accuracy': np.mean(strict),
        'loose_accuracy': np.mean(loose),
        'valid_format_rate': np.mean(fmt),
        'reflection_rate': np.mean(reflection),
        'n': len(records),
    })
confirm_df = pd.DataFrame(summary_rows)
display(confirm_df)
print('Raw rows:', len(base_records), 'base and', len(adapter_records), 'adapter')
"""

confirm_plot_code = r"""
# Generate confirmation charts live from confirm_df.
fig, ax = plt.subplots(figsize=(7, 4.8))
x = np.arange(2)
base = confirm_df[confirm_df['mode'] == 'base'].iloc[0]
adapter = confirm_df[confirm_df['mode'] == 'adapter'].iloc[0]
ax.bar(x - 0.18, [100 * base['strict_accuracy'], 100 * base['loose_accuracy']], width=0.36, label='base')
ax.bar(x + 0.18, [100 * adapter['strict_accuracy'], 100 * adapter['loose_accuracy']], width=0.36, label='adapter')
ax.set_xticks(x)
ax.set_xticklabels(['Strict tagged', 'Loose numeric'])
ax.set_ylabel('Accuracy on 100-question confirmation (%)')
ax.set_title('Leader confirmation generated from raw predictions')
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(['base', 'adapter'], [100 * base['valid_format_rate'], 100 * adapter['valid_format_rate']])
ax.set_ylabel('Valid required format (%)')
ax.set_title('Format compliance from raw traces')
plt.tight_layout()
plt.show()
"""

live_inference_code = r"""
# Tiny live inference demo with the saved adapter.
# This runs on the cloud GPU when CUDA is available; otherwise it falls back gracefully.
RUN_LIVE_INFERENCE = True

if RUN_LIVE_INFERENCE:
    try:
        import torch
        if not torch.cuda.is_available():
            print('CUDA is not available, so live inference is skipped in this environment.')
        else:
            from datasets import load_dataset
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
            ADAPTER_DIR = ART / LEADER_RUN / 'adapter'
            indices = load_json(CONFIRM_DIR / 'gsm8k_test_indices.json')
            example_index = indices[0]
            example = load_dataset('openai/gsm8k', 'main', split='test').select([example_index])[0]
            gold = normalize_number(example['answer'].split('####')[-1])

            tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quantization,
                device_map={'': 0},
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            model.eval()

            messages = [
                {'role': 'system', 'content': SYSTEM_INSTRUCTION},
                {'role': 'user', 'content': example['question']},
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors='pt',
                return_dict=True,
            ).to(model.device)

            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=768,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            trace = tokenizer.decode(output[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True)
            scored = score_trace(trace, gold)

            print('Question:')
            print(example['question'])
            print('\nGold answer:', gold)
            print('\nGenerated adapter trace:')
            print(trace)
            print('\nParsed score:')
            print(json.dumps(scored, indent=2))

            del model, base_model
            torch.cuda.empty_cache()
    except Exception as exc:
        print('Live inference skipped because this environment could not run it:')
        print(type(exc).__name__, exc)
"""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Qwen2.5-1.5B DeepReasoning ablation\n\nThis is the presentation notebook. It shows the training/evaluation code, recomputes metrics and charts from saved artifacts, and runs a tiny live adapter inference demo when CUDA is available."),
    nbf.v4.new_code_cell(setup_code),
    nbf.v4.new_markdown_cell("## 1. Research question and conclusion"),
    nbf.v4.new_code_cell("display(Markdown((ART / 'QWEN15B_PRESENTATION_REPORT.md').read_text()))"),
    nbf.v4.new_markdown_cell("## 2. Training configuration\n\nThis is the actual configuration family. Full training is disabled here so the notebook remains safe to present."),
    nbf.v4.new_code_cell(training_config_code),
    nbf.v4.new_markdown_cell("## 3. Core QLoRA training code"),
    nbf.v4.new_code_cell(training_code),
    nbf.v4.new_markdown_cell("## 4. Evaluation/scoring code"),
    nbf.v4.new_code_cell(scoring_code),
    nbf.v4.new_markdown_cell("## 5. Recompute the 30-question screen from raw predictions"),
    nbf.v4.new_code_cell(screen_recompute_code),
    nbf.v4.new_markdown_cell("## 6. Generate screen charts from recomputed data"),
    nbf.v4.new_code_cell(screen_plot_code),
    nbf.v4.new_markdown_cell("## 7. Training curves and OOM/runtime evidence"),
    nbf.v4.new_code_cell(training_curves_code),
    nbf.v4.new_markdown_cell("## 8. Recompute 100-question confirmation from raw predictions"),
    nbf.v4.new_code_cell(confirm_recompute_code),
    nbf.v4.new_markdown_cell("## 9. Generate confirmation charts from recomputed data"),
    nbf.v4.new_code_cell(confirm_plot_code),
    nbf.v4.new_markdown_cell("## 10. Tiny live inference demo with the saved adapter"),
    nbf.v4.new_code_cell(live_inference_code),
    nbf.v4.new_markdown_cell("## 11. Final takeaway\n\nThe adapter is best when structured reasoning format is required. The base model remains slightly better on loose final numeric answer accuracy. This is the core tradeoff shown by the ablation."),
]
nbf.write(nb, ROOT / 'qwen15b_presentation.ipynb')
print('Wrote qwen15b_presentation.ipynb with', len(nb.cells), 'cells')
