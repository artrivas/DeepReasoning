
from __future__ import annotations

import json, re
from pathlib import Path
from textwrap import dedent

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

# Plots
plt.figure(figsize=(9, 4.8))
labels = [r.replace("qwen15_", "").replace("_e3_es", "") for r in df["run"]]
x = range(len(df))
plt.bar([i - 0.18 for i in x], 100 * df["strict_accuracy_30"], width=0.36, label="strict tagged")
plt.bar([i + 0.18 for i in x], 100 * df["loose_accuracy_30"], width=0.36, label="loose numeric")
plt.xticks(list(x), labels, rotation=20, ha="right")
plt.ylabel("Accuracy on 30-question screen (%)")
plt.title("Qwen2.5-1.5B screen accuracy")
plt.legend(); plt.tight_layout()
plt.savefig(PLOTS / "screen_accuracy.png", dpi=180)

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

## Objective

Repeat the DeepReasoning QLoRA training/evaluation pipeline with `Qwen/Qwen2.5-1.5B-Instruct`, because the 3B model was strong enough that the dataset did not stress it as clearly.

## Runtime decisions

- Used a separate artifact root: `artifacts_qwen15b`, so no 3B base cache leaked into 1.5B comparisons.
- Tried larger batches first to exploit the RTX PRO 6000 VRAM, but preserved OOM records:
  - batch 32 / grad accum 1 OOMed near full VRAM.
  - batch 16 / grad accum 2 OOMed during fp32 loss over full vocabulary logits.
  - batch 8 / grad accum 4 was stable, using about 56 GiB allocated during training.
- Used 3 epochs with early stopping enabled; all three screened runs improved through the full schedule.
- Used fast 30-question greedy screens for the grid, then 100-question no-retraining confirmation only for the leader.

## 30-question screen

{df.to_markdown(index=False, floatfmt='.4f')}

Headline: the best screened run was `{leader['run']}` with {fmt(leader['strict_accuracy_30'])} strict tagged accuracy and {fmt(leader['format_rate_30'])} valid format.

## 100-question confirmation for leader

| Metric | Base | Adapter | Adapter - base | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Strict tagged answer | {fmt(confirm['strict_tag_answer']['base_accuracy'])} | {fmt(confirm['strict_tag_answer']['adapter_accuracy'])} | {fmt(confirm['strict_tag_answer']['adapter_minus_base'])} | [{fmt(confirm['strict_tag_answer']['bootstrap_95_ci'][0])}, {fmt(confirm['strict_tag_answer']['bootstrap_95_ci'][1])}] |
| Loose numeric answer | {fmt(confirm['loose_numeric_answer']['base_accuracy'])} | {fmt(confirm['loose_numeric_answer']['adapter_accuracy'])} | {fmt(confirm['loose_numeric_answer']['adapter_minus_base'])} | [{fmt(confirm['loose_numeric_answer']['bootstrap_95_ci'][0])}, {fmt(confirm['loose_numeric_answer']['bootstrap_95_ci'][1])}] |
| Valid required format | {fmt(confirm['base_valid_format_rate'])} | {fmt(confirm['adapter_valid_format_rate'])} | — | — |
| Reflection present | {fmt(confirm['base_reflection_rate'])} | {fmt(confirm['adapter_reflection_rate'])} | — | — |

## Interpretation

The LoRA adapter clearly learned the requested `<thinking>/<reflection>/<answer>` protocol: base format compliance was 0%, while the adapter reached {fmt(confirm['adapter_valid_format_rate'])} on the 100-question confirmation. Under the strict report metric, this is a large win because unformatted answers are invalid.

However, under loose numeric answer extraction, the base model scored {fmt(confirm['loose_numeric_answer']['base_accuracy'])} and the adapter scored {fmt(confirm['loose_numeric_answer']['adapter_accuracy'])}. That means the adapter improved instruction/format following but did not improve raw math-answer accuracy; the likely tradeoff is that the supervised traces impose a verbose format and sometimes hurt direct answer reliability.

The best defensible configuration is therefore `{leader['run']}` if the assignment values structured reasoning traces. If the only objective is final GSM8K numeric accuracy, the base 1.5B model remains competitive or slightly better.

## Reproduction

- Screen grid: `./run_qwen15b_screen.sh`
- 100-question confirmation: `MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct ARTIFACT_ROOT=$PWD/artifacts_qwen15b ADAPTER_DIR=$PWD/artifacts_qwen15b/qwen15_len1024_r8_a16_lr5e5_e3_es/adapter .venv/bin/python evaluate_qwen15_confirm.py`
- Status helper: `./job_status_qwen15b.sh`
"""
(ART / "QWEN15B_PRESENTATION_REPORT.md").write_text(report, encoding="utf-8")

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Qwen2.5-1.5B DeepReasoning ablation\n\nExecuted presentation notebook generated from cloud artifacts."),
    nbf.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import Markdown, Image, display\nART = Path('artifacts_qwen15b')\ndf = pd.read_csv(ART / 'qwen15b_screen_summary.csv')\nconfirm = json.loads((ART / 'confirm_qwen15_len1024_r8_a16_lr5e5_e3_es_greedy100' / 'evaluation_summary.json').read_text())\ndisplay(df)"),
    nbf.v4.new_markdown_cell("## 30-question screen results\n\nThe screen compares strict required-format accuracy, loose numeric accuracy, validation loss, and format compliance."),
    nbf.v4.new_code_cell("display(Image(filename=str(ART / 'presentation_plots' / 'screen_accuracy.png')))\ndisplay(Image(filename=str(ART / 'presentation_plots' / 'loss_vs_accuracy.png')))"),
    nbf.v4.new_markdown_cell("## Training stability\n\nAll stable runs used batch 8 with gradient accumulation 4 after larger microbatches OOMed."),
    nbf.v4.new_code_cell("display(Image(filename=str(ART / 'presentation_plots' / 'validation_curves.png')))"),
    nbf.v4.new_markdown_cell("## 100-question confirmation\n\nThe leader is confirmed without retraining. Strict accuracy treats missing tags as invalid; loose numeric accuracy extracts the last number even from malformed traces."),
    nbf.v4.new_code_cell("display(Image(filename=str(ART / 'presentation_plots' / 'confirm_accuracy.png')))\ndisplay(Image(filename=str(ART / 'presentation_plots' / 'confirm_format_rate.png')))\nprint(json.dumps(confirm, indent=2))"),
    nbf.v4.new_markdown_cell(report),
]
nbf.write(nb, ROOT / "qwen15b_presentation.ipynb")
print(report)
