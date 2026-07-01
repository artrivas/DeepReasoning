"""Aggregate completed notebook runs into report-ready tables and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(__import__("os").environ.get("ARTIFACT_ROOT", ROOT / "artifacts"))
rows = []

for run_dir in sorted(ARTIFACT_ROOT.iterdir()):
    required = [
        run_dir / "COMPLETED",
        run_dir / "config.json",
        run_dir / "training_summary.json",
        run_dir / "evaluation_summary.json",
        run_dir / "tokenization_stats.json",
        run_dir / "parameter_stats.json",
    ]
    if not all(path.exists() for path in required):
        continue

    config = json.loads((run_dir / "config.json").read_text())
    training = json.loads((run_dir / "training_summary.json").read_text())
    evaluation = json.loads((run_dir / "evaluation_summary.json").read_text())
    tokens = json.loads((run_dir / "tokenization_stats.json").read_text())
    parameters = json.loads((run_dir / "parameter_stats.json").read_text())
    rows.append(
        {
            "run": config["run_name"],
            "max_len": config["max_len"],
            "lora_r": config["lora_r"],
            "lora_alpha": config["lora_alpha"],
            "trainable_parameters": parameters["trainable_parameters"],
            "train_truncation_rate": tokens["train"]["truncated_fraction"],
            "eval_loss": training["eval_loss"],
            "best_eval_loss": training["best_metric"],
            "train_minutes": training["train_wall_time_seconds"] / 60,
            "max_gpu_gib": training["max_gpu_memory_allocated_gib"],
            "base_greedy_accuracy": evaluation["base"]["greedy_accuracy"],
            "adapter_greedy_accuracy": evaluation["adapter"]["greedy_accuracy"],
            "adapter_minus_base": evaluation["paired_greedy_comparison"]["adapter_minus_base"],
            "adapter_format_rate": evaluation["adapter"]["valid_format_rate"],
            "adapter_reflection_rate": evaluation["adapter"]["reflection_rate"],
            "base_sc_n10": evaluation["base"]["self_consistency_accuracy"]["10"],
            "adapter_sc_n10": evaluation["adapter"]["self_consistency_accuracy"]["10"],
        }
    )

if not rows:
    raise SystemExit("No completed runs found.")

df = pd.DataFrame(rows).sort_values(["max_len", "lora_r"])
df.to_csv(ARTIFACT_ROOT / "ablation_summary.csv", index=False)

display = df.copy()
for column in [
    "train_truncation_rate",
    "base_greedy_accuracy",
    "adapter_greedy_accuracy",
    "adapter_minus_base",
    "adapter_format_rate",
    "adapter_reflection_rate",
    "base_sc_n10",
    "adapter_sc_n10",
]:
    display[column] = (100 * display[column]).map(lambda value: f"{value:.1f}%")

report = f"""# DeepReasoning ablation report

## Experiment matrix and headline results

{display.to_markdown(index=False)}

## Interpretation guardrails

- All runs use the same seeded 85/15 training split and the same held-out GSM8K indices.
- Base-model generations are cached and reused, preventing decoding noise across comparisons.
- `adapter_minus_base` is a paired difference on the exact same questions.
- Sequence-length conclusions should be read together with `train_truncation_rate`.
- Full traces remain in each run's `raw/` directory for qualitative error analysis.

## Reproduction

Run `./run_ablation.sh`. Completed configurations are skipped safely.
"""
(ARTIFACT_ROOT / "ABLATION_REPORT.md").write_text(report, encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
labels = df["run"]
axes[0].bar(labels, 100 * df["adapter_greedy_accuracy"])
axes[0].axhline(
    100 * df["base_greedy_accuracy"].iloc[0],
    color="black",
    linestyle="--",
    label="base",
)
axes[0].set_ylabel("Greedy exact-match accuracy (%)")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=25)

axes[1].bar(labels, df["eval_loss"])
axes[1].set_ylabel("Validation loss")
axes[1].tick_params(axis="x", rotation=25)
fig.suptitle("DeepReasoning QLoRA ablation")
fig.tight_layout()
fig.savefig(ARTIFACT_ROOT / "ablation_comparison.png", dpi=180)
fig.savefig(ARTIFACT_ROOT / "ablation_comparison.pdf")
print(report)
