# DeepReasoning QLoRA ablation results

## Main result

The best configuration is:

- Maximum sequence length: **1024**
- LoRA rank / alpha: **8 / 16**
- Learning rate: **5e-5**
- Epochs: **1**
- Microbatch / gradient accumulation: **8 / 4** (effective batch 32)
- Gradient checkpointing: **disabled**
- Precision: **4-bit NF4 base weights with bfloat16 computation**

On 100 fixed, paired GSM8K test questions, this adapter achieved **81% greedy
exact-match accuracy**, compared with **72% for the base model**. The paired
difference was **+9 percentage points**, with a 10,000-sample bootstrap 95%
confidence interval from **-1 to +19 points**. The adapter won 19 questions,
the base won 10, and they tied on 71.

The adapter also produced the required three-tag format on **98%** of examples.

## Ablation summary

| Configuration | Evaluation | Adapter accuracy | Base accuracy | Difference | Valid format |
|---|---:|---:|---:|---:|---:|
| 512, r16/a32, lr 2e-4, 2 epochs | 100 | 30% | 72% | -42 pp | 32% |
| 1024, r8/a16, lr 2e-4, 2 epochs | 100 | 29% | 72% | -43 pp | 31% |
| 1024, r16/a32, lr 2e-4, 2 epochs | 100 | 26% | 72% | -46 pp | 27% |
| 1024, r8/a16, lr 5e-5, 1 epoch | 30-screen | 76.7% | 70% | +6.7 pp | 96.7% |
| 1024, r8/a16, lr 5e-5, 1 epoch | 100-confirmation | **81%** | **72%** | **+9 pp** | **98%** |
| 1024, r8/a16, lr 2e-5, 1 epoch | 30-screen | 76.7% | 70% | +6.7 pp | 83.3% |

The original high-learning-rate runs degraded performance regardless of maximum
length or LoRA rank. Lowering the learning rate and training for one epoch was
the decisive change.

## Sequence length and generation budget

At `MAX_LEN=512`, **88.1%** of training examples were truncated. At
`MAX_LEN=1024`, truncation fell to **15.0%**. The fine-tuned models also learned
longer reasoning traces than the base model, so a 512-token generation cap often
ended before the final `<answer>` tag. The confirmation therefore used up to
1024 generated tokens.

## GPU utilization and faster workflow

The initial training runs used only 7–11 GiB of the 96 GiB GPU. The optimized
configuration used:

- microbatch 8;
- no activation checkpointing;
- approximately **82.2 GiB** peak allocated VRAM;
- approximately **4.9 minutes** for one epoch.

Microbatch 16 was tested and preserved as an OOM observation: it used 94.6 GiB
and failed while requesting another 688 MiB. Batch 8 is therefore the practical
maximum for this model, sequence length, and software stack.

The evaluation workflow was also changed. Weak candidates receive a
30-question greedy screen. Only promising adapters are promoted to 100 paired
questions. Expensive 10-trace self-consistency evaluation is not run for weak
candidates.

## Interpretation

The +9-point result is encouraging but its confidence interval includes a small
negative effect. It should be reported as evidence that the tuned model is
competitive and likely better on this sample, not as definitive proof of a
population-level improvement. Repeating the 100-question comparison with
additional fixed subsets or seeds would tighten the uncertainty.

All configurations, split indices, raw predictions, generated traces, logs,
telemetry, plots, adapters, and environment metadata were saved for audit and
future analysis.
