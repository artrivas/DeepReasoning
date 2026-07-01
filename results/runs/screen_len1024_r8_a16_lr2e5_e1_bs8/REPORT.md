# Run report: screen_len1024_r8_a16_lr2e5_e1_bs8

## Configuration

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Maximum sequence length: **1024**
- LoRA rank / alpha: **8 / 16**
- Training examples: **1700**
- Validation examples: **300**
- Held-out GSM8K problems: **30**
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`

## Training

- Final validation loss: **0.5376**
- Best validation loss: **0.5376**
- Wall time: **4.9 minutes**
- Maximum allocated GPU memory: **82.21 GiB**
- Training truncation rate: **15.0%**

## Held-out GSM8K

| Metric | Base | Adapter |
|---|---:|---:|
| Greedy exact match | 70.0% | 76.7% |
| Valid three-tag format | 76.7% | 83.3% |
| Reflection present | 93.3% | 93.3% |
| Self-consistency N=1 | 76.7% | 76.7% |

Adapter minus base greedy accuracy:
**+6.7 percentage points**
(paired bootstrap 95% CI:
-13.3 to
+26.7).

## Artifact inventory

- `config.json`: complete hyperparameter snapshot
- `environment.json`: software, git, CUDA, and GPU environment
- `tokenization_stats.json`, `tables/token_lengths.csv`: truncation audit
- `raw/trainer_log_history.json`: every Trainer log event
- `raw/gpu_telemetry.jsonl`: per-log GPU observations
- `raw/*_predictions.jsonl`: full generated traces and voting details
- `evaluation_summary.json`: scalar and paired evaluation metrics
- `adapter/`: final LoRA adapter and tokenizer
- `checkpoints/`: two most recent/best recoverable checkpoints
- `plots/`: PNG and PDF figures
