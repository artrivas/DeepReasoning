# Run report: len1024_r8_a16

## Configuration

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Maximum sequence length: **1024**
- LoRA rank / alpha: **8 / 16**
- Training examples: **1700**
- Validation examples: **300**
- Held-out GSM8K problems: **100**
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`

## Training

- Final validation loss: **0.3241**
- Best validation loss: **0.3241**
- Wall time: **12.3 minutes**
- Maximum allocated GPU memory: **11.35 GiB**
- Training truncation rate: **15.0%**

## Held-out GSM8K

| Metric | Base | Adapter |
|---|---:|---:|
| Greedy exact match | 72.0% | 29.0% |
| Valid three-tag format | 65.0% | 31.0% |
| Reflection present | 91.0% | 32.0% |
| Self-consistency N=10 | 91.0% | 50.0% |

Adapter minus base greedy accuracy:
**-43.0 percentage points**
(paired bootstrap 95% CI:
-54.0 to
-32.0).

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
