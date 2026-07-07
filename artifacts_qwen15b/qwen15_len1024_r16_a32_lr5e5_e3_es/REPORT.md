# Run report: qwen15_len1024_r16_a32_lr5e5_e3_es

## Configuration

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Maximum sequence length: **1024**
- LoRA rank / alpha: **16 / 32**
- Training examples: **1700**
- Validation examples: **300**
- Held-out GSM8K problems: **30**
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`

## Training

- Final validation loss: **0.3832**
- Best validation loss: **0.3832**
- Wall time: **9.2 minutes**
- Early stopping patience: **3**
- Maximum allocated GPU memory: **56.38 GiB**
- Training truncation rate: **15.0%**

## Held-out GSM8K

| Metric | Base | Adapter |
|---|---:|---:|
| Greedy exact match | 0.0% | 53.3% |
| Valid three-tag format | 0.0% | 80.0% |
| Reflection present | 0.0% | 83.3% |
| Self-consistency N=1 | 0.0% | 53.3% |

Adapter minus base greedy accuracy:
**+53.3 percentage points**
(paired bootstrap 95% CI:
+36.7 to
+70.0).

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
