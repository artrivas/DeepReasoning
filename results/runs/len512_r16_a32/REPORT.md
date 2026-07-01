# Run report: len512_r16_a32

## Configuration

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Maximum sequence length: **512**
- LoRA rank / alpha: **16 / 32**
- Training examples: **1700**
- Validation examples: **300**
- Held-out GSM8K problems: **100**
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`

## Training

- Final validation loss: **0.3013**
- Best validation loss: **0.3013**
- Wall time: **6.4 minutes**
- Maximum allocated GPU memory: **7.40 GiB**
- Training truncation rate: **88.1%**

## Held-out GSM8K

| Metric | Base | Adapter |
|---|---:|---:|
| Greedy exact match | 72.0% | 30.0% |
| Valid three-tag format | 65.0% | 32.0% |
| Reflection present | 91.0% | 36.0% |
| Self-consistency N=10 | 91.0% | 53.0% |

Adapter minus base greedy accuracy:
**-42.0 percentage points**
(paired bootstrap 95% CI:
-52.0 to
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
