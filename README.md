# DeepReasoning

Portable final notebook for the Deep-Reasoning assignment. The project trains a small 3B language model to imitate structured reasoning traces and evaluates test-time compute on GSM8K.

## What This Repository Contains

- `deepreasoning.ipynb`: portable Lightning AI / Colab / local Jupyter notebook.
- `sft_reasoning.jsonl`: earlier 500-example distilled dataset.
- `sft_reasoning_2k.jsonl`: raw 2k Gemini-distilled GSM8K training traces.
- `sft_reasoning_2k_clean.jsonl`: strict cleaned dataset generated from the 2k file.
- `tools/prepare_final_notebook.py`: maintenance script that rebuilds the final notebook and structurally cleans the 2k JSONL.

The final pipeline uses GSM8K only. MBPP is intentionally excluded from the final run to keep the domain, metric, and leakage story clean.

## Runtime Targets

The notebook is designed to run in:

- Lightning AI Notebook
- Google Colab
- Local Jupyter with a CUDA GPU

It does not require Kaggle, Google Drive, or Kaggle Secrets.

## Required Secret

Distillation and LLM-as-a-Judge require a Gemini API key.

Lightning AI or local Jupyter:

```python
import os
os.environ["GEMINI_API_KEY"] = "your-key-here"
```

Google Colab:

1. Open Colab Secrets.
2. Add `GEMINI_API_KEY`.
3. Enable notebook access to the secret.

The notebook does not hard-fail when the key is missing. It only raises when you run teacher generation or judge cells.

## How To Run

1. Open `deepreasoning.ipynb`.
2. Run setup and path detection cells.
3. Run GSM8K loading and dataset cleaning.
4. Confirm `sft_reasoning_2k_clean.jsonl` is used for training.
5. Run QLoRA fine-tuning.
6. Run the 50-case evaluation.
7. Run the blind pairwise judge if `GEMINI_API_KEY` is available.

Expected GPU: T4, L4, A10, A100, or similar CUDA GPU. The model is loaded in 4-bit NF4 and only LoRA adapters are trained.

## Expected Outputs

The notebook writes results under `outputs/results/`:

- `dataset_audit.csv`
- `dataset_drop_reasons.csv`
- `trainer_log_history.json`
- `training_loss_curve.png`
- `base_metrics.csv`
- `base_records.jsonl`
- `fine_tuned_metrics.csv`
- `fine_tuned_records.jsonl`
- `final_evaluation_metrics.csv`
- `accuracy_vs_n.png`
- `exact_win_rate.csv`
- `judge_pairwise_results.jsonl`
- `judge_failures.json`
- `judge_win_rate.csv`
- `judge_cot_rigor.csv`
- `judge_self_correction_distribution.csv`

Adapters are saved to `outputs/qwen-reasoning-lora-final/`.

## Final Evaluation Design

The notebook evaluates 50 fixed-seed examples from the GSM8K test split. Distillation and SFT use GSM8K train indices only, so the evaluation is held out.

Required comparisons:

- Base model, N=1
- Fine-tuned model, N=1
- Fine-tuned model, N=3
- Fine-tuned model, N=5
- Fine-tuned model, N=7

Optional if runtime permits:

- Base model, N=5
- Base model, N=7

Reported metrics:

- Exact numeric accuracy
- Valid answer rate
- Tie rate
- Average latency
- Average generated tokens
- Accuracy vs N
- Exact win rate: fine-tuned vs base
- Blind pairwise LLM-as-a-Judge win rate
- Average CoT rigor
- Self-correction category distribution

## Known Runtime And Cost Risks

Training Qwen2.5-3B with QLoRA on about 2k traces is feasible on common notebook GPUs, but runtime depends heavily on GPU type and `MAX_SEQ_LENGTH`.

Self-consistency multiplies inference cost by N. The required fine-tuned sweep uses N=1,3,5,7 over 50 cases. Judge evaluation makes one Gemini call per case plus occasional repair calls.

## Limitations

- The cleaned dataset is strict: traces with missing opening tags or extra text outside the required tags are dropped rather than silently repaired.
- Reflections often perform checking rather than real error correction. The judge schema separates `checks_only` from `detects_and_fixes_error`.
- Exact numeric accuracy is the primary objective metric; judge scores are secondary and should be reported with malformed-output failure counts.

## Lightning AI Run Plan And Time Estimates

Recommended machine: one CUDA GPU with at least 16 GB VRAM if available. A T4 can work with the default conservative settings; L4/A10/A100 will be faster.

### Gemini Free-Tier Configuration

The notebook defaults to:

```python
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_SLEEP_SECONDS = 8
```

You can override these before running judge/distillation cells:

```python
import os
os.environ["GEMINI_MODEL"] = "gemini-3.1-flash-lite"
os.environ["GEMINI_SLEEP_SECONDS"] = "8"
os.environ["GEMINI_API_KEY"] = "your-free-tier-key"
```

`GEMINI_SLEEP_SECONDS=8` is intentionally conservative for free-tier keys. It spaces calls at roughly 7.5 requests/minute. If AI Studio shows a lower active limit, increase it. If it shows a higher active limit, you can lower it. The final notebook does not need Gemini for training if you use the included distilled dataset; Gemini is needed only for optional new distillation and the final LLM-as-a-Judge.

### GPU vs CPU/API Steps

| Step | Needs GPU? | Uses Gemini API? | Typical time |
|---|---:|---:|---:|
| Dependency install | No | No | 2-8 min |
| GSM8K download/load | No | No | 1-3 min |
| Clean/validate 2k dataset | No, but downloads tokenizer/model metadata | No | 2-8 min |
| Optional new teacher distillation | No | Yes | depends on target rows and free-tier pacing |
| Load Qwen 3B 4-bit | Yes | No | 3-10 min first run |
| QLoRA fine-tuning on ~1979 rows | Yes | No | ~1-3 h on L4/A10, ~2-5 h on T4 |
| Save adapters/loss curve | Yes/No mixed | No | <5 min |
| Base N=1 evaluation, 50 cases | Yes | No | ~15-45 min |
| Fine-tuned N=1,3,5,7 evaluation | Yes | No | ~2-6 h depending GPU/output length |
| Accuracy/win-rate plots | No | No | <2 min |
| Blind pairwise judge, 50 cases | No | Yes | ~7-12 min at 8s/call plus retries |

### Approximate End-to-End Runtime

Conservative full run on a T4-style GPU:

- Setup + cleaning: 10-20 min
- QLoRA training: 2-5 h
- Required evaluation: 2.5-6.5 h
- Judge: 7-15 min
- Total: about 5-12 h

Better GPU such as L4/A10/A100:

- Setup + cleaning: 10-20 min
- QLoRA training: 1-3 h
- Required evaluation: 1.5-4 h
- Judge: 7-15 min
- Total: about 3-7 h

The largest variable is inference evaluation because self-consistency multiplies generations. Required evaluation uses 850 total sampled paths: 50 base paths plus 50 * (1 + 3 + 5 + 7) fine-tuned paths.

### Lightning AI Checklist

1. Upload or clone this repository into your Lightning AI studio.
2. Open `deepreasoning.ipynb`.
3. Select a GPU runtime.
4. Set the Gemini key only if you will run judge or optional distillation:

```python
import os
os.environ["GEMINI_API_KEY"] = "your-free-tier-key"
os.environ["GEMINI_MODEL"] = "gemini-3.1-flash-lite"
os.environ["GEMINI_SLEEP_SECONDS"] = "8"
```

5. Run setup and imports.
6. Confirm path detection prints your Lightning project path and `CUDA available: True`.
7. Run GSM8K loading.
8. Run dataset cleaning and confirm final kept rows are around 1979 and above the 500-row guard.
9. Run QLoRA training.
10. Confirm adapters save to `outputs/qwen-reasoning-lora-final/` and loss curve saves to `outputs/results/training_loss_curve.png`.
11. Run base N=1 evaluation.
12. Run fine-tuned N=1,3,5,7 evaluation.
13. Generate `accuracy_vs_n.png` and exact win-rate table.
14. If API key is set, run the blind pairwise judge.
15. Use the final interpretation section and README outputs list for the report.

### Cost Control Notes

- Training and model evaluation use GPU only; they do not call Gemini.
- The included 1979-row clean dataset avoids regenerating teacher traces, so no distillation API cost is needed for the final run.
- The judge is the main Gemini consumer: about 50 calls plus occasional repair calls.
- Keep `RUN_BASE_SC = False` unless you have extra GPU time; base self-consistency is optional and can add 600 more sampled paths for N=5 and N=7.
