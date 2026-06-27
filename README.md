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
