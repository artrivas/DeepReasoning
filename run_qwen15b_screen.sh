#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/sft_reasoning_2k.jsonl ]]; then
  echo "Missing data/sft_reasoning_2k.jsonl" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi

uv pip install --python .venv/bin/python   torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python   "transformers==4.57.3" "peft==0.17.1" "bitsandbytes==0.47.0"   "accelerate==1.10.1" "datasets==4.0.0" "pandas==2.3.2"   "matplotlib==3.10.6" "jupyter==1.1.1" "sentencepiece==0.2.1"   "tabulate==0.9.0"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
export DATA_PATH="$PWD/data/sft_reasoning_2k.jsonl"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PWD/artifacts_qwen15b}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export EVAL_STEPS="${EVAL_STEPS:-20}"
export SAVE_STEPS="${SAVE_STEPS:-20}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
export EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0005}"
export EVAL_PROBLEMS="${EVAL_PROBLEMS:-30}"
export N_MAX=1
export N_LIST=1
export RUN_SELF_CONSISTENCY=0
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"

mkdir -p "$ARTIFACT_ROOT"

run_one() {
  local name="$1" max_len="$2" r="$3" alpha="$4" lr="$5"
  if [[ -f "$ARTIFACT_ROOT/$name/COMPLETED" ]]; then
    echo "[skip] $name already completed"
    return 0
  fi
  mkdir -p "$ARTIFACT_ROOT/$name"
  echo "[run] $name max_len=$max_len r=$r alpha=$alpha lr=$lr"
  RUN_NAME="$name" MAX_LEN="$max_len" LORA_R="$r" LORA_ALPHA="$alpha" LEARNING_RATE="$lr"     .venv/bin/jupyter nbconvert       --to notebook       --execute deepreasoning.ipynb       --output executed_notebook.ipynb       --output-dir "$ARTIFACT_ROOT/$name"       --ExecutePreprocessor.timeout=-1       --ExecutePreprocessor.kernel_name=python3       2>&1 | tee "$ARTIFACT_ROOT/$name/execution.log"
}

run_one qwen15_len1024_r8_a16_lr5e5_e3_es 1024 8 16 5e-5
run_one qwen15_len1024_r8_a16_lr2e5_e3_es 1024 8 16 2e-5
run_one qwen15_len1024_r16_a32_lr5e5_e3_es 1024 16 32 5e-5

ARTIFACT_ROOT="$ARTIFACT_ROOT" .venv/bin/python summarize_ablation.py 2>&1 | tee "$ARTIFACT_ROOT/summarize_qwen15b.log"
