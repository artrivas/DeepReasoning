#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/sft_reasoning_2k.jsonl ]]; then
  mkdir -p data
  git show HEAD^:sft_reasoning_2k.jsonl > data/sft_reasoning_2k.jsonl
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi

# CUDA 12.8 is required for reliable RTX PRO 6000 Blackwell support.
uv pip install --python .venv/bin/python \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python \
  "transformers==4.57.3" "peft==0.17.1" "bitsandbytes==0.47.0" \
  "accelerate==1.10.1" "datasets==4.0.0" "pandas==2.3.2" \
  "matplotlib==3.10.6" "jupyter==1.1.1" "sentencepiece==0.2.1"
uv pip install --python .venv/bin/python "tabulate==0.9.0"

export DATA_PATH="${DATA_PATH:-$PWD/data/sft_reasoning_2k.jsonl}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PWD/artifacts}"
export EVAL_PROBLEMS="${EVAL_PROBLEMS:-100}"
export N_MAX="${N_MAX:-10}"
export N_LIST="${N_LIST:-1,3,5,7,10}"
export RUN_SELF_CONSISTENCY="${RUN_SELF_CONSISTENCY:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

run_one() {
  export RUN_NAME="$1"
  export MAX_LEN="$2"
  export LORA_R="$3"
  export LORA_ALPHA="$4"

  if [[ -f "$ARTIFACT_ROOT/$RUN_NAME/COMPLETED" ]]; then
    echo "Skipping completed run: $RUN_NAME"
    return
  fi

  mkdir -p "$ARTIFACT_ROOT/$RUN_NAME"
  .venv/bin/jupyter nbconvert \
    --to notebook \
    --execute deepreasoning.ipynb \
    --output "executed_notebook.ipynb" \
    --output-dir "$ARTIFACT_ROOT/$RUN_NAME" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    2>&1 | tee "$ARTIFACT_ROOT/$RUN_NAME/execution.log"
}

run_one len512_r16_a32 512 16 32
run_one len1024_r8_a16 1024 8 16
run_one len1024_r16_a32 1024 16 32

.venv/bin/python summarize_ablation.py
