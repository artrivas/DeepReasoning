#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export DATA_PATH="$PWD/data/sft_reasoning_2k.jsonl"
export ARTIFACT_ROOT="$PWD/artifacts"
export RUN_NAME="${RUN_NAME:-screen_len1024_r8_a16_lr5e5_e1}"
export MAX_LEN="${MAX_LEN:-1024}"
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"

# The RTX PRO 6000 has 96 GiB. A large microbatch and disabled activation
# checkpointing trade memory for substantially higher throughput.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

# Screening uses paired greedy decoding only. Full self-consistency is reserved
# for a candidate that approaches or beats the base model.
export EVAL_PROBLEMS="${EVAL_PROBLEMS:-30}"
export N_MAX=1
export N_LIST=1
export RUN_SELF_CONSISTENCY=0
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

mkdir -p "$ARTIFACT_ROOT/$RUN_NAME"
.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute deepreasoning.ipynb \
  --output executed_notebook.ipynb \
  --output-dir "$ARTIFACT_ROOT/$RUN_NAME" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  2>&1 | tee "$ARTIFACT_ROOT/$RUN_NAME/execution.log"
