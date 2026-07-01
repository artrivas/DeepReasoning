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

# CUDA 12.8 supports the RTX PRO 6000 Blackwell GPU used in the study.
uv pip install --python .venv/bin/python \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python \
  "transformers==4.57.3" "peft==0.17.1" "bitsandbytes==0.47.0" \
  "accelerate==1.10.1" "datasets==4.0.0" "pandas==2.3.2" \
  "matplotlib==3.10.6" "jupyter==1.1.1" "sentencepiece==0.2.1" \
  "tabulate==0.9.0"

export DATA_PATH="$PWD/data/sft_reasoning_2k.jsonl"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PWD/artifacts}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ARTIFACT_ROOT/best_len1024_r8_a16_lr5e5_e1"
.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute deepreasoning.ipynb \
  --output executed_notebook.ipynb \
  --output-dir "$ARTIFACT_ROOT/best_len1024_r8_a16_lr5e5_e1" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3
