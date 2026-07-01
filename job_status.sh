#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"

echo "=== DeepReasoning ablation job ==="
if [[ -f artifacts/ablation_job.pid ]]; then
  pid="$(cat artifacts/ablation_job.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    echo "status: RUNNING (PID $pid)"
  else
    echo "status: NOT RUNNING (last PID $pid)"
  fi
else
  echo "status: PID file not found"
fi

echo
echo "=== Completed runs ==="
for run in len512_r16_a32 len1024_r8_a16 len1024_r16_a32; do
  if [[ -f "artifacts/$run/COMPLETED" ]]; then
    echo "[done] $run"
  elif [[ -d "artifacts/$run" ]]; then
    echo "[work] $run"
  else
    echo "[wait] $run"
  fi
done

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader

echo
echo "=== Latest log ==="
tail -n 30 artifacts/ablation_job.log 2>/dev/null || echo "No log yet."
