#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD/artifacts_qwen15b"
echo "=== Qwen2.5-1.5B DeepReasoning job ==="
if [[ -f artifacts_qwen15b_job.pid ]]; then
  pid=$(cat artifacts_qwen15b_job.pid)
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "status: RUNNING pid=$pid"
  else
    echo "status: NOT RUNNING last_pid=$pid"
  fi
else
  echo "status: no pid file"
fi
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
echo
echo "=== Runs ==="
for d in "$ROOT"/qwen15_*; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  if [[ -f "$d/COMPLETED" ]]; then state=done; elif [[ -f "$d/execution.log" ]]; then state=running_or_failed; else state=created; fi
  echo "[$state] $name"
  if [[ -f "$d/training_summary.json" ]]; then D="$d" .venv/bin/python - <<'STATPY'
import json, os
p=os.path.join(os.environ['D'],'training_summary.json')
s=json.load(open(p))
print('  eval_loss=', s.get('eval_loss'), 'best=', s.get('best_metric'), 'steps=', s.get('global_step'), 'minutes=', round(s.get('train_wall_time_seconds',0)/60,2))
STATPY
  fi
  if [[ -f "$d/evaluation_summary.json" ]]; then D="$d" .venv/bin/python - <<'STATPY'
import json, os
p=os.path.join(os.environ['D'],'evaluation_summary.json')
e=json.load(open(p))
print('  base=', e['base']['greedy_accuracy'], 'adapter=', e['adapter']['greedy_accuracy'], 'delta=', e['paired_greedy_comparison']['adapter_minus_base'], 'format=', e['adapter']['valid_format_rate'])
STATPY
  fi
done
echo
echo "=== Latest log tail ==="
if [[ -f artifacts_qwen15b_job.log ]]; then tail -80 artifacts_qwen15b_job.log; fi
