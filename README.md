# DeepReasoning

QLoRA ablation study for `Qwen/Qwen2.5-3B-Instruct` on filtered GSM8K
reasoning traces.

The main artifact is [`deepreasoning.ipynb`](deepreasoning.ipynb). Its defaults
reproduce the winning configuration (`MAX_LEN=1024`, LoRA `r=8`, alpha `16`,
learning rate `5e-5`, one epoch). It saves a complete, auditable result bundle.

On an RTX PRO 6000 with 96 GiB VRAM, reproduce it with:

```bash
chmod +x run_best.sh
./run_best.sh
```

Run the original requested three-way study with:

```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

The runner uses `uv`, resumes safely by skipping completed runs, and evaluates:

| Run | `MAX_LEN` | LoRA `r` | LoRA `alpha` |
|---|---:|---:|---:|
| `len512_r16_a32` | 512 | 16 | 32 |
| `len1024_r8_a16` | 1024 | 8 | 16 |
| `len1024_r16_a32` | 1024 | 16 | 32 |

Outputs are written under `artifacts/`. Each run includes raw predictions,
Trainer history, GPU telemetry, checkpoints, the final adapter, CSV tables,
PNG/PDF plots, environment metadata, and a Markdown report. After all runs,
`summarize_ablation.py` creates the cross-run ablation table and figures.

The final report is in [`FINAL_ABLATION_REPORT.md`](FINAL_ABLATION_REPORT.md).
Curated raw evidence and the winning adapter are committed under `results/`.

For a detached remote run:

```bash
mkdir -p artifacts
nohup ./run_ablation.sh > artifacts/ablation_job.log 2>&1 &
echo $! > artifacts/ablation_job.pid
./job_status.sh
```
