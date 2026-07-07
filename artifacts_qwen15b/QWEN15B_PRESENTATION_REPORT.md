# Qwen2.5-1.5B DeepReasoning ablation report

## Objective

Repeat the DeepReasoning QLoRA training/evaluation pipeline with `Qwen/Qwen2.5-1.5B-Instruct`, because the 3B model was strong enough that the dataset did not stress it as clearly.

## Runtime decisions

- Used a separate artifact root: `artifacts_qwen15b`, so no 3B base cache leaked into 1.5B comparisons.
- Tried larger batches first to exploit the RTX PRO 6000 VRAM, but preserved OOM records:
  - batch 32 / grad accum 1 OOMed near full VRAM.
  - batch 16 / grad accum 2 OOMed during fp32 loss over full vocabulary logits.
  - batch 8 / grad accum 4 was stable, using about 56 GiB allocated during training.
- Used 3 epochs with early stopping enabled; all three screened runs improved through the full schedule.
- Used fast 30-question greedy screens for the grid, then 100-question no-retraining confirmation only for the leader.

## 30-question screen

| run                                |   rank |   alpha |     lr |   epochs |   train_batch |   grad_accum |   trainable_params_m |   truncation_rate |   eval_loss |   best_eval_loss |   train_minutes |   max_gpu_gib |   strict_accuracy_30 |   loose_accuracy_30 |   format_rate_30 |   reflection_rate_30 |   mean_latency_s_30 |
|:-----------------------------------|-------:|--------:|-------:|---------:|--------------:|-------------:|---------------------:|------------------:|------------:|-----------------:|----------------:|--------------:|---------------------:|--------------------:|-----------------:|---------------------:|--------------------:|
| qwen15_len1024_r8_a16_lr5e5_e3_es  |      8 |      16 | 0.0001 |   3.0000 |             8 |            4 |               9.2324 |            0.1500 |      0.4018 |           0.4018 |          9.2226 |       56.2719 |               0.6333 |              0.6333 |           0.9000 |               0.9000 |             15.3797 |
| qwen15_len1024_r8_a16_lr2e5_e3_es  |      8 |      16 | 0.0000 |   3.0000 |             8 |            4 |               9.2324 |            0.1500 |      0.4558 |           0.4558 |          9.2346 |       56.2719 |               0.5667 |              0.6333 |           0.8000 |               0.8000 |             17.3790 |
| qwen15_len1024_r16_a32_lr5e5_e3_es |     16 |      32 | 0.0001 |   3.0000 |             8 |            4 |              18.4648 |            0.1500 |      0.3832 |           0.3832 |          9.2382 |       56.3761 |               0.5333 |              0.5667 |           0.8000 |               0.8333 |             16.4729 |

Headline: the best screened run was `qwen15_len1024_r8_a16_lr5e5_e3_es` with 63.3% strict tagged accuracy and 90.0% valid format.

## 100-question confirmation for leader

| Metric | Base | Adapter | Adapter - base | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Strict tagged answer | 0.0% | 48.0% | 48.0% | [38.0%, 58.0%] |
| Loose numeric answer | 55.0% | 49.0% | -6.0% | [-19.0%, 6.0%] |
| Valid required format | 0.0% | 75.0% | — | — |
| Reflection present | 0.0% | 75.0% | — | — |

## Interpretation

The LoRA adapter clearly learned the requested `<thinking>/<reflection>/<answer>` protocol: base format compliance was 0%, while the adapter reached 75.0% on the 100-question confirmation. Under the strict report metric, this is a large win because unformatted answers are invalid.

However, under loose numeric answer extraction, the base model scored 55.0% and the adapter scored 49.0%. That means the adapter improved instruction/format following but did not improve raw math-answer accuracy; the likely tradeoff is that the supervised traces impose a verbose format and sometimes hurt direct answer reliability.

The best defensible configuration is therefore `qwen15_len1024_r8_a16_lr5e5_e3_es` if the assignment values structured reasoning traces. If the only objective is final GSM8K numeric accuracy, the base 1.5B model remains competitive or slightly better.

## Reproduction

- Screen grid: `./run_qwen15b_screen.sh`
- 100-question confirmation: `MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct ARTIFACT_ROOT=$PWD/artifacts_qwen15b ADAPTER_DIR=$PWD/artifacts_qwen15b/qwen15_len1024_r8_a16_lr5e5_e3_es/adapter .venv/bin/python evaluate_qwen15_confirm.py`
- Status helper: `./job_status_qwen15b.sh`
