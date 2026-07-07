# Qwen2.5-1.5B DeepReasoning ablation report

## One-slide conclusion

We repeated the QLoRA DeepReasoning pipeline with `Qwen/Qwen2.5-1.5B-Instruct` because the 3B model was less stressed by the dataset. The best adapter was `qwen15_len1024_r8_a16_lr5e5_e3_es`.

| Metric | Base | Adapter | Interpretation |
|---|---:|---:|---|
| Strict tagged answer, 100 questions | 0.0% | 48.0% | Adapter wins because the required format matters. |
| Loose numeric answer, 100 questions | 55.0% | 49.0% | Base remains slightly better at raw math answers. |
| Valid required format | 0.0% | 75.0% | LoRA mainly teaches format/instruction following. |

## Why this pipeline

- Same fixed training split and GSM8K evaluation indices across runs.
- Separate `artifacts_qwen15b` root so no cached 3B base generations contaminate 1.5B results.
- 30-question greedy screens for quick hyperparameter comparison.
- 100-question no-retraining confirmation only for the best screened adapter.
- Strict and loose metrics are both reported because format compliance and final-answer accuracy answer different questions.

## Runtime decisions

- Batch 32 / grad accum 1 OOMed near full VRAM.
- Batch 16 / grad accum 2 also OOMed during the fp32 full-vocabulary loss.
- Batch 8 / grad accum 4 was stable and kept the same effective batch size of 32.
- All stable runs used 3 epochs with early stopping enabled. In practice, validation loss kept improving, so all three ran the full 3 epochs.

## 30-question screen

| run                                |   rank |   alpha |     lr |   epochs |   train_batch |   grad_accum |   effective_batch |   trainable_params_m |   truncation_rate |   eval_loss |   best_eval_loss |   train_minutes |   max_gpu_gib |   strict_accuracy_30 |   loose_accuracy_30 |   format_rate_30 |   reflection_rate_30 |   mean_latency_s_30 |
|:-----------------------------------|-------:|--------:|-------:|---------:|--------------:|-------------:|------------------:|---------------------:|------------------:|------------:|-----------------:|----------------:|--------------:|---------------------:|--------------------:|-----------------:|---------------------:|--------------------:|
| qwen15_len1024_r8_a16_lr5e5_e3_es  |      8 |      16 | 0.0001 |   3.0000 |             8 |            4 |                32 |               9.2324 |            0.1500 |      0.4018 |           0.4018 |          9.2226 |       56.2719 |               0.6333 |              0.6333 |           0.9000 |               0.9000 |             15.3797 |
| qwen15_len1024_r8_a16_lr2e5_e3_es  |      8 |      16 | 0.0000 |   3.0000 |             8 |            4 |                32 |               9.2324 |            0.1500 |      0.4558 |           0.4558 |          9.2346 |       56.2719 |               0.5667 |              0.6333 |           0.8000 |               0.8000 |             17.3790 |
| qwen15_len1024_r16_a32_lr5e5_e3_es |     16 |      32 | 0.0001 |   3.0000 |             8 |            4 |                32 |              18.4648 |            0.1500 |      0.3832 |           0.3832 |          9.2382 |       56.3761 |               0.5333 |              0.5667 |           0.8000 |               0.8333 |             16.4729 |

The screen selected `qwen15_len1024_r8_a16_lr5e5_e3_es` because it had the best strict accuracy (63.3%) and best format rate (90.0%). The r=16 run had lower validation loss but worse answer accuracy, which is why we did not rely on validation loss alone.

## Final interpretation

The best defensible configuration is `qwen15_len1024_r8_a16_lr5e5_e3_es` if the assignment rewards structured reasoning traces. If the only objective is raw GSM8K numeric accuracy, the base 1.5B model is still slightly better. The experiment therefore demonstrates a format-following gain with a raw-answer tradeoff.
