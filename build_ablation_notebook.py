"""Build the presentation-ready DeepReasoning ablation notebook.

The generated notebook is intentionally self-contained: each execution trains one
configuration selected with environment variables and writes a complete artifact
bundle that can later be compared without rerunning inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "deepreasoning.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip()}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip(),
    }


cells = [
    markdown(
        """
        # DeepReasoning: QLoRA ablation study

        This notebook fine-tunes `Qwen/Qwen2.5-3B-Instruct` on filtered GSM8K
        reasoning traces and evaluates both the base model and the LoRA adapter.

        One execution corresponds to one configuration. The companion
        `run_ablation.sh` executes the requested grid:

        | Run | Maximum length | LoRA rank | LoRA alpha |
        |---|---:|---:|---:|
        | `len512_r16_a32` | 512 | 16 | 32 |
        | `len1024_r8_a16` | 1024 | 8 | 16 |
        | `len1024_r16_a32` | 1024 | 16 | 32 |

        Every run saves its configuration, fixed data split, token-length and
        truncation statistics, raw Trainer history, GPU telemetry, checkpoints,
        final adapter, deterministic predictions, self-consistency traces, scalar
        metrics, plots, package versions, and the executed notebook. This is
        deliberately redundant: raw saved observations are more valuable than a
        plot that cannot be reconstructed later.
        """
    ),
    markdown("## 1. Imports, reproducibility, and experiment configuration"),
    code(
        """
        import csv
        import gc
        import hashlib
        import json
        import math
        import os
        import platform
        import random
        import re
        import subprocess
        import sys
        import time
        from collections import Counter
        from contextlib import nullcontext
        from datetime import datetime, timezone
        from pathlib import Path

        import bitsandbytes
        import datasets
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import peft
        import torch
        import transformers
        from datasets import Dataset, load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )

        MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        DATA_PATH = Path(os.getenv("DATA_PATH", "data/sft_reasoning_2k.jsonl")).resolve()
        ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts")).resolve()
        RUN_NAME = os.getenv("RUN_NAME", "best_len1024_r8_a16_lr5e5_e1")

        MAX_LEN = int(os.getenv("MAX_LEN", "1024"))
        LORA_R = int(os.getenv("LORA_R", "8"))
        LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
        SEED = int(os.getenv("SEED", "42"))
        NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "1"))
        LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
        TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
        EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "8"))
        GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
        GRADIENT_CHECKPOINTING = os.getenv("GRADIENT_CHECKPOINTING", "0") == "1"
        EVAL_PROBLEMS = int(os.getenv("EVAL_PROBLEMS", "100"))
        N_MAX = int(os.getenv("N_MAX", "1"))
        N_LIST = [int(x) for x in os.getenv("N_LIST", "1").split(",")]
        MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
        RUN_SELF_CONSISTENCY = os.getenv("RUN_SELF_CONSISTENCY", "0") == "1"

        RUN_DIR = ARTIFACT_ROOT / RUN_NAME
        CHECKPOINT_DIR = RUN_DIR / "checkpoints"
        ADAPTER_DIR = RUN_DIR / "adapter"
        PLOT_DIR = RUN_DIR / "plots"
        TABLE_DIR = RUN_DIR / "tables"
        RAW_DIR = RUN_DIR / "raw"
        for path in (RUN_DIR, CHECKPOINT_DIR, ADAPTER_DIR, PLOT_DIR, TABLE_DIR, RAW_DIR):
            path.mkdir(parents=True, exist_ok=True)

        random.seed(SEED)
        np.random.seed(SEED)
        set_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        assert torch.cuda.is_available(), "A CUDA GPU is required for this notebook."
        GPU_NAME = torch.cuda.get_device_name(0)
        COMPUTE_CAPABILITY = torch.cuda.get_device_capability(0)
        USE_BF16 = bool(torch.cuda.is_bf16_supported())
        COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

        SYSTEM_INSTRUCTION = '''You are a meticulous reasoning tutor.
        For every problem, answer using EXACTLY these tags in order:
        <thinking>
        Reason step by step. Show every intermediate calculation.
        </thinking>
        <reflection>
        Re-check your reasoning. Look for arithmetic slips or wrong assumptions.
        </reflection>
        <answer>
        Give only the final answer.
        </answer>'''

        config = {
            "run_name": RUN_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "data_path": str(DATA_PATH),
            "max_len": MAX_LEN,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "seed": SEED,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "effective_batch_size": TRAIN_BATCH_SIZE * GRAD_ACCUM,
            "eval_problems": EVAL_PROBLEMS,
            "self_consistency_n_max": N_MAX,
            "self_consistency_n_list": N_LIST,
            "max_new_tokens": MAX_NEW_TOKENS,
            "run_self_consistency": RUN_SELF_CONSISTENCY,
            "gpu": GPU_NAME,
            "compute_capability": list(COMPUTE_CAPABILITY),
            "compute_dtype": str(COMPUTE_DTYPE),
        }
        (RUN_DIR / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(config, indent=2, ensure_ascii=False))
        """
    ),
    markdown("## 2. Capture the software and hardware environment"),
    code(
        """
        def command_output(command):
            try:
                return subprocess.run(
                    command, check=False, capture_output=True, text=True
                ).stdout.strip()
            except Exception as exc:
                return f"unavailable: {exc}"

        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "datasets": datasets.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": GPU_NAME,
            "nvidia_smi": command_output(["nvidia-smi"]),
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
        }
        (RUN_DIR / "environment.json").write_text(
            json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Environment captured.")
        """
    ),
    markdown("## 3. Load data and create a fixed, auditable split"),
    code(
        """
        assert DATA_PATH.exists(), f"Training data not found: {DATA_PATH}"
        raw_bytes = DATA_PATH.read_bytes()
        dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        rows = [
            json.loads(line)
            # Split only on the JSONL record delimiter. str.splitlines() also
            # splits valid Unicode separators that may occur inside a JSON string.
            for line in raw_bytes.decode("utf-8").split("\\n")
            if line.strip()
        ]

        # The split is based on stable row IDs, so every ablation sees identical data.
        indices = list(range(len(rows)))
        split_rng = random.Random(SEED)
        split_rng.shuffle(indices)
        n_eval = max(1, round(0.15 * len(indices)))
        eval_indices = sorted(indices[:n_eval])
        train_indices = sorted(indices[n_eval:])
        split_manifest = {
            "seed": SEED,
            "dataset_sha256": dataset_sha256,
            "n_total": len(rows),
            "n_train": len(train_indices),
            "n_eval": len(eval_indices),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
        }
        (RAW_DIR / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2), encoding="utf-8"
        )

        train_rows = [rows[i] for i in train_indices]
        eval_rows = [rows[i] for i in eval_indices]
        print(f"Loaded {len(rows)} rows: {len(train_rows)} train / {len(eval_rows)} validation")
        print("SHA-256:", dataset_sha256)
        """
    ),
    markdown("## 4. Tokenization, assistant-only loss mask, and truncation audit"),
    code(
        """
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        def tokenize_example(example):
            prompt_messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": example["question"]},
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": example["trace"]}
            ]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=True
            )
            full_ids_untruncated = tokenizer.apply_chat_template(
                full_messages, add_generation_prompt=False, tokenize=True
            )
            full_ids = full_ids_untruncated[:MAX_LEN]
            labels = [-100] * min(len(prompt_ids), len(full_ids))
            labels += full_ids[len(labels):]
            return {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
                "original_length": len(full_ids_untruncated),
                "prompt_length": len(prompt_ids),
                "trained_tokens": sum(label != -100 for label in labels),
                "was_truncated": len(full_ids_untruncated) > MAX_LEN,
            }

        train_ds = Dataset.from_list(train_rows).map(tokenize_example)
        eval_ds = Dataset.from_list(eval_rows).map(tokenize_example)
        keep_columns = ["input_ids", "attention_mask", "labels"]
        train_tokenized = train_ds.remove_columns(
            [c for c in train_ds.column_names if c not in keep_columns]
        )
        eval_tokenized = eval_ds.remove_columns(
            [c for c in eval_ds.column_names if c not in keep_columns]
        )

        def length_rows(dataset, split):
            return [
                {
                    "split": split,
                    "row": i,
                    "original_length": ex["original_length"],
                    "prompt_length": ex["prompt_length"],
                    "trained_tokens": ex["trained_tokens"],
                    "was_truncated": bool(ex["was_truncated"]),
                }
                for i, ex in enumerate(dataset)
            ]

        length_df = pd.DataFrame(
            length_rows(train_ds, "train") + length_rows(eval_ds, "validation")
        )
        length_df.to_csv(TABLE_DIR / "token_lengths.csv", index=False)
        token_stats = {}
        for split, part in length_df.groupby("split"):
            token_stats[split] = {
                "count": int(len(part)),
                "truncated_count": int(part["was_truncated"].sum()),
                "truncated_fraction": float(part["was_truncated"].mean()),
                "original_length_mean": float(part["original_length"].mean()),
                "original_length_p50": float(part["original_length"].quantile(0.50)),
                "original_length_p90": float(part["original_length"].quantile(0.90)),
                "original_length_p95": float(part["original_length"].quantile(0.95)),
                "original_length_p99": float(part["original_length"].quantile(0.99)),
                "trained_tokens_mean": float(part["trained_tokens"].mean()),
                "zero_trained_examples": int((part["trained_tokens"] == 0).sum()),
            }
        (RUN_DIR / "tokenization_stats.json").write_text(
            json.dumps(token_stats, indent=2), encoding="utf-8"
        )
        assert token_stats["train"]["zero_trained_examples"] == 0

        sample = train_ds[0]
        ignored = [t for t, label in zip(sample["input_ids"], sample["labels"]) if label == -100]
        trained = [t for t, label in zip(sample["input_ids"], sample["labels"]) if label != -100]
        (RAW_DIR / "loss_mask_example.txt").write_text(
            "=== MASKED PROMPT ===\\n"
            + tokenizer.decode(ignored)
            + "\\n\\n=== TRAINED ASSISTANT TOKENS ===\\n"
            + tokenizer.decode(trained),
            encoding="utf-8",
        )
        print(json.dumps(token_stats, indent=2))
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 4.5))
        for split, part in length_df.groupby("split"):
            plt.hist(part["original_length"], bins=35, alpha=0.55, label=split)
        plt.axvline(MAX_LEN, color="black", linestyle="--", label=f"MAX_LEN={MAX_LEN}")
        plt.xlabel("Tokens before truncation")
        plt.ylabel("Examples")
        plt.title(f"Token-length distribution — {RUN_NAME}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "token_length_distribution.png", dpi=180)
        plt.savefig(PLOT_DIR / "token_length_distribution.pdf")
        plt.show()
        """
    ),
    markdown("## 5. Load Qwen in 4-bit and attach LoRA"),
    code(
        """
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=COMPUTE_DTYPE,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING
        )
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        if GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        parameter_stats = {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
        }
        (RUN_DIR / "parameter_stats.json").write_text(
            json.dumps(parameter_stats, indent=2), encoding="utf-8"
        )
        model.print_trainable_parameters()
        """
    ),
    markdown("## 6. Train while preserving raw logs and GPU telemetry"),
    code(
        """
        class ArtifactCallback(TrainerCallback):
            def _save(self, state):
                (RAW_DIR / "trainer_log_history.json").write_text(
                    json.dumps(state.log_history, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                pd.DataFrame(state.log_history).to_csv(
                    TABLE_DIR / "trainer_log_history.csv", index=False
                )

            def on_log(self, args, state, control, logs=None, **kwargs):
                self._save(state)
                gpu = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "nvidia_smi_csv": command_output([
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ]),
                }
                with (RAW_DIR / "gpu_telemetry.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(gpu) + "\\n")

            def on_train_end(self, args, state, control, **kwargs):
                self._save(state)

        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        training_args = TrainingArguments(
            output_dir=str(CHECKPOINT_DIR),
            run_name=RUN_NAME,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="paged_adamw_8bit",
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=USE_BF16,
            fp16=not USE_BF16,
            tf32=True,
            logging_strategy="steps",
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=25,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            seed=SEED,
            data_seed=SEED,
            remove_unused_columns=True,
            disable_tqdm=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[ArtifactCallback()],
        )

        train_started = time.time()
        train_result = trainer.train()
        train_seconds = time.time() - train_started
        eval_metrics = trainer.evaluate()
        trainer.save_model(str(ADAPTER_DIR))
        tokenizer.save_pretrained(str(ADAPTER_DIR))
        trainer.save_state()

        training_summary = {
            **train_result.metrics,
            **eval_metrics,
            "train_wall_time_seconds": train_seconds,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "global_step": trainer.state.global_step,
            "max_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        (RUN_DIR / "training_summary.json").write_text(
            json.dumps(training_summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(training_summary, indent=2))
        """
    ),
    markdown("## 7. Training curves"),
    code(
        """
        history = pd.DataFrame(trainer.state.log_history)
        train_history = history.dropna(subset=["loss"]) if "loss" in history else pd.DataFrame()
        eval_history = history.dropna(subset=["eval_loss"]) if "eval_loss" in history else pd.DataFrame()

        plt.figure(figsize=(8, 4.5))
        if not train_history.empty:
            plt.plot(train_history["step"], train_history["loss"], alpha=0.4, label="train loss")
            smooth = train_history["loss"].rolling(10, min_periods=1).mean()
            plt.plot(train_history["step"], smooth, linewidth=2, label="train loss (10-step mean)")
        if not eval_history.empty:
            plt.plot(eval_history["step"], eval_history["eval_loss"], "o-", label="validation loss")
        plt.xlabel("Optimizer step")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"Training curves — {RUN_NAME}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "training_curves.png", dpi=180)
        plt.savefig(PLOT_DIR / "training_curves.pdf")
        plt.show()
        """
    ),
    markdown(
        """
        ## 8. Held-out GSM8K evaluation

        The evaluation records every generated trace, extracted answer, exact-match
        result, format compliance, reasoning length, latency, and voting result.
        The same fixed test indices and decoding seed are used in every run.
        """
    ),
    code(
        """
        gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
        eval_count = min(EVAL_PROBLEMS, len(gsm8k_test))
        shared_base = ARTIFACT_ROOT / "shared_base_predictions.jsonl"
        cached_base_records = None
        if shared_base.exists():
            all_cached = [
                json.loads(line)
                for line in shared_base.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(all_cached) >= eval_count:
                cached_base_records = all_cached[:eval_count]
                test_indices = [row["dataset_index"] for row in cached_base_records]
            else:
                raise ValueError("Base cache has fewer rows than this evaluation requests.")
        else:
            test_rng = random.Random(SEED)
            test_indices = sorted(test_rng.sample(range(len(gsm8k_test)), eval_count))
        gsm8k_subset = gsm8k_test.select(test_indices)
        (RAW_DIR / "gsm8k_test_indices.json").write_text(
            json.dumps(test_indices, indent=2), encoding="utf-8"
        )

        def extract_tag(text, tag):
            match = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        def normalize_number(text):
            cleaned = str(text).replace(",", "").replace("$", "").strip()
            numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", cleaned)
            if not numbers:
                return ""
            value = numbers[-1]
            try:
                numeric = float(value)
                return str(int(numeric)) if numeric.is_integer() else str(numeric)
            except ValueError:
                return value

        def trace_features(text):
            thinking = extract_tag(text, "thinking")
            reflection = extract_tag(text, "reflection")
            answer = extract_tag(text, "answer")
            return {
                "predicted_answer": normalize_number(answer),
                "has_thinking": bool(thinking),
                "has_reflection": bool(reflection),
                "has_answer": bool(answer),
                "valid_format": bool(thinking and reflection and answer),
                "thinking_words": len(thinking.split()),
                "reflection_words": len(reflection.split()),
                "trace_words": len(text.split()),
            }

        @torch.inference_mode()
        def generate_traces(question, n, do_sample, seed_offset=0):
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": question},
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            # Transformers 4.57 validates model kwargs strictly and no longer
            # forwards a per-call Generator. Reset both RNGs immediately before
            # generation to keep the sampled traces reproducible.
            torch.manual_seed(SEED + seed_offset)
            torch.cuda.manual_seed_all(SEED + seed_offset)
            kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "num_return_sequences": n,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": do_sample,
            }
            if do_sample:
                kwargs.update(
                    {"temperature": 0.8, "top_p": 0.95}
                )
            started = time.time()
            outputs = model.generate(**encoded, **kwargs)
            elapsed = time.time() - started
            prompt_length = encoded["input_ids"].shape[1]
            traces = tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            return traces, elapsed

        def majority_vote(answers):
            nonempty = [answer for answer in answers if answer]
            return Counter(nonempty).most_common(1)[0][0] if nonempty else ""

        def evaluate_mode(mode, adapter_enabled):
            output_path = RAW_DIR / f"{mode}_predictions.jsonl"
            output_path.unlink(missing_ok=True)
            records = []
            # disable_adapter() is a PEFT context manager; the enabled path needs
            # no state change because the adapter is active by default.
            context = nullcontext() if adapter_enabled else model.disable_adapter()
            with context:
                model.eval()
                for local_i, example in enumerate(gsm8k_subset):
                    gold = normalize_number(example["answer"].split("####")[-1])
                    greedy, greedy_seconds = generate_traces(
                        example["question"], n=1, do_sample=False, seed_offset=local_i
                    )
                    if RUN_SELF_CONSISTENCY:
                        sampled, sampled_seconds = generate_traces(
                            example["question"], n=N_MAX, do_sample=True,
                            seed_offset=10_000 + local_i,
                        )
                    else:
                        # Fast screening avoids a redundant stochastic generation.
                        sampled, sampled_seconds = greedy, 0.0
                    sampled_features = [trace_features(trace) for trace in sampled]
                    record = {
                        "mode": mode,
                        "dataset_index": test_indices[local_i],
                        "question": example["question"],
                        "gold_answer": gold,
                        "greedy_trace": greedy[0],
                        "greedy": trace_features(greedy[0]),
                        "greedy_correct": trace_features(greedy[0])["predicted_answer"] == gold,
                        "sampled_traces": sampled,
                        "sampled_features": sampled_features,
                        "greedy_latency_seconds": greedy_seconds,
                        "sampled_latency_seconds": sampled_seconds,
                        "votes": {},
                    }
                    answers = [features["predicted_answer"] for features in sampled_features]
                    for n in N_LIST:
                        prediction = majority_vote(answers[:n])
                        record["votes"][str(n)] = {
                            "prediction": prediction,
                            "correct": prediction == gold,
                            "vote_counts": dict(Counter(a for a in answers[:n] if a)),
                        }
                    records.append(record)
                    with output_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
                    print(
                        f"{mode}: {local_i + 1}/{eval_count} | "
                        f"greedy={record['greedy_correct']} | "
                        f"N={N_MAX}: {record['votes'][str(N_MAX)]['correct']}"
                    )
            return records

        # Base predictions are cached once and reused by the other ablation runs.
        if cached_base_records is not None:
            base_records = cached_base_records
            expected = [(record["dataset_index"], record["question"]) for record in base_records]
            current = [(test_indices[i], ex["question"]) for i, ex in enumerate(gsm8k_subset)]
            if expected != current:
                raise ValueError("Cached base predictions use a different test subset.")
            print("Reusing cached base predictions.")
        else:
            base_records = evaluate_mode("base", adapter_enabled=False)
            shared_base.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in base_records),
                encoding="utf-8",
            )

        adapter_records = evaluate_mode("adapter", adapter_enabled=True)
        """
    ),
    markdown("## 9. Metrics, paired comparison, and evaluation plots"),
    code(
        """
        def summarize_records(records):
            greedy_correct = np.array([row["greedy_correct"] for row in records], dtype=float)
            summary = {
                "n_examples": len(records),
                "greedy_accuracy": float(greedy_correct.mean()),
                "valid_format_rate": float(np.mean([row["greedy"]["valid_format"] for row in records])),
                "reflection_rate": float(np.mean([row["greedy"]["has_reflection"] for row in records])),
                "mean_thinking_words": float(np.mean([row["greedy"]["thinking_words"] for row in records])),
                "mean_reflection_words": float(np.mean([row["greedy"]["reflection_words"] for row in records])),
                "mean_greedy_latency_seconds": float(np.mean([row["greedy_latency_seconds"] for row in records])),
                "self_consistency_accuracy": {},
            }
            for n in N_LIST:
                summary["self_consistency_accuracy"][str(n)] = float(
                    np.mean([row["votes"][str(n)]["correct"] for row in records])
                )
            return summary

        def bootstrap_paired_difference(base_records, adapter_records, repetitions=10_000):
            base = np.array([row["greedy_correct"] for row in base_records], dtype=float)
            adapter = np.array([row["greedy_correct"] for row in adapter_records], dtype=float)
            rng = np.random.default_rng(SEED)
            indices = rng.integers(0, len(base), size=(repetitions, len(base)))
            differences = (adapter[indices] - base[indices]).mean(axis=1)
            return {
                "adapter_minus_base": float((adapter - base).mean()),
                "bootstrap_95_ci": [
                    float(np.quantile(differences, 0.025)),
                    float(np.quantile(differences, 0.975)),
                ],
                "adapter_wins": int(np.sum((adapter == 1) & (base == 0))),
                "base_wins": int(np.sum((adapter == 0) & (base == 1))),
                "ties": int(np.sum(adapter == base)),
            }

        evaluation = {
            "base": summarize_records(base_records),
            "adapter": summarize_records(adapter_records),
            "paired_greedy_comparison": bootstrap_paired_difference(
                base_records, adapter_records
            ),
        }
        (RUN_DIR / "evaluation_summary.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )

        flat_rows = []
        for mode, records in [("base", base_records), ("adapter", adapter_records)]:
            for row in records:
                flat = {
                    "mode": mode,
                    "dataset_index": row["dataset_index"],
                    "gold_answer": row["gold_answer"],
                    "greedy_prediction": row["greedy"]["predicted_answer"],
                    "greedy_correct": row["greedy_correct"],
                    "valid_format": row["greedy"]["valid_format"],
                    "thinking_words": row["greedy"]["thinking_words"],
                    "reflection_words": row["greedy"]["reflection_words"],
                }
                for n in N_LIST:
                    flat[f"sc_n{n}_prediction"] = row["votes"][str(n)]["prediction"]
                    flat[f"sc_n{n}_correct"] = row["votes"][str(n)]["correct"]
                flat_rows.append(flat)
        pd.DataFrame(flat_rows).to_csv(TABLE_DIR / "evaluation_examples.csv", index=False)
        print(json.dumps(evaluation, indent=2))
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 4.5))
        for mode, summary in [("base", evaluation["base"]), ("adapter", evaluation["adapter"])]:
            values = [100 * summary["self_consistency_accuracy"][str(n)] for n in N_LIST]
            plt.plot(N_LIST, values, "o-", linewidth=2, label=mode)
        plt.xlabel("Number of sampled traces used in majority vote")
        plt.ylabel("Exact-match accuracy (%)")
        plt.title(f"Self-consistency — {RUN_NAME}")
        plt.xticks(N_LIST)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "self_consistency_accuracy.png", dpi=180)
        plt.savefig(PLOT_DIR / "self_consistency_accuracy.pdf")
        plt.show()

        labels = ["Accuracy", "Valid format", "Reflection"]
        base_values = [
            evaluation["base"]["greedy_accuracy"],
            evaluation["base"]["valid_format_rate"],
            evaluation["base"]["reflection_rate"],
        ]
        adapter_values = [
            evaluation["adapter"]["greedy_accuracy"],
            evaluation["adapter"]["valid_format_rate"],
            evaluation["adapter"]["reflection_rate"],
        ]
        x = np.arange(len(labels))
        width = 0.36
        plt.figure(figsize=(7, 4.5))
        plt.bar(x - width / 2, np.array(base_values) * 100, width, label="base")
        plt.bar(x + width / 2, np.array(adapter_values) * 100, width, label="adapter")
        plt.xticks(x, labels)
        plt.ylabel("Rate (%)")
        plt.ylim(0, 100)
        plt.title(f"Greedy generation quality — {RUN_NAME}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "base_vs_adapter.png", dpi=180)
        plt.savefig(PLOT_DIR / "base_vs_adapter.pdf")
        plt.show()
        """
    ),
    markdown("## 10. Human-readable run report and completion marker"),
    code(
        """
        final_eval_loss = training_summary.get("eval_loss")
        best_eval_loss = training_summary.get("best_metric")
        if best_eval_loss is None:
            best_eval_loss = final_eval_loss

        report = f'''# Run report: {RUN_NAME}

        ## Configuration

        - Model: `{MODEL_ID}`
        - Maximum sequence length: **{MAX_LEN}**
        - LoRA rank / alpha: **{LORA_R} / {LORA_ALPHA}**
        - Training examples: **{len(train_rows)}**
        - Validation examples: **{len(eval_rows)}**
        - Held-out GSM8K problems: **{eval_count}**
        - GPU: `{GPU_NAME}`

        ## Training

        - Final validation loss: **{final_eval_loss:.4f}**
        - Best validation loss: **{best_eval_loss:.4f}**
        - Wall time: **{train_seconds / 60:.1f} minutes**
        - Maximum allocated GPU memory: **{training_summary["max_gpu_memory_allocated_gib"]:.2f} GiB**
        - Training truncation rate: **{100 * token_stats["train"]["truncated_fraction"]:.1f}%**

        ## Held-out GSM8K

        | Metric | Base | Adapter |
        |---|---:|---:|
        | Greedy exact match | {100 * evaluation["base"]["greedy_accuracy"]:.1f}% | {100 * evaluation["adapter"]["greedy_accuracy"]:.1f}% |
        | Valid three-tag format | {100 * evaluation["base"]["valid_format_rate"]:.1f}% | {100 * evaluation["adapter"]["valid_format_rate"]:.1f}% |
        | Reflection present | {100 * evaluation["base"]["reflection_rate"]:.1f}% | {100 * evaluation["adapter"]["reflection_rate"]:.1f}% |
        | Self-consistency N={N_MAX} | {100 * evaluation["base"]["self_consistency_accuracy"][str(N_MAX)]:.1f}% | {100 * evaluation["adapter"]["self_consistency_accuracy"][str(N_MAX)]:.1f}% |

        Adapter minus base greedy accuracy:
        **{100 * evaluation["paired_greedy_comparison"]["adapter_minus_base"]:+.1f} percentage points**
        (paired bootstrap 95% CI:
        {100 * evaluation["paired_greedy_comparison"]["bootstrap_95_ci"][0]:+.1f} to
        {100 * evaluation["paired_greedy_comparison"]["bootstrap_95_ci"][1]:+.1f}).

        ## Artifact inventory

        - `config.json`: complete hyperparameter snapshot
        - `environment.json`: software, git, CUDA, and GPU environment
        - `tokenization_stats.json`, `tables/token_lengths.csv`: truncation audit
        - `raw/trainer_log_history.json`: every Trainer log event
        - `raw/gpu_telemetry.jsonl`: per-log GPU observations
        - `raw/*_predictions.jsonl`: full generated traces and voting details
        - `evaluation_summary.json`: scalar and paired evaluation metrics
        - `adapter/`: final LoRA adapter and tokenizer
        - `checkpoints/`: two most recent/best recoverable checkpoints
        - `plots/`: PNG and PDF figures
        '''
        (RUN_DIR / "REPORT.md").write_text(report, encoding="utf-8")
        (RUN_DIR / "COMPLETED").write_text(
            datetime.now(timezone.utc).isoformat() + "\\n", encoding="utf-8"
        )
        print(report)
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "source_notebook": "/Users/leonardocandio/Downloads/deepreasoning (5).ipynb",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
