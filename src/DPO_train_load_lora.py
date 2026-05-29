"""
Direct Preference Optimization (DPO) Training Script

This script sets up and runs DPO training for causal language models using Hugging Face's 
`trl` (Transformer Reinforcement Learning) library. It supports loading a base model, 
optionally initializing from an existing LoRA checkpoint, training with LoRA adapters, 
and evaluating preference datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, PeftModel

TORCH_DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "sdpa"
MAX_LENGTH = 3072

# load lora v1 param: 1e-5 wamrup 0.1 beta 0.05

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--train_file", type=str, default="../data/DPO_train.jsonl")
    parser.add_argument("--validation_file", type=str, default="../data/DPO_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./output/dpo_model")
    parser.add_argument("--init_lora_path", type=str, default=None, help="Path to initial LoRA checkpoint, if any.")

    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5) # 1e-5 5e-5 1e-6 5e-6
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.05) # 0.1 0.01

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def maybe_get_last_checkpoint(output_dir: str) -> Optional[str]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = []
    for p in output_path.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-")[-1])
            checkpoints.append((step, str(p)))
        except Exception:
            continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    if args.init_lora_path is not None:
        model = PeftModel.from_pretrained(
            model,
            args.init_lora_path,
            is_trainable=True,
        )

    model.config.use_cache = False
    return model, tokenizer


def load_dpo_dataset(path: str):
    dataset = load_dataset("json", data_files=path, split="train")

    required_columns = {"prompt", "chosen", "rejected"}
    if not required_columns.issubset(set(dataset.column_names)):
        raise ValueError(f"{path} must contain columns: {required_columns}")

    def _check_messages(name, messages):
        if not isinstance(messages, list):
            raise ValueError(f"{name} must be a list")
        if len(messages) == 0:
            raise ValueError(f"{name} must not be empty")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"{name}[{i}] must be a dict")
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"{name}[{i}] must contain 'role' and 'content'")

    def _check_example(example):
        _check_messages("prompt", example["prompt"])
        _check_messages("chosen", example["chosen"])
        _check_messages("rejected", example["rejected"])

        if example["prompt"][-1]["role"] != "user":
            raise ValueError("prompt last message should be user")

        if example["chosen"][0]["role"] != "assistant":
            raise ValueError("chosen should start with assistant")

        if example["rejected"][0]["role"] != "assistant":
            raise ValueError("rejected should start with assistant")

        return example

    dataset = dataset.map(_check_example)
    return dataset


def plot_training_curves(log_history: List[Dict], save_dir: str) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(log_history)
    df.to_csv(save_dir / "trainer_log_history.csv", index=False)

    if "loss" in df.columns:
        train_df = df[df["loss"].notna()]
        if not train_df.empty:
            plt.figure()
            plt.plot(train_df["step"], train_df["loss"])
            plt.title("Train DPO Loss")
            plt.savefig(save_dir / "train_loss.png")
            plt.close()

    if "eval_loss" in df.columns:
        eval_df = df[df["eval_loss"].notna()]
        if not eval_df.empty:
            plt.figure()
            plt.plot(eval_df["step"], eval_df["eval_loss"])
            plt.title("Validation DPO Loss")
            plt.savefig(save_dir / "val_loss.png")
            plt.close()

    if "eval_rewards/accuracies" in df.columns:
        eval_acc_df = df[df["eval_rewards/accuracies"].notna()]
        if not eval_acc_df.empty:
            plt.figure()
            plt.plot(eval_acc_df["step"], eval_acc_df["eval_rewards/accuracies"])
            plt.title("Validation Reward Accuracy")
            plt.savefig(save_dir / "val_reward_accuracy.png")
            plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("🚀 Start DPO Training")
    print(f"Model: {args.model_name_or_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Beta: {args.beta}")
    print("=" * 60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), output_dir / "run_config.json")

    # ===== Data =====
    train_dataset = load_dpo_dataset(args.train_file)
    val_dataset = load_dpo_dataset(args.validation_file)

    print("\n📊 Dataset Loaded")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Columns: {train_dataset.column_names}")

    # ===== Model =====
    model, tokenizer = make_model_and_tokenizer(args)

    print("\n🧠 Model Loaded")
    print(f"dtype: {TORCH_DTYPE}")
    print(f"attention: {ATTN_IMPLEMENTATION}")
    print(f"padding_side: {tokenizer.padding_side}")

    # ===== Training Configuration =====
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=75,
        save_strategy="steps",
        save_steps=75,
        save_total_limit=12,
        bf16=True,
        fp16=False,
        max_length=MAX_LENGTH,
        beta=args.beta,
        remove_unused_columns=False,
        report_to="none",
        save_only_model=False,
        seed=args.seed,
    )

    # ===== LoRA Configuration =====
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=None if args.init_lora_path is not None else peft_config,
    )

    # ===== checkpoint =====
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is None:
        resume_ckpt = maybe_get_last_checkpoint(str(output_dir))

    if resume_ckpt:
        print("\n♻️ Resume from checkpoint:")
        print(resume_ckpt)
    else:
        print("\n🆕 Start from base model")

    print("\n🏋️ Start Training...")

    trainer.train(
        resume_from_checkpoint=resume_ckpt if resume_ckpt else None
    )

    print("\n✅ Training Finished")

    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    trainer.save_state()

    val_metrics = trainer.evaluate()

    print("\n📈 Validation Results")
    print(val_metrics)

    trainer.save_metrics("eval", val_metrics)
    plot_training_curves(trainer.state.log_history, str(output_dir))

    print("\n🎉 All Done!")


if __name__ == "__main__":
    main()