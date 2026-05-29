"""
Supervised Fine-Tuning (SFT) Training Script

This script sets up and runs SFT for causal language models using Hugging Face's 
`trl` (Transformer Reinforcement Learning) library. It includes LoRA adapter configuration,
custom dataset processing for conversational formats, and automated training curve plotting.
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
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

TORCH_DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "sdpa"
MAX_LENGTH = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--train_file", type=str, default="../data/SFT_train.jsonl")
    parser.add_argument("--validation_file", type=str, default="../data/SFT_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./output/sft_model")

    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    model.config.use_cache = False
    return model, tokenizer


def load_messages_dataset(path: str):
    dataset = load_dataset("json", data_files=path, split="train")

    #------------------special data process-----------------------#
    columns = dataset.column_names

    if "messages" in columns:
        dataset = dataset.remove_columns([col for col in columns if col != "messages"])
    elif "user_prompt" in columns and "answer" in columns:
        def convert_to_messages(example):
            msgs = []
            if example.get("system_prompt"):
                msgs.append({"role": "system", "content": example["system_prompt"]})
                
            msgs.append({"role": "user", "content": example.get("user_prompt", "")})
            msgs.append({"role": "assistant", "content": example.get("answer", "")})
            
            return {"messages": msgs}

        dataset = dataset.map(convert_to_messages, remove_columns=columns)
    else:
        raise ValueError(f"{path} must contain a top-level 'messages' field")
    
    #------------------special data process end-----------------------#

    def _check_example(example):
        messages = example["messages"]
        if not isinstance(messages, list):
            raise ValueError("Each example['messages'] must be a list")
        if len(messages) == 0:
            raise ValueError("messages must not be empty")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"messages[{i}] must be a dict")
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"messages[{i}] must contain 'role' and 'content'")
        if messages[-1]["role"] != "assistant":
            raise ValueError("The last message must be assistant for SFT training")
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
            plt.title("Train Loss")
            plt.savefig(save_dir / "train_loss.png")
            plt.close()

    if "eval_loss" in df.columns:
        eval_df = df[df["eval_loss"].notna()]
        if not eval_df.empty:
            plt.figure()
            plt.plot(eval_df["step"], eval_df["eval_loss"])
            plt.title("Validation Loss")
            plt.savefig(save_dir / "val_loss.png")
            plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("🚀 Start Training")
    print(f"Model: {args.model_name_or_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Max length: {MAX_LENGTH}")
    print("=" * 60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), output_dir / "run_config.json")

    # ===== Data =====
    train_dataset = load_messages_dataset(args.train_file)
    val_dataset = load_messages_dataset(args.validation_file)

    print("\n📊 Dataset Loaded")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Columns: {train_dataset.column_names}")

    # ===== Model =====
    model, tokenizer = make_model_and_tokenizer(args)

    print("\n🧠 Model Loaded")
    print(f"dtype: {TORCH_DTYPE}")
    print(f"attention: {ATTN_IMPLEMENTATION}")

    # ===== Training Configuration =====
    training_args = SFTConfig(
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
        # eval_strategy="epoch",
        # save_strategy="epoch",
        eval_strategy="steps",               
        eval_steps=75,         
        save_strategy="steps",               
        save_steps=75,         
        save_total_limit=12,
        bf16=True,
        fp16=False,
        max_length=MAX_LENGTH,
        packing=False,
        assistant_only_loss=False,           
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,  
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