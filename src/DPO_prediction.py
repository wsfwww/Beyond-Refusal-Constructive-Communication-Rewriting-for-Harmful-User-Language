"""
vLLM Inference Script for DPO Checkpoints

This script automates the evaluation of multiple LoRA checkpoints generated during
Direct Preference Optimization (DPO) or standard SFT. It iterates through saved 
checkpoints, loads them via vLLM, and generates predictions for test and validation 
datasets (supporting both standard conversational 'messages' and 'dpo' formats).
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)


def load_jsonl(path: str | Path, data_format: str) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_idx}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Each line must be a JSON object in {path} at line {line_idx}, got {type(obj)}"
                )

            if data_format == "messages":
                if "messages" not in obj:
                    raise ValueError(f"Missing key 'messages' in {path} at line {line_idx}")
                if not isinstance(obj["messages"], list) or len(obj["messages"]) == 0:
                    raise ValueError(f"'messages' must be a non-empty list in {path} at line {line_idx}")

            elif data_format == "dpo":
                for key in ["prompt", "chosen", "rejected"]:
                    if key not in obj:
                        raise ValueError(f"Missing key '{key}' in {path} at line {line_idx}")
                    if not isinstance(obj[key], list) or len(obj[key]) == 0:
                        raise ValueError(f"'{key}' must be a non-empty list in {path} at line {line_idx}")

            else:
                raise ValueError(f"Unknown data_format: {data_format}")

            rows.append(obj)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM inference for every DPO checkpoint."
    )

    parser.add_argument(
        "--model_root",
        type=str,
        default="./output/dpo_model_checkpoints",
        help="Directory containing checkpoint-* subdirectories.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model path or HF repo id, used for tokenizer/chat template.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../data/test_data.jsonl",
        help="Path to normal SFT-style test jsonl with top-level messages.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default="",
        help="Optional path to DPO-format validation jsonl with prompt/chosen/rejected.",
    )
    parser.add_argument(
        "--only_final",
        default=False,
        action="store_true",
        help="If True, only evaluate the final_model and ignore all checkpoints.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=[],
        help="List of specific checkpoint folder names to evaluate. If empty, evaluates all.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="How many examples to generate for each split. <=0 means all.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/predictions",
        help="Where to save generations.",
    )
    parser.add_argument(
        "--include_final_model",
        default=False,
        action="store_true",
        help="Also evaluate final_model if it exists.",
    )

    return parser.parse_args()


def maybe_slice_examples(data: List[Dict], max_samples: int) -> List[Dict]:
    if max_samples is None or max_samples <= 0:
        return data
    return data[:max_samples]


def find_checkpoints(
    model_root: str,
    include_final_model: bool = False,
    only_final: bool = False,
) -> List[Path]:
    root = Path(model_root)
    if not root.exists():
        raise FileNotFoundError(f"Model root does not exist: {root}")

    if only_final:
        final_model_dir = root / "final_model"
        if final_model_dir.exists() and final_model_dir.is_dir():
            return [final_model_dir]
        raise ValueError(f"final_model directory not found under {root}")

    checkpoints: List[Tuple[int, Path]] = []
    for p in root.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-")[-1])
            checkpoints.append((step, p))
        except Exception:
            continue

    checkpoints.sort(key=lambda x: x[0])
    ordered = [p for _, p in checkpoints]

    if include_final_model:
        final_model_dir = root / "final_model"
        if final_model_dir.exists() and final_model_dir.is_dir():
            ordered.append(final_model_dir)

    if not ordered:
        raise ValueError(f"No checkpoint-* directories found under {root}")

    return ordered


def try_parse_json(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.rfind("```")
        if end > start:
            candidate = text[start:end].strip()
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            candidate = text[start:end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    return None


def split_prediction(pred_text: str) -> Dict[str, str]:
    pred_text = (pred_text or "").strip()
    m = _THINK_RE.match(pred_text)

    reasoning_prefix = ""
    answer_part = pred_text

    if m is not None:
        reasoning_prefix = m.group(1).strip()
        answer_part = m.group(2).strip()

    parsed_json = try_parse_json(answer_part)
    if parsed_json is None:
        parsed_json = try_parse_json(pred_text)

    pred_reasoning = reasoning_prefix
    pred_answer = answer_part

    if parsed_json is not None:
        pred_reasoning = str(parsed_json.get("nvc_cot_analysis", reasoning_prefix)).strip()
        pred_answer = json.dumps(parsed_json, ensure_ascii=False)

    return {
        "pred_reasoning": pred_reasoning,
        "pred_answer": pred_answer,
    }


def parse_gold_fields(gold_text: str) -> Dict[str, str]:
    parsed_json = try_parse_json(gold_text)

    if parsed_json is None:
        return {
            "gold_reasoning": "",
            "gold_answer": gold_text.strip(),
        }

    return {
        "gold_reasoning": str(parsed_json.get("nvc_cot_analysis", "")).strip(),
        "gold_answer": json.dumps(parsed_json, ensure_ascii=False),
    }


def get_assistant_text(messages: List[Dict]) -> str:
    for msg in messages:
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def build_prompt_from_example(tokenizer, ex: Dict, data_format: str) -> str:
    if data_format == "messages":
        prompt_messages = ex["messages"]
    elif data_format == "dpo":
        prompt_messages = ex["prompt"]
    else:
        raise ValueError(f"Unknown data_format: {data_format}")

    filtered_messages = [
        msg for msg in prompt_messages
        if msg.get("role") in ["system", "user"]
    ]

    present_roles = {msg.get("role") for msg in filtered_messages}

    if "user" not in present_roles:
        raise ValueError(f"Missing user role. Found roles: {present_roles}")

    if filtered_messages[-1].get("role") != "user":
        raise ValueError(f"Last prompt message must be user, got {filtered_messages[-1].get('role')}")

    return tokenizer.apply_chat_template(
        filtered_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prepare_requests(
    tokenizer,
    data: List[Dict],
    data_format: str,
) -> Tuple[List[str], List[Dict]]:
    prompts: List[str] = []
    rows: List[Dict] = []

    for ex in data:
        prompt = build_prompt_from_example(tokenizer, ex, data_format=data_format)
        prompts.append(prompt)
        rows.append(ex)

    return prompts, rows


def save_jsonl(rows: List[Dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_one_split(
    llm: LLM,
    tokenizer,
    split_name: str,
    data_file: str,
    output_file: str,
    max_samples: int,
    sampling_params: SamplingParams,
    lora_path: str,
    data_format: str,
) -> None:
    raw_data = load_jsonl(data_file, data_format=data_format)
    raw_data = maybe_slice_examples(raw_data, max_samples)

    print("=" * 80)
    print(f"Running split: {split_name}")
    print(f"Data format: {data_format}")
    print(f"Data file: {data_file}")
    print(f"Num examples: {len(raw_data)}")
    print(f"Output file: {output_file}")

    prompts, rows = prepare_requests(tokenizer, raw_data, data_format=data_format)

    lora_request = LoRARequest("target_lora", 1, lora_path)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    save_rows: List[Dict] = []

    for idx, (ex, prompt, out) in enumerate(zip(rows, prompts, outputs)):
        if data_format == "messages":
            messages = ex["messages"]

            gold_text = ""
            if len(messages) > 0 and messages[-1].get("role") == "assistant":
                gold_text = str(messages[-1].get("content", "")).strip()

            gold_fields = parse_gold_fields(gold_text) if gold_text else {
                "gold_reasoning": "",
                "gold_answer": "",
            }

            chosen = []
            rejected = []
            chosen_scores = []
            rejected_scores = []

        elif data_format == "dpo":
            messages = ex["prompt"]
            chosen = ex.get("chosen", [])
            rejected = ex.get("rejected", [])
            chosen_scores = ex.get("chosen_scores", [])
            rejected_scores = ex.get("rejected_scores", [])

            chosen_text = get_assistant_text(chosen)
            gold_fields = {
                "gold_reasoning": "",
                "gold_answer": chosen_text,
            }

        else:
            raise ValueError(f"Unknown data_format: {data_format}")

        system_prompt = ""
        user_prompt = ""

        for m in messages:
            if m.get("role") == "system":
                system_prompt += str(m.get("content", "")) + "\n"
            elif m.get("role") == "user":
                user_prompt += str(m.get("content", "")) + "\n"

        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()

        pred_text = out.outputs[0].text if len(out.outputs) > 0 else ""
        finish_reason = out.outputs[0].finish_reason if len(out.outputs) > 0 else ""
        generated_tokens = len(out.outputs[0].token_ids) if len(out.outputs) > 0 else 0

        pred_fields = split_prediction(pred_text)

        save_rows.append(
            {
                "id": ex.get("id", f"sample_{idx:07d}"),
                "task_type": ex.get("task_type", ""),
                "checkpoint_prompt": prompt,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,

                "gold_reasoning": gold_fields["gold_reasoning"],
                "gold_answer": gold_fields["gold_answer"],

                "chosen": chosen,
                "rejected": rejected,
                "chosen_scores": chosen_scores,
                "rejected_scores": rejected_scores,

                "analysis": ex.get("analysis", ""),
                "meta": ex.get("meta", {}),

                "pred_text": pred_text,
                "pred_reasoning": pred_fields["pred_reasoning"],
                "pred_answer": pred_fields["pred_answer"],
                "finish_reason": finish_reason,
                "generated_tokens": generated_tokens,
            }
        )

    save_jsonl(save_rows, output_file)
    print(f"Saved {len(save_rows)} rows to: {output_file}")


def cleanup_llm(llm: Optional[LLM]) -> None:
    if llm is not None:
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    checkpoints = find_checkpoints(
        model_root=args.model_root,
        include_final_model=args.include_final_model,
        only_final=args.only_final,
    )

    if args.checkpoints and not args.only_final:
        filtered_checkpoints = [ckpt for ckpt in checkpoints if ckpt.name in args.checkpoints]
        checkpoints = filtered_checkpoints
        if not checkpoints:
            print(f"[WARNING] Specified checkpoints {args.checkpoints} not found in the directory!")
            return

    print("=" * 80)
    print("Found checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")
    print("=" * 80)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading HF tokenizer from base model: {args.base_model}")
    hf_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )

    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    for ckpt_dir in checkpoints:
        ckpt_name = ckpt_dir.name
        ckpt_output_dir = output_root / ckpt_name
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)

        test_done = (ckpt_output_dir / "test_predictions.jsonl").exists()
        val_done = (
            (ckpt_output_dir / "validation_predictions.jsonl").exists()
            if args.validation_file
            else True
        )

        if test_done and val_done:
            print(f"[SKIP] Predictions for {ckpt_name} already exist, skipping inference.")
            continue

        print("\n" + "#" * 100)
        print(f"Loading checkpoint: {ckpt_dir}")
        print(f"Using tokenizer path for vLLM: {args.base_model}")
        print("#" * 100)

        llm = None
        try:
            llm = LLM(
                model=args.base_model,
                enable_lora=True,
                max_lora_rank=128,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                max_model_len=8192,
            )

            if args.validation_file:
                run_one_split(
                    llm=llm,
                    tokenizer=hf_tokenizer,
                    split_name="validation",
                    data_file=args.validation_file,
                    output_file=str(ckpt_output_dir / "validation_predictions.jsonl"),
                    max_samples=args.max_samples,
                    sampling_params=sampling_params,
                    lora_path=str(ckpt_dir),
                    data_format="dpo",
                )

            run_one_split(
                llm=llm,
                tokenizer=hf_tokenizer,
                split_name="test",
                data_file=args.test_file,
                output_file=str(ckpt_output_dir / "test_predictions.jsonl"),
                max_samples=args.max_samples,
                sampling_params=sampling_params,
                lora_path=str(ckpt_dir),
                data_format="messages",
            )

        finally:
            cleanup_llm(llm)

    print("\n🎉 All checkpoints finished.")


if __name__ == "__main__":
    main()