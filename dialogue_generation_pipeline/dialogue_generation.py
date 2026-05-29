"""
Asynchronous and batched generation script using OpenAI API and vLLM.
Supports both cloud API and local model inference with JSON format extraction.
"""

import json
import re
import asyncio
import argparse
import random
from typing import Any, Dict
from tqdm import tqdm
from openai import AsyncOpenAI
from vllm import LLM, SamplingParams


# =========================
# Async call API
# =========================

async def call_api(client: AsyncOpenAI, messages, model: str, temperature: float):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


# =========================
# Process one item
# =========================

async def process_item(client, item, model, temperature, semaphore):
    async with semaphore:
        messages = item["messages"]
        out_item = item.copy()
        out_item.pop("messages", None)

        try:
            content = await call_api(client, messages, model, temperature)

            try:
                dialogue = json.loads(content)
                out_item["dialogue"] = dialogue
                return out_item

            except Exception as e:
                out_item["error"] = "JSON parse failed"
                out_item["raw_content"] = content
                return out_item

        except Exception:
            out_item["error"] = "Generation failed"
            return out_item

# =========================
# vLLM Generation
# =========================

def run_vllm(items, model, temperature, tensor_parallel_size=1, quantization=None, gpu_memory_utilization=0.9):

    print(f"Initializing vLLM with model: {model}")
    llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=tensor_parallel_size, quantization=quantization, gpu_memory_utilization=gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()

    prompts = []
    for item in items:
        messages = item["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=temperature, max_tokens=2048)
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        item = items[i]
        
        out_item = item.copy()
        out_item.pop("messages", None)
        
        content = output.outputs[0].text
        
        clean_content = content
        
        # 1. Coarse filter: Strip the reasoning process by extracting content after </think> tag.
        if "</think>" in clean_content:
            clean_content = clean_content.split("</think>")[-1].strip()
        # 2. Fine filter: Remove markdown formatting (e.g., ```json ... ```) around the JSON output.
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', clean_content)
        if json_match:
            clean_content = json_match.group(0)
        

        try:
            dialogue = json.loads(clean_content)
            out_item["dialogue"] = dialogue
            results.append(out_item)
        except Exception as e:
            out_item["error"] = f"JSON parse failed: {str(e)}"
            out_item["raw_content"] = content
            out_item["clean_content"] = clean_content
            results.append(out_item)
            
    return results

# =========================
# Main async generation
# =========================

async def generate(
    input_path: str,
    output_path: str,
    model: str,
    temperature: float,
    max_num_gen: int = 500,
    max_concurrency: int = 20,
    backend: str = "openai",
    tensor_parallel_size: int = 1,
    quantization: str = None,
    gpu_memory_utilization: float = 0.9,
):
    # client = AsyncOpenAI()  # Moved inside else block

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = random.sample(lines, min(max_num_gen, len(lines)))

    items = []
    for line in lines:
        item = json.loads(line)
        # Filter out samples containing Chinese characters using regex on the messages string.
        if not re.search(r'[\u4e00-\u9fff]', json.dumps(item.get("messages", ""), ensure_ascii=False)):
            items.append(item)

    if backend == "vllm":
        # vLLM local inference (synchronous/batch)
        outputs = run_vllm(items, model, temperature, tensor_parallel_size, quantization, gpu_memory_utilization)
        # Write results to file
        with open(output_path, "w", encoding="utf-8") as f:
            for obj in outputs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"Saved {len(outputs)} generations to {output_path}")
    else:
        # OpenAI API inference (asynchronous/concurrent)
        semaphore = asyncio.Semaphore(max_concurrency)
        client = AsyncOpenAI() 
        semaphore = asyncio.Semaphore(max_concurrency)

        tasks = [
            process_item(client, item, model, temperature, semaphore)
            for item in items
        ]

        outputs = []
        open(output_path, "w", encoding="utf-8").close()
        
        # Open in append mode to write and flush to disk immediately after each request
        with open(output_path, "a", encoding="utf-8") as f:
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                result = await coro
                outputs.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
        
        print(f"Saved {len(outputs)} generations to {output_path}")




# =========================
# Entry point
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path for the output JSONL file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B",
                        help="Model name. Large models: Qwen/Qwen3-32B gpt-4.1 deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_num_gen", type=int, default=10000)
    parser.add_argument("--max_concurrency", type=int, default=20)

    # vLLM-specific arguments
    parser.add_argument("--backend", type=str, default="vllm", choices=["openai", "vllm"], help="Backend to use: openai or vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs to use for vLLM (required for 70B+ models)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="GPU memory utilization (0.0 to 1.0)")

    args = parser.parse_args()

    asyncio.run(
        generate(
            input_path=args.input_path,
            output_path=args.output_path,
            model=args.model,
            temperature=args.temperature,
            max_num_gen=args.max_num_gen,
            max_concurrency=args.max_concurrency,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    )