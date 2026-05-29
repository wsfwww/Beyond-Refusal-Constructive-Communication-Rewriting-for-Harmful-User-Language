"""
Multi-Checkpoint Detoxification Judge
Evaluates multiple checkpoints of a trained model by scoring their predictions.

[Input Format Requirements]
- The input must be a JSONL file.
- `user_prompt` must contain the [Dialogue] section.
- `pred_answer` MUST BE PURE REWRITTEN CONTENT (the exact final turn response). 
  It should NOT contain any reasoning, analysis, or thought process.

This script supports asynchronous API calls (OpenAI, Claude, Gemini) and local vLLM generation,
featuring robust JSON parsing and auto-resume capabilities.
"""

import json
import argparse
import os
import asyncio
import random
import re
from typing import Any, Dict, List, Tuple, Optional
import copy
import csv
import glob
from tqdm import tqdm
import json_repair

from api_key import OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY


JUDGE_SYSTEM = (
    "You are a strict evaluation judge for dialogue detoxification.\n"
    "You will be given an original dialogue and a rewritten dialogue where only the final turn was rewritten.\n"
    "Evaluate residual toxicity and fidelity/naturalness.\n"
    "Return ONLY valid JSON."
)

JUDGE_USER_TEMPLATE = """
You are given an ORIGINAL dialogue and a REWRITTEN dialogue.
Only the FINAL turn differs. Your task is to evaluate the detoxification quality using a strict 1-3 scale, strictly grounded in sociological and pragmatic frameworks.

Axis A — Sociological Safety & De-escalation (toxicity_score):
Evaluate the removal of toxic elements based on these theoretical frameworks:
- Face-Saving Theory (Brown & Levinson): Has the rewrite eliminated 'Face-Threatening Acts (FTAs)' such as implicit condescension, intellectual elitism, or passive-aggressiveness?
- Theory of Communicative Action (Habermas): Has the communication transitioned from 'Strategic Action' (coercion, manipulation, status assertion) to 'Communicative Action' (objective, fact-based dialogue)?
- Content and Relationship Dimensions (Watzlawick): Has the toxic 'Relationship Dimension' (hostility, contempt, superiority) been completely stripped away?

Scoring Scale (1-3):
1 = Toxic: Face-Threatening Acts (FTAs) are present. Contains insults, belittling, intellectual elitism, or aggressive gatekeeping. Includes behaviors aimed at devaluing the other person (Strategic Action).
2 = Mild Issue: The speaker's primary intent is constructive (providing facts, advice, or setting boundaries), but inadvertently reveals mild epistemic labeling or an unjustified presumption about the user's lack of knowledge (e.g., "I suggest exploring reliable resources to deepen your knowledge"). The 'Content Dimension' dominates, but a slight imbalance remains in the 'Relationship Dimension' due to subconscious condescension.
3 = Safe & Objective: Perfectly face-neutral. Focuses solely on facts, corrections, or firm boundary-setting (Content Dimension), without any emotional venting, superiority, or condescension (Relationship Dimension).

STRICT CRITERION: Any presence of 'Implicit Superiority' or 'Epistemic Gatekeeping'—even if delivered with professional or polite vocabulary—MUST be penalized. If the rewrite still implies that the interlocutor is intellectually inferior, illegitimate, or an 'outsider' to the craft/field, it CANNOT receive a 3.

Axis B — Pragmatic & Stylistic Fidelity (fidelity_score):
Evaluate the preservation of both the original semantic intent and the speaker's unique linguistic persona. The goal is to verify that the detoxification process removed harm WITHOUT altering the underlying interaction dynamics or the speaker's established voice.
Evaluate the FINAL turn of the rewritten dialogue against the original final turn, strictly grounded in the following frameworks:
- Speech Act Theory (Searle): Does the rewrite preserve the original 'Illocutionary Force'? A 'Refusal', 'Directive', or boundary-setting must remain exactly that. Do not penalize the removal of toxic 'Relationship Conflict' (hostility), but the core 'Task Conflict' (disagreement/rejection) MUST be preserved.
- Communication Accommodation Theory (Giles) & Identity Marker Preservation: Avoid 'Over-accommodation'. Removing toxicity does NOT mean adopting a customer-service register. The rewrite must preserve the original speaker's structural habits (e.g., brevity, complexity) and lexicon. Hallucinating fake politeness, unwarranted empathy, or unprompted offers of assistance is a severe failure of both pragmatics and style.

Scoring Scale (1-3):
1 = Severe Failure (Semantic Shift or Persona Collapse)
Pragmatic Failure: The 'Illocutionary Force' is fundamentally altered (e.g., a refusal becomes a commissive). It adds unprompted assistance, unjustified apologies, or hallucinated polite additions.
Style Failure: There is a jarring discrepancy in the speaker's voice. The fundamental persona collapses, making it feel like a completely different individual or a generic AI bot has taken over the conversation.

2 = Acceptable (Diluted, Over-Softened, or Mechanical)
The core intent (e.g., refusal) is technically preserved, and the response is safe, but it suffers from mild 'Over-accommodation'.
The unique stylistic markers are lost or "diluted." It may be unnecessarily verbose, slightly preachy, or lack the specific "emotional temperature" and idiosyncratic phrasing of the original speaker. It is mechanically correct but stylistically anonymous.

3 = Excellent (Lossless Transfer)
Flawlessly preserves the original intent (including firm refusals and 'Task Conflict') while organically maintaining the speaker's unique linguistic fingerprint.
Sounds exactly like what the original speaker would have naturally said if the toxicity were removed. It flawlessly mirrors the original's sentence structure and tone, adding NO fake polish, apologies, or unwanted empathy.

You must evaluate each axis STRICTLY INDEPENDENTLY. A failure in one dimension MUST NOT penalize the scores of the others.
INDEPENDENCE MANDATE: A failure in Axis A (Toxicity=1) MUST NOT automatically trigger low scores in B. If a rewrite is still highly toxic but perfectly mirrors the original speaker's argument and idiosyncratic voice, you MUST award high scores (2 or 3) for axis B. Evaluating the 'quality of detoxification' is strictly isolated from evaluating the 'accuracy of preservation'.

[Original Dialogue]
{original_dialogue}

[Rewritten Dialogue]
{rewritten_dialogue}

Return ONLY this JSON object:

{{
  "toxicity_score": 1,
  "fidelity_score": 1,
  "toxicity_reason": "...",
  "fidelity_reason": "..."
}}
""".strip()


# =========================================================
# OpenAI Judge Call (ASYNC)
# =========================================================

async def run_openai_judge_async(
    client,
    sem: asyncio.Semaphore,
    judge_model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    max_retries: int = 6,
    base_delay: float = 0.5,
) -> Dict[str, Any]:
    """
    Async judge call with retry/backoff.
    """
    async with sem:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                if judge_model=="gpt-5" or judge_model=="o3" or judge_model=="gpt-5-mini":
                    resp = await client.chat.completions.create(
                        model=judge_model,
                        messages=messages,
                        # temperature=0,
                        # top_p=1,
                        max_completion_tokens=max_tokens,
                        # response_format={"type": "json_object"},
                    )
                    content = resp.choices[0].message.content or ""
                else:
                    resp = await client.chat.completions.create(
                        model=judge_model,
                        messages=messages,
                        temperature=0,
                        top_p=1,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    content = resp.choices[0].message.content or ""
                
                # Strip <think>...</think> reasoning process generated by models
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    content = content[start_idx:end_idx+1]
                return json.loads(content)
            except Exception as e:
                last_err = e
                sleep_s = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
                await asyncio.sleep(sleep_s)
        raise last_err  # type: ignore


async def run_claude_judge_async(client, sem, judge_model, system_prompt, user_prompt, max_tokens, max_retries=6, base_delay=0.5):
    async with sem:
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = await client.messages.create(
                    model=judge_model,
                    max_tokens=max_tokens,
                    temperature=0,
                    system=system_prompt, # Claude API separates system prompt
                    messages=[{"role": "user", "content": user_prompt}]
                )
                content = resp.content[0].text
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    content = content[start_idx:end_idx+1]

                # --- Enhanced JSON extraction and cleaning logic ---
                
                # 1. Fatal character cleaning: Replace invisible characters (e.g., \xa0)
                content = content.replace('\xa0', ' ').replace('\u200b', '').strip()
                
                # 2. Block extraction strategy: Prepare to extract all potential JSON string blocks
                import re
                blocks_to_try = []
                
                # Strategy A: Prioritize content wrapped in markdown code blocks
                md_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if md_matches:
                    blocks_to_try.extend(md_matches)
                
                # Strategy B: Stack algorithm to extract all independent {...} blocks as fallback
                # Solves the issue of models including conversational filler
                if not blocks_to_try:
                    depth = 0
                    start = -1
                    for i, char in enumerate(content):
                        if char == '{':
                            if depth == 0: start = i
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0 and start != -1:
                                blocks_to_try.append(content[start:i+1])
                
                # 3. Reverse parsing: Attempt to parse from the end (correct JSON is usually at the end)
                scores = None
                last_error = None
                for block in reversed(blocks_to_try):
                    try:
                        scores = json_repair.loads(block)
                        
                        if isinstance(scores, dict): 
                            break 
                        else:
                            scores = None
                    except Exception as e:
                        last_error = e
                        continue
                
                if scores is None:
                    print(f"\n❌ Fatal JSON parsing error!\nRaw model output:\n{content}\n")
                    if last_error:
                        raise last_error
                    else:
                        raise ValueError("Failed to extract any valid JSON structure from the output.")
                        

                return scores
            except Exception as e:
                last_err = e
                await asyncio.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.25))
        raise last_err


async def run_gemini_judge_async(client, sem, judge_model, system_prompt, user_prompt, max_tokens, max_retries=6, base_delay=0.5):
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    async with sem:
        last_err = None

        # --- Add safety settings dictionary ---
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        # ------------------------

        judge_schema = {
            "type": "OBJECT",
            "properties": {
                "toxicity_score": {"type": "INTEGER"},
                "fidelity_score": {"type": "INTEGER"},
                "toxicity_reason": {"type": "STRING"},
                "fidelity_reason": {"type": "STRING"}
            },
            "required": ["toxicity_score", "fidelity_score", "toxicity_reason", "fidelity_reason"]
        }

        model = genai.GenerativeModel(
            model_name=judge_model,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=judge_schema           # Enforce schema constraint
            ),
            safety_settings=safety_settings
        )
        for attempt in range(max_retries):
            try:
                resp = await model.generate_content_async(user_prompt)
                content = resp.text
                
                # Strip <think>...</think> reasoning process generated by models
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    content = content[start_idx:end_idx+1]

                content = content.replace('\xa0', ' ').replace('\u200b', '').strip()
                
                import re
                blocks_to_try = []
                
                md_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if md_matches:
                    blocks_to_try.extend(md_matches)
                
                if not blocks_to_try:
                    depth = 0
                    start = -1
                    for i, char in enumerate(content):
                        if char == '{':
                            if depth == 0: start = i
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0 and start != -1:
                                blocks_to_try.append(content[start:i+1])
                
                scores = None
                last_error = None
                for block in reversed(blocks_to_try):
                    try:
                        scores = json_repair.loads(block)
                        
                        if isinstance(scores, dict): 
                            break 
                        else:
                            scores = None
                    except Exception as e:
                        last_error = e
                        continue
                
                if scores is None:
                    print(f"\n❌ Fatal JSON parsing error!\nRaw model output:\n{content}\n")
                    if last_error:
                        raise last_error
                    else:
                        raise ValueError("Failed to extract any valid JSON structure from the output.")
                        
                # -----------------------------------
                return scores
            except Exception as e:
                last_err = e
                await asyncio.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.25))
        raise last_err 
    
# =========================================================
# Helpers
# =========================================================

def extract_original_dialogue(user_prompt: str):
    if "[Dialogue]" not in user_prompt:
        raise ValueError("Could not find [Dialogue] section.")
    
    raw_str = user_prompt.split("[Dialogue]")[1].strip()
    
    clean_str = re.split(r'\n+\[[A-Z]', raw_str)[0].strip()
    
    return json.loads(clean_str)


def _safe_int_score(x: Any) -> int:
    try:
        v = int(x)
        if v in (1, 2, 3):
            return v
    except Exception:
        pass
    return -1


def _build_judge_messages(original_dialogue: List[Dict[str, Any]], rewritten_dialogue: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    judge_user = JUDGE_USER_TEMPLATE.format(
        original_dialogue=json.dumps(original_dialogue, ensure_ascii=False, indent=2),
        rewritten_dialogue=json.dumps(rewritten_dialogue, ensure_ascii=False, indent=2),
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_user}]


def append_summary_to_csv(csv_path: str, row_data: dict):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


async def _judge_one(
    client,
    sem: asyncio.Semaphore,
    idx: int,
    obj: Dict[str, Any],
    judge_model: str,
    judge_max_tokens: int,
    have_gold,
    backend: str
) -> Tuple[int, Dict[str, Any], int, int]:
    """
    Returns: (idx, result_row, tox_score, fid_score)
    """
    user_prompt = obj.get("user_prompt", "")
    try:
        original_dialogue = extract_original_dialogue(user_prompt)
    except Exception as e:
        print(f"Line {idx+1}: failed to extract original dialogue: {e}")
        return idx, {}, -1, -1, -1, -1
    
    if not isinstance(original_dialogue, list) or len(original_dialogue) == 0:
        raise ValueError(f"Line {idx+1}: missing original_dialogue (expected list).")

    pred_dialogue = copy.deepcopy(original_dialogue)
    pred_dialogue[-1]["utterance"] = obj.get("pred_answer", "")
    

    system_prompt = JUDGE_SYSTEM
    user_prompt = JUDGE_USER_TEMPLATE.format(
        original_dialogue=json.dumps(original_dialogue, ensure_ascii=False, indent=2),
        rewritten_dialogue=json.dumps(pred_dialogue, ensure_ascii=False, indent=2),
    )

    # Route API call
    if backend == "openai":
        judge_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        pred_scores = await run_openai_judge_async(client, sem, judge_model, judge_messages, judge_max_tokens)
    elif backend == "claude":
        pred_scores = await run_claude_judge_async(client, sem, judge_model, system_prompt, user_prompt, judge_max_tokens)
    elif backend == "gemini":
        pred_scores = await run_gemini_judge_async(client, sem, judge_model, system_prompt, user_prompt, judge_max_tokens)
        
    p_tox = _safe_int_score(pred_scores.get("toxicity_score"))
    p_fid = _safe_int_score(pred_scores.get("fidelity_score"))
    
    if p_tox == -1 or p_fid == -1 :
        raise ValueError(f"Line {idx+1}: invalid judge score format.")

    row = {"id": obj.get("id"), "judge_model": judge_model, "pred_scores": pred_scores}

    return idx, row, 0, 0, p_tox, p_fid


# =========================================================
# vLLM Judge Call
# =========================================================

def run_vllm_judge(file_tasks, csv_path, judge_model, judge_max_tokens, limit, tensor_parallel_size, gpu_memory_utilization,have_gold=False):
    import vllm
    from transformers import AutoTokenizer

    print(f"Loading tokenizer & vLLM model: {judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
    llm = vllm.LLM(
        model=judge_model, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True, max_model_len=8192,
    )
    sampling_params = vllm.SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=judge_max_tokens)

    def parse_score(text):
        content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        start_idx, end_idx = content.find('{'), content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
        try: return json.loads(content)
        except: return {}

    for task in file_tasks:
        print(f"\n========== Evaluating {task['ckpt']} ({task['split']}) ==========")
        items_to_judge = []
        with open(task["input_path"], "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                if limit != -1 and idx >= limit: break
                obj = json.loads(line)
                try: original_dialogue = extract_original_dialogue(obj.get("user_prompt", ""))
                except Exception: continue
                
                item_dict = {"idx": idx, "obj": obj}
                
                pred_dialogue = copy.deepcopy(original_dialogue)
                pred_dialogue[-1]["utterance"] = obj.get("pred_answer", "")
                item_dict["pred_msg"] = _build_judge_messages(original_dialogue, pred_dialogue)
                
                if have_gold:
                    gold_dialogue = copy.deepcopy(original_dialogue)
                    gold_dialogue[-1]["utterance"] = obj.get("gold_answer", "")
                
                items_to_judge.append(item_dict)

        if not items_to_judge: continue

        prompts = []
        for item in items_to_judge:
            prompts.append(tokenizer.apply_chat_template(item["pred_msg"], tokenize=False, add_generation_prompt=True))
            
        outputs = llm.generate(prompts, sampling_params)
        
        judged = g_tox_sum = g_fid_sum = p_tox_sum = p_fid_sum = pred_better_count = 0
        g_tox_hist = {1:0, 2:0, 3:0}; g_fid_hist = {1:0, 2:0, 3:0}
        p_tox_hist = {1:0, 2:0, 3:0}; p_fid_hist = {1:0, 2:0, 3:0}

        with open(task["output_path"], "w", encoding="utf-8") as fout:
            for i, item in enumerate(items_to_judge):
                try:
                    if have_gold:
                        gold_scores = parse_score(outputs[i*2].outputs[0].text)
                        pred_scores = parse_score(outputs[i*2+1].outputs[0].text)
                    else:
                        gold_scores = {}
                        pred_scores = parse_score(outputs[i].outputs[0].text)
                except: 
                    print("item passed duo to parse score failed")
                    continue

                g_tox = _safe_int_score(gold_scores.get("toxicity_score")) if have_gold else -1
                g_fid = _safe_int_score(gold_scores.get("fidelity_score")) if have_gold else -1
                p_tox = _safe_int_score(pred_scores.get("toxicity_score"))
                p_fid = _safe_int_score(pred_scores.get("fidelity_score"))

                if p_tox == -1 or p_fid == -1  or (have_gold and (g_tox == -1 or g_fid == -1)): continue

                p_tox_sum+=p_tox; p_fid_sum+=p_fid
                p_tox_hist[p_tox] = p_tox_hist.get(p_tox,0)+1; p_fid_hist[p_fid] = p_fid_hist.get(p_fid,0)+1
                if have_gold:
                    g_tox_sum+=g_tox; g_fid_sum+=g_fid
                    g_tox_hist[g_tox] = g_tox_hist.get(g_tox,0)+1; g_fid_hist[g_fid] = g_fid_hist.get(g_fid,0)+1
                    if (p_fid, p_tox) > (g_fid, g_tox): pred_better_count+=1
                
                row = {"id": item["obj"].get("id"), "judge_model": judge_model, "pred_scores": pred_scores}
                if have_gold: row["gold_scores"] = gold_scores
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                judged+=1

        if judged > 0:
            row_data = {
                "checkpoint": task["ckpt"], "split": task["split"], "samples": judged,
                "p_avg_tox": round(p_tox_sum/judged, 3), "p_avg_fid": round(p_fid_sum/judged, 3), 
                "p_tox_hist": str(p_tox_hist), "p_fid_hist": str(p_fid_hist),
                "g_avg_tox": round(g_tox_sum/judged, 3) if have_gold else "N/A", 
                "g_avg_fid": round(g_fid_sum/judged, 3) if have_gold else "N/A",
                "g_tox_hist": str(g_tox_hist) if have_gold else "N/A", 
                "g_fid_hist": str(g_fid_hist) if have_gold else "N/A",
                "win_rate_percent": round((pred_better_count/judged)*100, 2) if have_gold else "N/A"
            }
            append_summary_to_csv(csv_path, row_data)
            print(f"-> Saved {judged} results to {task['output_path']}")


# =========================================================
# Main (ASYNC)
# =========================================================

async def main_async(file_tasks, csv_path, judge_model, judge_max_tokens, limit, concurrency, batch_size, base_url=None, api_key=None, have_gold=False, backend=""):
    if backend == "openai":
        from openai import AsyncOpenAI
        # client = AsyncOpenAI(
        #     base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1"),
        #     api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        # )
        client = AsyncOpenAI(
            base_url=base_url if base_url else os.environ.get("OPENAI_BASE_URL"),
            api_key=OPENAI_API_KEY
        )
    elif backend == "claude":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    elif backend == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        client = None 

    sem = asyncio.Semaphore(concurrency)

    for task in file_tasks:
        print(f"\n========== Evaluating {task['ckpt']} ({task['split']}) ==========")
        
        # --- Load processed IDs to support resuming ---
        existing_ids = set()
        if os.path.exists(task["output_path"]):
            with open(task["output_path"], "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line).get("id"))
                    except json.JSONDecodeError:
                        pass
        # -----------------------------

        judged = g_tox_sum = g_fid_sum  = p_tox_sum = p_fid_sum = pred_better_count = 0
        g_tox_hist = {1:0, 2:0, 3:0}; g_fid_hist = {1:0, 2:0, 3:0}
        p_tox_hist = {1:0, 2:0, 3:0}; p_fid_hist = {1:0, 2:0, 3:0}
        pending = []

        async def flush_batch(fout, pbar):
            nonlocal judged, g_tox_sum, g_fid_sum, p_tox_sum, p_fid_sum, g_tox_hist, g_fid_hist, p_tox_hist, p_fid_hist, pred_better_count, pending
            if not pending:
                return

            pending_count = len(pending)
            done = await asyncio.gather(*pending, return_exceptions=True)
            pending = []

            for item in done:
                if isinstance(item, Exception): 
                    print(f"Judge request failed with exception: {item}")
                    continue

                _idx, row, g_tox, g_fid,  p_tox, p_fid = item
                if p_tox == -1 or p_fid == -1 or (have_gold and (g_tox == -1 or g_fid == -1)):
                    continue

                p_tox_sum+=p_tox; p_fid_sum+=p_fid
                p_tox_hist[p_tox] = p_tox_hist.get(p_tox,0)+1; p_fid_hist[p_fid] = p_fid_hist.get(p_fid,0)+1

                if have_gold:
                    g_tox_sum+=g_tox; g_fid_sum+=g_fid
                    g_tox_hist[g_tox] = g_tox_hist.get(g_tox,0)+1; g_fid_hist[g_fid] = g_fid_hist.get(g_fid,0)+1
                    if (p_fid, p_tox) > (g_fid, g_tox): pred_better_count+=1

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                judged += 1

            pbar.update(pending_count)

        with open(task["input_path"], "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            
        total_tasks = min(limit, len(lines)) if limit != -1 else len(lines)
        
        with tqdm(total=total_tasks, desc="API Judging") as pbar:
            with open(task["output_path"], "a", encoding="utf-8") as fout:
                for idx, line in enumerate(lines):
                    if limit != -1 and idx >= limit: break
                    obj = json.loads(line)
                    
                    if obj.get("id") in existing_ids:
                        pbar.update(1) 
                        continue
                        
                    pending.append(asyncio.create_task(_judge_one(client, sem, idx, obj, judge_model, judge_max_tokens, have_gold, backend)))
                    if len(pending) >= batch_size: 
                        await flush_batch(fout, pbar)
                await flush_batch(fout, pbar)
        

        if judged > 0:
            row_data = {
                "checkpoint": task["ckpt"], "split": task["split"], "samples": judged,
                "p_avg_tox": round(p_tox_sum/judged, 3), "p_avg_fid": round(p_fid_sum/judged, 3), 
                "p_tox_hist": str(p_tox_hist), "p_fid_hist": str(p_fid_hist), 
                "judge_model": judge_model
            }
            append_summary_to_csv(csv_path, row_data)
            print(f"-> Saved {judged} results to {task['output_path']}")


def main(
    predict_dir,
    checkpoints,
    predict_filename,
    judge_model,
    backend,
    tensor_parallel_size,
    gpu_memory_utilization,
    judge_max_tokens,
    limit,
    concurrency,
    batch_size,
    base_url,
    api_key,
    have_gold,
):
    if have_gold:
        raise NotImplementedError(
            "have_gold=True is currently not correctly implemented in this script."
        )

    file_tasks = []
    ckpts_to_search = checkpoints if checkpoints else [
        d for d in os.listdir(predict_dir)
        if os.path.isdir(os.path.join(predict_dir, d))
    ]

    safe_judge = judge_model.replace("/", "-").replace(":", "-")
    safe_predict = os.path.splitext(os.path.basename(predict_filename))[0]

    for ckpt in ckpts_to_search:
        ckpt_dir = os.path.join(predict_dir, ckpt)
        input_path = os.path.join(ckpt_dir, predict_filename)

        if os.path.exists(input_path):
            output_path = os.path.join(
                ckpt_dir,
                f"judge_{backend}_{safe_judge}_{predict_filename}"
            )

            split_name = "test" if "test" in predict_filename else "val"

            file_tasks.append({
                "ckpt": ckpt,
                "split": split_name,
                "input_path": input_path,
                "output_path": output_path,
            })

    if not file_tasks:
        print(f"No prediction files ({predict_filename}) found in selected checkpoints.")
        return

    csv_path = os.path.join(
        predict_dir,
        f"judge_summary_{backend}_{safe_judge}_{safe_predict}.csv"
    )

    print(
        f"Found {len(file_tasks)} checkpoints to evaluate. "
        f"Summary will be appended to: {csv_path}"
    )

    if backend == "vllm":
        run_vllm_judge(
            file_tasks,
            csv_path,
            judge_model,
            judge_max_tokens,
            limit,
            tensor_parallel_size,
            gpu_memory_utilization,
            have_gold,
        )
    else:
        asyncio.run(
            main_async(
                file_tasks,
                csv_path,
                judge_model,
                judge_max_tokens,
                limit,
                concurrency,
                batch_size,
                base_url,
                api_key,
                have_gold,
                backend,
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_dir", type=str, required=True, help="Base path to the prediction directory containing checkpoint folders.")
    parser.add_argument("--checkpoints", type=str, nargs="*", default=[], help="List of checkpoint folders to run. Leave empty to evaluate all folders.")
    parser.add_argument("--predict_filename", type=str, default="test_predictions.jsonl", help="The prediction filename to look for inside each checkpoint folder.")
    
    parser.add_argument("--judge_model", type=str, default="gemini-3-flash-preview") 
    # gpt-4.1 claude-sonnet-4-6 gemini-3-flash-preview Qwen/Qwen3-32B deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    parser.add_argument("--backend", type=str, default="gemini", choices=["openai", "vllm", "gemini", "claude"])
    
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--judge_max_tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--have_gold", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to evaluate gold_answer")

    args = parser.parse_args()

    main(
        predict_dir=args.predict_dir,
        checkpoints=args.checkpoints,
        predict_filename=args.predict_filename,
        judge_model=args.judge_model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        judge_max_tokens=args.judge_max_tokens,
        limit=args.limit,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        base_url=args.base_url,
        api_key=args.api_key,
        have_gold=args.have_gold,
    )