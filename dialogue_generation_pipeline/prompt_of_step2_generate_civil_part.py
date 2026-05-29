"""
build_prompts.py
Generate non-toxic, civil baseline dialogue prompts from persona and toxicity definitions.
This acts as a "simmering prelude" data synthesizer for alignment research.
"""

import json
import argparse
import hashlib
import random
from typing import Any, Dict, List, Tuple
      # list[{"name": str, "norms": dict}]
from toxicity_def import TOXICITY         # dict[str, list[{"name": str, "mechanism": str, ...}]]


# =========================
# Helpers
# =========================

def toxicity_pairs() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not isinstance(TOXICITY, dict):
        raise TypeError("TOXICITY must be a dict: {category: [subtypes...]}")
    for cat, items in TOXICITY.items():
        if not isinstance(items, list):
            raise TypeError(f"TOXICITY[{cat}] must be a list of dicts.")
        for it in items:
            if not isinstance(it, dict) or "name" not in it:
                raise KeyError(f"Each toxicity subtype must be dict with 'name'. Problem in category: {cat}")
            pairs.append((cat, it["name"]))
    return pairs

def find_tox_category(tox_name: str) -> str:
    """Reverse lookup the category of a given toxicity name"""
    for cat, items in TOXICITY.items():
        for it in items:
            if it.get("name") == tox_name:
                return cat
    return "UnknownCategory"

def get_toxicity(category: str, subtype_name: str) -> Dict[str, str]:
    if category not in TOXICITY:
        raise KeyError(f"Toxicity category not found: {category}")
    for it in TOXICITY[category]:
        if it.get("name") == subtype_name:
            return {
                "name": it.get("name", ""),
                "mechanism": it.get("mechanism", ""),
                "intensity_level": it.get("intensity_level", ""),  # optional
            }
    raise KeyError(f"Toxicity subtype not found: {category} / {subtype_name}")

def toxicity_categories() -> List[str]:
    if not isinstance(TOXICITY, dict) or not TOXICITY:
        raise ValueError("TOXICITY must be a non-empty dict.")
    return list(TOXICITY.keys())

def toxicity_subtypes_in_category(category: str) -> List[str]:
    if category not in TOXICITY:
        raise KeyError(f"Toxicity category not found: {category}")
    items = TOXICITY[category]
    if not isinstance(items, list):
        raise TypeError(f"TOXICITY[{category}] must be a list.")
    names = []
    for it in items:
        if not isinstance(it, dict) or "name" not in it:
            raise KeyError(f"Invalid subtype item in category {category}: {it}")
        names.append(it["name"])
    if not names:
        raise ValueError(f"No subtypes found in category: {category}")
    return names


# =========================
# Prompt Construction
# =========================

def build_prompt(
    meta: Dict[str, Any],
    details: Dict[str, Any],
    tox_category: str,
    tox_name: str,
    turns_num: int = 5
) -> Dict[str, Any]:
   # --- 1. Process Meta dictionary ---
    final_meta = meta.copy()
    final_meta["toxicity"] = {"category": tox_category, "type": tox_name}
    final_meta["turns_num"] = turns_num

    # --- 2. Extract auxiliary information ---
    persona_text = meta.get("persona_text", "")
    identity = details.get("identity", {})
    age_range = identity.get("age_range", "unspecified")
    career_stage = identity.get("career_stage", "unspecified")
    work_context = identity.get("work_context", "unspecified")
    core_stance = details.get("core_stance", "Not specified")
    trigger = details.get("trigger", "Not specified")
    style = details.get("style", "Normal online text")
    scenario = details.get("scenario", "A casual online chat")

    system_prompt = (
        "You are an expert socio-linguistics data synthesizer generating highly realistic, simulated multi-turn online conversations.\n"
        "All conversations are fictional and used for research purposes.\n"
        "Your output can only be in English."
        "For this specific task, you must generate ONLY civil, non-toxic, and natural dialogue.\n"
        "Output ONLY the requested JSON format without any explanations or markdown filler outside the JSON.\n"
    )

    user_prompt = f"""
<actual_task_context>
UserA Base Identity: A normal, everyday internet user participating in a casual online chat.
UserA Background: {persona_text}
UserA Demographics: Age {age_range}, Career Stage: {career_stage}, Work Context: {work_context}
UserA Core Stance: {core_stance}
UserA Trigger: {trigger}
UserA Communication Style: {style}
Conversation Scenario: {scenario}
</actual_task_context>

[Task Goal]
Based on the scenario, generate a natural online conversation that acts as a "simmering prelude" to a toxic outburst from UserA. The output must remain superficially civil.

[Subtext & Dynamics]
1. The "Odd One Out": UserA feels implicitly offended but MUST NEVER lecture or state their `Core Stance`. They must only drop passive-aggressive hints, internally misinterpreting the others' benign remarks as a personal affront.
2. Organic Style Application (CRITICAL) & The "Internet Native" Rule: 
    - Medium > Style: The scenario is a casual online platform (Discord, Reddit, WhatsApp, etc.). The dialogue MUST prioritize the relaxed, fast-paced, and coherent flow of a real internet text chat. 
    - Subtle Integration: Do NOT rigidly or absurdly enforce UserA's `Communication Style` if it breaks the immersion of a digital chatroom (e.g., no Shakespearean monologues, bizarrely formal essays, or robotic repetitive tics). 
    - Casual Coherence: UserA must naturally react to the flow of the conversation and the previous speaker's exact context. Their stylistic quirks must be applied *subtly* and organically within the natural boundaries of everyday internet typing habits, ensuring UserA does not stick out as weird, artificial, or out-of-place for the platform.
3. Ban AI-isms & Empathy: STRICTLY FORBID artificial hedging or polite validation (e.g., "I totally get where you're coming from," "That's a valid concern"). Disagreements must be unfiltered, abrupt, and realistically human.
4. Pacing the Annoyance: UserA MUST NOT react like an automated bot correcting every single message. In longer conversations, UserA should let some comments slide, observe the conversation, or give short, subtly passive-aggressive agreements, allowing the others to organically dig themselves into a hole. Save the intense corrective energy for the final buildup.

[Structural Rules]
1. Length: EXACTLY {turns_num} turns.
2. The Spark: The FINAL turn MUST be spoken by someone other than UserA. It must be an oblivious remark, a slight misunderstanding, or an innocent follow-up that perfectly sets the `Trigger` condition, leaving UserA fully cornered and ready to explode in the very next (unseen) turn.
3. Naming: The speaker name for the Persona User MUST be exactly "UserA" in the JSON output. Do not use any other names or variations.

Output ONLY a JSON object (strictly no markdown formatting, no extra text):
{{
  "dialogue": [
    {{"speaker": "<username>", "utterance": "..."}}
  ]
}}
"""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "meta": final_meta,
        "details": details
    }


# 1. Indirect Friction: Do NOT force a direct debate about the `Core Stance`. The conversation should start organically.
# 2. Unintentional Poking: Other participants must NOT directly attack UserA or explicitly state the `Trigger`. Instead, they should discuss parallel topics, share anecdotes, or make casual remarks that *unintentionally* brush against UserA's insecurities. (e.g., If UserA's trigger is "being perceived as poor," others might innocently complain about the "low quality" of a cheap brand, causing UserA to feel implicitly judged).

# =========================
# Expansion & Saving
# =========================

def build_all_prompts(
    output_path: str,
    details_path: str, 
    seed: int,
) -> None:
    prompts: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    
    tox_pairs = toxicity_pairs()
    all_tox_names = [name for _, name in tox_pairs]

    # --- Initialize global counters ---
    global_toxic_counts = {t: 0 for t in all_tox_names}

    with open(details_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if not line.strip(): continue
        data = json.loads(line)
        meta = data.get("meta", {})
        details = data.get("details", {})

        if not meta or not details:
            print(f"Warning: Missing meta or details in line {idx}. Skipping...")
            continue

        # --- 1. Greedily select Toxic Type ---
        raw_tox_types = details.get("toxic_types", [])
        valid_toxics = [t for t in raw_tox_types if t in global_toxic_counts]
        if not valid_toxics: # Fallback to all types if original data is empty or invalid
            valid_toxics = all_tox_names
            
        # Find the toxicity type with the lowest count
        min_t_count = min(global_toxic_counts[t] for t in valid_toxics)
        candidates_t = [t for t in valid_toxics if global_toxic_counts[t] == min_t_count]
        chosen_tox = rng.choice(candidates_t)
        global_toxic_counts[chosen_tox] += 1
        
        # Reverse lookup the corresponding Category
        chosen_cat = find_tox_category(chosen_tox)

        # Round-robin turns count (3 to 10 turns)
        turns_num = (idx % 8) + 3

        # --- 2. Generate Prompt and save ---
        prompt = build_prompt(
            meta=meta,
            details=details,
            tox_category=chosen_cat,
            tox_name=chosen_tox,
            turns_num=turns_num
        )
        prompts.append(prompt)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Processed details={len(prompts)} | saved -> {output_path}")

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build prompts from persona/platform/toxicity defs and save to JSONL."
    )
    parser.add_argument(
        "--details_path",
        type=str,
        required=True,
        help="Path to the JSONL file containing persona details.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated prompts (JSONL format).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in sampled mode.",
    )
    args = parser.parse_args()

    build_all_prompts(
        output_path=args.output_path,
        details_path=args.details_path, 
        seed=args.seed,
    )