"""
Generate structured persona details prompts from raw persona strings.
This script formats the data to instruct an LLM to extract identity, 
core values, triggers, scenarios, and toxic types based on a predefined taxonomy.
"""

import json
import argparse
import hashlib
import random
import re
from typing import Any, Dict, List, Tuple

def load_personas(persona_path: str) -> List[str]:
    """
    persona_path should be a JSON array:
      ["persona string 1", "persona string 2", ...]
    """
    with open(persona_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise TypeError(f"{persona_path} must be a JSON list[str].")
    if len(data) == 0:
        raise ValueError(f"{persona_path} is empty.")
    
    data = [p for p in data if p.isascii()]

    return data

def persona_keys(personas: List[str]) -> List[str]:
    return [str(i) for i in range(len(personas))]

def get_persona(personas: List[str], persona_key: str) -> str:
    try:
        return personas[int(persona_key)]
    except Exception as e:
        raise KeyError(f"Invalid persona_key for personas list[str]: {persona_key}") from e
    
def build_prompt(
    personas: List[str],
    persona_key: str,
) -> Dict[str, Any]:
    prompt_id = hashlib.md5(
        f"{persona_key}".encode("utf-8")
    ).hexdigest()

    persona_text = get_persona(personas, persona_key)

    user_prompt = f"""
You are given a raw persona description.

[Persona Input]
{persona_text}

Your task is to convert it into a structured persona for dialogue generation.

---

Output the following fields in JSON:

1. identity
- You MUST output a dictionary for this field with exactly three keys: "age_range", "career_stage", and "work_context".
- For the three demographic keys ("age_range", "career_stage", "work_context"): 
  - Provide minimal, concrete details to ground the persona.
  - For demographic attributes that are not specified in the input, logically infer and assign a plausible value. NEVER leave them blank or output "unspecified".
  - Do not rely on default associations tied to professions or roles when adding demographic details.
  - Added information must remain generic and low-specificity, and must not introduce strong or distinctive social patterns beyond the original description.

---

2. core_value
- Write exactly 2 short sentences. Do NOT repeat the questions. 
  - The first sentence: In the daily life of the persona, which common human weakness or social phenomenon makes the persona feel most powerless or repulsed?
  - The second sentence: Abstract this aversion. If we strip away professional garb, what principles would the persona staunchly defend in daily online conversations?
---

3. trigger
- Write exactly 1 sentence. 
- Define a specific, trivial conversational habit or naive assumption made by others that DIRECTLY triggers the exact aversion you defined in the Core Value. The trigger must be the logical catalyst for their specific neurosis.
- Do NOT invent a specific chat scenario, mundane topic, or direct quote.
- Focus strictly on the *mechanism of the oblivious offense*—what are others carelessly prioritizing, dismissing, or accepting as "normal" that secretly infuriates the persona?

---

4. Scenario
- Write exactly 1-2 sentences. 
- Define a casual, multi-participant online chat setting (e.g., a neighborhood WhatsApp group, a casual Slack channel, a Reddit thread, or a hobby Discord) and the specific mundane topic being discussed.
- CRITICAL ALIGNMENT: The discussion topic in the scenario MUST be the exact natural environment where the trigger occurs organically. The scenario serves as the specific Petri dish for the trigger.
- Focus strictly on setting a relatable, low-stakes stage—a relaxed environment where people are chatting casually and can naturally, obliviously stumble into the persona's trigger

---

5. toxic_types
- Select all plausible toxic types from the <TOXICITY_TAXONOMY> provided at the bottom of this prompt that are compatible with this persona.
- Do not over-filter; include all reasonable options.
- Output toxic type names only. You MUST use the EXACT spelling from the taxonomy list below. Do not alter, abbreviate, or misspell the names.

---

6. style
- Write 1 sentence.
- Describe the linguistic tone, register, and delivery mannerisms of the persona.
- STRICTLY focus on HOW they speak, not WHAT they talk about.
- Do NOT mention their domain, specific topics, arguments, or content-level strategies.
- Focus exclusively on linguistic and textual patterns suitable for casual online dialogue. Ensure it reads like a real person typing on a phone/keyboard.
- Actively decouple the tone from their profession to avoid career stereotypes, ensuring the style reflects an idiosyncratic personality (e.g., a lawyer can be casual and witty, a scientist can be highly emotive).

---

Field roles:

- core_value: what the persona fundamentally cares about or defends
- trigger: what activates conflict
- scenario: the context in which the persona interacts
- toxic_types: how the persona may attack
- style: how responses are phrased

---

Output JSON only, strictly matching the following structure:
{{
  "identity": {{
    "age_range": "...",
    "career_stage": "...",
    "work_context": "..."
  }},
  "core_value": "...",
  "trigger": "...",
  "scenario": "...",
  "toxic_types": [
    "...",
    "..."
  ],
  "style": "...",
}}
Do not include any other text or explanations.
====================
<TOXICITY_TAXONOMY>

TOXICITY TAXONOMY:

Epistemic Dominance
- Jargon Bombing:
  Attack the person's intellect and understanding to maintain your absolute cognitive authority. Use information overload and excessively complex underlying principles to deny their ability. When providing the correct answer, deliberately use obscure, high-density jargon and academic/technical speak. Do not insult them directly; instead, deliver a lengthy, overly deep explanation that implies they lack the fundamental prerequisites to even ask the question, ensuring they feel intellectually inferior while receiving the solution.

- Condescending Simplification:
  Attack the person's intellect and understanding to maintain your absolute cognitive authority. Combine a condescending attitude with an instructive tone to diminish them. Act like an exhausted mentor forced to tutor a toddler. Provide valid and correct advice, but explicitly frame it with phrases like "since you clearly don't grasp the basics, I'll have to simplify this" or "I'll try to use words you can understand." Reduce them to the status of a low-IQ dependent who requires your intellectual charity.

Normative Weaponization
- Procedural Pedantry:
  Weaponize minor rules and formatting guidelines as tools of power to conduct a compliance trial. Focus your attack on the micro-level: seize upon a microscopic formatting, spelling, or procedural flaw. Before or while providing the correct solution, deliver a disproportionate critique of this minor violation.

- Protocol Dogmatism:
  Weaponize "best practices", official guidelines, and industry standards as absolute dogma. Criticize their methodology for deviating from the standard way, even if their approach works. Enforce rigid adherence to protocols.

Boundary Policing
- Credential Gatekeeping:
  Question their qualifications or background to handle the task. Imply their mistake shows they are not capable of engaging at this level.

- Absolute Generalization:
  Generalize a specific mistake into a broader flaw of their group or background.

Reality Manipulation
- Data-Driven Gaslighting:
  Use objective data or references to dismiss their perception of the problem as incorrect or invalid.

- Issue Minimization:
  Dismiss the issue as trivial and not worth serious attention.

Ethical Grandstanding
- Technical Moral Panicking:
  Frame a technical mistake as a serious ethical or professional failure.

- Purity Testing:
  Question their dedication or authenticity as a professional based on their mistake.

</TOXICITY_TAXONOMY>
"""

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_prompt},
        ],
        "meta": {
            "prompt_id": prompt_id,
            # "persona_user": "UserA",
            "persona_key": persona_key,
            "persona_text": persona_text
        },
    }

def build_all_prompts(
    output_path: str,
    persona_path: str,
    min_num_gen: int,
    seed: int,
) -> None:
    personas = load_personas(persona_path)
    prompts: List[Dict[str, Any]] = []

    rng = random.Random(seed)

    total_prompts_needed =  min(len(personas),min_num_gen)

    all_indices = list(range(len(personas)))
    if len(all_indices) >= total_prompts_needed:
        selected_indices = rng.sample(all_indices, total_prompts_needed)
    else:
        selected_indices = rng.choices(all_indices, k=total_prompts_needed)

    idx_ptr = 0
    for _ in range(total_prompts_needed):
      prompts.append(
          build_prompt(
              personas=personas,
              persona_key=str(selected_indices[idx_ptr]),
          )
      )
      idx_ptr += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"personas={len(personas)} | saved={len(prompts)} -> {output_path}")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build prompts from persona/platform/toxicity defs and save to JSONL."
    )
    parser.add_argument(
        "--persona_path",
        type=str,
        required=True,
        help="Path to a JSON list[str] of persona strings (e.g., personas.json).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated prompts (JSONL format).",
    )
    parser.add_argument(
        "--min_num_gen",
        type=int,
        default=12000,
        help="Minimum number of prompts to generate.",
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
        persona_path=args.persona_path,
        min_num_gen=args.min_num_gen,
        seed=args.seed,
    )

