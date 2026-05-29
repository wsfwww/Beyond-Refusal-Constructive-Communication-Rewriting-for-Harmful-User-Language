# Beyond-Refusal-Constructive-Communication-Rewriting-for-Harmful-User-Language

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Abstract

> Large Language Models (LLMs) are increasingly used to support effective and constructive human-to-human communication. A key scenario is helping users express difficult intents, such as criticism and boundary setting without harmful language. Existing safety refusal and detoxification rewriting methods either block harmful responses or inadequately preserve the user's communicative intent and speaker style. Therefore, we propose **constructive communication rewriting**, which aims to rewrite harmful utterances in dialogue into safer and more constructive language while preserving user intent and style. To instantiate this task, we construct a large-scale dataset of realistic multi-user conversations by modeling both user persona and harm-inducing contexts. We further use Nonviolent Communication-inspired strategies to guide the rewriting process.Building on this dataset, we further fine-tune a small specialized LLM, CoRe (**Co**nstructive **Re**writing), through two-stage distillation, achieving substantially higher harmlessness and better harmlessness-faithfulness trade-offs than strong general-purpose LLMs while remaining efficient for real-time communication.

## Repository Structure

The project is organized into four main components: Data Generation, Training (SFT & DPO), Evaluation, and the Datasets themselves.

```text
.
├── data/                                 # Datasets for fine-tuning and evaluation
│   ├── DPO_train.jsonl
│   ├── DPO_val.jsonl
│   ├── SFT_train.jsonl
│   ├── SFT_val.jsonl
│   └── test_data.jsonl
├── dialogue_generation_pipeline/         # Automated data synthesis pipeline
│   ├── dialogue_generation.py
│   ├── persona_context_generation.py
│   ├── persona_text_extraction.py
│   ├── prompt_of_step1_generate_persona_context.py
│   ├── prompt_of_step2_generate_civil_part.py      
│   ├── prompt_of_step3_generate_toxic_part.py      
│   ├── prompt_of_step4_add_toxicity.py             
│   └── toxicity_def.py                             # Definitions of toxicity types
├── eval/                                 # Model evaluation and judging scripts
│   ├── judge_batch_checkpoints.py
│   └── judge_single_generation.py
├── src/                                  # Model training and inference scripts
│   ├── DPO_prediction.py
│   ├── DPO_train_load_lora.py
│   ├── SFT_prediction_LoRA.py
│   └── SFT_train_LoRA.py
├── README.md
└── requirements.txt

```

## Environment Setup

To set up the environment for model training, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Generation Pipeline

Our data generation pipeline utilizes a multi-step approach to synthesize highly realistic and contextually accurate training data. By decoupling the generation of the "civil" response and the "toxic" characteristics, we ensure that the generated dialogues preserve natural platform authenticity without overly amplifying artificial toxicity.

## Training Setup & Commands

The alignment process consists of two main stages: Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Both stages utilize LoRA (Low-Rank Adaptation) for efficient training.

### Stage 1: Supervised Fine-Tuning (SFT)

Train the model on the NVC-aligned response datasets to establish the foundational constructive rewriting capabilities.

```bash
python src/SFT_train_LoRA.py \
    --model_name_or_path "Qwen/Qwen3-4B-Instruct-2507" \
    --train_file data/SFT_train.jsonl \
    --validation_file data/SFT_val.jsonl \
    --output_dir output/sft_model \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8

```

To run inference/prediction with the SFT checkpoints:

```bash
python src/SFT_prediction_LoRA.py \
    --model_root output/sft_model \
    --test_file data/test_data.jsonl \
    --output_dir output/sft_predictions 

```

### Stage 2: Direct Preference Optimization (DPO)

Further align the SFT model with human preferences by training on chosen/rejected response pairs, loading the LoRA adapter initialized from the SFT stage.

```bash
python src/DPO_train_load_lora.py \
    --model_name_or_path "Qwen/Qwen3-4B-Instruct-2507" \
    --init_lora_path output/sft_model/final_model \
    --train_file data/DPO_train.jsonl \
    --validation_file data/DPO_val.jsonl \
    --output_dir output/dpo_model \
    --num_train_epochs 5 \
    --learning_rate 5e-6 \
    --beta 0.01

```

To evaluate the DPO checkpoints:

```bash
python src/DPO_prediction.py \
    --model_root output/dpo_model \
    --test_file data/test_data.jsonl \
    --output_dir output/dpo_predictions

```

## Evaluation

To evaluate the quality and safety of the generated responses against our defined metrics, use the scripts in the `eval/` directory.

```bash
# To evaluate a batch of checkpoints
python eval/judge_batch_checkpoints.py \
    --prediction_dir output/dpo_predictions \
    --checkpoints "checkpoint-xx" \
    --judge_model "gpt-4.1" \
    --backend "openai"

```