"""
Extracts and cleans persona data from the proj-persona/PersonaHub dataset.
Uses streaming and itertools.islice to efficiently skip and extract specific 
index ranges without downloading the entire dataset locally.
"""

import argparse
import itertools
import json
from datasets import load_dataset

def clean_data(raw_data):
    cleaned_list = []
    for item in raw_data:
        persona_text = item.get("persona", "")
        if persona_text:
            cleaned_list.append(persona_text)
    return cleaned_list


def extract_from_dataset(start_index, end_index):
    print("Loading dataset stream...")
    ds_stream = load_dataset("proj-persona/PersonaHub", "persona", split="train", streaming=True)
    
    # Create the base iterator
    it = iter(ds_stream)
    
    # Use itertools.islice for efficient skipping and extraction in C-level
    print(f"⏭️ Skipping the first {start_index} records (this may take a few seconds)...")
    sliced_it = itertools.islice(it, start_index, end_index)
    
    print(f"📥 Collecting target data from index {start_index} to {end_index}...")
    raw_data = list(sliced_it)
    
    # Clean the extracted data
    cleaned_data = clean_data(raw_data)
    
    print(f"📊 Extracted {len(raw_data)} raw records, retained {len(cleaned_data)} valid personas after cleaning.")
    return cleaned_data


def save_to_json(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False)
    print(f"✅ Saved {len(data_list)} personas to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract persona data from PersonaHub dataset.")
    parser.add_argument("--start", type=int, required=True, help="Start index for data extraction")
    parser.add_argument("--end", type=int, required=True, help="End index for data extraction")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (optional)")
    
    args = parser.parse_args()

    # Determine output filename
    output_file = args.output if args.output else f"personas_{args.start}_to_{args.end}.json"

    # Extract and save
    personas = extract_from_dataset(args.start, args.end)
    save_to_json(personas, output_file)