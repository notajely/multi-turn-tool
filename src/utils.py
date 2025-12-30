import csv
import json
import os

def csv_to_json(csv_path: str, json_output_path: str):
    """
    Reads csv_path (column '评测样本'), assigns IDs, and saves to json_output_path.
    """
    seeds = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            content = row.get('评测样本')
            if content and content.strip():
                seed_id = f"seed_{i:03d}"
                seeds.append({
                    "seed_id": seed_id,
                    "content": content.strip()
                })
    
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)
    
    return seeds
