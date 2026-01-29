import json
import sys

notebooks = [
    'notebooks/03_ground_truth_generation.ipynb',
    'notebooks/04_blocking_strategies.ipynb',
    'notebooks/05_record_linkage_rules.ipynb'
]

print("Verifying notebooks...")
print("=" * 60)

for nb_path in notebooks:
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ {nb_path}: {len(data['cells'])} celle - OK")
    except Exception as e:
        print(f"✗ {nb_path}: ERRORE - {e}")

print("=" * 60)
