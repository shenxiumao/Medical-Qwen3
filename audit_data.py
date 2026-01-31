import json
import os
import glob
from collections import Counter
import hashlib

DATA_DIR = "/root/workspace/train/data/medical/finetune"
DATASETS = {
    "F3 (Train En)": "train_en_1.json",
    "F4 (Train Zh)": "train_zh_0.json",
    "F5 (Valid En)": "valid_en_1.json",
    "F6 (Valid Zh)": "valid_zh_0.json"
}

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Maybe jsonl?
            f.seek(0)
            return [json.loads(line) for line in f]

def get_hash(text):
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

def analyze_datasets():
    print("| Dataset | Samples | Avg Input Len | Avg Output Len | Source Distribution |")
    print("| :--- | ---: | ---: | ---: | :--- |")
    
    hashes = {} # dataset_name -> set of hashes
    
    for name, filename in DATASETS.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"| {name} | Not Found | - | - | - |")
            continue
            
        data = load_data(path)
        samples = len(data)
        
        input_lens = []
        output_lens = []
        sources = Counter()
        dataset_hashes = set()
        
        for item in data:
            # Assuming 'instruction'/'input' and 'output' fields
            inp = (item.get('instruction', '') + item.get('input', '')).strip()
            out = item.get('output', '').strip()
            
            input_lens.append(len(inp))
            output_lens.append(len(out))
            
            # Simple source heuristic if not present
            src = "unknown"
            if 'source' in item:
                src = item['source']
            # Try to guess from content if source missing? Or just skip
            sources[src] += 1
            
            dataset_hashes.add(get_hash(inp + out))
            
        avg_in = sum(input_lens) / len(input_lens) if input_lens else 0
        avg_out = sum(output_lens) / len(output_lens) if output_lens else 0
        
        # Top 3 sources
        top_src = ", ".join([f"{k}({v})" for k, v in sources.most_common(3)])
        
        print(f"| {name} | {samples} | {avg_in:.1f} | {avg_out:.1f} | {top_src} |")
        
        hashes[name] = dataset_hashes

    print("\n## Leakage Check (Exact Match Hash)")
    train_hashes = hashes.get("F3 (Train En)", set()) | hashes.get("F4 (Train Zh)", set())
    valid_hashes_en = hashes.get("F5 (Valid En)", set())
    valid_hashes_zh = hashes.get("F6 (Valid Zh)", set())
    
    leak_en = len(train_hashes.intersection(valid_hashes_en))
    leak_zh = len(train_hashes.intersection(valid_hashes_zh))
    
    print(f"- Train (F3+F4) vs Valid En (F5): {leak_en} / {len(valid_hashes_en)} ({leak_en/len(valid_hashes_en)*100:.2f}%) overlap")
    print(f"- Train (F3+F4) vs Valid Zh (F6): {leak_zh} / {len(valid_hashes_zh)} ({leak_zh/len(valid_hashes_zh)*100:.2f}%) overlap")

if __name__ == "__main__":
    analyze_datasets()
