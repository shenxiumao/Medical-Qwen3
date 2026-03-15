import argparse
import json
import os
import glob
from tqdm import tqdm

def merge_shards(output_dir):
    pattern = os.path.join(output_dir, "predictions_gpu*_shard*.jsonl")
    files = glob.glob(pattern)
    print(f"Found {len(files)} shard files.")
    
    merged_path = os.path.join(output_dir, "predictions_merged.jsonl")
    
    seen_keys = set()
    total_lines = 0
    duplicates = 0
    
    with open(merged_path, 'w') as out_f:
        for fp in tqdm(files, desc="Merging files"):
            with open(fp, 'r') as in_f:
                for line in in_f:
                    try:
                        data = json.loads(line)
                        # Unique key: id + model + template + preset
                        key = (data['id'], data['model_id'], data['template_name'], data['preset_name'])
                        
                        if key in seen_keys:
                            duplicates += 1
                            continue
                        
                        seen_keys.add(key)
                        out_f.write(line)
                        total_lines += 1
                    except Exception as e:
                        print(f"Error reading line in {fp}: {e}")
                        continue
                        
    print(f"Merged {total_lines} unique records to {merged_path}")
    print(f"Skipped {duplicates} duplicates.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    
    merge_shards(args.output_dir)
