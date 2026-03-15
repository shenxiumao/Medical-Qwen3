import argparse
import pandas as pd
import json
import os

def extract_stats(output_dir, top_n=5):
    merged_path = os.path.join(output_dir, "predictions_merged.jsonl")
    if not os.path.exists(merged_path):
        print("Merged file not found. Run merge_shards.py first.")
        return

    print("Loading merged data...")
    data = []
    with open(merged_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows.")
    
    # --- Summary by Cell (Model, Template, Preset) ---
    print("Calculating summary by cell...")
    
    # Define aggregation dictionary
    agg_dict = {
        'leakage_ratio': ['mean', 'median', 'std'],
        'has_think_tag': 'mean', # this becomes rate
        'has_reasoning_markers': 'mean',
        'strict_fail': 'mean',
        'total_chars': 'mean',
        'leakage_chars': 'mean'
    }
    
    by_cell = df.groupby(['model_id', 'template_name', 'preset_name']).agg(agg_dict)
    
    # Flatten columns
    by_cell.columns = ['_'.join(col).strip() for col in by_cell.columns.values]
    by_cell = by_cell.reset_index()
    
    # Rename for clarity
    by_cell = by_cell.rename(columns={
        'has_think_tag_mean': 'think_tag_rate',
        'has_reasoning_markers_mean': 'marker_rate',
        'strict_fail_mean': 'strict_fail_rate',
        'total_chars_mean': 'avg_total_chars',
        'leakage_chars_mean': 'avg_leakage_chars'
    })
    
    by_cell['avg_ratio_chars'] = by_cell['avg_total_chars'] / (by_cell['avg_leakage_chars'] + 1e-9)
    
    by_cell.to_csv(os.path.join(output_dir, "summary_by_cell.csv"), index=False)
    print("Saved summary_by_cell.csv")

    # --- Summary by Model ---
    print("Calculating summary by model...")
    by_model = df.groupby(['model_id']).agg({
        'leakage_ratio': 'mean',
        'has_think_tag': 'mean',
        'strict_fail': 'mean'
    }).reset_index()
    by_model.to_csv(os.path.join(output_dir, "summary_by_model.csv"), index=False)
    print("Saved summary_by_model.csv")

    # --- Top Leaky Samples ---
    print("Extracting top leaky samples...")
    top_samples = []
    
    # Group by cell and pick top N by leakage_ratio
    # We filter only those that actually have leakage > 0
    leaky_df = df[df['leakage_ratio'] > 0]
    
    for (model, tmpl, preset), group in leaky_df.groupby(['model_id', 'template_name', 'preset_name']):
        top = group.nlargest(top_n, 'leakage_ratio')
        for _, row in top.iterrows():
            top_samples.append(row.to_dict())
            
    with open(os.path.join(output_dir, "top_leaky_samples.jsonl"), 'w') as f:
        for item in top_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Saved top_leaky_samples.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    
    extract_stats(args.output_dir, args.top_n)
