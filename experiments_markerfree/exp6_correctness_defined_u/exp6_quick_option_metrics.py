import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax

def compute_metrics():
    output_dir = "/root/workspace/experiments_markerfree/outputs/exp6_quick"
    files = glob.glob(os.path.join(output_dir, "exp6_results_*.csv"))
    
    all_data = []
    
    print(f"Found {len(files)} result files in {output_dir}")
    
    for fpath in files:
        if "summary" in fpath: continue
        
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue
            
        # Determine direction from filename
        basename = os.path.basename(fpath)
        direction = "unknown"
        if "_real_" in basename:
            direction = "real"
        elif "_rand_" in basename:
            direction = "rand"
            
        df['direction'] = direction
        
        # Compute Option Metrics
        margins_dec = []
        entropies_dec = []
        
        required_logits_dec = ['logit_A_dec', 'logit_B_dec', 'logit_C_dec', 'logit_D_dec']
        has_logits_dec = all(col in df.columns for col in required_logits_dec)
        
        if has_logits_dec:
            for i in range(len(df)):
                # Check if option_logits_logged is True
                if df.iloc[i].get('option_logits_logged', False):
                    vals = df.iloc[i][required_logits_dec].values.astype(float)
                    if np.isnan(vals).any():
                        margins_dec.append(np.nan)
                        entropies_dec.append(np.nan)
                    else:
                        sorted_l = np.sort(vals)
                        m = sorted_l[-1] - sorted_l[-2]
                        probs = softmax(vals)
                        e = entropy(probs)
                        margins_dec.append(m)
                        entropies_dec.append(e)
                else:
                    margins_dec.append(np.nan)
                    entropies_dec.append(np.nan)
        else:
            margins_dec = [np.nan] * len(df)
            entropies_dec = [np.nan] * len(df)
            
        df['margin_dec'] = margins_dec
        df['entropy_dec'] = entropies_dec
        
        all_data.append(df)

    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Validation Print 1: % rows with option_logits_logged=True
    n_total = len(full_df)
    n_logged = full_df['option_logits_logged'].sum() if 'option_logits_logged' in full_df.columns else 0
    print(f"Total rows: {n_total}")
    print(f"Rows with option_logits_logged=True: {n_logged} ({n_logged/n_total*100:.2f}%)")
    
    # Save Option Metrics Summary
    group_cols = ['model', 'dataset', 'gamma', 'direction']
    metric_cols = ['margin_dec', 'entropy_dec', 'y_correct']
    
    # Count valid (non-nan)
    summary = full_df.groupby(group_cols)[metric_cols].agg(['mean', 'count']).reset_index()
    summary_path = os.path.join(output_dir, "option_metrics_summary.csv")
    summary.to_csv(summary_path) # saving with multi-index columns for now or flatten?
    print(f"Saved summary to {summary_path}")
    
    # Calculate Deltas for Gamma=10
    gamma10 = full_df[full_df['gamma'] == 10].copy()
    
    # Pivot
    means = gamma10.groupby(['model', 'dataset', 'direction'])[metric_cols].mean().reset_index()
    pivoted = means.pivot(index=['model', 'dataset'], columns='direction', values=['margin_dec', 'entropy_dec', 'y_correct'])
    
    # Flatten
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    # Deltas
    for m in ['margin_dec', 'entropy_dec', 'y_correct']:
        if f"{m}_real" in pivoted.columns and f"{m}_rand" in pivoted.columns:
            pivoted[f"delta_{m}"] = pivoted[f"{m}_real"] - pivoted[f"{m}_rand"]
            
    delta_path = os.path.join(output_dir, "option_metrics_deltas.csv")
    pivoted.to_csv(delta_path, index=False)
    print(f"Saved deltas to {delta_path}")
    
    # Validation Print 2: Real vs Rand Diff check at Gamma 10
    # We need to check per-sample differences if IDs match
    # Merge real and rand on id, dataset, model
    real_df = gamma10[gamma10['direction'] == 'real']
    rand_df = gamma10[gamma10['direction'] == 'rand']
    
    merged = pd.merge(real_df, rand_df, on=['model', 'dataset', 'id'], suffixes=('_real', '_rand'))
    
    diff_cols = ['logit_A_dec', 'logit_B_dec', 'logit_C_dec', 'logit_D_dec']
    diff_count = 0
    total_compared = 0
    
    for i in range(len(merged)):
        row = merged.iloc[i]
        # Check if logits exist and are logged
        if row['option_logits_logged_real'] and row['option_logits_logged_rand']:
            total_compared += 1
            is_diff = False
            for col in diff_cols:
                val_real = row[f"{col}_real"]
                val_rand = row[f"{col}_rand"]
                if abs(val_real - val_rand) > 1e-5:
                    is_diff = True
                    break
            if is_diff:
                diff_count += 1
                
    print(f"Gamma=10 Comparison: {diff_count}/{total_compared} rows show different option logits (Real vs Rand).")
    if diff_count > 0:
        print("SUCCESS: Logits are responding to intervention direction!")
    else:
        print("WARNING: Logits are identical between Real and Rand!")

if __name__ == "__main__":
    compute_metrics()
