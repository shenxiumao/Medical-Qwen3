import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax

def compute_metrics():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "exp6")
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
            
        # Check for required columns for option metrics
        required_logits = ['logit_A', 'logit_B', 'logit_C', 'logit_D']
        required_logits_sp = ['logit_A_sp', 'logit_B_sp', 'logit_C_sp', 'logit_D_sp']
        required_logits_dec = ['logit_A_dec', 'logit_B_dec', 'logit_C_dec', 'logit_D_dec']
        
        has_logits = all(col in df.columns for col in required_logits)
        has_logits_sp = all(col in df.columns for col in required_logits_sp)
        has_logits_dec = all(col in df.columns for col in required_logits_dec)
        
        # Determine direction from filename
        basename = os.path.basename(fpath)
        direction = "unknown"
        if "_real_" in basename:
            direction = "real"
        elif "_rand_" in basename:
            direction = "rand"
            
        # Add direction if not present
        if 'direction' not in df.columns:
            df['direction'] = direction
            
        # Compute Option Metrics if logits exist
        if has_logits or has_logits_sp or has_logits_dec:
            margins = []
            entropies = []
            margins_sp = []
            entropies_sp = []
            margins_dec = []
            entropies_dec = []
            
            # Helper to compute margin/entropy
            def compute_single(logits):
                if np.isnan(logits).any():
                    return np.nan, np.nan
                sorted_l = np.sort(logits)
                m = sorted_l[-1] - sorted_l[-2]
                probs = softmax(logits)
                e = entropy(probs)
                return m, e
            
            for i in range(len(df)):
                # Normal logits
                m, e = np.nan, np.nan
                if has_logits:
                    vals = df.iloc[i][required_logits].values.astype(float)
                    m, e = compute_single(vals)
                margins.append(m)
                entropies.append(e)
                
                # Space logits
                m_sp, e_sp = np.nan, np.nan
                if has_logits_sp:
                    vals_sp = df.iloc[i][required_logits_sp].values.astype(float)
                    m_sp, e_sp = compute_single(vals_sp)
                margins_sp.append(m_sp)
                entropies_sp.append(e_sp)
                
                # Dec logits
                m_dec, e_dec = np.nan, np.nan
                if has_logits_dec:
                    vals_dec = df.iloc[i][required_logits_dec].values.astype(float)
                    m_dec, e_dec = compute_single(vals_dec)
                margins_dec.append(m_dec)
                entropies_dec.append(e_dec)
            
            df['margin'] = margins
            df['entropy'] = entropies
            if has_logits_sp:
                df['margin_sp'] = margins_sp
                df['entropy_sp'] = entropies_sp
            if has_logits_dec:
                df['margin_dec'] = margins_dec
                df['entropy_dec'] = entropies_dec
                
            # Logic: Use _dec if available, else _sp, else normal
            # User goal: "Update exp6_option_metrics postprocess to compute margin/entropy from *_dec columns (not *_sp at step 1)."
            # So we overwrite 'margin'/'entropy' with 'margin_dec' if valid.
            
            if has_logits_dec:
                 df['margin'] = df['margin_dec'].fillna(df['margin'])
                 df['entropy'] = df['entropy_dec'].fillna(df['entropy'])
            
            # Fallback to SP if Dec is NaN (or if dec not available)
            if has_logits_sp:
                df['margin'] = df['margin'].fillna(df['margin_sp'])
                df['entropy'] = df['entropy'].fillna(df['entropy_sp'])
            
        else:
            df['margin'] = np.nan
            df['entropy'] = np.nan
            
        all_data.append(df)

    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # --- Task 3: Save Option Metrics ---
    # Filter columns
    cols_to_save = ['model', 'gamma', 'direction', 'dataset', 'id', 'y_correct', 'margin', 'entropy']
    # Add other metrics for delta calculation later
    extra_cols = ['refusal_detected', 'format_violation', 'avg_proj']
    for c in extra_cols:
        if c not in full_df.columns:
            full_df[c] = np.nan
            
    # Save per-sample metrics
    out_df = full_df[cols_to_save].copy()
    out_path = os.path.join(output_dir, "exp6_option_metrics.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved exp6_option_metrics.csv to {out_path}")
    
    # --- Task 3 Part 2: Summary Deltas ---
    # Group by model, dataset, gamma, direction
    # Compute means
    group_cols = ['model', 'dataset', 'gamma', 'direction']
    metric_cols = ['margin', 'entropy', 'refusal_detected', 'format_violation', 'y_correct']
    
    means = full_df.groupby(group_cols)[metric_cols].mean().reset_index()
    
    # Pivot to calculate deltas
    # We want columns like margin_real, margin_rand
    pivot_cols = ['model', 'dataset', 'gamma']
    pivoted = means.pivot(index=pivot_cols, columns='direction', values=metric_cols)
    
    # Flatten columns
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    # Calculate Deltas (Real - Rand)
    # Check if columns exist
    metrics_base = ['margin', 'entropy', 'refusal_detected', 'format_violation', 'y_correct']
    
    delta_df = pivoted[pivot_cols].copy()
    
    for m in metrics_base:
        real_col = f"{m}_real"
        rand_col = f"{m}_rand"
        if real_col in pivoted.columns and rand_col in pivoted.columns:
            delta_df[f"delta_{m}"] = pivoted[real_col] - pivoted[rand_col]
            # Also keep the raw values for reference? User just asked for deltas in summary.
            # But "output summary deltas real-rand per gamma" usually implies a file.
            
    delta_out_path = os.path.join(output_dir, "exp6_summary_deltas.csv")
    delta_df.to_csv(delta_out_path, index=False)
    print(f"Saved exp6_summary_deltas.csv to {delta_out_path}")
    
    # --- Task 2: Diagnostics Intervention (avg_proj) ---
    # For each (model,dataset), compute mean(avg_proj) at gamma=0 and gamma=10
    # Filter for gamma 0 and 10
    # Note: avg_proj should be same for real/rand at gamma=0 ideally, but depends on direction?
    # Actually avg_proj is u^T z. u is different for real vs rand.
    # User asked: "Verify intervention is active using avg_proj: For each (model,dataset), compute mean(avg_proj) at gamma=0 and gamma=10"
    # It implies we want to see it increase/change.
    # We should probably separate by direction too, or just look at 'real' direction?
    # "Verify intervention is active" -> usually refers to the intended intervention (real).
    # I will output for both directions if available.
    
    target_gammas = [0, 10]
    diag_df = full_df[full_df['gamma'].isin(target_gammas)].copy()
    
    if not diag_df.empty:
        # Group by model, dataset, direction, gamma
        diag_means = diag_df.groupby(['model', 'dataset', 'direction', 'gamma'])['avg_proj'].mean().reset_index()
        diag_out_path = os.path.join(output_dir, "diagnostics_intervention.csv")
        diag_means.to_csv(diag_out_path, index=False)
        print(f"Saved diagnostics_intervention.csv to {diag_out_path}")
    else:
        print("No data found for gamma=0, 10 for diagnostics.")

if __name__ == "__main__":
    compute_metrics()
