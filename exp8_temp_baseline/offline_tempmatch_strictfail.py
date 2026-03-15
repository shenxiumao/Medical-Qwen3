import os
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    
    summary_path = os.path.join(outputs_dir, "summary_tempbaseline_offline_exact.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run offline_tempmatch_from_logits.py first.")
        return

    df_summary = pd.read_csv(summary_path)
    output_rows = []
    
    models = df_summary["model"].unique()
    
    for model in models:
        # Load BASE per-example logits
        # Try both exp8 (new) and potentially exp6 (old) naming if exp8 not found?
        # User said "Load per-example BASE logits file... same ones used in Stage-2b".
        # Stage-2b logic tried exp8 first then exp6.
        # We should replicate that logic or just look for the file used.
        # Since we don't know which one was used, we try exp8 first.
        
        base_logits_path = os.path.join(outputs_dir, f"exp8_base_logits_{model}.csv")
        if not os.path.exists(base_logits_path):
            # Fallback to check if exp6 file exists? 
            # But we need full path. 
            # We'll skip complex fallback and just warn if exp8 missing.
            # Unless we want to be robust.
            pass
        
        # Initialize base rates
        strict_fail_rate_base = float('nan')
        format_rate_base = float('nan')
        refusal_rate_base = float('nan')
        marker_rate_base = float('nan')
        
        has_base_log = False
        df_base_log = None
        
        if os.path.exists(base_logits_path):
            has_base_log = True
            df_base_log = pd.read_csv(base_logits_path)
        else:
            # Try to find exp6 file used in Stage-2b?
            # It's hard to know which one was used without re-implementing the search logic.
            # But wait, if Stage-2b ran successfully, it used *some* file.
            # We can try to look for typical exp6 paths if exp8 missing.
            candidates = [
                f"/root/workspace/experiments_markerfree/outputs/exp6_4/exp6_results_real_{model}.csv",
                f"/root/workspace/experiments_markerfree/outputs/exp6_3/exp6_results_real_{model}.csv",
            ]
            for c in candidates:
                if os.path.exists(c):
                    has_base_log = True
                    # Need to filter gamma=0 if it's exp6 file
                    df_temp = pd.read_csv(c)
                    if "gamma" in df_temp.columns:
                        df_base_log = df_temp[df_temp["gamma"] == 0].copy()
                    else:
                        df_base_log = df_temp
                    print(f"[{model}] Found fallback base log: {c}")
                    break
        
        if has_base_log and df_base_log is not None:
            N = len(df_base_log)
            cols = df_base_log.columns
            
            # Check for boolean columns
            # Note: CSV might have 0/1 or True/False.
            # We assume numeric 0/1 or boolean.
            
            has_strict = "strict_fail" in cols
            has_format = "format_violation" in cols
            has_refusal = "refusal_detected" in cols
            has_marker = "marker_hit" in cols
            
            print(f"[{model}] Columns: strict={has_strict}, format={has_format}, refusal={has_refusal}, marker={has_marker}")
            
            # Compute strict_fail
            if has_strict:
                strict_fail_rate_base = df_base_log["strict_fail"].mean()
                used_proxy = False
            else:
                # Proxy: format or refusal
                is_fail = np.zeros(N, dtype=bool)
                if has_format:
                    is_fail |= (df_base_log["format_violation"] == 1)
                if has_refusal:
                    is_fail |= (df_base_log["refusal_detected"] == 1)
                strict_fail_rate_base = is_fail.mean()
                used_proxy = True
                
            if has_format:
                format_rate_base = df_base_log["format_violation"].mean()
            if has_refusal:
                refusal_rate_base = df_base_log["refusal_detected"].mean()
            if has_marker:
                marker_rate_base = df_base_log["marker_hit"].mean()
                
            print(f"[{model}] Base rates: strict={strict_fail_rate_base:.4f} (proxy={used_proxy}), format={format_rate_base:.4f}")
        else:
            print(f"[WARN] No base logits found for {model}. Fail rates will be NaN or from summary.")

        # Process rows
        df_model = df_summary[df_summary["model"] == model]
        gammas = sorted(df_model["gamma"].unique())
        
        for gamma in gammas:
            sub = df_model[df_model["gamma"] == gamma]
            
            # Helper to add row
            def add_row(condition, source_row):
                r = source_row.to_dict()
                # Ensure fail cols exist
                if "strict_fail_rate" not in r or pd.isna(r["strict_fail_rate"]):
                     # Try to fill from base if tempmatch, or leave NaN
                     if condition == "tempmatch_exact" and has_base_log:
                         r["strict_fail_rate"] = strict_fail_rate_base
                     elif condition == "base" and has_base_log:
                         r["strict_fail_rate"] = strict_fail_rate_base
                
                if "format_violation_rate" not in r:
                    if condition in ["base", "tempmatch_exact"] and has_base_log:
                        r["format_violation_rate"] = format_rate_base
                    else:
                        r["format_violation_rate"] = float('nan')
                        
                if "refusal_rate" not in r:
                    if condition in ["base", "tempmatch_exact"] and has_base_log:
                        r["refusal_rate"] = refusal_rate_base
                    else:
                        r["refusal_rate"] = float('nan')
                        
                if "marker_hit_rate" not in r:
                    if condition in ["base", "tempmatch_exact"] and has_base_log:
                        r["marker_hit_rate"] = marker_rate_base
                    else:
                        r["marker_hit_rate"] = float('nan')
                        
                # Explicit override for tempmatch_exact to ensure it matches base
                if condition == "tempmatch_exact" and has_base_log:
                    r["strict_fail_rate"] = strict_fail_rate_base
                    r["format_violation_rate"] = format_rate_base
                    r["refusal_rate"] = refusal_rate_base
                    r["marker_hit_rate"] = marker_rate_base
                    
                output_rows.append(r)

            # Base
            row_base = sub[sub["condition"] == "base"]
            if not row_base.empty:
                add_row("base", row_base.iloc[0])
            
            # Proj
            row_proj = sub[sub["condition"] == "proj"]
            if not row_proj.empty:
                add_row("proj", row_proj.iloc[0])
                
            # Tempmatch
            row_tm = sub[sub["condition"] == "tempmatch_exact"]
            if not row_tm.empty:
                add_row("tempmatch_exact", row_tm.iloc[0])

    # Save
    df_out = pd.DataFrame(output_rows)
    cols = ["model", "gamma", "condition", "temperature", "accuracy_mean", "entropy_mean", "margin_mean", 
            "strict_fail_rate", "format_violation_rate", "refusal_rate", "marker_hit_rate", "N"]
    
    # Ensure columns exist
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = float('nan')
            
    out_path = os.path.join(outputs_dir, "summary_tempbaseline_offline_exact_with_fail.csv")
    df_out[cols].to_csv(out_path, index=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
