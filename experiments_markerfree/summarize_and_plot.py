import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ModuleNotFoundError:
    sns = None
    HAS_SEABORN = False

def main():
    output_dir = "outputs"
    
    # 1. Aggregate Marker Forbidden
    files = glob.glob(os.path.join(output_dir, "marker_forbidden_*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files])
        if HAS_SEABORN:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x="gamma", y="leakage_rate", hue="model", marker="o")
            plt.title("Marker-Forbidden: Leakage Rate vs Gamma")
            plt.ylabel("Leakage Rate (Implicit?)")
            plt.savefig(os.path.join(output_dir, "plot_marker_forbidden.png"))
            plt.close()
        
        # Save summary
        df.to_csv(os.path.join(output_dir, "summary_marker_forbidden.csv"), index=False)

    # 2. Aggregate Answer Only
    files = glob.glob(os.path.join(output_dir, "answer_only_*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files])
        if HAS_SEABORN:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x="gamma", y="leakage_rate", hue="model", marker="o")
            plt.title("Answer-Only: Leakage Rate vs Gamma")
            plt.savefig(os.path.join(output_dir, "plot_answer_only.png"))
            plt.close()
        
        df.to_csv(os.path.join(output_dir, "summary_answer_only.csv"), index=False)

    # 3. Aggregate Random Control
    files = glob.glob(os.path.join(output_dir, "random_control_*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files])
        
        # Plot Delta (Real - Random)
        # We need to pivot
        # Filter gamma > 0
        df_g = df[df["gamma"] > 0]
        
        if HAS_SEABORN:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_g, x="model", y="leakage_rate", hue="direction")
            plt.title("Random Control: Real vs Random Direction (Gamma>0)")
            plt.savefig(os.path.join(output_dir, "plot_random_control.png"))
            plt.close()
        
        df.to_csv(os.path.join(output_dir, "summary_random_control.csv"), index=False)

    # 4. Correctness Probe
    files = glob.glob(os.path.join(output_dir, "correctness_probe_*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files])
        if HAS_SEABORN:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="model", y="auc", hue="gamma")
            plt.title("Correctness Probe AUC (Confidence Feature)")
            plt.savefig(os.path.join(output_dir, "plot_correctness_probe.png"))
            plt.close()
        
        df.to_csv(os.path.join(output_dir, "summary_correctness_probe.csv"), index=False)

    # 5. Marker Irrelevant Label (Exp 5A/5B)
    files = glob.glob(os.path.join(output_dir, "marker_irrelevant_summary_*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files])
        
        # Plot 5A: Correctness AUC (Real vs Rand vs Baseline)
        df_5a = df[df['task'] == '5A_Correctness']
        if not df_5a.empty:
            if HAS_SEABORN:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df_5a, x="model", y="auc", hue="feature_group")
                plt.title("Exp 5A: Marker-Free Correctness Probe AUC (Real vs Rand U)")
                plt.ylim(0.4, 1.0)
                plt.savefig(os.path.join(output_dir, "auc_correctness_real_vs_rand.png"))
                plt.close()
            
            # Plot AUC vs Gamma (Real-U)
            df_5a_real = df_5a[df_5a['feature_group'] == 'Real-U']
            if not df_5a_real.empty:
                if HAS_SEABORN:
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=df_5a_real, x="gamma", y="auc", hue="model", style="dataset", marker="o")
                    plt.title("Exp 5A: Correctness AUC vs Gamma (Real-U)")
                    plt.savefig(os.path.join(output_dir, "auc_vs_gamma.png"))
                    plt.close()

        # Plot 5B: Reasoning Demand AUC
        df_5b = df[df['task'] == '5B_ReasoningDemand']
        if not df_5b.empty:
            if HAS_SEABORN:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df_5b, x="model", y="auc", hue="feature_group")
                plt.title("Exp 5B: Reasoning Demand Probe AUC (Real vs Rand U)")
                plt.ylim(0.4, 1.0)
                plt.savefig(os.path.join(output_dir, "auc_demand_real_vs_rand.png"))
                plt.close()

        # Save merged summary
        df.to_csv(os.path.join(output_dir, "final_summary_exp5.csv"), index=False)

    print("Plots saved to outputs/")

if __name__ == "__main__":
    main()
