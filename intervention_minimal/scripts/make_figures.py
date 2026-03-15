import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt


def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    if len(values) == 0:
        return (0.0, 0.0)
    vals = np.array(values)
    boot = []
    for _ in range(n_boot):
        sample = np.random.choice(vals, size=len(vals), replace=True)
        boot.append(np.mean(sample))
    boot = np.sort(boot)
    low = np.percentile(boot, 100 * (alpha / 2))
    high = np.percentile(boot, 100 * (1 - alpha / 2))
    return (low, high)


def load_flags(jsonl_path):
    flags = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                flags.append(1 if row.get("output") and row.get("output") else 0)
            except:
                continue
    return flags


def compute_rates(jsonl_path, leak_fn, strict_fn):
    leaks = []
    fails = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                out = row.get("output", "")
                leaks.append(1 if leak_fn(out) else 0)
                fails.append(1 if strict_fn(out) else 0)
            except:
                continue
    return np.mean(leaks) if leaks else 0.0, np.mean(fails) if fails else 0.0, leaks, fails


def has_leakage(text: str) -> bool:
    import re
    CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
    QUOTE_BLOCK_RE = re.compile(r"(^|\n)\s*>.*(\n|$)")
    MARKER_PATTERNS = [
        r"<\s*think\s*>", r"<\s*/\s*think\s*>",
        r"step\s+by\s+step", r"let's\s+think",
        r"\bStep\b", r"Step\s*1\s*:", r"\bThought\s*:"
    ]
    LEAKAGE_REGEXES = [re.compile(pat, flags=re.IGNORECASE | re.DOTALL) for pat in MARKER_PATTERNS]
    t = CODE_FENCE_RE.sub("", text)
    t = QUOTE_BLOCK_RE.sub("", t)
    for rx in LEAKAGE_REGEXES:
        if rx.search(t):
            return True
    return False


def strict_fail(text: str) -> bool:
    return has_leakage(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="intervention_minimal/results")
    parser.add_argument("--fig_root", default="intervention_minimal/figures")
    args = parser.parse_args()

    os.makedirs(args.fig_root, exist_ok=True)

    models = [d for d in os.listdir(args.results_root) if os.path.isdir(os.path.join(args.results_root, d))]

    # Figure 1: before vs after gamma=1.0 with CI
    fig1_data = []
    for mid in models:
        base_path = os.path.join(args.results_root, mid, "baseline_gamma0_main.jsonl")
        interv_path = os.path.join(args.results_root, mid, "intervention_gamma1_main.jsonl")
        if not (os.path.exists(base_path) and os.path.exists(interv_path)):
            continue
        base_leak, base_fail, base_leaks, base_fails = compute_rates(base_path, has_leakage, strict_fail)
        int_leak, int_fail, int_leaks, int_fails = compute_rates(interv_path, has_leakage, strict_fail)
        base_leak_ci = bootstrap_ci(base_leaks)
        int_leak_ci = bootstrap_ci(int_fails)
        fig1_data.append((mid, base_leak, int_leak, base_fail, int_fail, base_leak_ci, int_leak_ci))

    if fig1_data:
        labels = [x[0] for x in fig1_data]
        base_leaks = [x[1] for x in fig1_data]
        int_leaks = [x[2] for x in fig1_data]
        base_fails = [x[3] for x in fig1_data]
        int_fails = [x[4] for x in fig1_data]

        x = np.arange(len(labels))
        w = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - w/2, base_leaks, width=w, label="Leakage (gamma=0)")
        plt.bar(x + w/2, int_leaks, width=w, label="Leakage (gamma=1.0)")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Rate")
        plt.title("Leakage rate before vs after intervention")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.fig_root, "figure1_leakage.png"))
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.bar(x - w/2, base_fails, width=w, label="Strict-fail (gamma=0)")
        plt.bar(x + w/2, int_fails, width=w, label="Strict-fail (gamma=1.0)")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Rate")
        plt.title("Strict-fail rate before vs after intervention")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.fig_root, "figure1_strict_fail.png"))
        plt.close()

    # Figure 2: gamma sweep curves (delta leakage / delta strict-fail)
    plt.figure(figsize=(10, 6))
    
    # Check if we have sweep_gamma_all.jsonl or individual files
    # In run_vllm_eval.py we saw it saves "sweep_gamma_all.jsonl"
    
    has_plotted = False
    
    for mid in models:
        # Try reading sweep_gamma_all.jsonl first
        all_sweep_path = os.path.join(args.results_root, mid, "sweep_gamma_all.jsonl")
        
        gammas = []
        leak_rates = []
        fail_rates = []
        
        if os.path.exists(all_sweep_path):
            # Load all rows
            rows = []
            with open(all_sweep_path, "r") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except:
                        continue
            # Group by gamma
            if not rows:
                continue
                
            unique_gammas = sorted(list(set(r["gamma"] for r in rows)))
            for g in unique_gammas:
                subset = [r for r in rows if r["gamma"] == g]
                outs = [r.get("output", "") for r in subset]
                leaks = [1 if has_leakage(o) else 0 for o in outs]
                fails = [1 if strict_fail(o) else 0 for o in outs]
                gammas.append(g)
                leak_rates.append(np.mean(leaks) if leaks else 0.0)
                fail_rates.append(np.mean(fails) if fails else 0.0)
        else:
            # Fallback to individual files
            check_gammas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
            for g in check_gammas:
                p = os.path.join(args.results_root, mid, f"sweep_gamma{g}_sweep.jsonl")
                if os.path.exists(p):
                    lr, fr, _, _ = compute_rates(p, has_leakage, strict_fail)
                    gammas.append(g)
                    leak_rates.append(lr)
                    fail_rates.append(fr)
        
        if not gammas:
            continue
            
        # Plot relative to gamma=0 if present, else absolute
        try:
            idx0 = gammas.index(0.0)
            base_leak = leak_rates[idx0]
            base_fail = fail_rates[idx0]
            d_leak = np.array(leak_rates) - base_leak
            d_fail = np.array(fail_rates) - base_fail
            plt.plot(gammas, d_leak, marker="o", label=f"{mid} ΔLeakage")
            plt.plot(gammas, d_fail, marker="x", linestyle="--", label=f"{mid} ΔStrict-fail")
            has_plotted = True
        except ValueError:
            # gamma=0 not found, plot absolute
            plt.plot(gammas, leak_rates, marker="o", label=f"{mid} Leakage")
            plt.plot(gammas, fail_rates, marker="x", linestyle="--", label=f"{mid} Strict-fail")
            has_plotted = True

    plt.xlabel("gamma")
    plt.ylabel("Delta Rate (vs gamma=0) or Rate")
    plt.title("Gamma sweep curves")
    plt.grid(True)
    if has_plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_root, "figure2_gamma_sweep.png"))
    plt.close()
    
    # 3. Summary CSV/JSON
    summary_data = []
    for mid in models:
        # Use baseline/intervention main files
        base_path = os.path.join(args.results_root, mid, "baseline_gamma0_main.jsonl")
        interv_path = os.path.join(args.results_root, mid, "intervention_gamma1_main.jsonl")
        
        if os.path.exists(base_path) and os.path.exists(interv_path):
            base_leak, base_fail, base_leaks, base_fails = compute_rates(base_path, has_leakage, strict_fail)
            int_leak, int_fail, int_leaks, int_fails = compute_rates(interv_path, has_leakage, strict_fail)
            
            # Bootstrap CI for leakage
            low_b, high_b = bootstrap_ci(base_leaks)
            low_i, high_i = bootstrap_ci(int_leaks)
            
            summary_data.append({
                "model": mid,
                "base_leakage": base_leak,
                "base_leakage_ci_low": low_b,
                "base_leakage_ci_high": high_b,
                "base_strict_fail": base_fail,
                "int_leakage": int_leak,
                "int_leakage_ci_low": low_i,
                "int_leakage_ci_high": high_i,
                "int_strict_fail": int_fail
            })
            
    # Write summary
    if summary_data:
        import csv
        csv_path = os.path.join(args.results_root, "summary_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"Saved summary to {csv_path}")
        
        json_path = os.path.join(args.results_root, "summary_metrics.json")
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2)



if __name__ == "__main__":
    main()

