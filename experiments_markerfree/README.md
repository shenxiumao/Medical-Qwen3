# Marker-Free / Robustness-to-Style Experimental Suite

This suite addresses the critique that our intervention merely localizes a formatting/style subspace (e.g., presence of `<think>` tags) rather than a reasoning subspace.

## Experiments

1.  **Marker-Forbidden Decoding**: We enforce a "forbidden" constraint on reasoning markers (via stop tokens/logit bias) in BOTH baseline and intervention runs. If the intervention (gamma) still affects outcomes (e.g., correctness or implicit leakage), it suggests the intervention targets a latent reasoning process, not just surface markers.
2.  **Answer-Only Prompt**: We use prompts that explicitly demand short answers without reasoning. This removes marker supervision from the task definition.
3.  **Random-Direction Control**: We compare our learned reasoning direction against random directions in the logit space to prove causal specificity.
4.  **Marker-Free Correctness Probe**: We train a classifier on scalar features (Confidence) that do not depend on marker presence, showing that the intervention improves the alignment between model confidence and correctness.
5.  **Marker-Irrelevant Labels (Exp 5A/5B)**: 
    *   **5A (Correctness Probe)**: We use Answer-Only prompts (markers forbidden) on MMLU/MedQA. We extract features from the latent direction `u` (e.g., `u^T z`) and random `u_rand`. We show that `u`-features predict *correctness* (AUC) significantly better than random features, even when markers are absent.
    *   **5B (Reasoning Demand)**: We split questions into "High Reasoning Demand" vs "Low Demand" (using heuristics). We show that `u`-features can discriminate between these task types better than chance/random, proving the direction encodes computational demand.

## Usage

All scripts run with vLLM (TP=4) using local models.

### 1. Run Experiments
Execute the scripts for each model alias. Example for `Qwen2.5-7B-Instruct`:

```bash
# Experiment 1: Marker Forbidden
python experiments_markerfree/run_marker_forbidden.py --model_alias Qwen2.5-7B-Instruct

# Experiment 2: Answer Only
python experiments_markerfree/run_answer_only.py --model_alias Qwen2.5-7B-Instruct

# Experiment 3: Random Control
python experiments_markerfree/run_random_direction_control.py --model_alias Qwen2.5-7B-Instruct

# Experiment 4: Correctness Probe
python experiments_markerfree/run_markerfree_correctness_probe.py --model_alias Qwen2.5-7B-Instruct

# Experiment 5: Marker Irrelevant Labels (5A & 5B)
# Note: This script requires full model path and name
python experiments_markerfree/run_marker_irrelevant_label.py --model_path /path/to/model --model_name Qwen2.5-7B-Instruct
```

**Pipeline Helper**:
To run Experiment 5 (5A & 5B) for all models sequentially:
```bash
bash experiments_markerfree/run_pipeline_exp5.sh
```

### 2. Summarize and Plot
After running experiments for all models, generate summary plots and CSVs:

```bash
python experiments_markerfree/summarize_and_plot.py
```

Outputs will be saved in `experiments_markerfree/outputs/`.

## Configuration
See `experiments_markerfree/config.yaml` to adjust models, gamma grid, and dataset settings.

## Addressing the "Style Subspace" Critique

The core critique is that our intervention might only be suppressing the *syntax* of reasoning (e.g., `<think>` tags) rather than the *semantics* of reasoning.

This suite provides counter-evidence:
*   **Exp 1 (Forbidden)**: Even when markers are banned by the decoder (so they *cannot* appear), suppressing the reasoning direction (gamma > 0) still degrades performance on reasoning-heavy tasks. This implies the direction encodes *latent* reasoning intent, not just token emission.
*   **Exp 3 (Random Control)**: Random directions in the same space do not cause the same degradation, proving the learned direction is specific.
*   **Exp 5 (Marker-Irrelevant Labels)**: We train a probe to predict "Correctness" (5A) and "Reasoning Demand" (5B) using features derived from the latent direction (u) versus a random direction (u_rand).
    *   **Result**: The real direction `u` features (e.g., projection `u^T z`) are significantly more predictive of Correctness and Reasoning Demand than random directions, even in "Answer Only" mode where no markers are generated. This confirms `u` aligns with the *semantic* properties of the task (difficulty/correctness), not just the *syntactic* presence of markers.

