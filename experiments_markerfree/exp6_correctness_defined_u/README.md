# Experiment 6: Correctness-Defined Direction (Marker-Free)

This experiment investigates whether a direction vector $u$ defined purely by **correctness** (without any specific marker tokens) can effectively steer the model.

## Overview

- **Direction Definition**: $u_{corr} = \text{normalize}(\mu_{pos} - \mu_{neg})$, where $\mu_{pos}$ is the mean of first-token logit vectors for correctly answered questions, and $\mu_{neg}$ is for incorrect ones.
- **Control**: $u_{rand}$ is a random direction vector of the same dimension.
- **Method**: Logit bias intervention sweeping $\gamma$ from -50 to 50.
- **Models**: Same set as Experiment 5.

## Directory Structure

This folder contains a self-contained copy of the experiment code, modified for Exp6.
- `build_u_corr.py`: Computes $u_{corr}$ and $u_{rand}$ from MCQ datasets.
- `run_marker_irrelevant_label.py`: Runs the intervention sweep (supports `real` and `rand` directions).
- `summarize_and_plot.py`: Aggregates results and generates comparison plots.
- `run_pipeline_exp6.sh`: End-to-end pipeline script.

## How to Run

### 1. Run the Full Pipeline (Recommended)

To run the entire experiment for all models (build $u$, run interventions, plot results):

```bash
bash run_pipeline_exp6.sh
```

### 2. Manual Steps

**Step 1: Build Direction Vectors**
```bash
python3 build_u_corr.py \
    --model_path "/path/to/model" \
    --model_name "ModelName"
```
*Sanity Checks*: Look for the output showing sample counts and vector norm:
```
Total samples: 100
Correct: 45
Incorrect: 55
Marker Hit Rate: 0.0000
Norm of (mu_pos - mu_neg): 12.345
```

**Step 2: Run Intervention (Real Direction)**
```bash
python3 run_marker_irrelevant_label.py \
    --model_path "/path/to/model" \
    --model_name "ModelName" \
    --direction_type "real"
```

**Step 3: Run Intervention (Random Control)**
```bash
python3 run_marker_irrelevant_label.py \
    --model_path "/path/to/model" \
    --model_name "ModelName" \
    --direction_type "rand"
```

**Step 4: Summarize and Plot**
```bash
python3 summarize_and_plot.py
```

## Outputs

All outputs are saved to `../outputs/exp6/`:
- `u_corr_*.npy`, `u_rand_*.npy`: Direction vectors.
- `diagnostics_u_corr_*.json`: Sanity diagnostics for $u_{corr}$.
- `exp6_results_real_*.csv`, `exp6_results_rand_*.csv`: Detailed intervention results.
- `summary_exp6_plus.csv`: Aggregated metrics including behavioral indicators.
- `acc_vs_gamma_real_vs_rand.png`: Accuracy comparison plot.
- `strictfail_vs_gamma_real_vs_rand.png`: Failure rate comparison plot.
- `delta_choice_margin_real_minus_rand.png`: Difference in choice confidence.
- `delta_format_violation_real_minus_rand.png`: Difference in adherence to format.
- `delta_refusal_real_minus_rand.png`: Difference in refusal rates.

## Metrics Interpretation

Since raw accuracy may be insensitive to small interventions in the latent space, we track additional behavioral metrics:

1.  **Choice Margin**: $\text{LogProb}(\text{Top1}) - \text{LogProb}(\text{Top2})$ (or proxy entropy). A drop in margin indicates the model is less certain about its answer, even if the top-1 choice hasn't flipped yet.
2.  **Format Violation**: Rate of answers not adhering to the "Answer ONLY with A/B/C/D" constraint. An increase suggests the intervention is disrupting instruction following capabilities.
3.  **Refusal Rate**: Fraction of outputs containing refusal patterns ("I can't", "As an AI"). This measures if the intervention vector triggers safety refusals.

We plot the **Delta (Real - Rand)** for these metrics. A significant deviation from 0 implies the "Correctness" direction has a specific causal effect distinct from random noise.
