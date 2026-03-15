#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_USE_V1=0

# Define models (Path | Name) - ONLY DeepSeek
MODELS=(
    "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|DeepSeek-R1-Distill-Qwen-7B"
)

echo "Starting Experiment 6 Pipeline (DeepSeek Only)..."
echo "Results will be saved to experiments_markerfree/outputs/exp6/"

for entry in "${MODELS[@]}"; do
    IFS="|" read -r path name <<< "$entry"
    echo "=================================================="
    echo "Running Exp 6 for $name"
    echo "Path: $path"
    echo "=================================================="
    
    # Step 1: Build u_corr (and u_rand)
    echo "[Step 1] Building u_corr..."
    python3 "$SCRIPT_DIR/build_u_corr.py" \
        --model_path "$path" \
        --model_name "$name"
    
    # Step 2: Run Intervention (Real U)
    echo "[Step 2] Running Intervention (Real U)..."
    python3 "$SCRIPT_DIR/run_marker_irrelevant_label.py" \
        --model_path "$path" \
        --model_name "$name" \
        --direction_type "real"
        
    # Step 3: Run Intervention (Rand U)
    echo "[Step 3] Running Intervention (Rand U)..."
    python3 "$SCRIPT_DIR/run_marker_irrelevant_label.py" \
        --model_path "$path" \
        --model_name "$name" \
        --direction_type "rand"
        
    echo "Finished $name"
    echo ""
done

echo "Generating Plots..."
python3 "$SCRIPT_DIR/summarize_and_plot.py"

echo "Calculating Option Metrics & Diagnostics..."
python3 "$SCRIPT_DIR/exp6_option_metrics_from_z1.py"

echo "Pipeline Completed!"
