#!/bin/bash
set -e

# Experiment 5 Pipeline: Marker-Irrelevant Supervision (5A & 5B)
# Runs run_marker_irrelevant_label.py for all models and then generates plots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_USE_V1=0

# Define models (Path | Name)
MODELS=(
    "/root/workspace/model/meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct"
    "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|DeepSeek-R1-Distill-Qwen-7B"
    "/root/workspace/model/Qwen/Qwen3-8B|Qwen3-8B"
    "/root/workspace/model/Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct"
)

echo "Starting Experiment 5 Pipeline (5A & 5B)..."
echo "Results will be saved to outputs/"

for entry in "${MODELS[@]}"; do
    IFS="|" read -r path name <<< "$entry"
    echo "=================================================="
    echo "Running Exp 5 for $name"
    echo "Path: $path"
    echo "=================================================="
    
    python3 "$SCRIPT_DIR/run_marker_irrelevant_label.py" \
        --model_path "$path" \
        --model_name "$name"
        
    echo "Finished $name"
    echo ""
done

echo "Generating Plots..."
python3 "$SCRIPT_DIR/summarize_and_plot.py"

echo "Pipeline Completed!"
