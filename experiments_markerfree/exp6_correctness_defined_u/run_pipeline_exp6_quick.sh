#!/bin/bash
set -e

# Config
MODEL_PATH="/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR="/root/workspace/experiments_markerfree/outputs/exp6_quick"
CONFIG_PATH="/root/workspace/experiments_markerfree/exp6_correctness_defined_u/config_quick.yaml"

mkdir -p "$OUTPUT_DIR"

echo "Step 1: Build u_corr (diagnostics check)"
python /root/workspace/experiments_markerfree/exp6_correctness_defined_u/build_u_corr.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR"

echo "Step 2 & 3: Run Sweeps (Real and Rand)"
# Real
python /root/workspace/experiments_markerfree/exp6_correctness_defined_u/run_marker_irrelevant_label.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --direction_type "real" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR"

# Rand
python /root/workspace/experiments_markerfree/exp6_correctness_defined_u/run_marker_irrelevant_label.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --direction_type "rand" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR"

echo "Pipeline finished."
