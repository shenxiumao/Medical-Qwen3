#!/bin/bash
set -e

# Path to python environment
PYTHON_BIN="/opt/miniconda3/envs/lit_reason/bin/python"
SCRIPT_PATH="/root/workspace/intervention_robustness/scripts/run_robustness.py"

echo "Starting Robustness Experiment (Gamma Sanity + Marker Ablation)..."
echo "Output Directory: /root/workspace/intervention_robustness/results"

# Run the master script (which spawns workers for each model)
$PYTHON_BIN $SCRIPT_PATH

echo "Done! Figures are in /root/workspace/intervention_robustness/figures"
