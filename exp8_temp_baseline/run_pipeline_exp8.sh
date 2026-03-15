#!/bin/bash
set -e

# Stage-2: Exp8 温度 / logit-scaling 基线一键跑完四个模型
# 在 /root/workspace/exp8_temp_baseline 下执行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export VLLM_USE_V1=0

GAMMA_GRID="0,2,5,10"
TEMP_GRID="0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0"
N=500
SEED=0
MATCH_TARGET="entropy"

declare -a MODELS=(
  "/root/workspace/model/meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct"
  "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|DeepSeek-R1-Distill-Qwen-7B"
  "/root/workspace/model/Qwen/Qwen3-8B|Qwen3-8B"
  "/root/workspace/model/Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct"
)

echo "=== Exp8 温度基线：开始跑 4 个模型 ==="

for entry in "${MODELS[@]}"; do
  IFS="|" read -r MODEL_PATH MODEL_NAME <<< "$entry"
  echo "--------------------------------------------------"
  echo "Running Exp8 Temp Baseline for: $MODEL_NAME"
  echo "Model path: $MODEL_PATH"
  echo "Gamma grid: $GAMMA_GRID"
  echo "Temp  grid: $TEMP_GRID"
  echo "--------------------------------------------------"

  python run_temp_baseline.py \
    --model "$MODEL_PATH" \
    --N "$N" \
    --seed "$SEED" \
    --gamma_grid "$GAMMA_GRID" \
    --temp_grid "$TEMP_GRID" \
    --match_target "$MATCH_TARGET" \
    --out "outputs/tempbaseline_${MODEL_NAME}.csv"
done

echo "=== 所有模型跑完，开始汇总并画图 ==="
python summarize_and_plot.py

echo "=== Exp8 完成 ==="
echo "输出文件："
echo "  - outputs/tempbaseline_*.csv"
echo "  - outputs/summary_tempbaseline_all.csv"
echo "  - outputs/fig_proj_vs_tempmatch_accuracy.png"
echo "  - outputs/fig_proj_vs_tempmatch_strictfail.png"
echo "  - outputs/fig_proj_vs_tempmatch_entropy.png"
echo "  - outputs/fig_proj_vs_tempmatch_margin.png"
