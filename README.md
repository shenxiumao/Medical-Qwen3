# Medical-Qwen3（arXiv:2601.18350）实验代码

本目录包含论文 [Adapter Merging Reactivates Latent Reasoning Traces: A Mechanism Analysis](https://arxiv.org/abs/2601.18350) 的实验用到的全部代码与脚本（包含我们使用的 LLaMA-Factory 版本与各类实验/分析 pipeline）。

## 引用

arXiv: [2601.18350](https://arxiv.org/abs/2601.18350) · DOI: https://doi.org/10.48550/arXiv.2601.18350

```bibtex
@article{zou2026adaptermerging,
  title   = {Adapter Merging Reactivates Latent Reasoning Traces: A Mechanism Analysis},
  author  = {Zou, Junyi},
  journal = {arXiv preprint arXiv:2601.18350},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.18350}
}
```

## 仓库结构

顶层目录：

- [LLaMA-Factory-Qwen3/](file://LLaMA-Factory-Qwen3)：训练/对齐/评测框架（论文实验使用的代码基线与 `llamafactory-cli`）。
- [reasoning_trace_project/](file://reasoning_trace_project)：跨模型/模板/解码配置的 trace leakage 统计、作图与机制分析（含 layer-wise CKA/PCA/probe），入口见 [reasoning_trace_project/README.md](file://reasoning_trace_project/README.md)。
- [intervention_minimal/](file://intervention_minimal)：vLLM 内部实现的 logit-space rank-1 干预最小复现版本（含 prompts、评测、作图与汇总）。
- [intervention_robustness/](file://intervention_robustness)：干预鲁棒性实验（marker ablation、gamma sanity 等）。
- [experiments_markerfree/](file://experiments_markerfree)：marker-forbidden / answer-only / correctness-defined direction 等“trace leakage”与干预实验（包含 Exp6 与作图脚本），入口见 [experiments_markerfree/README.md](file://experiments_markerfree/README.md)。
- [exp7_markerfree_from_logged/](file://exp7_markerfree_from_logged)：仅基于 logged decision-space signals 重建 marker-free 指标与作图（无需跑模型），入口见 [exp7_markerfree_from_logged/README.md](file://exp7_markerfree_from_logged/README.md)。
- [exp8_temp_baseline/](file://exp8_temp_baseline)：temperature/logit-scaling baseline，用于检验投影干预是否可由“有效温度变化”解释，入口见 [exp8_temp_baseline/README.md](file://exp8_temp_baseline/README.md)。
- [exp9_rankk_ablation/](file://exp9_rankk_ablation)：rank-k 方向（k>1）消融实验与作图，入口见 [exp9_rankk_ablation/README.md](file://exp9_rankk_ablation/README.md)。
- [exp9_rankk_ablation_offline_match/](file://exp9_rankk_ablation_offline_match)：离线 entropy/margin match 的 rank-k 对照与作图，入口见 [exp9_rankk_ablation_offline_match/README.md](file://exp9_rankk_ablation_offline_match/README.md)。
- [subspace_aware_merge_toy/](file://subspace_aware_merge_toy)：toy setting 的 geometry-aware merge / 投影合并验证（含 checkpoint 与评测脚本）。
- [experiments/](file://experiments)：部分实验产物/脚本的聚合目录（当前包含 `subspace_aware_merge_toy/` 的一份副本）。

顶层脚本（训练/合并/评测/工具）：

- [merge_qwen_adapters_scaled.py](file://merge_qwen_adapters_scaled.py)：LoRA(PT/SFT) 缩放合并到基座模型并导出。
- [verify_merge.py](file://verify_merge.py)：合并数值校验（MAE / Max Error），产出 `verify_report.json`。
- [run_merge_and_verify.py](file://run_merge_and_verify.py) / [run_merge_and_verify_v2.py](file://run_merge_and_verify_v2.py)：批量合并 + 校验的 stage-1 pipeline（多 GPU worker）。
- [run_eval_only.py](file://run_eval_only.py) / [run_eval_only_v2.py](file://run_eval_only_v2.py)：对预合并模型做 stage-2 评测（通过 `llamafactory-cli`）。
- [run_ablation_pipeline.py](file://run_ablation_pipeline.py)：旧版全流程消融脚本（合并 + 评测）。
- [medical_benchmark.py](file://medical_benchmark.py) / [prepare_medical_bench.py](file://prepare_medical_bench.py)：医疗基准评测与数据准备脚本。
- [audit_data.py](file://audit_data.py)：训练/评测数据审计脚本。
- [compare_merged_models_cpu.py](file://compare_merged_models_cpu.py)：CPU 端权重比对工具（低资源环境快速检查）。

## 快速开始（最小复现）

本仓库中部分脚本带有作者环境下的绝对路径（例如 `/root/workspace/...`）。复现前先把路径改成你本地的模型与数据位置。

### 1) 安装依赖

LLaMA-Factory 依赖在子目录中维护：

```bash
cd LLaMA-Factory-Qwen3
pip install -r requirements.txt
pip install -e .
```

marker-free 与投影干预实验（`experiments_markerfree/`）额外依赖 vLLM 与常见科学计算包：

```bash
pip install vllm scikit-learn
```

### 2) Stage-1：适配器合并与校验

单次合并（缩放后顺序 merge）：

```bash
python merge_qwen_adapters_scaled.py \
  --base_model <BASE_MODEL_DIR_OR_HF_ID> \
  --pt_adapter <PT_ADAPTER_DIR> \
  --sft_adapter <SFT_ADAPTER_DIR> \
  --alpha_pt 0.3 \
  --alpha_sft 0.7 \
  --export_dir <EXPORT_DIR>
```

批量合并 + 校验：

- 修改 [run_merge_and_verify.py](file://run_merge_and_verify.py) 或 [run_merge_and_verify_v2.py](file://run_merge_and_verify_v2.py) 文件头部的 `BASE_MODEL / PT_ADAPTER / SFT_ADAPTER / MODELS_DIR` 等路径后运行。
- 校验逻辑在 [verify_merge.py](file://verify_merge.py)。

### 3) Stage-2：评测（LLaMA-Factory）

- 修改 [run_eval_only.py](file://run_eval_only.py) 或 [run_eval_only_v2.py](file://run_eval_only_v2.py) 文件头部的 `BASE_MODEL / MODELS_DIR / OUTPUT_DIR / DATASET_DIR` 等路径后运行。
- 评测通过 `llamafactory-cli` 触发，`run_eval_only_v2.py` 支持 `think_on / think_off`、确定性/随机解码预设，以及写入 `run_meta.json` 记录元数据。

### 4) Trace leakage 诊断与干预（marker-forbidden / answer-only / correctness direction）

入口与说明在：

- [experiments_markerfree/README.md](file://experiments_markerfree/README.md)
- Exp6（correctness-defined direction + logit-space rank-1 intervention）在 [experiments_markerfree/exp6_correctness_defined_u/](file://experiments_markerfree/exp6_correctness_defined_u)

运行前通常需要先改：

- [experiments_markerfree/config.yaml](file://experiments_markerfree/config.yaml) 里的 `models` 与 `dataset` 路径
- Exp6 的 [config.yaml](file://experiments_markerfree/exp6_correctness_defined_u/config.yaml)（模型与数据路径、gamma grid 等）

示例命令（以某个 `model_alias` 为例）：

```bash
python experiments_markerfree/run_marker_forbidden.py --model_alias Qwen2.5-7B-Instruct
python experiments_markerfree/run_answer_only.py --model_alias Qwen2.5-7B-Instruct
python experiments_markerfree/run_random_direction_control.py --model_alias Qwen2.5-7B-Instruct
python experiments_markerfree/run_markerfree_correctness_probe.py --model_alias Qwen2.5-7B-Instruct
```

Exp6（correctness-defined direction）可用 pipeline 脚本一键跑（先按需修改脚本内的模型路径）：

```bash
bash experiments_markerfree/exp6_correctness_defined_u/run_pipeline_exp6.sh
```

### 5) 离线重建与 baseline

- Exp7：仅用 `experiments_markerfree/outputs` 的 CSV/日志重建 marker-free 指标与图，见 [exp7_markerfree_from_logged/README.md](file://exp7_markerfree_from_logged/README.md)。
- Exp8：温度匹配 baseline，见 [exp8_temp_baseline/README.md](file://exp8_temp_baseline/README.md)。

Exp8 一键运行：

```bash
bash exp8_temp_baseline/run_pipeline_exp8.sh
```

### 6) 机制分析（Layer-wise CKA / PCA / Probe）

机制分析代码在 [reasoning_trace_project/mechanism_analysis/](file://reasoning_trace_project/mechanism_analysis)：

- 抽取激活：[00_collect_activations.py](file://reasoning_trace_project/mechanism_analysis/scripts/00_collect_activations.py)
- Layer-wise CKA：[01_layerwise_cka.py](file://reasoning_trace_project/mechanism_analysis/scripts/01_layerwise_cka.py)
- Layer-wise PCA：[02_layerwise_pca.py](file://reasoning_trace_project/mechanism_analysis/scripts/02_layerwise_pca.py)
- Layer-wise Probe：[03_layerwise_probe.py](file://reasoning_trace_project/mechanism_analysis/scripts/03_layerwise_probe.py)

### 7) 其它实验入口

- 最小 vLLM 干预复现：见 [intervention_minimal/](file://intervention_minimal)（`scripts/run_vllm_eval.py`、`scripts/logits_intervention.py`、`scripts/make_figures.py`）。
- 干预鲁棒性：见 [intervention_robustness/run.sh](file://intervention_robustness/run.sh) 与 [intervention_robustness/scripts/run_robustness.py](file://intervention_robustness/scripts/run_robustness.py)。
- Rank-k 消融：见 [exp9_rankk_ablation/](file://exp9_rankk_ablation) 与 [exp9_rankk_ablation_offline_match/](file://exp9_rankk_ablation_offline_match)。
- Toy geometry-aware merge：见 [subspace_aware_merge_toy/](file://subspace_aware_merge_toy)。
