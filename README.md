# Medical-Qwen3 Tools

[2601.18350] When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs
https://arxiv.org/abs/2601.18350

本文件夹用于整理并上传与 Medical-Qwen3 训练/合并相关的关键工具脚本。

## 目录

- [Merge Scripts](#merge-scripts)
- [Evaluation Pipelines](#evaluation-pipelines)
- [Verification & Analysis](#verification--analysis)
- [Benchmarks](#benchmarks)

## Merge Scripts

### `merge_qwen_adapters_scaled.py`
核心合并脚本。对 PT/SFT 两个 LoRA 适配器进行缩放后顺序合并到基座模型，并导出一个可部署的合并权重目录。

```bash
python merge_qwen_adapters_scaled.py \
  --base_model <BASE_MODEL_DIR_OR_HF_ID> \
  --pt_adapter <PT_ADAPTER_DIR> \
  --sft_adapter <SFT_ADAPTER_DIR> \
  --alpha_pt 0.3 \
  --alpha_sft 0.7 \
  --export_dir <EXPORT_DIR>
```

### `run_merge_and_verify.py` / `run_merge_and_verify_v2.py`
Pipeline 第一阶段脚本。
- **功能**：自动化执行多组权重的合并任务，并调用验证脚本。
- **V2 区别**：V2 版本针对特定的 SFT Adapter 和精简的权重组合（pt0.0, pt0.3, pt1.0）进行了优化，支持 metadata 记录。

## Evaluation Pipelines

### `run_eval_only.py` / `run_eval_only_v2.py`
Pipeline 第二阶段脚本。
- **功能**：加载预合并的模型，在指定数据集上运行 LLaMA-Factory 评测。
- **V2 特性**：
    - 支持 `think_on` / `think_off` 模板切换。
    - 支持 Deterministic / Stochastic 双解码配置。
    - 自动禁用 SwanLab/WandB 以避免阻塞。
    - 生成详细的 `run_meta.json` 记录实验元数据。

### `run_ablation_pipeline.py`
（旧版）全流程消融实验脚本，包含合并和评测逻辑。

## Verification & Analysis

### `verify_merge.py`
合并验证工具。
- **功能**：对比合并后的模型权重与理论计算值的误差（MAE/Max Error）。
- **输出**：生成 `verify_report.json`，确保合并过程的数值精度。

### `compare_merged_models_cpu.py`
CPU 端模型比对工具，用于在低资源环境下快速检查模型一致性。

### `audit_data.py`
数据审计工具，用于检查训练数据的分布和质量。

## Benchmarks

### `medical_benchmark.py`
医疗领域基准测试运行脚本。

### `prepare_medical_bench.py`
基准测试数据准备脚本，用于格式化和预处理评测数据集。
