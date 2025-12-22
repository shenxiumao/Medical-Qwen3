# Medical-Qwen3 Tools

本文件夹用于整理并上传与 Medical-Qwen3 训练/合并相关的关键工具脚本。

## merge_qwen_adapters_scaled.py

对 PT/SFT 两个 LoRA 适配器进行缩放后顺序合并到基座模型，并导出一个可部署的合并权重目录。

### 依赖

- Python 3.10+
- `torch`
- `transformers`
- `peft`
- `safetensors`

### 用法示例

```bash
python merge_qwen_adapters_scaled.py \
  --base_model <BASE_MODEL_DIR_OR_HF_ID> \
  --pt_adapter <PT_ADAPTER_DIR> \
  --sft_adapter <SFT_ADAPTER_DIR> \
  --alpha_pt 0.3 \
  --alpha_sft 0.7 \
  --export_dir <EXPORT_DIR>
```

可选参数：

- `--alpha_attn_pt` / `--alpha_attn_sft`：仅对 attention 相关层（`q_proj/k_proj/v_proj/o_proj`）使用独立缩放系数
- `--alpha_mlp_pt` / `--alpha_mlp_sft`：仅对 MLP 相关层（`up_proj/down_proj/gate_proj/gate_up_proj`）使用独立缩放系数
- `--dtype`：`bfloat16` / `float16` / `float32`
- `--trust_remote_code`：需要加载自定义模型代码时开启
