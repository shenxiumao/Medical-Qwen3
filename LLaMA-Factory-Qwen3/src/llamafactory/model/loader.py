# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.ktransformers import load_kt_pretrained_model
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _infer_hf_repo_id(model_name_or_path: str) -> Optional[str]:
    path = Path(model_name_or_path)
    if not path.is_dir():
        return None

    readme_path = path / "README.md"
    if readme_path.exists():
        try:
            readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            readme_text = ""

        for pattern in [
            r"""model_id\s*=\s*["']([^"']+)["']""",
            r"""model\s*=\s*["']([^"']+)["']""",
            r"""huggingface\.co/([\w.-]+/[\w.-]+)""",
        ]:
            match = re.search(pattern, readme_text)
            if match:
                candidate = match.group(1).strip()
                if "/" in candidate:
                    return candidate

    if path.parent.name and path.name:
        candidate = f"{path.parent.name}/{path.name}"
        if candidate.count("/") == 1:
            return candidate

    return None


def _peek_safetensors_dims(model_name_or_path: str, tensor_key: str) -> Optional[tuple[int, ...]]:
    index_path = Path(model_name_or_path) / "model.safetensors.index.json"
    if not index_path.exists():
        return None

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    weight_map = index.get("weight_map") or {}
    shard_name = weight_map.get(tensor_key)
    if not shard_name:
        return None

    shard_path = Path(model_name_or_path) / shard_name
    if not shard_path.exists():
        return None

    try:
        from safetensors import safe_open
    except Exception:
        return None

    try:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            shape = list(f.get_slice(tensor_key).get_shape())  # type: ignore[attr-defined]
            return tuple(int(x) for x in shape)
    except Exception:
        return None


def _peek_safetensors_shape(model_name_or_path: str, tensor_key: str) -> Optional[tuple[int, int]]:
    dims = _peek_safetensors_dims(model_name_or_path, tensor_key)
    if dims is None or len(dims) != 2:
        return None

    return dims[0], dims[1]


def _infer_gemma3_text_config_from_local_weights(model_name_or_path: str):
    if not os.path.isdir(model_name_or_path):
        return None

    index_path = Path(model_name_or_path) / "model.safetensors.index.json"
    if not index_path.exists():
        return None

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    weight_map = index.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        return None

    embed_shape = _peek_safetensors_shape(model_name_or_path, "model.embed_tokens.weight")
    if embed_shape is None:
        return None

    q_proj_shape = _peek_safetensors_shape(model_name_or_path, "model.layers.0.self_attn.q_proj.weight")
    k_proj_shape = _peek_safetensors_shape(model_name_or_path, "model.layers.0.self_attn.k_proj.weight")
    gate_proj_shape = _peek_safetensors_shape(model_name_or_path, "model.layers.0.mlp.gate_proj.weight")
    q_norm_dims = _peek_safetensors_dims(model_name_or_path, "model.layers.0.self_attn.q_norm.weight")
    if (
        q_proj_shape is None
        or k_proj_shape is None
        or gate_proj_shape is None
        or q_norm_dims is None
        or len(q_norm_dims) != 1
    ):
        return None

    max_layer = -1
    for key in weight_map:
        if not key.startswith("model.layers."):
            continue

        rest = key[len("model.layers.") :]
        layer_id = rest.split(".", 1)[0]
        if layer_id.isdigit():
            max_layer = max(max_layer, int(layer_id))

    if max_layer < 0:
        return None

    vocab_size, hidden_size = embed_shape
    num_hidden_layers = max_layer + 1
    intermediate_size, _ = gate_proj_shape
    head_dim = int(q_norm_dims[0])
    num_attention_heads = q_proj_shape[0] // head_dim
    num_key_value_heads = k_proj_shape[0] // head_dim
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        return None

    try:
        from transformers import Gemma3TextConfig
    except Exception:
        return None

    return Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        query_pre_attn_scalar=head_dim,
    )


def _maybe_reload_config_for_local_weights(
    config: "PretrainedConfig", model_name_or_path: str, init_kwargs: dict[str, Any]
) -> "PretrainedConfig":
    if not os.path.isdir(model_name_or_path):
        return config

    candidate_keys = (
        "model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
        "transformer.wte.weight",
    )
    embed_shape = None
    for key in candidate_keys:
        embed_shape = _peek_safetensors_shape(model_name_or_path, key)
        if embed_shape is not None:
            break

    if embed_shape is None:
        return config

    vocab_size, hidden_size = embed_shape
    cfg_vocab_size = getattr(config, "vocab_size", None)
    cfg_hidden_size = getattr(config, "hidden_size", None)
    if cfg_vocab_size is None or cfg_hidden_size is None:
        return config

    if cfg_vocab_size == vocab_size and cfg_hidden_size == hidden_size:
        return config

    inferred = _infer_gemma3_text_config_from_local_weights(model_name_or_path)
    if inferred is not None:
        inferred_vocab = getattr(inferred, "vocab_size", None)
        inferred_hidden = getattr(inferred, "hidden_size", None)
        if inferred_vocab == vocab_size and inferred_hidden == hidden_size:
            logger.warning_rank0(
                "Detected config/weights mismatch for local model files. Inferred config from weights."
            )
            return inferred

    repo_id = _infer_hf_repo_id(model_name_or_path)
    if not repo_id:
        raise RuntimeError(
            "Config mismatch with local weights. "
            f"Config vocab_size/hidden_size=({cfg_vocab_size}, {cfg_hidden_size}), "
            f"but local embedding shape=({vocab_size}, {hidden_size}). "
            "Provide the correct `config.json` in your local model directory."
        )

    try:
        fixed = AutoConfig.from_pretrained(repo_id, **init_kwargs)
    except Exception as e:
        raise RuntimeError(
            "Config mismatch with local weights, and failed to fetch a compatible config from the Hub. "
            f"Config vocab_size/hidden_size=({cfg_vocab_size}, {cfg_hidden_size}), "
            f"local embedding shape=({vocab_size}, {hidden_size}), repo_id={repo_id}."
        ) from e

    logger.warning_rank0(
        "Detected config/weights mismatch for local model files. "
        f"Reloaded config from `{repo_id}` to match local weights."
    )
    return fixed


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    try:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        return _maybe_reload_config_for_local_weights(config, model_args.model_name_or_path, init_kwargs)
    except OSError as e:
        inferred = _infer_gemma3_text_config_from_local_weights(model_args.model_name_or_path)
        if inferred is not None:
            logger.warning_rank0("Failed to load config from local directory. Inferred config from local weights.")
            return inferred

        repo_id = _infer_hf_repo_id(model_args.model_name_or_path)
        if repo_id and ("config.json" in str(e) or "configuration.json" in str(e)):
            return AutoConfig.from_pretrained(repo_id, **init_kwargs)
        raise


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    config = _maybe_reload_config_for_local_weights(config, model_args.model_name_or_path, init_kwargs)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_kt:
        from ktransformers.sft.monkey_patch_torch_module import install_patch

        install_patch()
        model = load_kt_pretrained_model(config, model_args)
    elif model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    # Borrowing the kernel plugins ability of v1 to temporarily apply the NPU fusion operator to v0,
    # it is turned off by default, and can be discarded after the transition period ends.
    if model_args.use_v1_kernels and is_trainable:
        logger.warning_rank0(
            "You are try to using future feature about kernels, please note that this feature "
            "is not supported for all models. If get any error, please disable this feature, or report the issue."
        )
        from ..v1.plugins.model_plugins.kernels.registry import apply_available_kernels

        model = apply_available_kernels(model)

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
