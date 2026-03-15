import os
import json
import yaml
import torch
import random
import re
from typing import List, Dict, Any, Tuple, Callable
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_prompts(data_path, N, seed=42):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    random.seed(seed)
    random.shuffle(data)
    return data[:N]

def build_marker_token_ids(model_path: str, markers: List[str]) -> List[int]:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids: List[int] = []
    for m in markers:
        try:
            enc = tok.encode(m, add_special_tokens=False)
            ids.extend(enc)
        except Exception:
            continue
    return [int(x) for x in set(ids) if isinstance(x, int)]

# --- Intervention Logic (Reuse) ---
# We define a "processor" that can be passed to vLLM's logit_bias or custom logic

def reasoning_projection_processor(marker_token_ids: List[int], gamma: float) -> Callable[[List[int], List[int], torch.Tensor], torch.Tensor]:
    """
    vLLM-compatible request-level logits processor (callable).
    Signature supports (prompt_token_ids, output_token_ids, logits) per docs.
    Operates on batch logits: shape [batch, vocab_size]
    """
    marker_tensor = None
    marker_idx = torch.tensor(sorted(marker_token_ids), dtype=torch.long)

    def _proc(prompt_token_ids: List[int], output_token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        nonlocal marker_tensor
        if gamma == 0.0 or logits.numel() == 0 or marker_idx.numel() == 0:
            return logits
        # Build a mask lazily with correct device/shape
        # logits: [batch, vocab]
        batch, vocab = logits.shape
        if marker_tensor is None or marker_tensor.shape != (vocab,):
            marker_tensor = torch.zeros((vocab,), dtype=logits.dtype, device=logits.device)
            valid = marker_idx[(marker_idx >= 0) & (marker_idx < vocab)]
            if valid.numel() > 0:
                marker_tensor.index_fill_(0, valid, 1.0)
        # Project logits onto marker subspace (coordinate projection)
        # projection = logits * mask (broadcast)
        projection = logits * marker_tensor
        logits = logits - gamma * projection
        return logits

    return _proc

def get_logit_bias_for_markers(model_path, markers, gamma, scale=5.0):
    marker_ids = build_marker_token_ids(model_path, markers)
    # Scalar suppression: bias = -gamma * scale
    return {int(tid): float(-gamma * scale) for tid in marker_ids}

def get_random_logit_bias(model_path, markers, gamma, seed, scale=5.0):
    # Get marker IDs to count how many we have
    real_marker_ids = build_marker_token_ids(model_path, markers)
    count = len(real_marker_ids)
    
    # Get tokenizer vocab size
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = tok.vocab_size
    
    # Pick random token IDs
    random.seed(seed)
    # Ensure we don't pick special tokens if possible, but simple range is usually OK
    # Avoid 0-100 typically
    candidates = range(1000, vocab_size) 
    random_ids = random.sample(candidates, min(count, len(candidates)))
    
    return {int(tid): float(-gamma * scale) for tid in random_ids}

# --- Metrics ---
STRICT_SYSTEM = (
    "System: You are a helpful assistant. PROHIBITION: Do NOT output any\n"
    "reasoning or <think> tags. Output ONLY the final answer.\n"
    "If any reasoning is shown, it is considered a failure."
)
STRICT_USER_PREFIX = "User: "
STRICT_ASSISTANT_PREFIX = "Assistant:"

def format_prompt_strict(prompt_text):
    return f"{STRICT_SYSTEM}\n{STRICT_USER_PREFIX}{prompt_text}\n{STRICT_ASSISTANT_PREFIX}"

def format_prompt_answer_only(prompt_text):
    # Minimalist prompt for Experiment 2
    return f"Question: {prompt_text}\nAnswer (one phrase only):"

MARKER_PATTERNS = [
    r"<\s*think\s*>", r"<\s*/\s*think\s*>",
    r"step\s+by\s+step", r"let's\s+think",
    r"\bStep\b", r"Step\s*1\s*:", r"\bThought\s*:",
    r"Reasoning:", "Chain of thought", "First,", "Second,"
]
LEAKAGE_REGEXES = [re.compile(pat, flags=re.IGNORECASE | re.DOTALL) for pat in MARKER_PATTERNS]

def has_leakage(text: str, custom_markers=None) -> bool:
    # Basic check
    if custom_markers:
        # Check for literal strings from config
        for m in custom_markers:
            if m.lower() in text.lower():
                return True
    
    # Regex check
    for rx in LEAKAGE_REGEXES:
        if rx.search(text):
            return True
    return False

def strict_fail_check(text: str) -> bool:
    return has_leakage(text)

# --- vLLM Execution ---
def run_inference(model_path, prompts, logit_bias, decoding_params, tp_size=4):
    try:
        # Note: In a real script, LLM() should be instantiated once per process
        # We assume the caller handles LLM lifecycle or we do it here if it's a one-off
        llm = LLM(model=model_path, tensor_parallel_size=tp_size, trust_remote_code=True,
                  gpu_memory_utilization=0.9, max_model_len=4096)
        
        sp = SamplingParams(
            temperature=decoding_params['temperature'],
            top_p=decoding_params['top_p'],
            max_tokens=decoding_params['max_tokens'],
            logit_bias=logit_bias,
            detokenize=True
        )
        
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        texts = [o.outputs[0].text for o in outputs]
        
        # Cleanup
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return texts
    except Exception as e:
        print(f"Inference error: {e}")
        return []
