import torch
from typing import List, Callable, Dict, Any

try:
    from vllm.tokenizers import get_tokenizer
except Exception:
    get_tokenizer = None


def build_marker_token_ids(model_path: str, markers: List[str]) -> List[int]:
    tok = None
    if get_tokenizer is not None:
        tok = get_tokenizer(model_path, trust_remote_code=True)
    else:
        # Fallback: try HF, but caller should ensure vLLM environment provides tokenizer
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids: List[int] = []
    for m in markers:
        try:
            enc = tok.encode(m, add_special_tokens=False)
            # Add all token pieces to cover BPE splits
            ids.extend(enc)
        except Exception:
            continue
    # Deduplicate and ensure valid
    ids = [int(x) for x in set(ids) if isinstance(x, int)]
    return ids


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


def build_reasoning_processor(model_path: str, gamma: float, option: str = "A") -> Dict[str, Any]:
    """
    Builds a reasoning suppression processor and metadata.
    option:
      "A" -> projection onto subspace spanned by marker token logits
      "B" -> direct suppression with scalar direction (uniform over markers)
    """
    # Marker strings per submission (strict leakage markers)
    marker_strings = [
        "<think>", "</think>", "Step", "Step 1:", "Thought:", "let's think", "step by step"
    ]
    marker_ids = build_marker_token_ids(model_path, marker_strings)
    if option == "A":
        proc = reasoning_projection_processor(marker_ids, gamma)
        return {"processor": proc, "marker_ids": marker_ids, "gamma": gamma, "option": "A"}
    else:
        # Option B: return logit_bias map for built-in LogitBiasLogitsProcessor
        # Use a strong negative bias scaled by gamma.
        bias_scale = 5.0
        logit_bias = {int(tid): float(-gamma * bias_scale) for tid in marker_ids}
        return {"logit_bias": logit_bias, "marker_ids": marker_ids, "gamma": gamma, "option": "B"}
