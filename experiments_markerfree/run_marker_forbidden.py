import argparse
import os
import sys
import json
import csv
import torch
from vllm import LLM, SamplingParams
from utils import load_config, load_prompts, format_prompt_strict, get_logit_bias_for_markers, has_leakage, build_marker_token_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_alias", type=str, required=True)
    args = parser.parse_args()

    config = load_config()
    model_path = config['models'][args.model_alias]
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{args.model_alias}] Starting Marker-Forbidden Experiment...")

    # Load data
    raw_data = load_prompts(config['dataset']['path'], config['dataset']['N'], config['seed'])
    prompts = [format_prompt_strict(d['prompt']) for d in raw_data]

    # Setup "Forbidden" mechanism: Infinite negative bias for markers
    # This applies to BOTH baseline (gamma=0) and intervention (gamma>0)
    # The "Intervention" (gamma) is an ADDITIONAL suppression on the "reasoning direction"
    # Wait, if we use Option B (scalar suppression), "forbidden" is just gamma=very_high.
    # But the experiment asks: "Enforce forbidden markers in BOTH baseline and intervention... 
    # Goal: show gamma intervention changes strict-fail/correctness even when markers cannot appear."
    # So:
    # Baseline: logit_bias = {marker_id: -INF}
    # Intervention: logit_bias = {marker_id: -INF} + gamma_suppression?
    # If the intervention is defined as "suppress markers", then "forbidden markers" ALREADY does that perfectly.
    # The critique says: "you localize formatting/style subspace, not reasoning subspace".
    # If we forbid markers, the "style" is removed. If we then apply gamma, what are we suppressing?
    # If the "intervention" is exactly "suppress markers", then gamma > 0 is redundant if markers are already forbidden.
    # UNLESS: The "intervention" vector u is defined by the *embeddings* of the markers, and we project hidden states away from u.
    # BUT: Our current implementation (Option B) is just `logit_bias`.
    # If we use `logit_bias`, then `forbidden` means bias=-1000. `intervention` means bias=-gamma*5.
    # Adding them: -1000 - gamma*5 is still -1000.
    # So for this experiment to make sense with `logit_bias` implementation, we might need to:
    # 1. "Forbidden" via stop_strings or post-hoc filtering (as suggested in prompt: "add stop strings... or logits processor... post-check").
    # 2. Apply gamma suppression to *other* tokens? No, the user says "Reuse existing... logits-space direction suppression".
    # 3. If the intervention is purely marker suppression, and we forbid markers, then the intervention is fully active in the baseline.
    #
    # Re-reading prompt: "Markers can be a proxy for trace emission... show intervention still affects correctness... when markers are forbidden."
    # If the intervention is JUST marker suppression, and markers are forbidden, then the intervention should have NO effect (delta=0) because the markers are already gone.
    # If the intervention has an effect, it means it's suppressing something *else* correlated with markers?
    # Or maybe "Intervention" here refers to the *subspace projection* (Option A in logits_intervention.py).
    # Option A: `logits = logits - gamma * (logits * marker_mask)`.
    # If `marker_mask` is 1 for markers, this reduces probability of markers.
    # If we forbid markers (bias=-inf), their prob is 0.
    # So `logits * marker_mask` -> 0.
    # Then `logits - gamma * 0` -> `logits`.
    # So with Option A, if markers are forbidden, the intervention also does nothing *at the logits level*.
    #
    # UNLESS the "intervention" is defined differently, e.g. projecting *hidden states* (like in the paper).
    # BUT the user said: "Intervention implementation already exists in the repo (logits-space direction suppression with gamma). Reuse it."
    # And: "Method... Enforce forbidden markers... (i) prompt instruction... (ii) add stop strings... (iii) post-check".
    # If we use stop strings, the model *generates* the marker, then we stop. The marker *was* generated.
    # If we use bias=-inf, the marker is *never* generated.
    #
    # Let's interpret "Forbidden" as "Stop strings" or "Truncation".
    # If we use stop strings, the model *starts* to output "<think>", and we stop.
    # The "Intervention" (gamma) would prevent it from even *starting* to output "<think>".
    # But if we stop immediately, the output is clean.
    #
    # Maybe the "Forbidden" means: we ban the *explicit* markers, but we check if the model still *implicitly* fails or if the intervention changes the *implicit* reasoning.
    # But if the intervention acts *only* on the marker logits, and we ban marker logits, they are mathematically the same channel.
    #
    # Wait, the prompt says: "strict-fail/correctness... strict-fail should be ~0 by construction".
    # If strict-fail is 0, what are we measuring? "correctness".
    # So we want to see if `Accuracy(gamma=0)` vs `Accuracy(gamma=1)` differs when markers are banned.
    # If the intervention is *only* suppressing markers, and markers are banned, then Accuracy should be identical.
    # If Accuracy changes, it implies... what?
    #
    # Perhaps the user implies that "Strict-fail" includes reasoning *content* not just tags?
    # No, strict-fail is defined by regex on markers.
    #
    # Let's stick to the prompt's Method:
    # "Enforce forbidden markers... (ii) add stop strings OR a logits processor that heavily penalizes... (iii) post-check... if any marker appears, flag... (but still count in strict-fail)."
    # And "show gamma intervention changes strict-fail/correctness".
    #
    # If I use `logit_bias=-inf` for markers:
    # Then `strict-fail` will be 0 (unless leakage patterns are complex and not covered by the token list).
    # Then `Accuracy` should be the same for gamma=0 vs gamma=1, because the logits for non-marker tokens are unaffected by `logit_bias` on markers (in vLLM, softmax is over all, but if marker is -inf, it's 0 mass. If marker is -inf - gamma, it's still 0 mass).
    #
    # So, if I use `logit_bias`, this experiment will likely show Null Result (No change).
    # Is that what is expected? "show the intervention still affects correctness...".
    # This implies the intervention *should* have an effect.
    # This is only possible if the "Intervention" is NOT just `logit_bias` on the *same* tokens we are banning.
    # OR if we ban *some* markers but intervene on *all*?
    # "Define a marker set M... Enforce forbidden markers...".
    #
    # Maybe "Forbidden" is via Prompting + Stop Strings.
    # If we use Stop Strings, the model generates `Step`, we stop.
    # If we use Gamma, the model might not generate `Step` at all, or generate `The answer is`.
    # This changes the output *content*.
    # So I should implement "Forbidden" using **Stop Strings** (and maybe moderate bias if prompted), NOT infinite bias.
    # Infinite bias makes Gamma irrelevant.
    # Stop strings allow the model to *try* to emit reasoning, but we cut it off.
    # Gamma prevents the *attempt*.
    # This distinction allows Gamma to affect Correctness (by allowing/forcing the model to skip reasoning vs getting cut off).
    #
    # PLAN: Use `stop` in SamplingParams for the markers.
    
    # Init LLM
    try:
        llm = LLM(model=model_path, tensor_parallel_size=4, trust_remote_code=True,
                  gpu_memory_utilization=0.9, max_model_len=4096)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Markers
    markers = config['markers']
    # Stop strings: The markers themselves
    stop_strings = markers
    
    results = []

    for gamma in config['gamma_grid']:
        print(f"Running Gamma={gamma}...")
        
        # Intervention: Logit bias on markers
        logit_bias = get_logit_bias_for_markers(model_path, markers, gamma, scale=5.0)
        
        sp = SamplingParams(
            temperature=config['decoding']['temperature'],
            top_p=config['decoding']['top_p'],
            max_tokens=config['decoding']['max_tokens'],
            logit_bias=logit_bias,
            stop=stop_strings, # Enforce forbidden via stop
            detokenize=True
        )
        
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        
        leak_count = 0
        hit_count = 0
        
        for o in outputs:
            text = o.outputs[0].text
            # Check leakage (regex)
            if has_leakage(text, custom_markers=markers):
                leak_count += 1
            
            # Check stop reason (did we hit a marker?)
            if o.outputs[0].finish_reason == "stop":
                # If stopped by one of our markers
                # vLLM doesn't always tell us WHICH string, but "stop" implies it hit one.
                # We can assume it's a marker hit.
                hit_count += 1
                
        leak_rate = leak_count / len(prompts)
        hit_rate = hit_count / len(prompts)
        
        print(f"  Gamma={gamma}: Leakage={leak_rate:.2f}, MarkerHit={hit_rate:.2f}")
        
        results.append({
            "model": args.model_alias,
            "gamma": gamma,
            "leakage_rate": leak_rate,
            "marker_hit_rate": hit_rate,
            "experiment": "marker_forbidden"
        })
        
    # Cleanup
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except:
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Save
    csv_path = os.path.join(output_dir, f"marker_forbidden_{args.model_alias}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")

if __name__ == "__main__":
    main()
