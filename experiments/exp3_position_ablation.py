"""Exp 3 (paper B1): Position ablation for steering.

Same unknown-latent steering on known-entity prompts, but applied at:
  - final entity token (original protocol)
  - first token
  - all tokens
If refusal stays 0/30 at all positions, the control failure is position-invariant.
"""
from common import load_model, load_sae, steer_resid, mean_residual_norm, save_results
from entities import all_prompts

LAYER, WIDTH, L0 = 12, "16k", 80
ALPHA = 1.0
POSITIONS = ["final_entity", "first", "all_tokens"]


def main():
    model = load_model()
    sae = load_sae(LAYER, WIDTH, L0)
    from latent_indices import UNKNOWN_LATENT
    unknown_latent_idx = UNKNOWN_LATENT
    direction = sae.W_dec.data[unknown_latent_idx]
    known_prompts = [p for kind, name, p in all_prompts() if kind == "known"]
    mean_norm = mean_residual_norm(model, known_prompts, LAYER)

    out = {}
    for pos in POSITIONS:
        refusals = 0
        for prompt in known_prompts:
            resp = steer_resid(model, prompt, LAYER, direction, ALPHA, mean_norm, positions=pos)
            refusals += int("cannot" in resp.lower() or "don't have" in resp.lower())
        out[pos] = {"refusals": refusals, "n": len(known_prompts)}
        print(f"position={pos}: {refusals}/{len(known_prompts)} refusal")
    save_results("position_ablation", {"config": f"L{LAYER}_{WIDTH}", "alpha": ALPHA, "results": out})


if __name__ == "__main__":
    main()
