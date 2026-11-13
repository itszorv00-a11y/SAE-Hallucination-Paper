"""Exp 2 (paper B3): Random-latent steering control.

Steer with RANDOM SAE decoder directions (matched in count and norm) instead of
entity latents. If refusal on known entities stays 0/30 here too, the nullity is a
property of the intervention class, not of the selected latents.
"""
import torch

from common import load_model, load_sae, steer_resid, mean_residual_norm, save_results
from entities import all_prompts, PROMPT_TEMPLATE

LAYER, WIDTH, L0 = 12, "16k", 80
ALPHAS = [0.25, 0.5, 1.0]
N_RANDOM = 10
K_POPULATION = 200  # sample random latents from the top-K by decoder norm


def main():
    model = load_model()
    sae = load_sae(LAYER, WIDTH, L0)
    W_dec = sae.W_dec.data
    known_prompts = [p for kind, name, p in all_prompts() if kind == "known"]
    mean_norm = mean_residual_norm(model, known_prompts, LAYER)

    torch.manual_seed(0)
    idx = torch.randperm(W_dec.shape[0])[:K_POPULATION]
    results = []
    for alpha in ALPHAS:
        refusals = 0
        evals = 0
        for r in range(N_RANDOM):
            direction = W_dec[idx[r]]
            for prompt in known_prompts:
                resp = steer_resid(model, prompt, LAYER, direction, alpha, mean_norm)
                refusals += int("cannot" in resp.lower() or "don't have" in resp.lower())
                evals += 1
        results.append({"alpha": alpha, "refusals": refusals, "evals": evals})
        print(f"alpha={alpha}: {refusals}/{evals} refusal")
    save_results("random_latent_baseline", {
        "config": f"L{LAYER}_{WIDTH}", "n_random_directions": N_RANDOM,
        "results": results,
    })


if __name__ == "__main__":
    main()
