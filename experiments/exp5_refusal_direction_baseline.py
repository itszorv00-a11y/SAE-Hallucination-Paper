"""Exp 5 (paper A5 — the decisive baseline): Arditi-style refusal direction.

Compute the refusal direction by difference-in-means over harmful vs harmless
prompts (Arditi et al., 2024), then steer known-entity prompts with it at the same
norm-matched scales. If THIS induces refusal while SAE entity latents do not,
the paper's core claim is upgraded to: "refusal IS steerable — but not by SAE
entity-recognition latents."
"""
import torch

from common import load_model, mean_residual_norm, steer_resid, save_results
from entities import all_prompts

LAYER = 12
ALPHAS = [0.25, 0.5, 1.0]

HARMFUL = [
    # TODO: ~30 harmful requests (e.g., from jailbreakbench-style public lists)
]
HARMLESS = [
    # TODO: ~30 matched harmless requests
]


def diff_in_means_direction(model, layer: int) -> torch.Tensor:
    _, cache = model.run_with_cache(
        HARMFUL + HARMLESS,
        names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post",
    )
    h = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :].float()
    n_h = len(HARMFUL)
    return (h[:n_h].mean(0) - h[n_h:].mean(0))


def main():
    model = load_model()
    direction = diff_in_means_direction(model, LAYER)
    known_prompts = [p for kind, name, p in all_prompts() if kind == "known"]
    mean_norm = mean_residual_norm(model, known_prompts, LAYER)

    results = []
    for alpha in ALPHAS:
        refusals = 0
        for prompt in known_prompts:
            resp = steer_resid(model, prompt, LAYER, direction, alpha, mean_norm)
            refusals += int("cannot" in resp.lower() or "don't have" in resp.lower())
        results.append({"alpha": alpha, "refusals": refusals, "n": len(known_prompts)})
        print(f"refusal-direction alpha={alpha}: {refusals}/{len(known_prompts)}")
    save_results("arditi_baseline", {"layer": LAYER, "results": results})


if __name__ == "__main__":
    main()
