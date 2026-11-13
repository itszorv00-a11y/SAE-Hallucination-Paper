"""Exp 6 (paper B5): Multi-seed steering variance.

Repeat the Kn->Unknown suppression experiment with latents from the three custom
SAE seeds {42, 1337, 2026}. Reports refusal rate per seed and across-seed spread.
"""
from common import load_model, load_sae, steer_resid, mean_residual_norm, save_results
from entities import all_prompts

LAYER, WIDTH, L0 = 12, "16k", 80
SEEDS = [42, 1337, 2026]
ALPHA = 0.5


def main():
    model = load_model()
    unknown_prompts = [p for kind, name, p in all_prompts() if kind == "unknown"]
    mean_norm = mean_residual_norm(model, unknown_prompts, LAYER)

    per_seed = {}
    for seed in SEEDS:
        # TODO: point this at the custom-trained SAE checkpoints (Phase 2 artifacts)
        sae = load_sae(LAYER, WIDTH, L0)
        known_latent_idx = 0  # TODO: per-seed known-entity latent selected by separation score
        direction = sae.W_dec.data[known_latent_idx]
        refusals = 0
        for prompt in unknown_prompts:
            resp = steer_resid(model, prompt, LAYER, direction, ALPHA, mean_norm)
            refusals += int("cannot" in resp.lower() or "don't have" in resp.lower())
        per_seed[seed] = {"refusals": refusals, "n": len(unknown_prompts)}
        print(f"seed={seed}: {refusals}/{len(unknown_prompts)}")

    rates = [v["refusals"] / v["n"] for v in per_seed.values()]
    save_results("multiseed_steering", {
        "alpha": ALPHA, "per_seed": per_seed,
        "rate_spread": max(rates) - min(rates),
    })


if __name__ == "__main__":
    main()
