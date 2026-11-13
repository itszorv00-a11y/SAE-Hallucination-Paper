"""Common utilities for the SAE steering audit experiments.

Requires: GPU (molab RTX Pro 6000), transformer_lens, sae-lens, torch, transformers.
Run on molab:  uv pip install -r requirements.txt  then python <experiment>.py
"""
import json
import math
import os
from pathlib import Path

import torch

MODEL_NAME = "google/gemma-2-2b-it"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN", "")
if HUGGINGFACE_TOKEN:
    os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN


def load_model():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device="cuda")
    model.eval()
    return model


def load_sae(layer: int, width: str, l0: int):
    from sae_lens import SAE
    release = "gemma-scope-2b-it-res" if width != "65k" else "gemma-scope-2b-it-res"
    sae_id = f"{release}/{layer}-{width}-l0-{l0}"
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device="cuda")[0]
    return sae


def mean_residual_norm(model, prompts, layer: int) -> float:
    _, cache = model.run_with_cache(prompts, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post")
    h = cache[f"blocks.{layer}.hook_resid_post"]
    return h.norm(dim=-1).mean().item()


def steer_resid(model, prompt: str, layer: int, direction: torch.Tensor, alpha_scale: float,
                mean_norm: float, positions: str = "final_entity"):
    """Run generation with an additive steering vector on the residual stream.

    positions: "final_entity" | "first" | "all_tokens"
    """
    vec = direction / direction.norm() * (alpha_scale * mean_norm)

    def hook(resid, hook):
        if positions == "all_tokens":
            resid[:, :, :] = resid[:, :, :] + vec
        elif positions == "first":
            resid[:, 0, :] = resid[:, 0, :] + vec
        else:
            resid[:, -1, :] = resid[:, -1, :] + vec
        return resid

    return model.run_with_hooks(
        prompt, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)],
        max_new_tokens=60, temperature=0.0,
    )


REFUSAL_MARKERS = [
    "i cannot", "i can't", "i'm unable", "i am unable", "i don't have",
    "i do not have", "i'm not familiar", "i am not familiar", "no information",
    "not aware of", "unable to provide", "cannot provide",
]


def score_refusal_string(response: str) -> bool:
    r = response.lower()
    return any(m in r for m in REFUSAL_MARKERS)


def score_refusal_llm_judge(response: str, judge) -> bool:
    """Second scorer: pass a callable judge(response) -> bool. Wire to any LLM API."""
    return judge(response)


def binomial_ci_0(n: int, conf: float = 0.95):
    """Clopper-Pearson upper bound for k=0 successes."""
    upper = 1.0 - (1.0 - conf) ** (1.0 / n)
    return 0.0, round(upper * 100, 1)


def save_results(name: str, payload: dict):
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"saved {path}")
    return path
