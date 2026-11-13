"""Exp 4 (paper B4): Separate refusal suppression from generation collapse at alpha=1.0.

Metrics per steered generation:
  - refusal (string match)
  - fluency proxy: mean token log-prob of the generated text under the UNSTEERED model
If suppression at alpha=1.0 coincides with a large fluency drop, suppression is
confounded by collapse.
"""
import torch

from common import load_model, load_sae, steer_resid, mean_residual_norm, save_results
from entities import all_prompts

LAYER, WIDTH, L0 = 12, "16k", 80


def mean_logprob(model, text: str) -> float:
    toks = model.to_tokens(text)
    with torch.no_grad():
        logits = model(toks)
    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = toks[0, 1:]
    return lp.gather(-1, tgt.unsqueeze(-1)).mean().item()


def main():
    model = load_model()
    sae = load_sae(LAYER, WIDTH, L0)
    known_latent_idx = 0  # TODO: paper's known-entity latent
    direction = sae.W_dec.data[known_latent_idx]
    unknown_prompts = [p for kind, name, p in all_prompts() if kind == "unknown"]
    mean_norm = mean_residual_norm(model, unknown_prompts, LAYER)

    rows = []
    for prompt in unknown_prompts:
        resp = steer_resid(model, prompt, LAYER, direction, 1.0, mean_norm)
        refusal = int("cannot" in resp.lower() or "don't have" in resp.lower())
        fluency = mean_logprob(model, resp)
        rows.append({"prompt": prompt, "response": resp, "refusal": refusal, "fluency": fluency})
        print(f"refusal={refusal} fluency={fluency:.2f} :: {resp[:60]}")

    ref = [r["refusal"] for r in rows]
    flu = [r["fluency"] for r in rows]
    save_results("collapse_disentangle", {
        "n": len(rows),
        "refusal_rate": sum(ref) / len(ref),
        "mean_fluency": sum(flu) / len(flu),
        "rows": rows,
    })


if __name__ == "__main__":
    main()
