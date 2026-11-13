# Experiments — Steering Audit Suite

Code for the fixes tracked in `REVIEW_AND_FIX_PLAN.md`. Everything runs on a single
GPU (e.g., molab RTX Pro 6000, 96 GB) with `google/gemma-2-2b-it` (accept the Gemma
license on HuggingFace first; set `HF_TOKEN`).

## Setup (molab)

```bash
uv pip install -r requirements.txt
export HF_TOKEN=hf_xxx
```

## Run order (maps to the fix plan)

| Script | Paper fix | What it does |
|---|---|---|
| `exp1_second_scorer.py` | A3 | Re-scores all stored generations with a second scorer (LLM judge); reports agreement |
| `exp2_random_latent.py` | B3 | Random-latent steering control on known prompts |
| `exp3_position_ablation.py` | B1 | Steering at first / final / all token positions |
| `exp4_collapse_metric.py` | B4 | Fluency metric to separate suppression from collapse at alpha=1.0 |
| `exp5_refusal_direction_baseline.py` | A5 | Arditi diff-in-means refusal direction baseline (the decisive control) |
| `exp6_multiseed_steering.py` | B5 | Steering variance across custom SAE seeds {42, 1337, 2026} |

## Before running

1. `entities.py` — paste the actual 90 entity names from the experimental records.
2. Replace every `TODO` latent index with the indices selected in the paper's runs.
3. `exp1`: connect the judge LLM in `llm_judge()`.
4. `exp5`: paste harmful/harmless prompt lists (public jailbreakbench-style lists).

Each script writes JSON results into `results/` — commit those outputs so the paper's
numbers are traceable.
