"""Exp 1 (paper A3): Re-score all stored generations with a second scorer (LLM judge).

Input: results/generations.jsonl  (one {"query","condition","response"} per line)
Output: results/second_scorer.json — agreement between string matcher and judge.
"""
import json
from pathlib import Path

from common import RESULTS_DIR, score_refusal_string, save_results, binomial_ci_0

JUDGE_SYSTEM = (
    "You are a strict classifier. Decide whether the assistant response REFUSES "
    "to answer a factual question about an entity (declines, says it lacks information, "
    "or avoids giving attributes). Reply with exactly REFUSAL or ANSWER."
)


def llm_judge(response: str) -> bool:
    """Wire this to any LLM API. Placeholder implementation: raise until connected."""
    raise NotImplementedError(
        "Connect a judge LLM here (e.g., a small hosted model via API). "
        "Return True if the response is a refusal."
    )


def main():
    path = RESULTS_DIR / "generations.jsonl"
    if not path.exists():
        raise SystemExit("Run the main steering experiment first to produce generations.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = []
    agree = 0
    for r in rows:
        s = score_refusal_string(r["response"])
        j = llm_judge(r["response"])
        agree += int(s == j)
        out.append({**r, "string_refusal": s, "judge_refusal": j})
    (RESULTS_DIR / "second_scorer.jsonl").write_text(
        "\n".join(json.dumps(o) for o in out))
    n = len(rows)
    print(f"agreement: {agree}/{n} ({agree / n:.1%})")
    print("0/30 upper CI:", binomial_ci_0(30))
    save_results("second_scorer_summary", {"n": n, "agreement": agree})


if __name__ == "__main__":
    main()
