# molab_runner.py — SAE steering audit: one-file marimo notebook
# Upload this file directly to molab (Add file / upload), or open with: marimo edit molab_runner.py
# Attach a GPU (notebook specs) BEFORE running the experiments.

import marimo

__generated_with = "0.0.0"

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import subprocess
    import sys
    from pathlib import Path

    EXP_DIR = Path("experiments").resolve()
    if not EXP_DIR.exists():
        EXP_DIR = Path(__file__).parent.resolve() / "experiments"
    RESULTS = EXP_DIR / "results"
    RESULTS.mkdir(exist_ok=True)

    setup_btn = mo.ui.run_button(label="1 Install deps (5 min)")
    exp2_btn = mo.ui.run_button(label="2 Exp2: random-latent baseline")
    exp3_btn = mo.ui.run_button(label="3 Exp3: position ablation")
    exp4_btn = mo.ui.run_button(label="4 Exp4: collapse metric")
    exp5_btn = mo.ui.run_button(label="5 Exp5: Arditi baseline (IMPORTANT)")
    show_btn = mo.ui.run_button(label="6 Show all results")

    mo.vstack([setup_btn, exp2_btn, exp3_btn, exp4_btn, exp5_btn, show_btn])
    return EXP_DIR, RESULTS, mo, json, subprocess, sys, setup_btn, exp2_btn, exp3_btn, exp4_btn, exp5_btn, show_btn


@app.cell
def _(EXP_DIR, mo, setup_btn, subprocess, sys):
    status1 = "press button 1"
    if setup_btn.value:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch", "transformer-lens", "sae-lens", "transformers", "accelerate"],
            cwd=EXP_DIR,
        )
        status1 = "SETUP DONE" if r.returncode == 0 else "SETUP FAILED"
        print("NOTE: fill entities.py TODOs and latent indices before experiments!")
    print(status1)
    return ()


@app.cell
def _(EXP_DIR, exp2_btn, mo, subprocess, sys):
    status2 = "press button 2"
    if exp2_btn.value:
        subprocess.run([sys.executable, "exp2_random_latent.py"], cwd=EXP_DIR, check=True)
        status2 = "exp2_random_latent.py DONE"
    print(status1)
    return ()


@app.cell
def _(EXP_DIR, exp3_btn, mo, subprocess, sys):
    status3 = "press button 3"
    if exp3_btn.value:
        subprocess.run([sys.executable, "exp3_position_ablation.py"], cwd=EXP_DIR, check=True)
        status3 = "exp3_position_ablation.py DONE"
    print(status1)
    return ()


@app.cell
def _(EXP_DIR, exp4_btn, mo, subprocess, sys):
    status4 = "press button 4"
    if exp4_btn.value:
        subprocess.run([sys.executable, "exp4_collapse_metric.py"], cwd=EXP_DIR, check=True)
        status4 = "exp4_collapse_metric.py DONE"
    print(status1)
    return ()


@app.cell
def _(EXP_DIR, exp5_btn, mo, subprocess, sys):
    status5 = "press button 5"
    if exp5_btn.value:
        subprocess.run([sys.executable, "exp5_refusal_direction_baseline.py"], cwd=EXP_DIR, check=True)
        status5 = "exp5_refusal_direction_baseline.py DONE"
    print(status1)
    return ()


@app.cell
def _(RESULTS, json, show_btn):
    if show_btn.value:
        for f in sorted(RESULTS.glob("*.json")):
            print("===", f.name, "===")
            print(json.dumps(json.loads(f.read_text()), indent=2)[:1500])
    else:
        print("press button 6 after experiments finish")
    return ()


if __name__ == "__main__":
    app.run()
