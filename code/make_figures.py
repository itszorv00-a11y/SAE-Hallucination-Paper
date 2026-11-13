import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "/workspace/paper_figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

CONFIGS = ["L12_16k", "L12_65k", "L9_16k", "L9_65k"]
KNOWN = [0, 0, 0, 1]
CHIMERA = [17, 16, 20, 20]
FAKE = [29, 30, 28, 25]
ACT_KNOWN = [0.00, 0.00, 0.00, 0.11]
ACT_CHIM = [3.92, 2.40, 5.55, 3.05]
ACT_FAKE = [12.86, 6.55, 10.53, 4.36]

x = np.arange(len(CONFIGS))
w = 0.26

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(x - w, [k / 30 for k in KNOWN], w, label="Known (Real)", color="#2a9d8f")
ax.bar(x, [c / 30 for c in CHIMERA], w, label="Chimera (Hybrid Real Tokens)", color="#e9c46a")
ax.bar(x + w, [f / 30 for f in FAKE], w, label="Synthetic Unknown", color="#e76f51")
ax.set_xticks(x, CONFIGS)
ax.set_ylabel("Activation Probability  P(a = 1)")
ax.set_title("(A) Firing Frequency on Entity Types")
ax.legend(fontsize=7)
ax.set_ylim(0, 1.05)
fig.savefig(f"{OUT}/fig1a.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(x - w, ACT_KNOWN, w, label="Known (Real)", color="#2a9d8f")
ax.bar(x, ACT_CHIM, w, label="Chimera (Hybrid Real Tokens)", color="#e9c46a")
ax.bar(x + w, ACT_FAKE, w, label="Synthetic Unknown", color="#e76f51")
ax.set_xticks(x, CONFIGS)
ax.set_ylabel("Mean Latent Activation")
ax.set_title("(B) Continuous Activation Strength")
ax.legend(fontsize=7)
fig.savefig(f"{OUT}/fig1b.png")
plt.close(fig)

ALPHAS = ["0.1", "0.25", "0.5", "1.0"]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for ci, cfg in enumerate(CONFIGS):
    ax.bar(x[ci] + (np.arange(4) - 1.5) * 0.2, [0, 0, 0, 0], 0.18, label=cfg)
ax.set_xticks(x, CONFIGS)
ax.set_ylabel("Refusal Rate (n=30)")
ax.set_ylim(0, 1.05)
ax.set_title("(A) Known Entities + Unknown Direction\n(Total Resistance to Refusal Steering: 0/30 at all scales)")
ax.text(0.5, 0.55, "0/30 refusal across all configs\nand all norm ratios", ha="center", transform=ax.transAxes, fontsize=9, color="#555555")
fig.savefig(f"{OUT}/fig2a.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.0))
conds = ["baseline", "k=1, r=0.5", "k=1, r=1.0", "k=3, r=1.0", "k=5, r=1.0", "k=10, r=1.0"]
l12 = [13 / 30, 15 / 30, 7 / 30, 11 / 30, 10 / 30, 11 / 30]
l9 = [13 / 30, 14 / 30, 4 / 30, 14 / 30, 13 / 30, 11 / 30]
xx = np.arange(len(conds))
ax.bar(xx - 0.18, l12, 0.34, label="L12_16k", color="#264653")
ax.bar(xx + 0.18, l9, 0.34, label="L9_16k", color="#2a9d8f")
ax.axhline(13 / 30, color="#e76f51", linestyle="--", linewidth=1, label="Baseline refusal (13/30)")
ax.set_xticks(xx, conds, rotation=20, ha="right")
ax.set_ylabel("Refusal Rate (n=30)")
ax.set_ylim(0, 1.05)
ax.set_title("(B) Unknown Entities + Known Direction\n(Suppression Only at r=1.0, With Generation Collapse)")
ax.legend(fontsize=7)
fig.savefig(f"{OUT}/fig2b.png")
plt.close(fig)

pairs = ["42 vs 1337", "42 vs 2026", "1337 vs 2026"]
cos = [0.0787, 0.0787, 0.0786]
RANDOM = 0.0208
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(pairs, cos, 0.5, color="#264653", label="Observed mean cosine")
ax.axhline(RANDOM, color="#e76f51", linestyle="--", linewidth=1.2, label="Random baseline 1/sqrt(2304) = 0.0208")
ax.axhline(0.70, color="#999999", linestyle=":", linewidth=1.2, label="Functional threshold 0.70")
ax.set_ylabel("Hungarian-Matched Mean Cosine")
ax.set_ylim(0, 0.85)
ax.set_title("(A) Hungarian Bipartite Matching Alignment Across Seeds\nM=8192 JumpReLU SAEs, Layer 12")
ax.legend(fontsize=7)
fig.savefig(f"{OUT}/fig3a.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(pairs, [0, 0, 0], 0.5, color="#264653")
ax.set_ylabel("Overlap at cosine >= 0.70")
ax.set_ylim(0, 1.05)
ax.set_title("(B) The Shattered Assumption of Seed Invariance\n0.0% functional overlap across all seed pairs")
ax.text(0.5, 0.55, "0.0% overlap at threshold >= 0.70\nfor all three seed pairs", ha="center", transform=ax.transAxes, fontsize=9, color="#555555")
fig.savefig(f"{OUT}/fig3b.png")
plt.close(fig)

print("figures done:", sorted(os.listdir(OUT)))
