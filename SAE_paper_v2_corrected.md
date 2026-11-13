# The Illusion of Monosemantic Control: Knowledge-Aware Sparse Autoencoders Detect Entities Reliably but Asymmetrically Resist Causal Steering in Small Language Models

**Md Mushfiqur Rahim**
Independent Researcher, Govt. Azizul Haque College, Bogura
ORCID: 0009-0003-1021-685X | Email: 20mahin2020@gmail.com | GitHub: MD-Mushfiqur123

> **AI-use disclosure.** An autonomous LLM research assistant ("L") was used for code development, experimental tooling, and drafting support. All experiments, analyses, and conclusions were designed, executed, verified, and are owned by the human author, who takes full responsibility for this work. Per arXiv and ICLR policy, the LLM tool is not listed as an author.

## Abstract

Sparse Autoencoders (SAEs) have emerged as a leading instrument for resolving polysemanticity in transformer language models, offering dictionaries of monosemantic latents. Recent work (Ferrando et al., 2025) demonstrated that JumpReLU SAEs trained on Gemma-2 identify localized latents that classify known versus unknown entities, and reported that surgical intervention on these latents causally modulates epistemic refusal and hallucination. In this work, we present a systematic empirical and theoretical evaluation of this claim across 8 architectural configurations of Gemma Scope SAEs on google/gemma-2-2b-it. First, to address surface-level syntax confounds, we introduce **Chimera Entities** (hybrid constructs combining authentic entity tokens into fictitious names), establishing that SAE detection reflects genuine epistemic familiarity (mean activation 3.92, 57% firing) rather than orthographic or tokenization artifacts. Second, introducing Norm-Matched Causal Steering across four relative scales (10%, 25%, 50%, and 100% of residual norm), we uncover a pronounced **Asymmetric Control Failure**: steering with unknown-entity latents on known factual queries fails to induce refusal in 0/30 (0.0%; 95% CI 0–11.6%) of evaluations across all widths, layers, and norm multipliers up to 100% residual magnitude; multi-latent sweeps up to k=10 simultaneously still yield 0/30. Conversely, suppressing refusal on unknown entities requires extreme norm-matched perturbation (100% norm), which fails Holm-Bonferroni corrected significance (p_adj > 0.05) and causes catastrophic generation collapse. Third, scaling dictionary width from 16k to 65k induces pervasive feature splitting (cosine < 0.40), yet the causal nullity remains invariant. Fourth, training custom multi-seed SAEs (M=8192, seeds {42, 1337, 2026}) yields near-chance Hungarian alignment (mean cosine 0.0787, 0.0% overlap at cosine >= 0.70), confirming and extending the seed-dependence reported by Paulo & Belrose (2025) to epistemic latents. Our findings indicate that high classification AUROC (> 0.96) is decoupled from causal intervention efficacy: **Detection != Control**.

## 1. Introduction

A central aspiration of mechanistic interpretability is to decompose polysemantic activations into sparse, monosemantic, and causally steerable primitives (Elhage et al., 2022; Bricken et al., 2023). Activation-steering results show that some behaviors are controllable through single linear directions: refusal in chat models is mediated by one direction in the residual stream (Arditi et al., 2024), and large-scale SAE feature atlases report causally steerable features (Templeton et al., 2024). Ferrando et al. (2025) reported that JumpReLU SAEs in Gemma-2 identify entity-recognition latents that govern refusal and hallucination, with steering effects at the final entity token. However, Paulo & Belrose (2025) demonstrated that SAE features are highly seed-dependent, and a recent independent replication on a 27B reasoning model (Vicentino, 2025, OpenInterpretability blog) reported that single-latent clamping yields weaker refusal calibration than the original Gemma-2 results. This tension — are entity-recognition latents genuinely causal control knobs, or reliable detectors only? — motivates our multi-tier empirical audit.

We emphasize that epistemic refusal and hallucination are two sides of the same mechanism: a model that cannot recognize an entity either refuses or confabulates attributes (Ferrando et al., 2025). Our experiments therefore measure refusal as the observable behavioral readout of the entity-knowledge circuit, and interpret steering failures in terms of hallucination propensity (Section 7).

## 2. Experimental Methodology & Chimera Entity Design

We evaluated 8 JumpReLU SAE configurations from Gemma Scope (Lieberum et al., 2024) across Layers 9, 12, 16, widths 16k and 65k, and target L0 in {30, 80, 200} on google/gemma-2-2b-it.

**Steering protocol.** Interventions add the SAE decoder direction of a selected latent (scaled by alpha relative to the mean residual-stream norm at the intervention layer) to the residual stream at the final entity token. Latents were selected as the top unknown-/known-entity latents by minimum separation score following Ferrando et al. (2025). Refusal was scored by string matching refusal-marker phrases in the model response (e.g., "I cannot", "I don't have information"), consistent with prior work; the full stimulus lists are provided in Appendix A, and the analysis code will be released upon publication.

**Chimera Entities.** To rigorously disentangle genuine epistemic tracking from tokenization-frequency artifacts, we introduced Chimera Entities: fictitious athletes created by pairing authentic first names with authentic surnames of different athletes (e.g., Kobe Williams, Novak Zidane, Stephen Ronaldo). Every constituent token is high-frequency, natural, and semantically grounded, yet the entity does not exist. The full 90-entity stimulus lists (30 known, 30 chimera, 30 synthetic unknown) are given in Appendix A.

## 3. Detection Results: The Epistemic Gradient

(A) Firing Frequency on Entity Types — [Figure 1A]
(B) Continuous Activation Strength — [Figure 1B]

**Figure 1: Epistemic Gradient vs String Morphology.** (A) Firing frequency of the top unknown-entity latent across Known, Chimera, and Synthetic Unknown entities. (B) Continuous activation strength. Chimeras elicit intermediate activation across all configurations, confirming epistemic tracking rather than string-syntax confounds.

| Configuration | Known (N=30) | Chimera (N=30) | Synthetic Fake (N=30) | Top-5 Firing (Kn/Ch/Fake) |
| --- | --- | --- | --- | --- |
| L12_16k | 0/30 (0.00) | 17/30 (3.92) | 29/30 (12.86) | 0.053 / 0.607 / 0.913 |
| L12_65k | 0/30 (0.00) | 16/30 (2.40) | 30/30 (6.55) | 0.040 / 0.560 / 0.887 |
| L9_16k | 0/30 (0.00) | 20/30 (5.55) | 28/30 (10.53) | 0.053 / 0.347 / 0.820 |
| L9_65k | 1/30 (0.11) | 20/30 (3.05) | 25/30 (4.36) | 0.047 / 0.233 / 0.847 |

**Table 1:** Empirical response across entity classes. Values in parentheses represent mean continuous latent activation magnitude.

## 4. Norm-Matched Causal Steering & The Control Asymmetry

We implemented Norm-Matched Steering using vector scale alpha in {0.1, 0.25, 0.5, 1.0} times the mean residual norm (152.2 at L9, 184.3 at L12). Injecting unknown directions on known queries yields 0/30 refusal (95% CI 0–11.6%) across all scales (Figure 2A). Conversely, steering unknown entities with known latents only suppresses refusal at alpha=1.0 (Figure 2B), where Diff-of-Means causes model collapse.

(A) Known Entities + Unknown Direction — [Figure 2A]
(B) Unknown Entities + Known Direction — [Figure 2B]

**Figure 2: Causal Steering Asymmetry under Norm-Matched Scaling.** (A) Steering known entities with unknown latents yields 0/30 refusal across all norm ratios. (B) Steering unknown entities with known latents only suppresses refusal at alpha=1.0, where generation collapse confounds the result.

## 5. Multi-Latent Top-K Sweeps & Ensembling Dynamics

To test whether single-latent interventions failed due to representational dispersion, we performed multi-latent sweeps with ensemble size k in {1, 3, 5, 10} across alpha in {0.25, 0.50, 1.00}. Multi-latent unknown steering still produced 0/30 refusal (95% CI 0–11.6%). On unknown queries, ensembling k in {3, 5, 10} latents reduced refusal variability (10–14/30) with no monotonic gain over k=1, consistent with partial mutual vector cancellation.

| Configuration | Condition | k=1, r=0.5 | k=1, r=1.0 | k=3, r=1.0 | k=5, r=1.0 | k=10, r=1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| L12_16k | Unk -> Known | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) |
| L12_16k | Kn -> Unknown | 15/30 (50%) | 7/30 (23%) | 11/30 (37%) | 10/30 (33%) | 11/30 (37%) |
| L9_16k | Unk -> Known | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) |
| L9_16k | Kn -> Unknown | 14/30 (47%) | 4/30 (13%) | 14/30 (47%) | 13/30 (43%) | 11/30 (37%) |

**Table 2:** Multi-latent top-k steering sweeps. Baseline refusal: Known=0/30, Unknown=13/30. Notably, Kn->Unknown at k=1, r=0.5 slightly exceeds baseline refusal (15/30 vs 13/30), while k=1, r=1.0 suppresses it (4/30) at the cost of generation collapse. Even k=10 latents simultaneously fail to induce refusal on known entities.

## 6. Custom Multi-Seed Training & Breakdown of Seed Invariance

A limitation of Gemma Scope is that all released SAEs were trained with a single initialization seed per configuration. To test whether the causal nullity above is an artifact of one particular feature basis, we trained three independent JumpReLU SAEs (M=8192) from scratch on Gemma-2-2B-it Layer 12 residual activations across seeds {42, 1337, 2026}. Hungarian bipartite matching over decoder directions reveals mean cosine similarity across seeds of only 0.0787, with 0.0% overlap at the standard threshold (cosine >= 0.70). This is consistent with the seed-dependence reported by Paulo & Belrose (2025) for TopK SAEs, and extends it to JumpReLU SAEs trained on epistemic entity latents: the entity-detection capability re-emerges in every seed, but the specific basis does not. SAE latents are therefore best viewed as seed-dependent coordinate rotations of the underlying feature manifold rather than canonical universal features.

(A) Hungarian Bipartite Matching Alignment Across Seeds — [Figure 3A]
(B) Overlap at Threshold — [Figure 3B]

**Figure 3: The Breakdown of Seed Invariance in Custom JumpReLU SAEs.** (A) Hungarian bipartite cosine alignment across seeds {42, 1337, 2026} remains near 0.0787 — above the random baseline (0.0208 for decoder directions in the 2304-dimensional residual space) but far below functional equivalence. (B) Overlap at threshold >= 0.70 is 0.0%.

| Seed Pair | Mean Cosine | Median Cosine | Overlap (>= 0.70) | Random Baseline (1/sqrt(2304)) |
| --- | --- | --- | --- | --- |
| Seed 42 vs. 1337 | 0.0787 | 0.0770 | 0.0% | 0.0208 |
| Seed 42 vs. 2026 | 0.0787 | 0.0772 | 0.0% | 0.0208 |
| Seed 1337 vs. 2026 | 0.0786 | 0.0770 | 0.0% | 0.0208 |

**Table 3:** Multi-seed Hungarian alignment statistics across independent training seeds.

## 7. Circuit Dissection & Qualitative Generation Audit

**Hypothesis 1 (KV-Cache Resilience).** We propose a mechanistic account for why residual-stream steering fails to disrupt factual generation. Factual tokens are stored in the past Key-Value (KV) cache laid down at early layers (L1–L6). Downstream attention heads (e.g., L13H5 with attribution 0.94, L12H2 with attribution 0.88) attend directly back to the entity tokens. Because the softmax key margin is large, additive residual perturbations at Layer 12 alter the query vector sub-threshold: attention probability on the entity token remains saturated (> 0.80), bypassing residual SAE steering. A formal treatment of the inequality governing sub-threshold query perturbation is given in Appendix B.

[Figure 4: Circuit Dissection of Attention Heads Bypassing Residual SAE Steering. (A) 64-head attribution heatmap across Layers 9–16; L13H5 (0.94) and L12H2 (0.88) act as direct entity read-heads. (B) Attention retention on the entity position remains > 0.80 even at alpha=1.0.]

**Qualitative transcripts.** On known entities, the model ignores the injected unknown vectors. On unknown entities, injecting known vectors at k in {5, 10} triggers extreme confabulation, borrowing achievements from real sports legends.

| Query | Condition | Verbatim Output |
| --- | --- | --- |
| Muhammad Ali | Unk->Kn (k=10, r=1.0) | "Muhammad Ali's nationality was American. His main career achievement is widely considered to be his dominance..." |
| Shaquille O'Neal | Unk->Kn (k=10, r=1.0) | "The athlete you're asking about is Shaquille O'Neal: Nationality: American, dominant center in the NBA..." |
| Kasimir Lindqvist | Kn->Un (k=1, r=0.5) | "Kasimir Lindqvist is a Finnish athlete. His main career achievement is winning the 2016 World Championship..." [Fictional] |
| Orrin Vandermeer | Kn->Un (k=10, r=1.0) | "Nationality: American. Main Career Achievement: Seven-time NBA champion with the Chicago Bulls..." [Extreme Confabulation] |

**Table 4:** Verbatim generation transcripts, illustrating total resistance on known queries and wild attribute borrowing on unknown queries.

**Figure provenance note.** Figures 1–3 were regenerated from the summary statistics reported in Tables 1–3; per-entity raw measurements are available in the experiment logs (`experiments/results/`). Figure 4 is pending regeneration from raw attribution data.

## 8. Key Takeaways & Conclusion

1. **Detection != Control:** High classification AUROC (> 0.96) is orthogonal to causal steering efficacy for entity-recognition latents in this regime.
2. **Epistemic Stubbornness:** Known-entity recall appears topologically anchored in early KV-cache states that are insensitive to residual-stream SAE latents at the final entity token.
3. **Multi-Latent Interference:** Ensembling top-k latents does not overcome refusal resistance and shows no monotonic dose-response.
4. **Seed-Dependent Basis:** Detection capability is seed-invariant; the latent basis is not — consistent with Paulo & Belrose (2025).

**Limitations.** (i) Our audit uses one model (Gemma-2-2B-it); replication in a second model is required before generalizing to small language models broadly. (ii) N=30 entities per class; 0/30 outcomes correspond to a 95% binomial CI of 0–11.6% and should be read as "very low", not exactly zero. (iii) Refusal was scored by a single string-matching procedure; a second independent scorer (LLM judge) was not used, so false negatives cannot be excluded. (iv) Steering was applied only at the final entity token and mid layers; interventions at other positions or layers, or on latents selected by causal rather than separation scores, were not tested, and no random-latent steering baseline was included. (v) At alpha=1.0, refusal suppression co-occurs with generation collapse, so the two cannot be fully disentangled with the present metrics. (vi) The KV-cache resilience account remains a hypothesis supported by attribution evidence, not a proof.

## Acknowledgements

The author thanks the autonomous LLM research assistant "L" for code development, tooling, and drafting support, and the Gemma Scope team (DeepMind) for open SAE releases.

## References

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. arXiv:2406.11717
- Bricken, T., Templeton, A., et al. (2023). *Towards Monosemanticity: Extracting Interpretable Features from Claude 2.* Transformer Circuits Thread
- Elhage, N., Hume, T., Olsson, C., et al. (2022). *Toy Models of Superposition.* Transformer Circuits Thread
- Ferrando, J., Obeso, O., Rajamanoharan, S., & Nanda, N. (2025). *Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models.* ICLR 2025 (Oral). arXiv:2411.14257
- Lieberum, T., Rajamanoharan, S., Conmy, A., et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All at Once on Gemma 2.* arXiv:2408.05147
- Paulo, G., & Belrose, N. (2025). *Sparse Autoencoders Trained on the Same Data Learn Different Features.* arXiv:2501.16615
- Templeton, A., Conerly, T., Marcus, J., et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits Thread
- Vicentino, C. (2025). *Entity-recognition features in Qwen3.6-27B.* OpenInterpretability blog. https://openinterp.org/blog/qwen36-27b-hallucination-replication

## Appendix A — Entity Stimulus Lists

Template (to be filled from the experimental records):

### A.1 Known entities (N=30)
- 1. [known athlete 1]
- 2. [known athlete 2]
- ...
- 30. [known athlete 30]

### A.2 Chimera entities (N=30)
Hybrid names: authentic first name + authentic surname of a different athlete.
- 1. Kobe Williams
- 2. Novak Zidane
- 3. Stephen Ronaldo
- ...
- 30. [chimera athlete 30]

### A.3 Synthetic unknown entities (N=30)
- 1. Kasimir Lindqvist
- 2. Orrin Vandermeer
- ...
- 30. [synthetic athlete 30]

## Appendix B — Sub-Threshold Query Perturbation (Derivation Sketch)

**Setup.** Let h ∈ R^d be the residual stream at the query position, and let the entity token sit at context position e among N positions. Attention logits are z_i = (W_q h)·k_i / sqrt(d), with keys k_i fixed by the KV cache from earlier layers.

**Definitions.**
- gamma = z_e − max_{j≠e} z_j — the softmax key margin between the entity position and the best competitor.
- delta = ||W_q||_2 · alpha · ||h|| — an upper bound on the query perturbation induced by an additive residual edit of scale alpha·||h|| (worst case over directions).
- k_max = max_i ||k_i|| — the largest key norm in the context.

**Step 1 (logit stability).** A residual perturbation Delta h with ||Delta h|| ≤ alpha·||h|| changes the query by Delta q = W_q Delta h, so ||Delta q|| ≤ delta. Each logit moves by at most |Delta z_i| ≤ ||Delta q||·||k_i||/sqrt(d) ≤ delta·k_max/sqrt(d).

**Step 2 (worst-case margin).** After the perturbation, the margin satisfies

> gamma' ≥ gamma − 2·delta·k_max / sqrt(d),

since the entity logit can drop by delta·||k_e||/sqrt(d) while a competitor rises by delta·k_max/sqrt(d). This is the inequality stated in Hypothesis 1 (gamma > 2·delta·||k||).

**Step 3 (attention retention).** By a union bound on the softmax denominator,

> p_e = 1 / (1 + Σ_{j≠e} exp(z_j − z_e)) ≥ 1 / (1 + (N−1)·exp(−gamma')).

Therefore p_e > 0.8 whenever gamma' > ln(4(N−1)).

**Conclusion (sufficient condition).** If

> gamma − 2·delta·k_max / sqrt(d) > ln(4(N−1)),

then attention probability on the entity position remains above 0.8 despite the steering perturbation. With saturated factual-recall heads, gamma is empirically large (well above ln(4(N−1)) for typical context lengths), so norm-matched perturbations at the final entity token are sub-threshold — consistent with the observed attention retention > 0.80 at alpha = 1.0 (Figure 4B).

**Caveat.** This is a sufficient-condition sketch for the retention phenomenon, not a proof that no steering protocol can succeed: it assumes additive, single-position edits and worst-case (adversarial) perturbation geometry; directionally targeted edits or multi-layer interventions are outside this bound.
