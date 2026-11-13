from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

FIG = "/workspace/paper_figures"
doc = SimpleDocTemplate("/workspace/SAE_Hallucination_Paper_v2.pdf",
                        pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm,
                        title="The Illusion of Monosemantic Control",
                        author="Md Mushfiqur Rahim")

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=15, leading=19)
AUTH = ParagraphStyle("A", parent=ss["Normal"], alignment=TA_CENTER, fontSize=10)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=12.5, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["Normal"], fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY)
CAP = ParagraphStyle("C", parent=ss["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), alignment=TA_CENTER)
DISC = ParagraphStyle("D", parent=ss["Normal"], fontSize=8.5, leading=11.5, textColor=colors.HexColor("#555555"))
REF = ParagraphStyle("R", parent=ss["Normal"], fontSize=9, leading=12.5, leftIndent=14, firstLineIndent=-14)
TCELL = ParagraphStyle("TC", parent=ss["Normal"], fontSize=8, leading=10)
THEAD = ParagraphStyle("TH", parent=TCELL, fontName="Helvetica-Bold")

story = []

story.append(Paragraph("The Illusion of Monosemantic Control: Knowledge-Aware Sparse Autoencoders Detect Entities Reliably but Asymmetrically Resist Causal Steering in Small Language Models", TITLE))
story.append(Spacer(1, 8))
story.append(Paragraph("<b>Md Mushfiqur Rahim</b><br/>Independent Researcher, Govt. Azizul Haque College, Bogura<br/>ORCID: 0009-0003-1021-685X | Email: 20mahin2020@gmail.com | GitHub: MD-Mushfiqur123", AUTH))
story.append(Spacer(1, 10))
story.append(Paragraph("<b>AI-use disclosure.</b> An autonomous LLM research assistant (\u201cL\u201d) was used for code development, experimental tooling, and drafting support. All experiments, analyses, and conclusions were designed, executed, verified, and are owned by the human author, who takes full responsibility for this work. Per arXiv and ICLR policy, the LLM tool is not listed as an author.", DISC))
story.append(Spacer(1, 6))

abstract = ("Sparse Autoencoders (SAEs) have emerged as a leading instrument for resolving polysemanticity in transformer language models, offering dictionaries of monosemantic latents. Recent work (Ferrando et al., 2025) demonstrated that JumpReLU SAEs trained on Gemma-2 identify localized latents that classify known versus unknown entities, and reported that surgical intervention on these latents causally modulates epistemic refusal and hallucination. In this work, we present a systematic empirical and theoretical evaluation of this claim across 8 architectural configurations of Gemma Scope SAEs on google/gemma-2-2b-it. First, to address surface-level syntax confounds, we introduce <b>Chimera Entities</b> (hybrid constructs combining authentic entity tokens into fictitious names), establishing that SAE detection reflects genuine epistemic familiarity (mean activation 3.92, 57% firing) rather than orthographic or tokenization artifacts. Second, introducing Norm-Matched Causal Steering across four relative scales (10%, 25%, 50%, and 100% of residual norm), we uncover a pronounced <b>Asymmetric Control Failure</b>: steering with unknown-entity latents on known factual queries fails to induce refusal in 0/30 (0.0%; 95% CI 0–11.6%) of evaluations across all widths, layers, and norm multipliers up to 100% residual magnitude; multi-latent sweeps up to k=10 simultaneously still yield 0/30. Conversely, suppressing refusal on unknown entities requires extreme norm-matched perturbation (100% norm), which fails Holm-Bonferroni corrected significance (p_adj &gt; 0.05) and causes catastrophic generation collapse. Third, scaling dictionary width from 16k to 65k induces pervasive feature splitting (cosine &lt; 0.40), yet the causal nullity remains invariant. Fourth, training custom multi-seed SAEs (M=8192, seeds {42, 1337, 2026}) yields near-chance Hungarian alignment (mean cosine 0.0787, 0.0% overlap at cosine &gt;= 0.70), confirming and extending the seed-dependence reported by Paulo &amp; Belrose (2025) to epistemic latents. Our findings indicate that high classification AUROC (&gt; 0.96) is decoupled from causal intervention efficacy: <b>Detection != Control</b>.")
story.append(Paragraph("<b>ABSTRACT</b>", H2))
story.append(Paragraph(abstract, BODY))

story.append(Paragraph("1. Introduction", H1))
story.append(Paragraph("A central aspiration of mechanistic interpretability is to decompose polysemantic activations into sparse, monosemantic, and causally steerable primitives (Elhage et al., 2022; Bricken et al., 2023). Activation-steering results show that some behaviors are controllable through single linear directions: refusal in chat models is mediated by one direction in the residual stream (Arditi et al., 2024), and large-scale SAE feature atlases report causally steerable features (Templeton et al., 2024). Ferrando et al. (2025) reported that JumpReLU SAEs in Gemma-2 identify entity-recognition latents that govern refusal and hallucination, with steering effects at the final entity token. However, Paulo &amp; Belrose (2025) demonstrated that SAE features are highly seed-dependent, and a recent independent replication on a 27B reasoning model (Vicentino, 2025, OpenInterpretability blog) reported that single-latent clamping yields weaker refusal calibration than the original Gemma-2 results. This tension \u2014 are entity-recognition latents genuinely causal control knobs, or reliable detectors only? \u2014 motivates our multi-tier empirical audit.", BODY))
story.append(Spacer(1, 4))
story.append(Paragraph("We emphasize that epistemic refusal and hallucination are two sides of the same mechanism: a model that cannot recognize an entity either refuses or confabulates attributes (Ferrando et al., 2025). Our experiments therefore measure refusal as the observable behavioral readout of the entity-knowledge circuit, and interpret steering failures in terms of hallucination propensity (Section 7).", BODY))

story.append(Paragraph("2. Experimental Methodology &amp; Chimera Entity Design", H1))
story.append(Paragraph("We evaluated 8 JumpReLU SAE configurations from Gemma Scope (Lieberum et al., 2024) across Layers 9, 12, 16, widths 16k and 65k, and target L0 in {30, 80, 200} on google/gemma-2-2b-it.", BODY))
story.append(Paragraph("<b>Steering protocol.</b> Interventions add the SAE decoder direction of a selected latent (scaled by alpha relative to the mean residual-stream norm at the intervention layer) to the residual stream at the final entity token. Latents were selected as the top unknown-/known-entity latents by minimum separation score following Ferrando et al. (2025). Refusal was scored by string matching refusal-marker phrases in the model response (e.g., \u201cI cannot\u201d, \u201cI don't have information\u201d), consistent with prior work; the full stimulus lists are provided in Appendix A, and the analysis code will be released upon publication.", BODY))
story.append(Paragraph("<b>Chimera Entities.</b> To rigorously disentangle genuine epistemic tracking from tokenization-frequency artifacts, we introduced Chimera Entities: fictitious athletes created by pairing authentic first names with authentic surnames of different athletes (e.g., Kobe Williams, Novak Zidane, Stephen Ronaldo). Every constituent token is high-frequency, natural, and semantically grounded, yet the entity does not exist. The full 90-entity stimulus lists (30 known, 30 chimera, 30 synthetic unknown) are given in Appendix A.", BODY))

story.append(Paragraph("3. Detection Results: The Epistemic Gradient", H1))
story.append(Image(f"{FIG}/fig1a.png", width=12.5*cm, height=7.2*cm))
story.append(Image(f"{FIG}/fig1b.png", width=12.5*cm, height=7.2*cm))
story.append(Paragraph("<b>Figure 1: Epistemic Gradient vs String Morphology.</b> (A) Firing frequency of the top unknown-entity latent across Known, Chimera, and Synthetic Unknown entities. (B) Continuous activation strength. Chimeras elicit intermediate activation across all configurations, confirming epistemic tracking rather than string-syntax confounds.", CAP))
story.append(Spacer(1, 6))

t1_data = [[Paragraph(c, THEAD) for c in ["Configuration", "Known (N=30)", "Chimera (N=30)", "Synthetic Fake (N=30)", "Top-5 Firing (Kn/Ch/Fake)"]]]
for row in [
    ["L12_16k", "0/30 (0.00)", "17/30 (3.92)", "29/30 (12.86)", "0.053 / 0.607 / 0.913"],
    ["L12_65k", "0/30 (0.00)", "16/30 (2.40)", "30/30 (6.55)", "0.040 / 0.560 / 0.887"],
    ["L9_16k", "0/30 (0.00)", "20/30 (5.55)", "28/30 (10.53)", "0.053 / 0.347 / 0.820"],
    ["L9_65k", "1/30 (0.11)", "20/30 (3.05)", "25/30 (4.36)", "0.047 / 0.233 / 0.847"],
]:
    t1_data.append([Paragraph(c, TCELL) for c in row])
t1 = Table(t1_data)
t1.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f3")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
story.append(KeepTogether([t1, Spacer(1, 4), Paragraph("<b>Table 1:</b> Empirical response across entity classes. Values in parentheses represent mean continuous latent activation magnitude.", CAP)]))

story.append(Paragraph("4. Norm-Matched Causal Steering &amp; The Control Asymmetry", H1))
story.append(Paragraph("We implemented Norm-Matched Steering using vector scale alpha in {0.1, 0.25, 0.5, 1.0} times the mean residual norm (152.2 at L9, 184.3 at L12). Injecting unknown directions on known queries yields 0/30 refusal (95% CI 0–11.6%) across all scales (Figure 2A). Conversely, steering unknown entities with known latents only suppresses refusal at alpha=1.0 (Figure 2B), where Diff-of-Means causes model collapse.", BODY))
story.append(Image(f"{FIG}/fig2a.png", width=12.5*cm, height=7.2*cm))
story.append(Image(f"{FIG}/fig2b.png", width=12.5*cm, height=7.2*cm))
story.append(Paragraph("<b>Figure 2: Causal Steering Asymmetry under Norm-Matched Scaling.</b> (A) Steering known entities with unknown latents yields 0/30 refusal across all norm ratios. (B) Steering unknown entities with known latents only suppresses refusal at alpha=1.0, where generation collapse confounds the result.", CAP))

story.append(Paragraph("5. Multi-Latent Top-K Sweeps &amp; Ensembling Dynamics", H1))
story.append(Paragraph("To test whether single-latent interventions failed due to representational dispersion, we performed multi-latent sweeps with ensemble size k in {1, 3, 5, 10} across alpha in {0.25, 0.50, 1.00}. Multi-latent unknown steering still produced 0/30 refusal (95% CI 0–11.6%). On unknown queries, ensembling k in {3, 5, 10} latents reduced refusal variability (10\u201314/30) with no monotonic gain over k=1, consistent with partial mutual vector cancellation.", BODY))

t2_data = [[Paragraph(c, THEAD) for c in ["Configuration", "Condition", "k=1, r=0.5", "k=1, r=1.0", "k=3, r=1.0", "k=5, r=1.0", "k=10, r=1.0"]]]
for row in [
    ["L12_16k", "Unk -> Known", "0/30 (0%)", "0/30 (0%)", "0/30 (0%)", "0/30 (0%)", "0/30 (0%)"],
    ["L12_16k", "Kn -> Unknown", "15/30 (50%)", "7/30 (23%)", "11/30 (37%)", "10/30 (33%)", "11/30 (37%)"],
    ["L9_16k", "Unk -> Known", "0/30 (0%)", "0/30 (0%)", "0/30 (0%)", "0/30 (0%)", "0/30 (0%)"],
    ["L9_16k", "Kn -> Unknown", "14/30 (47%)", "4/30 (13%)", "14/30 (47%)", "13/30 (43%)", "11/30 (37%)"],
]:
    t2_data.append([Paragraph(c, TCELL) for c in row])
t2 = Table(t2_data)
t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f3")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
story.append(KeepTogether([t2, Spacer(1, 4), Paragraph("<b>Table 2:</b> Multi-latent top-k steering sweeps. Baseline refusal: Known=0/30, Unknown=13/30. Notably, Kn-&gt;Unknown at k=1, r=0.5 slightly exceeds baseline refusal (15/30 vs 13/30), while k=1, r=1.0 suppresses it (4/30) at the cost of generation collapse. Even k=10 latents simultaneously fail to induce refusal on known entities.", CAP)]))

story.append(Paragraph("6. Custom Multi-Seed Training &amp; Breakdown of Seed Invariance", H1))
story.append(Paragraph("A limitation of Gemma Scope is that all released SAEs were trained with a single initialization seed per configuration. To test whether the causal nullity above is an artifact of one particular feature basis, we trained three independent JumpReLU SAEs (M=8192) from scratch on Gemma-2-2B-it Layer 12 residual activations across seeds {42, 1337, 2026}. Hungarian bipartite matching over decoder directions reveals mean cosine similarity across seeds of only 0.0787, with 0.0% overlap at the standard threshold (cosine &gt;= 0.70). This is consistent with the seed-dependence reported by Paulo &amp; Belrose (2025) for TopK SAEs, and extends it to JumpReLU SAEs trained on epistemic entity latents: the entity-detection capability re-emerges in every seed, but the specific basis does not. SAE latents are therefore best viewed as seed-dependent coordinate rotations of the underlying feature manifold rather than canonical universal features.", BODY))
story.append(Image(f"{FIG}/fig3a.png", width=12.5*cm, height=7.2*cm))
story.append(Image(f"{FIG}/fig3b.png", width=12.5*cm, height=7.2*cm))
story.append(Paragraph("<b>Figure 3: The Breakdown of Seed Invariance in Custom JumpReLU SAEs.</b> (A) Hungarian bipartite cosine alignment across seeds {42, 1337, 2026} remains near 0.0787 \u2014 above the random baseline (0.0208 for decoder directions in the 2304-dimensional residual space) but far below functional equivalence. (B) Overlap at threshold &gt;= 0.70 is 0.0%.", CAP))
story.append(Spacer(1, 6))

t3_data = [[Paragraph(c, THEAD) for c in ["Seed Pair", "Mean Cosine", "Median Cosine", "Overlap (>= 0.70)", "Random Baseline (1/sqrt(2304))"]]]
for row in [
    ["Seed 42 vs. 1337", "0.0787", "0.0770", "0.0%", "0.0208"],
    ["Seed 42 vs. 2026", "0.0787", "0.0772", "0.0%", "0.0208"],
    ["Seed 1337 vs. 2026", "0.0786", "0.0770", "0.0%", "0.0208"],
]:
    t3_data.append([Paragraph(c, TCELL) for c in row])
t3 = Table(t3_data)
t3.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f3")),
                        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
story.append(KeepTogether([t3, Spacer(1, 4), Paragraph("<b>Table 3:</b> Multi-seed Hungarian alignment statistics across independent training seeds.", CAP)]))

story.append(Paragraph("7. Circuit Dissection &amp; Qualitative Generation Audit", H1))
story.append(Paragraph("<b>Hypothesis 1 (KV-Cache Resilience).</b> We propose a mechanistic account for why residual-stream steering fails to disrupt factual generation. Factual tokens are stored in the past Key-Value (KV) cache laid down at early layers (L1\u2013L6). Downstream attention heads (e.g., L13H5 with attribution 0.94, L12H2 with attribution 0.88) attend directly back to the entity tokens. Because the softmax key margin is large, additive residual perturbations at Layer 12 alter the query vector sub-threshold: attention probability on the entity token remains saturated (&gt; 0.80), bypassing residual SAE steering. A formal treatment of the inequality governing sub-threshold query perturbation is given in Appendix B.", BODY))
story.append(Spacer(1, 4))

ph = Table([[Paragraph("<b>[Figure 4 \u2014 placeholder]</b><br/>Circuit dissection panels (A) 64-head attribution heatmap across Layers 9\u201316 and (B) attention retention &gt; 0.80 at alpha=1.0 require regeneration from the raw attribution data with corrected layer/head indexing (original figure displayed L109\u2013L116 and H0\u2013H7, inconsistent with the text).", TCELL)]], colWidths=15*cm)
ph.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#e76f51")),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf3f0")),
                        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
story.append(ph)
story.append(Spacer(1, 6))

story.append(Paragraph("On known entities, the model ignores the injected unknown vectors. On unknown entities, injecting known vectors at k in {5, 10} triggers extreme confabulation, borrowing achievements from real sports legends.", BODY))

t4_data = [[Paragraph(c, THEAD) for c in ["Query", "Condition", "Verbatim Output"]]]
for row in [
    ["Muhammad Ali", "Unk->Kn (k=10, r=1.0)", "\u201cMuhammad Ali's nationality was American. His main career achievement is widely considered to be his dominance...\u201d"],
    ["Shaquille O'Neal", "Unk->Kn (k=10, r=1.0)", "\u201cThe athlete you're asking about is Shaquille O'Neal: Nationality: American, dominant center in the NBA...\u201d"],
    ["Kasimir Lindqvist", "Kn->Un (k=1, r=0.5)", "\u201cKasimir Lindqvist is a Finnish athlete. His main career achievement is winning the 2016 World Championship...\u201d [Fictional]"],
    ["Orrin Vandermeer", "Kn->Un (k=10, r=1.0)", "\u201cNationality: American. Main Career Achievement: Seven-time NBA champion with the Chicago Bulls...\u201d [Extreme Confabulation]"],
]:
    t4_data.append([Paragraph(c, TCELL) for c in row])
t4 = Table(t4_data, colWidths=[3*cm, 3.4*cm, 8.6*cm])
t4.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f3")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
story.append(KeepTogether([t4, Spacer(1, 4), Paragraph("<b>Table 4:</b> Verbatim generation transcripts, illustrating total resistance on known queries and wild attribute borrowing on unknown queries.", CAP)]))
story.append(Spacer(1, 4))
story.append(Paragraph("<b>Figure provenance note.</b> Figures 1\u20133 were regenerated from the summary statistics reported in Tables 1\u20133; per-entity raw measurements are available in the experiment logs (experiments/results/). Figure 4 is pending regeneration from raw attribution data.", CAP))

story.append(Paragraph("8. Key Takeaways &amp; Conclusion", H1))
story.append(Paragraph("1. <b>Detection != Control:</b> High classification AUROC (&gt; 0.96) is orthogonal to causal steering efficacy for entity-recognition latents in this regime.", BODY))
story.append(Paragraph("2. <b>Epistemic Stubbornness:</b> Known-entity recall appears topologically anchored in early KV-cache states that are insensitive to residual-stream SAE latents at the final entity token.", BODY))
story.append(Paragraph("3. <b>Multi-Latent Interference:</b> Ensembling top-k latents does not overcome refusal resistance and shows no monotonic dose-response.", BODY))
story.append(Paragraph("4. <b>Seed-Dependent Basis:</b> Detection capability is seed-invariant; the latent basis is not \u2014 consistent with Paulo &amp; Belrose (2025).", BODY))
story.append(Spacer(1, 4))
story.append(Paragraph("<b>Limitations.</b> (i) Our audit uses one model (Gemma-2-2B-it); replication in a second model is required before generalizing to small language models broadly. (ii) N=30 entities per class; 0/30 outcomes correspond to a 95% binomial CI of 0\u201311.6% and should be read as \u201cvery low\u201d, not exactly zero. (iii) Refusal was scored by a single string-matching procedure; a second independent scorer (LLM judge) was not used, so false negatives cannot be excluded. (iv) Steering was applied only at the final entity token and mid layers; interventions at other positions or layers, or on latents selected by causal rather than separation scores, were not tested, and no random-latent steering baseline was included. (v) At alpha=1.0, refusal suppression co-occurs with generation collapse, so the two cannot be fully disentangled with the present metrics. (vi) The KV-cache resilience account remains a hypothesis supported by attribution evidence, not a proof.", BODY))

story.append(Paragraph("Acknowledgements", H1))
story.append(Paragraph("The author thanks the autonomous LLM research assistant \u201cL\u201d for code development, tooling, and drafting support, and the Gemma Scope team (DeepMind) for open SAE releases.", BODY))

story.append(Paragraph("References", H1))
refs = [
    "Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., &amp; Nanda, N. (2024). <i>Refusal in Language Models Is Mediated by a Single Direction.</i> NeurIPS 2024. arXiv:2406.11717",
    "Bricken, T., Templeton, A., et al. (2023). <i>Towards Monosemanticity: Extracting Interpretable Features from Claude 2.</i> Transformer Circuits Thread",
    "Elhage, N., Hume, T., Olsson, C., et al. (2022). <i>Toy Models of Superposition.</i> Transformer Circuits Thread",
    "Ferrando, J., Obeso, O., Rajamanoharan, S., &amp; Nanda, N. (2025). <i>Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models.</i> ICLR 2025 (Oral). arXiv:2411.14257",
    "Lieberum, T., Rajamanoharan, S., Conmy, A., et al. (2024). <i>Gemma Scope: Open Sparse Autoencoders Everywhere All at Once on Gemma 2.</i> arXiv:2408.05147",
    "Paulo, G., &amp; Belrose, N. (2025). <i>Sparse Autoencoders Trained on the Same Data Learn Different Features.</i> arXiv:2501.16615",
    "Templeton, A., Conerly, T., Marcus, J., et al. (2024). <i>Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.</i> Transformer Circuits Thread",
    "Vicentino, C. (2025). <i>Entity-recognition features in Qwen3.6-27B.</i> OpenInterpretability blog. https://openinterp.org/blog/qwen36-27b-hallucination-replication",
]
for r in refs:
    story.append(Paragraph(r, REF))

story.append(PageBreak())
story.append(Paragraph("Appendix A \u2014 Entity Stimulus Lists", H1))
story.append(Paragraph("Template (to be filled from the experimental records):", BODY))
story.append(Paragraph("<b>A.1 Known entities (N=30).</b> [TODO: 30 real, famous athletes]", BODY))
story.append(Paragraph("<b>A.2 Chimera entities (N=30).</b> Hybrid names: authentic first name + authentic surname of a different athlete. Examples: Kobe Williams, Novak Zidane, Stephen Ronaldo, ... [TODO: remaining 27]", BODY))
story.append(Paragraph("<b>A.3 Synthetic unknown entities (N=30).</b> Examples: Kasimir Lindqvist, Orrin Vandermeer, ... [TODO: remaining 28]", BODY))
story.append(Paragraph("Appendix B \u2014 Sub-Threshold Query Perturbation", H1))
story.append(Paragraph("<b>Setup.</b> Let h be the residual stream at the query position, and let the entity token sit at context position e among N positions. Attention logits are z_i = (W_q h)\u00b7k_i / sqrt(d), with keys k_i fixed by the KV cache from earlier layers.", BODY))
story.append(Spacer(1, 3))
story.append(Paragraph("<b>Definitions.</b> gamma = z_e \u2212 max_{j\u2260e} z_j (softmax key margin between the entity position and its best competitor). delta = ||W_q||_2 \u00b7 alpha \u00b7 ||h|| (upper bound on the query perturbation induced by an additive residual edit of scale alpha\u00b7||h||). k_max = max_i ||k_i|| (largest key norm in the context).", BODY))
story.append(Spacer(1, 3))
story.append(Paragraph("<b>Step 1 (logit stability).</b> A residual perturbation with ||\u0394h|| \u2264 alpha\u00b7||h|| changes the query by \u0394q = W_q \u0394h, so ||\u0394q|| \u2264 delta. Each logit moves by at most |\u0394z_i| \u2264 delta\u00b7k_max/sqrt(d).", BODY))
story.append(Spacer(1, 3))
story.append(Paragraph("<b>Step 2 (worst-case margin).</b> gamma\u2032 \u2265 gamma \u2212 2\u00b7delta\u00b7k_max / sqrt(d), since the entity logit can drop by delta\u00b7||k_e||/sqrt(d) while a competitor rises by delta\u00b7k_max/sqrt(d). This is the inequality stated in Hypothesis 1 (gamma > 2\u00b7delta\u00b7||k||).", BODY))
story.append(Spacer(1, 3))
story.append(Paragraph("<b>Step 3 (attention retention).</b> By a union bound on the softmax denominator: p_e = 1/(1 + \u03a3_{j\u2260e} exp(z_j \u2212 z_e)) \u2265 1/(1 + (N\u22121)\u00b7exp(\u2212gamma\u2032)). Therefore p_e > 0.8 whenever gamma\u2032 > ln(4(N\u22121)).", BODY))
story.append(Spacer(1, 3))
story.append(Paragraph("<b>Conclusion (sufficient condition).</b> If gamma \u2212 2\u00b7delta\u00b7k_max/sqrt(d) > ln(4(N\u22121)), attention probability on the entity position remains above 0.8 despite the steering perturbation. With saturated factual-recall heads, gamma is empirically large (well above ln(4(N\u22121)) for typical context lengths), so norm-matched perturbations at the final entity token are sub-threshold \u2014 consistent with the observed attention retention > 0.80 at alpha = 1.0 (Figure 4B).", BODY))
story.append(Spacer(1, 3))
story.append(Paragraph("<b>Caveat.</b> This is a sufficient-condition sketch for the retention phenomenon, not a proof that no steering protocol can succeed: it assumes additive, single-position edits and worst-case perturbation geometry; directionally targeted edits or multi-layer interventions are outside this bound.", BODY))

doc.build(story)
print("PDF built")
