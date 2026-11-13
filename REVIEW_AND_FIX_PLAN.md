# REVIEW_AND_FIX_PLAN.md
## সম্পূর্ণ সৎ Review — সব দোষ + প্রতিটার Fix Plan
**Paper:** The Illusion of Monosemantic Control (v2) | **Current score: 52/100** | **Target: 80+**

---

# PART 1 — সব দোষ (সম্পূর্ণ সৎ তালিকা)

## 🔴 A. Fatal — এগুলো ঠিক না হলে paper টিকে না

### A1. "Code is provided in the repository" — কিন্তু repository তে experiment code নেই
- **দোষ:** Paper এর Methodology section এ লেখা "full lists and code are provided in the repository"। বাস্তবে repo তে শুধু figure/PDF build script আছে — যে code দিয়ে experiment গুলো চালানো হয়েছিল (steering, SAE training, Hungarian matching, refusal scoring) সেটা নেই।
- **কেন বিপজ্জনক:** এটা paper এর একটা factual claim, যা এখন মিথ্যা। একজন reviewer repo check করলেই ধরবে — আর একবার একটা claim মিথ্যা ধরা পড়লে বাকি সব claim এর উপরও সন্দেহ আসে।
- **Fix:** Experiment code + run logs repo তে push, অথবা বাক্যটা বদলে লেখো "code will be released upon publication" (দুর্বল, কিন্তু সৎ)।

### A2. n=30, কোনো confidence interval নেই
- **দোষ:** মূল ফলাফল "0/30" — কিন্তু কোথাও লেখা নেই এই sample size কী দেখার জন্য যথেষ্ট কিনা।
- **গণিত:** 0/30 হলে true refusal rate এর 95% binomial CI ≈ 0–11.6%। মানে "0/30" দেখলেও সত্যি করে বলা যায় না rate ঠিক 0 — শুধু "very low"। এই ভাষায় claim করতে হবে।
- **Fix:** প্রতিটা 0/30 এর পাশে CI লেখো। আর দাবির ভাষা বদলাও: "fails to induce refusal (0/30, 95% CI 0–11.6%)"।

### A3. Refusal মাপা হয়েছে একটাই পদ্ধতিতে (string matching)
- **দোষ:** String matching এ false negative থাকে — model অনেক সময় ভিন্ন ভাষায় refuse করে ("As an AI...", "I'm unable to...", বা প্রশ্ন এড়িয়ে যাওয়া)। একটা মিস হলেই "0/30" ভেঙে পড়ে।
- **Fix:** Second scorer যোগ করো — ছোট classifier বা LLM judge দিয়ে সব ১২০টা response re-score করো। দুটো scorer একরকম বললে ফলাফল অনেক শক্ত হয়।

### A4. Figure 4 placeholder — অথচ paper এর সবচেয়ে বড় তত্ত্ব তার উপর দাঁড়িয়ে আছে
- **দোষ:** Hypothesis 1 (KV-Cache Resilience) এর প্রমাণ হলো attribution 0.94/0.88 আর attention >0.80 — সবই Figure 4 এ, যেটা এখনো আঁকা হয়নি। আর পুরনো figure এ layer/head indexing ভুল ছিল (L109–L116, H0–H7)।
- **Fix:** Raw attribution data থেকে figure regenerate + method লেখা (কোন attribution technique, কোন prompt set)।

### A5. সবচেয়ে বড় counter-argument address করা হয়নি: Arditi et al.
- **দোষ:** Arditi et al. (2024) দেখিয়েছে refusal একটা single direction (diff-in-means) দিয়ে steerable। তাহলে তোমার SAE latent fail করার মানে হতে পারে **latent টাই ভুল direction** — circuit ভাঙা না। এই ব্যাখ্যা ছাড়া "Detection != Control" overreach।
- **Fix:** একই setup এ Arditi-style refusal direction baseline চালাও। কাজ করলে (সম্ভবত করবে) তোমার দাবি ঝালময় হয়ে যায়: "refusal IS steerable, কিন্তু SAE entity latents দিয়ে না" — এটা অনেক বেশি শক্তিশালী ও সুন্দর ফলাফল।

## 🟠 B. Methodological দোষ

### B1. Steering শুধু final entity token এ
- Intervention শুধু এক position এ। Position ablation (প্রথম token, entity এর সব token, generation চলাকালীন প্রতি token) ছাড়া "residual steering fails" বলা যায় না — শুধু "এই position এ fails"।

### B2. একটাই model (Gemma-2-2B-it)
- Title এ "Small Language Models" (plural), কিন্তু এক model। Ferrando et al. ৩টা model এ replicate করেছিল। অন্তত একটা আরেকটা model (Gemma-2-9B বা Llama-3.1-8B) এ মূল steering result টা দেখা দরকার।

### B3. Latent selection separation score দিয়ে, causal score দিয়ে না
- যে latents বাছা হয়েছে সেগুলো classification এ ভালো, কিন্তু causal relevance এর ভিত্তিতে বাছা হয়নি। Ferrando et al. এর random-latent steering baseline টাও তোমার paper এ নেই — যোগ করা দরকার (random latents দিয়ে steering করলেও কি 0/30 থেকে যায়? হলে বোঝা যায় intervention টাই দুর্বল)।

### B4. alpha=1.0 এ generation collapse confound
- "Suppression only at alpha=1.0, where model collapses" — collapse হলে refusal কমাটা কি suppression, নাকি model ভেঙে যাওয়া? এই দুটো আলাদা করে মাপতে হবে (যেমন: perplexity/fluency metric সাথে রিপোর্ট করা)।

### B5. Steering results এ seed variance নেই
- Phase 2 এ দেখালে basis seed-dependent, কিন্তু Phase 1 এর steering results একটা SAE seed থেকে। অন্তত ৩ seed এর latent দিয়ে steering repeat করলে ভালো হয়।

### B6. Statistics অসম্পূর্ণ
- Holm-Bonferroni correction উল্লেখ আছে কিন্তু কোন test, কোন p_adj values — কিছুই রিপোর্ট নেই। Error bars কোথাও নেই।

## 🟡 C. Presentation দোষ

- **C1.** "exhaustive" শব্দটা overclaim — 8 configs, 1 model। "systematic" লেখো।
- **C2.** Table 1 এ L9_65k Known=1/30 — কোন entity তে সেই firing? এক লাইনে বলো।
- **C3.** Baseline refusal 13/30 = unknown entities এ 57% ক্ষেত্রে refuse করে না — বাকি 57% কি hallucinate করে? এটা quantified করলে title এর "Hallucination" শব্দটা সত্যিকারের হয়।
- **C4.** Appendix A, B খালি — কিন্তু text দাবি করে "are given in Appendix A"।
- **C5.** Figure regeneration থেকে summary stats ব্যবহার হয়েছে (raw per-entity data না) — paper বা repo তে এটা স্পষ্ট করে লেখা উচিত।

## ✅ D. যা ঠিক আছে (এগুলো না ভাঙা)

- Chimera Entities design — সবচেয়ে original অংশ
- v2 এর numbering/references/citation verification
- AI-use disclosure (arXiv-safe)
- Norm-matched steering concept
- Multi-seed Hungarian matching methodology

---

# PART 2 — FIX PLAN (ধাপে ধাপে)

## Phase 0 — আজ, 0 compute, আমি করতে পারি ⚡
| # | কাজ | সময় | Score gain |
|---|---|---|---|
| 0.1 | "code provided" দাবি ঠিক করা (code push না হওয়া পর্যন্ত ভাষা বদলানো) | 5 min | +3 |
| 0.2 | সব 0/30 এর পাশে binomial CI যোগ | 30 min | +3 |
| 0.3 | "exhaustive" → "systematic" + বাকি overclaim নামানো | 15 min | +1 |
| 0.4 | C2 (L9_65k 1/30) ও C3 (57% non-refusal) এর জায়গা তৈরি + Limitations এ B1, B2, B3, B4 যোগ | 30 min | +2 |
| 0.5 | Appendix A এর জন্য entity list template বানানো (তোমার list দিলে ভরে দিবো) | 20 min | +2 |

**Phase 0 শেষে projected: ~63/100**

## Phase 1 — ছোট compute (molab free GPU তে চলবে) 🔬
| # | কাজ | সময় | Score gain |
|---|---|---|---|
| 1.1 | Experiment code + logs repo তে push (A1 পুরো সমাধান) | 1–2 hr | +5 |
| 1.2 | Second refusal scorer (LLM judge) দিয়ে সব response re-score (A3) | 2–3 hr | +6 |
| 1.3 | Random-latent steering baseline (B3) | 1–2 hr | +3 |
| 1.4 | Position ablation: first token / all tokens (B1) | 2 hr | +3 |
| 1.5 | alpha=1.0 collapse আলাদা করে মাপা (perplexity metric) (B4) | 1 hr | +2 |

**Phase 1 শেষে projected: ~80/100**

## Phase 2 — বড় win (বেশি compute, কিন্তু একই GPU তে সম্ভব) 🏆
| # | কাজ | সময় | Score gain |
|---|---|---|---|
| 2.1 | **Arditi refusal-direction baseline** একই setup এ (A5) — সবচেয়ে বড় single upgrade | 3–4 hr | +10 |
| 2.2 | দ্বিতীয় model এ মূল result replicate (Gemma-2-9B) (B2) | 4–6 hr | +5 |
| 2.3 | Multi-seed steering variance (B5) | 2 hr | +2 |
| 2.4 | Figure 4 raw data থেকে regenerate + attribution method লেখা (A4) | 2 hr | +5 |
| 2.5 | Appendix B derivation draft (আমি করতে পারি) | 1 hr | +2 |

**Phase 2 শেষে projected: ~95+/100**

## Phase 3 — Release 🚀
- Zenodo DOI + arXiv submission (endorsement লাগবে)
- Full stats table appendix এ (সব p_adj, CI)
- BindCraft প্রজেক্টের মতো একটা public demo notebook

## কে কী করবে
- **আমি (sandbox):** Phase 0 এর সব, Appendix B derivation, code cleanup, PDF rebuild, প্রতিটা ধাপে review
- **তুমি (molab GPU):** Phase 1 ও 2.1–2.3 এর experiment চালানো — আমি সব script লিখে দিবো
- **তোমার raw data:** 2.4 (Figure 4) আর Appendix A entity list

---
*Next immediate action: Phase 0 শুরু — বললেই আমি 0.1–0.4 এখনই করে PDF rebuild করে push করে দিচ্ছি।*
