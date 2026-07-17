# Reasoning in pretraining: under-reasoning (H1) & finding/exploiting reasoning-rich text (H2)

**Status: first-pass literature map. Full reads pending.** The summaries below come from a neutral
`deep-research` survey (run `wf_869397f2-d8b`, 2026-07-16: 6 angles → 27 sources → 125 claims → 25
adversarially verified) — **not** yet from me reading each paper end-to-end. Verification is marked per item:

- **✅ VERIFIED** — claim survived 3-vote adversarial verification (vote shown); the *specific claim* is
  confirmed, though I have not personally read the full paper yet.
- **◎ SEARCH-SUMMARY** — title + one-paragraph relevance from the search agent; plausible but neither
  independently verified nor full-read. Treat as a lead, not a settled fact.
- **📖 READ** — I have full-read this paper (in an earlier pass).

Per Dongwei's paper rule, before any *verdict* enters our EXPERIMENT_LOG "key findings," the in-scope papers get
full reads. Citation counts were not collected by the workflow; venue (ICLR/NeurIPS/ACL/EMNLP) is shown as the
credibility signal where captured.

---

## The two hypotheses (corrected, reasoning-only)

**H1 — UNDER-REASONING AND ITS PERSISTENCE.** In pretraining (next-token prediction), a model can satisfy the
objective by a **shortcut** — surface pattern-match, memorized association, plausible guess — instead of running
the full multi-step inference the text encodes. Call that **under-reasoning**. Two disjoint causes must be kept
separate:
- **(C) Can't** — the inference needs knowledge/information the model lacks → forced guess (a *knowledge* gap).
- **(W) Won't** — the model *has* what it needs, but next-token prediction is satisfied by a cheaper shortcut,
  so it never exercises or learns the full inference (an *incentive/shortcut* gap).

Claim: this under-reasoning — especially the **(W)** form — is learned in pretraining and **persists** through
SFT/RL. **(W) is the branch our completeness thread can act on** (make the reasoning explicit in the data → the
shortcut is removed → the model must learn the inference); **(C) is the confounder to hold constant.**

**H2 — FINDING & EXPLOITING REASONING-RICH CONTENT IN PRETRAINING TEXT.** Not about evaluating whether a *model*
reasons — about whether a *piece of text* contains reasoning and whether we can use it. Three parts: **(4)
identify** reasoning-rich content in corpora; **(5) exploit** it by augmenting text with explicit reasoning;
**(6) completeness** — how complete a reasoning chain must be (implicit premises / enthymemes made explicit) to
help; **(7)** can a **perplexity / weak-vs-strong-model gap** detect reasoning content?

---

## What the verification pass established (21 confirmed, 4 refuted)

**Confirmed (H1 — under-reasoning is real, mechanistic, and pretraining-laid):**
1. **Shortcuts + brittleness** — GSM-Symbolic: one irrelevant clause (NoOp) drops accuracy up to 65% (Phi-3-mini
   88%→22.4%, GPT-4o 95.2%→63.1%). Cao et al. independently frame LMs as relying on shortcuts. ✅ 3-0.
2. **Arithmetic = "bag of heuristics"**, not an algorithm; established early in pretraining and never replaced;
   ablating a prompt's heuristic neurons drops accuracy ~29pp; heuristics account for ~79% of circuit
   performance at intermediate Pythia checkpoints. ✅ 3-0.
3. **Long-horizon under-execution** — first-answer accuracy 63% (5 steps) → 20% (95 steps) across 15 models;
   *"strong final-answer performance does not necessarily reflect faithful execution."* ✅ 3-0 (single 2026 source).
4. **Latent multi-hop is REAL but PARTIAL & ASYMMETRIC** — genuine internal pathway, strong first hop (>80% for
   some relations), moderate second hop; "hopping too late" timing bottleneck, back-patching fixes 66%. ✅ 3-0.
5. **Shortcut-inflated & category-dependent** — SOCRATES: shortcut-free composability ~5× lower; country-bridge
   ~82% vs year-bridge ~6%; **conditioned on the model already knowing the 1-hop facts** → isolates a
   composition gap, not a knowledge gap. ✅ 3-0.
6. **Learnable only under extreme conditions** — implicit composition emerges only via *grokking*; required data
   grows *exponentially in hop-count*; composition fails OOD while comparison generalizes. ✅ 3-0.
7. **Bound by pretraining exposure (the can't-vs-won't separator)** — atomically-seen entities: 97% 1-hop but ~1%
   2-hop; compositionally-exposed: 83% 2-hop. Same 1-hop on both ⇒ *"the compositional gap is a pretraining
   failure, not a capacity limit."* ✅ 3-0 (quantitative claims; the universal "never under any augmentation"
   phrasing was the lone 2-1).

**Refuted (do NOT believe the strong anti-reasoning framings):**
- ✗ 0-3 "LLMs do not reason at all, only replicate memorized steps" (over-read of GSM-Symbolic).
- ✗ 0-3 "genuine latent composability is only ~7-8%, models overwhelmingly don't compose" (over-read of SOCRATES).
- ✗ 0-3 "multi-hop failure is a knowledge gap, not an inference deficit" (over-read of the exposure paper —
  the correct reading is the *opposite*: facts are present, composition is missing).
- ✗ 1-2 the strong single-forward-pass "bridge entity in early layers → second hop later" as a universal claim.

**Net H1 read (provisional):** under-reasoning via shortcuts is well-supported and *mechanistically* traced to
pretraining; the honest picture is **genuine-but-partial, shortcut-inflated, exposure-bound** reasoning — not
"no reasoning." Multiple independent lines **separate (C) from (W)** and land on **(W)/exposure** — which is
exactly the branch our completeness augmentation targets.

**H2 coverage note:** the verification budget went to H1, so H2.4 (identify) and H2.7 (perplexity-gap) produced
**no *verified* claims** — but the *papers exist* (below). That's an absence of verification, not absence of
literature. These are the priority full-reads.

---

## Full paper catalog (26 papers, by angle)

### H1.1 — reasoning shortcuts in language modeling
- **GSM-Symbolic** (Mirzadeh et al., Apple; ICLR'25) — `2410.05229` — 📖✅ template GSM8K variants; NoOp clause
  drops accuracy up to 65%; accuracy varies with numeric/name changes. *(model-eval; the definitional caveat:
  this measures the model, not text content.)*
- **Arithmetic Without Algorithms: Bag of Heuristics** (Nikankin et al.; ICLR'25) — `2410.21272` — ✅ ⭐ sparse
  late-layer neurons each = one interpretable heuristic; model sums them instead of computing; rules out both
  algorithm and memorization; forms early in pretraining.
- **Mitigating Shortcut Reasoning: A Gradient-Aware Training Approach** (2026) — `2603.20899` — ✅ ⭐ names the (W)
  cause: *"training paradigms that optimize primarily for answer correctness structurally favor shortcuts… that
  efficiently reduce training loss."*
- **Shortcut Learning of LLMs in NLU** (Du et al.; CACM / `2208.11857`) — ◎ foundational survey/taxonomy of
  shortcut learning; strong in-distribution, collapse under shift. *(secondary — skip full read.)*
- **Opening the Black Box: Survey on Multi-Step Reasoning Mechanisms** (2026) — `2601.14270` — ◎ survey;
  "shortcut neurons" (subject→answer, skipping hops); ablation drops perf ~3×. *(secondary — skip.)*
- **When LLMs Stop Following Steps** (2026) — `2605.00817` — ✅ ⭐ separates can't/won't: fails *even when the full
  procedure is given in-prompt*; 63%→20% over 5→95 steps; procedural-state failures, not arithmetic errors.

### H1.2 — latent multi-hop reasoning vs memorization
- **Do LLMs Latently Perform Multi-Hop Reasoning?** (Yang et al.; ACL'24) — `2402.16837` — ✅ ⭐ TwoHopFact (45,595
  prompts, 52 types); first-hop pathway >80% for some relations; second hop only moderate; first hop scales with
  size, second doesn't.
- **Hopping Too Late** (Biran, Yang et al.; EMNLP'24) — `2406.12775` — ✅ ⭐ second hop starts in too-late layers;
  back-patching a later state to an earlier layer fixes 66% of failures → causal timing bottleneck.
- **Grokked Transformers are Implicit Reasoners** (Wang et al.; NeurIPS'24) — `2405.15071` — ✅ ⭐ implicit
  composition only via grokking; generalizes OOD for comparison, fails for composition (distinct circuits).
- **Do LLMs Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts? (SOCRATES)** (Yang et al.; ACL'25) —
  `2411.16679` — ✅ ⭐ shortcut-free test; country-bridge ~80% vs year-bridge ~6%; conditions on knowing 1-hop
  facts.
- **Multi-Hop Knowledge Composition is Bound by Pretraining Exposure** (2026) — `2606.09338` — ✅ ⭐⭐ **the
  single most on-point paper**: 97% 1-hop / ~1% 2-hop for atomic-only entities vs 83% 2-hop for
  compositionally-exposed; same 1-hop ⇒ gap is exposure, not knowledge → directly our (W) branch and implies
  augmenting text *with the composition* is what helps.
- **LMs can learn implicit multi-hop reasoning, but only with lots of data** (Yao et al.; EMNLP'25) —
  `2505.17923` — ✅ ⭐ GPT-2-from-scratch on k-hop data; genuine single-pass k-hop is learnable but required data
  grows exponentially in k, depth linearly in k.

### H1.3 — persistence through post-training
- **Does RL Really Incentivize Reasoning Beyond the Base Model?** (Yue et al.; 2025) — `2504.13837` — 📖✅ RLVR
  beats base at small k, base matches/exceeds at large k; RL coverage stays inside base distribution.
- **ProRL** (NVIDIA; 2025) — `2505.24864` — 📖✅ prolonged stabilized RL beats base across pass@k incl. tasks base
  fails entirely; the counter to Yue.
- **The Debate on RLVR Boundary: Shrinkage, Expansion, or Both?** (2025) — `2510.04028` — ◎ ⭐ reconciles
  Yue-vs-ProRL: boundary both shrinks (early diversity loss) and expands (later exploration) by stage;
  single-snapshot pass@k misleads.
- **Front-Loading Reasoning: Synergy of Pretraining & Post-Training Data** (NVIDIA) — ◎ ⭐⭐ **most direct
  persistence evidence**: reasoning instilled in pretraining is *amplified* by SFT and survives SFT+RLVR;
  doubling SFT data (+4.09%) still can't match reasoning-pretrained models → post-training compounds with, does
  not replace, a pretraining foundation.

### H2.4 — identifying / selecting reasoning-rich content
- **AttentionInfluence: Weak-to-Strong Pretraining Data Selection** (2025) — `2505.07293` — ◎ ⭐ small (1.3B)
  model masks retrieval attention heads; docs where masking most raises loss = reasoning-rich; no supervised
  classifier; 73B-token subset → 7B model +0.8–3.5pp on MMLU/MMLU-Pro/GSM8K/HumanEval.
- **The FineWeb Datasets / FineWeb-Edu** — `2406.17557` — ◎ ⭐ educational-value scoring: Llama-3-70B labels 500k
  samples 0–5, BERT regressor (F1 82% @thr 3) filters 1.3T tokens; the canonical "reasoning/edu-density"
  classifier.
- **PreSelect: Predictive Data Selection** (HKUST-NLP; ICML'25) — `github/hkust-nlp/PreSelect` +
  `2025.findings-acl.216`? — ◎ ⭐ selects by "predictive strength" (per-doc loss↔benchmark correlation), then a
  fastText classifier scales it. Bridges H2.4 (learned classifier) and H2.7 (loss signal).
- **Autonomous Data Selection with Zero-shot Generative Classifiers for Math** (ACL Findings'25) —
  `2025.findings-acl.216` — ◎ ⭐ LLM as zero-shot scorer judging whether a doc has genuine math reasoning →
  continue-pretrain on high scorers. A concrete "this document requires reasoning" heuristic.

### H2.5/6 — augmenting text with reasoning + completeness
- **Reasoning to Learn from Latent Thoughts (BoLT)** — `2503.18866` — 📖 augments text with inferred latent
  thoughts via EM bootstrap; MATH 5.7%→25.4% vs raw; augments **uniformly** (no selection); corpus already
  math-heavy (FineMath).
- **Thinking Augmented Pre-training (TPT)** — `2509.20186` — 📖 prepends generated thinking trajectories; GSM8k
  19.2%→50.1%, MATH 9.1%→21.8%, 3× data efficiency; no selection, but trajectories run *longer* for Math/Physics
  (emergent density signal).
- **Faithfulness as Information Flow** (2026) — `2605.24286` — ◎ ⭐⭐ **defines COMPLETENESS**: a faithful CoT must
  absorb all answer-relevant prompt content into the trace; completeness failures = "residual shortcuts" (chain
  skipped steps the model used). Directly our thread's variable.
- **Making Implicit Premises Explicit in Enthymemes** (2026) — `2603.06114` — ◎ ⭐⭐ the **enthymeme /
  suppressed-premise** problem our framing named: LLM generates the missing intermediate premises, neuro-symbolic
  SAT checker verifies entailment.

### H2.7 — perplexity-gap detection of reasoning content
- **Improving Pretraining Data Using Perplexity Correlations** (Thrush et al.; 2024) — `2409.05816` — 📖 selects by
  correlation of per-doc perplexity with downstream scores across many models; NOT raw low perplexity.
- **Rho-1: Not All Tokens Are What You Need** (Lin, Gou et al.; NeurIPS'24) — `2404.07965` — 📖 excess-loss GAP
  (reference − training model) at token level; the closest analog to our reverse-filter; but "useful" = quality,
  NOT reasoning (authors' own framing).

---

## Coverage gaps & open questions (from the workflow's own caveats)

1. **Persistence through SFT/RLHF/RLVR is under-tested for *under-reasoning* specifically.** The only verified
   "persistence" is across *pretraining* checkpoints (bag-of-heuristics). Whether the (W) shortcut habit survives
   post-training is essentially untested here — **Front-Loading Reasoning is the lead** and needs a full read.
2. **Generalization of mechanistic findings.** Much is arithmetic-specific (heuristics) or synthetic/controlled
   (grokking, k-hop, atomic-biography). Whether it transfers to natural-corpus multi-hop text is an extrapolation
   the papers don't fully establish.
3. **Temporal sensitivity.** GSM-Symbolic brittleness (Oct 2024) may be attenuated in 2025–26 reasoning-tuned
   frontier models; report measured effects, not "current SOTA."
4. **Completeness threshold (our core question).** What completeness must an added rationale meet — filling
   enthymemes, faithful vs shortcut chains — to actually help learning, and how much of an augmentation gain (e.g.
   the 83% 2-hop for exposed individuals) is composition vs surface exposure? *Faithfulness-as-Information-Flow*
   and *Enthymemes* are the leads.

## Reading plan (in-scope full reads, priority order)

**⭐⭐ first (hit the hypotheses hardest):** Multi-Hop Bound by Pretraining Exposure (`2606.09338`), Front-Loading
Reasoning (NVIDIA), Faithfulness as Information Flow (`2605.24286`), Enthymemes (`2603.06114`).
**⭐ next:** the H1.2 multi-hop cluster (Yang `2402.16837`, Hopping-Too-Late `2406.12775`, Grokked `2405.15071`,
SOCRATES `2411.16679`, Yao `2505.17923`); bag-of-heuristics (`2410.21272`); When-LLMs-Stop-Following-Steps
(`2605.00817`); the H2.4 selection set (AttentionInfluence, FineWeb-Edu, PreSelect, Zero-shot-math-classifier);
RLVR-boundary reconciliation (`2510.04028`).
**Skip:** the two secondary surveys (CACM `2208.11857`, `2601.14270`).

---

*Provenance: neutral `deep-research` run `wf_869397f2-d8b` (2026-07-16), zero seed papers. Raw journal at
`subagents/workflows/wf_869397f2-d8b/journal.jsonl`; harvested titles/summaries at scratchpad
`paper_list.txt`. Prior knowledge-framing doc `PERSISTENCE_AND_USEFUL_REASONING.md` is superseded by this
reasoning-only framing.*
