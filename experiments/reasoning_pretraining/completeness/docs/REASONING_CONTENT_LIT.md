# Reasoning in pretraining: under-reasoning (H1) & finding/exploiting reasoning-rich text (H2)

**Status: FULL READS DONE (2026-07-21).** 16 in-scope papers were read end-to-end (workflow `wf_e16faf72-dc2`,
one agent per paper, HTML/PDF full text, body numbers + verbatim quotes + author/venue confirmed); 4 more
(TPT, BoLT, RHO-1, Quiet-STaR) were read earlier; 2 secondary surveys were skipped. Every cited paper below is
📖 (read), not ◎ (search-summary). Where a specific number is read off a figure rather than a table, it's flagged.
Provenance + method at the end. This supersedes the earlier abstract-only map.

---

## TL;DR — the reads SUPPORT H1 but COMPLICATE our core thesis, with one direct counter-result

The blunt finding: **almost every paper supports H1 (under-reasoning via shortcuts is real and pretraining-laid)
but cautions or contradicts the specific H2 thesis "augment pretraining text with *explicit, complete* reasoning
to remove the shortcut."** Three things you must not miss:

1. **🔴 A direct counter-result — the Exposure paper (2606.09338).** In a controlled GPT-2 study, **EXPLICIT**
   (complete, bridge-entity-named) formats matched the no-augmentation baseline (2-hop **0.08**), while **IMPLICIT**
   (bridge-omitted, incomplete) formats drove the gains (**0.79** RDF / **0.62** NL). Logit-lens kicker: the explicit
   condition *emits the bridge token strongly yet composes 8%*; the implicit condition *never emits it yet composes
   79%*. Their words: *"decodability of an intermediate result does not imply its use."* → **For latent multi-hop,
   making the implicit premise explicit did NOT help; matching the inference distribution mattered more.**

2. **🟡 But completeness DOES help a different regime — the Enthymeme paper (2603.06114).** Filling implicit
   premises monotonically improved logical-entailment verification: ANLI **0.53→0.73**, ARCT **0.29→0.56**, more
   steps = more gain. So **completeness helps for explicit symbolic entailment but fails for latent
   single-forward-pass composition.** *Which regime our thread targets is now the central design question.*

3. **🟢 The correction to our own reverse-filter — the perplexity-gap papers.** AttentionInfluence and PreSelect are
   weak-vs-strong loss-**gap** detectors that WORK for finding reasoning-rich text — **but they use a *delta between
   two models on the same text*, NOT the single-model zero-shot continuation-perplexity our reverse-filter used
   (which we found never drops).** That is the likely reason ours failed: single-model NLL carries the
   frequency/memorization confound; a *cross-model gap* is the signal that actually carries reasoning value.

**Net for the thread:** the naive "spell out every step" thesis is not supported and is partly contradicted for
latent reasoning. But reasoning-in-pretraining clearly persists and compounds (Front-Loading), explicit reasoning
reliably helps at inference (CoT), and there is a rigorous completeness definition to borrow (Faithfulness). The
center of gravity should shift from *completeness per se* toward *whether the encoding makes the model actually run
the inference* (necessity / inference-distribution match), plus re-running our reverse-filter as a cross-model gap.

---

## The two hypotheses (corrected, reasoning-only)

**H1 — UNDER-REASONING AND ITS PERSISTENCE.** In next-token prediction a model can satisfy the objective by a
**shortcut** (surface pattern-match, memorized association, plausible guess) instead of running the full inference
the text encodes. Two causes to keep separate: **(C) Can't** (lacks the knowledge → forced guess) vs **(W) Won't**
(has it, but a cheaper shortcut satisfies the loss, so the full inference is never exercised or learned). Claim:
under-reasoning — especially (W) — is learned in pretraining and **persists** through SFT/RL.

**H2 — FINDING & EXPLOITING REASONING-RICH CONTENT IN PRETRAINING TEXT.** (4) identify reasoning-rich content;
(5) augment text with reasoning; (6) **completeness** — how complete must the chain be (implicit premises /
enthymemes made explicit); (7) can a **perplexity / weak-vs-strong gap** detect reasoning content?

---

## H1 — well-supported: under-reasoning is real; can't-vs-won't is a *spectrum* of ≥4 mechanisms

The (C)/(W) split held up as a lens, but the reads reveal it's not binary — at least four distinct failure
mechanisms, each with different implications for whether *data* can fix it:

| Mechanism | Paper | Evidence | Fixable by better data? |
|---|---|---|---|
| **(W) shortcut satisfies loss first** | Grokked Transformers (2405.15071) | A "memorize" circuit forms fast and fits the loss; the genuine "generalize" circuit appears only after grokking (~50× the steps to fit); pre-grok ID acc **9.2%** → ~98% after grokking | Partly — ratio (φ) of reasoning:atomic examples accelerates grokking |
| **(W) never develops an algorithm** | Bag of Heuristics (2410.21272) | Arithmetic = a "bag of heuristics" (circuit faithfulness **0.96**; ablation −**29pp**); final heuristics = **79%** of contribution at *every* Pythia checkpoint — formed early, never replaced by an algorithm | Authors: "may require fundamental changes to training and architectures" |
| **Exposure-manufactured (C)** | Exposure (2606.09338) | Knowledge present (**97%** 1-hop), scale-invariant (2-hop **0.01** from 124M→774M), yet uncomposed for entities never seen in compositional contexts | Only by *exposing* the entities compositionally — and implicit-format beats explicit |
| **Architectural timing** | Hopping Too Late (2406.12775) | Model has both facts + wants to compose, but the 2nd hop starts too late in the layer stack; back-patching a later state to an earlier layer fixes **66%** of failures | No — inference-time architectural bottleneck |
| **Capacity / data-coverage (C)** | Yao k-hop (2505.17923) | Genuinely learnable but training data grows **exponentially in k**, depth **linearly in k** (`L ≥ k/(8pdH)`); below budget the model sits at the ~1% random baseline | Only with exponentially more data — or curriculum (×100→×5) |

**Two H1 takeaways:**
- **Under-reasoning is genuine, mechanistically located, and pretraining-laid** — not "models can't reason at all"
  (that framing was refuted in the earlier verification pass). The picture is *genuine-but-partial,
  shortcut-inflated, exposure/architecture-bound.*
- **Making reasoning explicit at *inference* (CoT) reliably rescues it**: SOCRATES latent **2.4–8.4%** → CoT
  **~85–92.8%** (GPT-4o **7.6%→92.8%**); Yang finds the model recalls the bridge (hop-1 **>80%**) but under-uses it
  (hop-2 **~0.61**, flat with scale). The gap our thread targets is real; the question is whether *training-text*
  encoding can install what CoT provides at inference.

**Persistence (H1's key claim) — strongest single result: Front-Loading Reasoning (NVIDIA, 2510.03264).**
Reasoning put in *pretraining* doesn't just survive post-training, it **amplifies**: base-pretraining lead +9.09%
→ **+9.3%** after SFT → **18.57%** after SFT+RLVR (AIME24 **12.29→45.21**). Doubling SFT data lifts the baseline
only +4.09% (29.92→34.01) — *still below even the weakest reasoning-pretrained model* (37.33). Quote:
*"front-loading reasoning data into pretraining is critical (19% average gain)… cannot be fully replicated by
later-stage SFT, even with more data."* A high-quality-pretraining benefit even stays **latent** and is *unlocked*
only by SFT (+4.25%). This is the cleanest evidence that a pretraining reasoning foundation persists and compounds
rather than being replaced — directly supporting the value of front-loading reasoning into the base.
*(RLVR-boundary 2510.04028 reconciles the Yue-vs-ProRL debate: standard RL can shrink coverage early — MMLU-Pro
Pass@256 100→90.6 — and expand only under sustained/diversity-preserving training — AIME25 46.7→66.7 — so the base
model's boundary is the binding ceiling for ordinary post-training.)*

---

## H2 — identify is solved-ish; completeness is regime-dependent; the perplexity-gap needs the *cross-model* form

### (4) Identify reasoning-rich content — three working recipes
- **AttentionInfluence (2505.07293)** — classifier-free: mask a small model's retrieval heads, score docs by the
  loss *gap* between masked and unmasked. Top-20% upsampled → 7B gains **HumanEval +3.5, GSM8K +2.7, MMLU-Pro +2.7pp**;
  beats a FineWeb-Edu classifier on GPT-4o "reasoning score" (OpenWebMath **0.88 vs 0.52**). Caveat: within-domain
  comparable only; also lifts pure-knowledge benchmarks.
- **AutoDS / AutoMathText (ACL-Findings 2025)** — zero-shot: a Qwen-72B base model's normalized YES-logit on two
  rubric questions ("mathematical intelligence?", "educational?"). Mistral-7B **MATH 12.9→16.1, GSM8K 38.8→45.4**;
  ~2.36× token efficiency. *Signal = single strong model's targeted logit — NOT perplexity, NOT a gap.*
- **FineWeb-Edu (2406.17557)** — LLM-labeled classifier (Llama-3-70B scores 0–5, BERT regressor F1 **82%**, 1.3T
  tokens). MMLU **37 vs 33**, ARC **57 vs 46**. But "educational" is deliberately *grade-school* targeted and
  *down-weights* arXiv/technical — so it's broader than and partly orthogonal to "reasoning-rich."

### (6) Completeness — the regime split + a formal definition + a necessity caveat
- **The split (headline #1 and #2 above):** completeness *fails* for latent multi-hop (Exposure: explicit 0.08 vs
  implicit 0.79) but *helps* for explicit symbolic entailment (Enthymeme: 0.53→0.73 monotonic in #steps).
- **A rigorous definition to borrow — Faithfulness as Information Flow (2605.24286):** completeness ⟺ `I(P;A|C) ≈ 0`
  — the chain-of-thought must "screen off" the prompt from the answer; *"a completeness failure indicates a residual
  P→A shortcut."* This is a measurable operationalization of exactly our variable.
- **The caveat that reshapes the thread — necessity:** their interventions make a shortcut *visible* in the CoT
  **without removing it**; a complete-*looking* chain can still be a rationalization while the answer is computed
  directly from the prompt. So they add a separate **Necessity** property (A must causally depend on C). Hinted-GPQA
  verbalized-faithful rate: **89.4%** (Qwen3-8B) vs **54.3%** (DeepSeek-R1-Distill-14B). → *Surface completeness is
  not sufficient; the encoding must make the chain actually **used**.*

### (7) Perplexity-gap detection — the correction to our reverse-filter
- **Cross-model gaps WORK:** AttentionInfluence (masked-vs-unmasked loss gap) and **PreSelect (2503.00808)** —
  "predictive strength" = whether the loss *ranking* across a ladder of models matches their ability ranking; 30B
  selected tokens beat 300B random (**10×**), distilled into a fastText classifier. Both are scalable, benchmark-
  predictive selection signals.
- **The lesson for us:** our reverse-filter used *single-model zero-shot continuation perplexity* (found to never
  drop — the frequency/memorization confound). The working methods all use a **delta between two models on the same
  text.** Re-running the reverse-filter as a **weak-vs-strong NLL gap** (our 1.4B vs Qwen-72B, but as a per-doc gap
  ranked within-domain) is the concrete, cheap fix these papers point to. Caveat: PreSelect targets *general* ability
  and AutoDS uses a single-model logit — so "gap ⇒ reasoning specifically" is not guaranteed; it must be validated.

---

## What this means for the thread (honest read)

1. **The naive completeness thesis is not supported and is partly contradicted for *latent* reasoning.** The Exposure
   paper is a direct counter; Faithfulness warns surface-completeness ≠ functional use.
2. **But reasoning-in-pretraining is worth it** — Front-Loading shows it persists and compounds through SFT+RL, and
   explicit CoT reliably rescues latent-composition failures at inference.
3. **The sharper, more novel question** is *encoding-for-necessity / inference-distribution-match*, not completeness
   per se: what text encoding makes the model actually **run** the inference (and *use* the chain), rather than merely
   contain it. The Exposure result (implicit > explicit) and the Necessity property both point here.
4. **A concrete, cheap next experiment on our own data:** re-run the reverse-filter as a **cross-model NLL gap**
   (à la AttentionInfluence/PreSelect) instead of single-model perplexity — the reads predict this fixes our null.

---

## Full paper catalog (all read — real numbers)

### H1.1 — reasoning shortcuts
- **📖 Bag of Heuristics** (Nikankin et al., ICLR'25) — `2410.21272`. Arithmetic circuit faithfulness **0.96**;
  ablating a prompt's heuristic neurons −**29pp** (from 95%); final heuristics = **79%** of contribution at every
  Pythia-6.9B checkpoint → formed early, never replaced by an algorithm. *"neither robust algorithms nor
  memorization… a bag of heuristics."* **Verdict: supports H1 (W-shortcut, persistent); complicates the augmentation
  remedy** (authors say fixing it "may require fundamental changes to training and architectures").
- **📖 When LLMs Stop Following Steps** (2026) — `2605.00817`. 15 models, 55k examples: first-answer **63%→20%** over
  5→95 steps; exact-step exec **70.88%→46.84%**, under-execution **24.25%→50.87%**; look-back depth −**18.43pp**.
  Procedure given *in-prompt* yet fails → isolates execution (W), not knowledge (C); **persists in RL/reasoning-tuned
  models**. **Verdict: supports H1; complicates completeness (a complete in-context chain isn't followed over long
  horizons).**
- **📖 GSM-Symbolic** (Apple, ICLR'25) — `2410.05229` (read earlier). NoOp clause −up to **65%**. *Model-eval, not
  content — kept only as shortcut/brittleness evidence.*

### H1.2 — latent multi-hop reasoning
- **📖 Do LLMs Latently Perform Multi-Hop Reasoning?** (Yang et al., ACL'24) — `2402.16837`. TwoHopFact (45,595
  prompts, 52 types). Hop-1 recall **0.71/0.72/0.78** (7B/13B/70B, *scales*); hop-2 utilization **0.64/0.65/0.61**
  (*flat*); "up to 23% of types show strong latent reasoning in >80% of cases." **Verdict: supports H1 — model has
  the bridge but under-uses it, and scale doesn't fix hop-2; supports completeness intuition mechanistically.**
- **📖 Hopping Too Late** (Biran et al., EMNLP'24) — `2406.12775`. 82,020 queries; selects cases where both hops are
  correct in isolation but the composition fails. Back-patching fixes **66%** (Pythia-6.9B). A *third* category:
  architectural timing bottleneck (has facts + wants to compose, runs out of layers). **Verdict: supports H1;
  complicates "data removes the shortcut" (this failure is architectural); weakly supports externalization.**
- **📖 Grokked Transformers are Implicit Reasoners** (Wang et al., NeurIPS'24) — `2405.15071`. Composition learned
  only via grokking (pre-grok **9.2%** → ~98% after ~50× steps); OOD composition **~0%** (fails) but OOD comparison
  **~98%** (succeeds). Two circuits: memorize (shortcut, first) vs generalize (grokking). Frontier models score
  **~28–37%** (≈random) on the hard implicit task. **Verdict: supports H1 (clean W→grok + hard-C OOD); neutral/
  complicating for augmentation (implicit reasoning internalizable WITHOUT explicit chains; some limits architectural).**
- **📖 SOCRATES (shortcut-free latent multi-hop)** (Yang et al., DeepMind) — `2411.16679`. Latent composability
  **2.4–8.4%** (conditioned on knowing both 1-hop facts) vs CoT **~85–92.8%** (GPT-4o **7.6→92.8**); shortcut
  inflation **~5×** (2.4 vs 11.6); country-bridge **82–85%** vs year-bridge **6–7%** (~14×); OLMo pretraining: only
  **~11%** of eligible cases ever show emergent latent 2-hop. **Verdict: strongly supports H1 (definition-quality
  shortcut demo, C excluded by conditioning); supports explicit-at-inference, cautions mere fact co-presence won't
  induce latent composition.**
- **📖 LMs can learn implicit multi-hop, but only with lots of data** (Yao et al., EMNLP'25) — `2505.17923`. GPT-2
  from scratch: 2-hop **99.8%** at ×1; 3-hop needs ×5–10; 4-hop needs ×20–100; **data ∝ exp(k)**, depth ∝ k
  (`L ≥ k/(8pdH)`). Curriculum cuts 4-hop budget **×100→×5**. *"can learn… even without explicit rationales."*
  **Verdict: mostly complicates the shortcut framing (bottleneck is C: coverage/depth); supports curriculum/
  intermediate-supervision; neutral on perplexity-gap.**

### H1.3 — persistence through post-training
- **📖 Front-Loading Reasoning** (NVIDIA/CMU, 2025) — `2510.03264`. *(numbers in the persistence section above)*
  **Verdict: SUPPORTS H1 persistence — the cleanest evidence reasoning-in-pretraining compounds through SFT+RL and
  SFT can't catch up.** Caveats: reasoning data = QA/long-CoT SFT-style mixed at 20% (not rewritten web text);
  "quality"≈CoT length; diversity beats curation at pretraining.
- **📖 Yue "RL beyond base?"** — `2504.13837` & **ProRL** — `2505.24864` (read earlier).
- **📖 RLVR Boundary: Shrinkage, Expansion, or Both?** (2025) — `2510.04028`. Two-stage dynamic reconciles
  Yue-shrinkage (MMLU-Pro Pass@256 **100→90.6**) and ProRL-expansion (AIME25 **46.7→66.7** with diversity-preserving
  RL). **Verdict: base boundary is the binding ceiling for ordinary post-training → weakly motivates reasoning-in-base;
  neutral on H2.**

### H2.4 — identify reasoning-rich content
- **📖 AttentionInfluence** — `2505.07293` · **📖 AutoDS/AutoMathText** — ACL-Findings'25 · **📖 FineWeb-Edu** —
  `2406.17557` · **📖 PreSelect** — `2503.00808`. *(numbers + the perplexity-gap correction in the H2 section above)*

### H2.5/6 — augment + completeness
- **📖 Faithfulness as Information Flow** (2026) — `2605.24286` — the completeness definition + necessity caveat.
- **📖 Making Implicit Premises Explicit in Enthymemes** (Feng & Hunter, UCL, 2026) — `2603.06114` — completeness
  helps symbolic entailment (ANLI 0.53→0.73, ARCT 0.29→0.56); LLM-generated premises beat dataset originals; verified
  neuro-symbolically (AMR→CNF→PySAT). **Verdict: supports completeness for explicit entailment; neutral on the
  pretraining/shortcut mechanism (no LM training, no perplexity).**
- **📖 Multi-Hop Composition Bound by Pretraining Exposure** (2026) — `2606.09338` — the counter-result +
  can't-vs-won't separator. *(numbers up top)* **Verdict: complicates the completeness thesis (explicit ≤ baseline,
  implicit wins for latent composition); supports separating exposure from capacity.** Caveats: fully synthetic,
  2 relations, GPT-2 scale, implicit-only.
- **📖 TPT** — `2509.20186` · **📖 BoLT** — `2503.18866` (read earlier): augment uniformly (no selection); TPT 3×
  data efficiency, BoLT MATH 5.7→25.4; both on already math-heavy corpora.
- **📖 Quiet-STaR** — `2403.09629` (read earlier): token-level rationales, perplexity gain on *difficult* tokens.

### H2.7 — perplexity-gap
- **📖 RHO-1** — `2404.07965` (read earlier): excess-loss (reference − training) token selection; "useful" = quality,
  not reasoning. · **📖 Perplexity Correlations** — `2409.05816` (read earlier). *(cross-model-gap lesson above)*

### Skipped (secondary surveys)
- Shortcut-Learning survey (`2208.11857`), Multi-Step-Reasoning survey (`2601.14270`).

---

## Open questions / proposed next steps

1. **Which regime does our thread target?** Latent single-forward-pass composition (where explicit *hurt* —
   Exposure) or explicit multi-step reasoning at inference (where completeness *helped* — Enthymeme, CoT)? This
   decides whether "completeness augmentation" is even the right lever.
2. **Re-run the reverse-filter as a cross-model NLL gap** (1.4B vs Qwen-72B per-doc, within-domain), not
   single-model perplexity — the reads predict this fixes our null. Cheap, uses data already on disk.
3. **Encoding-for-necessity, not completeness:** design an augmentation that makes the model *use* the chain (borrow
   Faithfulness's `I(P;A|C)` / necessity metric), and test implicit-inference-distribution-matching vs fully-explicit
   (the Exposure axis) on our ladder.
4. **Does under-reasoning persist through *our* post-training?** Still under-tested for the (W) form specifically;
   Front-Loading is the closest but uses SFT-style reasoning data, not rewritten text.

---

*Provenance: full-read workflow `wf_e16faf72-dc2` (2026-07-21), 16 agents (opus), one per paper, HTML/PDF full text,
schema-validated structured extraction (method / numbers / quotes / can't-vs-won't / completeness / limitations /
verdict). Raw journal: `subagents/workflows/wf_e16faf72-dc2/journal.jsonl`; full structured results in the session
task output. Neutral-search discovery via `wf_869397f2-d8b` (zero seed papers). Prior abstract-only map and the
knowledge-framing doc `PERSISTENCE_AND_USEFUL_REASONING.md` are both superseded by this.*
