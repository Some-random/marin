# Completeness of Reasoning (code → text) — Experiment Log

**Thread:** `experiments/reasoning_pretraining/completeness/`
**Owner:** Dongwei Jiang · **Started:** 2026-07-04
*Entries are newest-first. Layout: `EXPERIMENT_LOG.md` + `REASONING_CONTENT_LIT.md` (the literature map — top-level
thread artifacts) at the thread root; `docs/` (other write-ups), `scripts/` (`*.py`), `data/` (jsonl).*

## Framing (read first)

Code has "completeness of reasoning" — every step is explicit/executable, no suppressed premises, and it
bottoms out at the language's primitives (a natural stopping point). Natural-language text leaves most
premises implicit (**enthymemes**), with *no finite natural stopping point* for making them explicit ("what
premise made that premise usable?" → Lewis Carroll's regress / the frame-qualification problem).

**Hypothesis family:** code→text/reasoning transfer works partly because code teaches *complete, explicit,
multi-step* reasoning. If we augment pretraining **text** to carry the same property — implicit premises made
explicit to a *bounded* depth — a text-trained model should transfer better, approaching what code buys.

**The real knob** (not "make text complete" — impossible): choose a *stopping rule* that makes
completeness-augmentation of text tractable AND useful for transfer.

**Hard constraint:** NO pretraining launch without Dongwei's explicit sign-off (standing rule).

## Running key findings
1. **Transfer mechanism = STRUCTURE not executability** (Waheed 2025; Zhao 2026 code/math traces are the
   active ingredient; Petty 2024 helps math/structured, HARMS syntax/knowledge). → `docs/CODE_REASONING_LIT.md`
2. **"Augment text w/ reasoning, train normally" already works** (BoLT/TPT/Reasoning-CPT 2025). Honest novelty
   = **N1** completeness/stopping-rule as a controlled variable, **N2** reasoning-text as a code-substitute on
   OUR ladder, **N3** cheap "found-completeness" gate. → `docs/PRETRAIN_AUGMENTATION_LIT.md`, `docs/LITERATURE_REVIEW.md`
3. **Stopping rule is THE unsolved problem**; 6 implementable rules enumerated. → `docs/STOPPING_RULES.md`
4. **Found-completeness gate (N3):** only **11.6%** of DCLM warrants a rationale (347/3000). → `docs/RATIONALE_WARRANT_AUDIT.md`
5. **Zero-shot perplexity is confounded on raw/synthetic targets, BUT a clean reasoning-specific drop DOES
   exist on real reasoning-conclusions** (2026-07-08). On raw continuation / synthetic Q&A: no reasoning-specific
   drop (placebo-matched). On the ~1% of DCLM docs whose real continuation IS the doc's own conclusion
   (re-split at "Thus/Therefore/…"): **adding the correct rationale lowers the real conclusion's perplexity —
   real<placebo on 22/22 docs, placebo HURTS (+0.399)**. Genuinely the reasoning, not priming. → `PERPLEXITY_HUNT.md`,
   `docs/CONCLUSION_RESULTS.md`, `docs/PROBE_RESULTS.md`
6. **The reasoning-specific perplexity drop is REAL and universal on a reasoning-determined, self-contained
   target, but STRUCTURALLY capped on DCLM** (2026-07-10). Winogrande (200 ex, natural-prose rationale): the
   rationale lowers the true continuation's perplexity on all 9 judges (−0.60 to −0.84), monotonic in
   completeness, placebo≈0. DCLM has the SAME content benefit (~−0.6) yet only nets a drop for the smallest
   models — its mid-continuation insertion penalty (+0.4–0.7) plus low base perplexity (big judges already
   predict the continuation) cancel it. Completeness (complete−incomplete)≈0 everywhere — the model fills gaps
   itself. So DCLM's null was **format/structure, not reasoning-poverty.** → `docs/PREPEND_VS_MID.md`,
   `scripts/winogrande_score.py`

---

## 2026-07-23
- **Adversarial full re-read of ALL 24 papers in `REASONING_CONTENT_LIT.md`** (workflow `wf_13d49562-ffa`, 24 agents,
  24/24 full-text fetches, ~10.5 min) — every reader now REQUIRED to critique the paper's own eval methodology (the
  check that caught the Exposure misread), plus first full reads for the 8 papers that only had thin entries (TPT,
  BoLT, Quiet-STaR, RHO-1, GSM-Symbolic, Yue, ProRL, Perplexity Correlations). Doc updated throughout. Biggest deltas:
  1. **Enthymeme's 0.53→0.73 is confounded** — entailment-class-only recall (no specificity control; more steps
     mechanically loosen the SAT trigger) and the GOLD complete premise barely beats none (0.558 vs 0.530), so part of
     the gain is verifier plumbing. TL;DR claim "completeness helps explicit reasoning" downgraded to *suggestive*.
  2. **Front-Loading's "can't catch up" is not token-matched** — reasoning-pretrained models saw ~200B reasoning
     tokens; the catch-up baseline got a doubled SFT set (≪200B, counts never reported), and their own Table 6 shows
     naive SFT-doubling *harms* math (−4.9%). Compounding direction stands; irreplaceability claim dropped.
  3. **When-LLMs-Stop has the same no-scratchpad trap as Exposure** (prompt forbids written reasoning; thinking models
     get a hidden scratchpad, the one non-reasoning model floors), its under-execution metric counts obedient direct
     answers and correct batching as failures, and mult/div float drift explains much of the decline (add/sub ~98% vs
     mult/div ~43–53%). Also corrected our numbers: 14 models, 61%→20% (not 15/63%). Its "isolates Won't" role is
     substantially weakened.
  4. **TPT + BoLT (the "augmenting works" evidence) are strong-teacher-distillation-confounded** — TPT's from-scratch
     thinking is written by Qwen3-8B (weak-teacher ablation only run in mid-training); BoLT's honest isolated delta is
     +6 MATH vs teacher-matched WRAP-CoT (not 5.7→25.4), and its teacher-free bootstrap is modest (~13→20%, plateaus).
     Neither varies completeness → the completeness dose-response experiment remains unrun by anyone.
  5. **RHO-1's own self-reference ablation collapses its gain +16.5pp→+3.3pp** — ~80% of the headline is the curated
     reference distribution leaking through the token mask (distillation through a mask, not token selection). Flip
     side, an actionable idea: train the reference model on complete-reasoning text → the gap signal points at
     completeness.
  6. **AttentionInfluence weakened as data-selection evidence** (intervention is upsampling; selected docs ~2× longer,
     no matched-upsampling control; overall avg +0.75pp with commonsense regressions; "more reasoning than classifier"
     only in math/code domains; corrected "random heads barely move" — random masking still drops GSM8K ~30% rel.).
     Table-6 head-validation and the recipe-A plan stand.
  7. **Exposure correction CONFIRMED against full text + appendix** (no scratchpad cell anywhere; the paper rejects
     the paradigm explicitly), with one nuance to keep: its logit-lens is a partial mechanism-level defense (explicit
     model treats the bridge as a generation target, not an internal variable) — informative but no substitute for
     the missing experiment.
  8. New full reads of the RL-persistence trio (Yue / ProRL / Boundary-debate): pass@k-at-huge-k structurally favors
     the high-entropy base; ProRL evals the base at RL-optimal temperature with an already-distilled base; the
     Boundary paper's own Table 1 never reproduces the Yue crossover. "Post-training is bounded by the base" is
     directionally supported but softer than we'd written.
  Per-paper records: `subagents/workflows/wf_13d49562-ffa/journal.jsonl` (session 8eafed95); doc provenance updated.
- **Promoted `REASONING_CONTENT_LIT.md` from `docs/` to the completeness thread ROOT** (next to `EXPERIMENT_LOG.md`)
  via `git mv` — per Dongwei it's "one of the most important things for this project," so it's now a top-level thread
  artifact rather than a buried write-up. Updated the log's layout line, the historical path references in the
  July-14/July-21 entries, and the doc's internal ref to `docs/PERSISTENCE_AND_USEFUL_REASONING.md`.
- **Tier-3 code-level deep-dive of AttentionInfluence + PreSelect** (workflow `wf_1163664e-5a9`, read the actual repos/
  code + appendices). It OVERTURNED two of my earlier claims: (1) **AttentionInfluence is self-ablation, NOT a
  two-different-model gap** — one model vs. itself with top-5% retrieval heads masked to uniform attention; memorization
  cancels because both losses share weights (that's *why* it isolates reasoning). (2) **The 1.4B-vs-72B two-model
  magnitude gap I'd proposed is a documented near-failure** — PreSelect runs exactly it ("ScalingFilter"): +0.4 over
  random, selects short/easy junk, Spearman 0.05 vs the rank-match that works — which also *explains* our own original
  reverse-filter finding "knowledge not reasoning." PreSelect's working signal is a **multi-model rank-match** (sign
  over many pairs), not a magnitude gap. Two surviving recipes: **(A)** self-ablation on our own 1.4B (with a same-day
  Table-6-style go/no-go: does masking its retrieval heads collapse GSM8K/BBH?), **(B)** multi-model rank-match on the
  Qwen size ladder (0.5B→72B, same tokenizer).
- **CORRECTED the Exposure paper reading** (Dongwei read `2606.09338` and caught my error; his `correction.txt` merged
  into the doc then deleted). My "explicit/complete reasoning doesn't help (0.08 vs implicit 0.79)" was a **misread**:
  the eval forces a single forward pass with **no scratchpad**, so it's unfair to the explicit condition (explicit
  training relies on the bridge token being in the prefix, which the test deletes). Defensible narrow claim: explicit
  traces don't auto-compile into *latent* one-pass computation — says nothing about explicit + scratchpad (a **missing
  experiment**). What robustly holds is **exposure-boundedness** (compose only for compositionally-exposed entities).
- **Rewrote the affected `REASONING_CONTENT_LIT.md` sections** (TL;DR 🔴🟡🟢, H2.6/H2.7, Exposure + AttentionInfluence +
  PreSelect entries, "what this means," open questions) to correct these AND to **dial back over-confident cross-paper
  synthesis** — my synthesis was wrong three times this pass (Exposure, AttentionInfluence-as-gap, reverse-filter fix),
  so the doc is now hedged: H2 is "genuinely open," not "our thesis is contradicted."
- **Process lesson (Dongwei, repeated):** stop rushing to next-steps before fully reading; read the full paper/code
  first. The Exposure and AttentionInfluence errors both came from concluding off abstracts/framing.

## 2026-07-22
- **Enriched `REASONING_CONTENT_LIT.md` with paper metadata** (commit `e7662784f`): filled every author byline —
  including the 4 previously unsurfaced, verified from paper HTML: Exposure = Karmim (Inria) / Barrière (U. de Chile),
  Faithfulness = Jia (Michigan State / Anthropic) / Easley (Anthropic), When-LLMs-Stop = Panda / Singh (IIT
  Gandhinagar), RLVR-boundary = Yao (Renmin / Ant) / Liu (Renmin) — plus **first/last-author institutions** and
  **Semantic-Scholar citation counts** per paper, and a consolidated metadata table. Flagged the recency caveat:
  2026 preprints show 0 citations because they're too new, NOT weak — and the *most thread-relevant* results
  (Exposure, Faithfulness, When-LLMs-Stop, Front-Loading) are exactly those low/zero-cite recent papers.
- Restructured the July-21 entry's H1 findings to explicitly map the three investigation sub-questions
  (1)/(2)/(3) → H1.1 shortcuts / H1.2 latent multi-hop / H1.3 persistence.
- *(Note: this entry backfilled on 2026-07-23 — the July-22 nightly cron did not fire, session idle at 23:48 PDT.)*

## 2026-07-21
- **Full-read all 16 in-scope reasoning/pretraining papers** (workflow `wf_e16faf72-dc2`, 1 agent/paper, HTML/PDF
  full text, body numbers + verbatim quotes + author/venue confirmed; 16/16, 0 failures, ~747k tokens). Upgraded
  every catalog entry in `REASONING_CONTENT_LIT.md` from abstract-only (◎) to full read (📖), then rewrote the
  catalog a second time as plain-English per-paper narratives (what it is / what they did / what they found / why
  it matters) so the doc reads cold. Two commits: `fdbcee655` (full-read numbers) → `be3af2015` (readable rewrite).
- **The reads SUPPORT H1 but COMPLICATE the core H2 thesis, with one direct counter-result.** Headlines:
  - 🔴 **Exposure paper (`2606.09338`) is a counter-result to naive completeness.** Controlled GPT-2: spelling the
    bridge step **explicitly** (complete chain) matched the no-help baseline (2-hop **0.08**); leaving it **implicit**
    (incomplete) drove the gains (**0.79** RDF / **0.62** NL). Logit-lens: explicit *emits* the bridge yet composes
    8%; implicit *never emits* it yet composes 79% — *"decodability of an intermediate result does not imply its use."*
    For latent one-shot multi-hop, making the hidden step explicit did NOT help; matching the inference distribution did.
    The real lever was entity-level compositional **exposure**, not chain completeness (facts present at 97% 1-hop,
    scale-invariant 124M→774M).
  - 🟡 **Completeness DOES help a different regime.** Enthymeme paper (`2603.06114`): filling implicit premises
    monotonically improved symbolic entailment (ANLI 0.53→0.73, ARCT 0.29→0.56). So completeness helps explicit
    step-by-step argument but fails silent composition — *which regime the thread targets is now the key design question.*
  - 🟢 **Fix for our reverse-filter null.** The working detectors (AttentionInfluence `2505.07293`, PreSelect
    `2503.00808`) use a **weak-vs-strong-model loss GAP on the same text**, NOT the single-model continuation
    perplexity our reverse-filter used (which never dropped — the frequency/memorization confound). Re-running the
    filter as a cross-model NLL gap (our 1.4B vs Qwen-72B, within-domain) is the cheap next experiment.
- **H1 findings, mapped to the three original investigation sub-questions** (these are the doc's H1.1 / H1.2 / H1.3):
  - **(1) reasoning shortcuts → H1.1.** The shortcut is real, mechanistic, and pretraining-laid. Arithmetic runs on a
    "bag of heuristics" (`2410.21272`) that forms early and is *never* replaced by an algorithm (~79% of the mechanism
    at every checkpoint); and *When LLMs Stop* (`2605.00817`) shows faithful step-execution collapses 63%→20% over
    5→95 steps *even with the full procedure in-prompt* → isolates Won't/execution from Can't/knowledge.
  - **(2) latent (silent, one-forward-pass) multi-hop → H1.2.** This is the *mechanistic core of H1*, because
    pretraining reasoning IS silent reasoning (no scratchpad). Models hold the pieces but under-compose them silently:
    Yang (`2402.16837`) — bridge recalled >80% but *used* only ~61%, flat with scale; SOCRATES (`2411.16679`) —
    shortcut-free silent composition 2.4–8.4% vs explicit-CoT ~85–93% (GPT-4o **7.6%→92.8%**). **Can't-vs-Won't is a
    ≥4-mechanism spectrum**, not a binary: shortcut-wins-first (Grokked `2405.15071`) / no-algorithm-ever
    (`2410.21272`) / exposure-manufactured-Can't (Exposure `2606.09338`) / architectural-timing (Hopping-Too-Late
    `2406.12775`, back-patch fixes 66%) / capacity-coverage (Yao `2505.17923`, data ∝ exp(k)).
  - **(3) persistence through post-training → H1.3.** Front-Loading Reasoning (NVIDIA `2510.03264`) is the strongest
    evidence: reasoning-in-pretraining *compounds* (+9% pretrain → +9.3% SFT → **+18.5% RL**), and doubling SFT
    (+4.09%) can't match even the weakest reasoning-pretrained model — SFT can't substitute for a pretraining
    foundation. RLVR-boundary (`2510.04028`) reconciles Yue-vs-ProRL: base boundary is the ceiling for ordinary
    post-training. Caveat: Front-Loading's "reasoning data" is SFT-style long-CoT mixed at 20%, not rewritten web text.
- **Proposed pivot (not yet acted on):** shift the thread from *completeness-per-se* toward *encoding-for-necessity*
  (does the encoding make the model actually RUN the inference — borrow Faithfulness `2605.24286`'s "chain screens off
  prompt" / necessity metric), + the cross-model-gap reverse-filter re-run. **§ EXPERIMENT_LOG "key findings" NOT yet
  updated — reframe pending user decision** on (1) which reasoning regime to target, (2) green-light the gap re-run.

## 2026-07-20
- No completeness work today (session idle; full reads of the four ⭐⭐ leads still queued).

## 2026-07-19
- No completeness work today (session idle; full reads of the four ⭐⭐ leads still queued).

## 2026-07-18
- No completeness work today (session idle; full reads of the four ⭐⭐ leads still queued).

## 2026-07-17
- No completeness work today (session idle after the July-16 reasoning-only reframe + literature map).

## 2026-07-16
- **Reframed the two hypotheses to reasoning-only after catching two scoping errors.** (a) The old H1 was a
  *knowledge*-persistence claim mapped onto the RL-beyond-base debate (Yue/ProRL) — not what the thread is about.
  Rephrased H1 as **under-reasoning**: in next-token prediction the model satisfies the objective by a *shortcut*
  instead of running the full inference the text encodes, with two disjoint causes — **(C) can't** (lacks the
  knowledge → forced guess) vs **(W) won't** (has it, but a cheaper shortcut satisfies the loss). (W) is the
  branch completeness-augmentation can act on; (C) is the confounder to hold constant. (b) The old H2 "definition"
  section answered the *model-eval* question (GSM-Symbolic / Reasoning-or-Reciting) — wrong. Retargeted H2 to
  **find & exploit reasoning-rich CONTENT in pretraining text**: (4) identify, (5) augment, (6) completeness,
  (7) perplexity/weak-vs-strong-gap detection.
- **Ran a fresh NEUTRAL deep-research pass (zero seed papers) on both hypotheses** (run `wf_869397f2-d8b`);
  completed after a resume (first attempt hit the session limit; account upgraded, resume replayed 93 cached
  agents + finished the 17 that died, incl. synthesis). 6 angles → 27 sources → 125 claims → 25 adversarially
  verified: **21 confirmed / 4 refuted**. The neutral search surfaced strong papers we never seeded.
- **Key verified H1 findings.** Arithmetic = a "bag of heuristics" formed early in pretraining, never replaced by
  an algorithm; long-horizon under-execution (63%→20% over 5→95 steps, faithful-execution ≠ final-answer);
  latent multi-hop reasoning is **real but partial & asymmetric** (first hop >80%, second hop moderate, "hopping
  too late" — back-patching fixes 66%); **exposure-bound composition is the (C)-vs-(W) separator** — 97% 1-hop
  but ~1% 2-hop for atomically-seen entities vs 83% 2-hop for compositionally-exposed (same 1-hop ⇒ gap is
  *exposure, not knowledge*). **Refuted (0-3):** "LLMs don't reason at all," "genuine composability only ~7-8%,"
  and "multi-hop failure is a knowledge gap" — the honest picture is genuine-but-partial, shortcut-inflated,
  exposure-bound.
- **Wrote + committed `REASONING_CONTENT_LIT.md`** (`3280e183a`, then in `docs/`): corrected hypotheses + all **26 papers by
  angle with per-paper summaries + real titles + verification flags** (✅ verified-claim / ◎ search-summary /
  📖 read), coverage gaps, and a prioritized full-read plan. Labeled a **first-pass map — full reads pending**
  (summaries are search-derived, not yet my end-to-end reads). H2.4 (identify) and H2.7 (perplexity-gap) had no
  *verified* claims but the papers exist (AttentionInfluence, FineWeb-Edu, PreSelect, zero-shot math classifier;
  Perplexity-Correlations, RHO-1) — these are the priority reads.
- **Most on-point leads for the thread:** Multi-Hop-Bound-by-Pretraining-Exposure (`2606.09338`),
  Front-Loading-Reasoning (NVIDIA), Faithfulness-as-Information-Flow completeness (`2605.24286`), and
  Making-Implicit-Premises-Explicit-in-Enthymemes (`2603.06114`). Full reads of these four are next.
- Superseded the old knowledge-framing doc `PERSISTENCE_AND_USEFUL_REASONING.md` (left uncommitted on disk
  pending review); `REASONING_CONTENT_LIT.md` is the current framing.

## 2026-07-15
- **Recovered the quota-killed deep-research pass without re-running it.** The overnight `deep-research`
  workflow (H1 persistence / H2 useful-reasoning) had died on the session token limit before its synthesize
  step; hand-synthesized the report from the 109 harvested claims in the workflow journal — zero re-run.
- **Wrote `docs/PERSISTENCE_AND_USEFUL_REASONING.md`**, then **verified every one of the 26 cited papers against
  its primary source** (26 direct fetches, no subagent fan-out), fixing two wording errors (Superfiltering:
  perplexity *ordering* is consistent across models, *scale* varies → they use IFD not raw ppl; Physics 3.3:
  "data signal-to-noise ratio affects capacity"). All 26 now ✅.
- **Restructured H1 into three testable links** (flat FOR/AGAINST framing was unclear): **(A) origin** — a
  forced-guess knowledge/inference gap is planted at pretraining (Kalai singleton lower-bound, Physics 3.1/3.2,
  Reversal Curse) → **robust**; **(B-SFT)** — SFT can't fix it and forcing new knowledge in *backfires* (Gekhman
  linear ↑ hallucination, LIMA/URIAL superficial-alignment) → **robust**; **(B-RL)** — RL-fixability is
  **contested** (Yue "bounded by base", 0% AIME24-only-by-RL vs ProRL / Curriculum-RL +9.8 pass@256 / RL-Grokking
  0%→100%). Yue demoted from "the H1 anchor" to the B-RL link only.
- **Takeaways for the thread.** H1: knowledge-origin + SFT-can't-fix are well-established; RL moves the base
  boundary **only with external structure** (distillation / dense-reward curriculum / prolonged RL) — the seam
  our "augment pretraining text with complete reasoning" thesis targets (supply the structure at pretraining, in
  the data, not via an RL teacher). H2: our reverse-filter perplexity-gap null is a **known confound** (Razeghi
  frequency >70% gap, Small-Model Learnability Gap, Superfiltering, Perplexity-Correlations, "Perplexity Cannot
  Tell Right from Wrong"); the field defines reasoning by **perturbation-invariance** (GSM-Symbolic NoOp −65%),
  not loss; RHO-1's excess-loss selects for *quality* not reasoning; TPT/BoLT/Quiet-STaR augment **uniformly** →
  N1 novelty should be the **stopping-rule/completeness** control, not a detector.
- Report written + fully verified; **file left uncommitted pending Dongwei's review** (the log entry commits, the
  doc does not yet).

## 2026-07-14
- **Reframed the investigation into two sub-hypotheses.** The reasoning / knowledge / arbitrary rubric
  used for the reverse-filter triage (see 2026-07-12) does not capture the real question. The intuition
  we actually care about: pretraining forces the model to **guess at gaps it lacks the knowledge or
  inferential ability to fill**, and that forced-guess deficiency **persists through post-training and RL**.
  Split into **H1 (persistence):** do pretraining-planted forced-guess deficiencies linger through SFT/RL,
  i.e. is post-training bounded by the base model? — and **H2 (definition):** how to define/identify
  genuinely reasoning-dependent (vs knowledge-recall vs boilerplate) content in pretraining text.
- **Ran a deep-research pass on H1/H2** (`deep-research` workflow, run `wf_2175e331-d8d`). Fanned out
  6 search angles → **26 sources → 109 extracted claims → 25 adversarially verified**
  (**10 confirmed**, 8 refuted, 7 left unverified — the 7 only because their verify-votes errored when the
  run hit the session token quota before the final synthesize step). Raw results recoverable from the
  workflow journal (`subagents/workflows/wf_2175e331-d8d/journal.jsonl`); full write-up pending user review.
- **Preliminary headline (from the verified claims, synthesis not yet written):** "pretraining deficiencies
  linger through post-training/RL" is **CONTESTED, not settled.** The Yue et al. "RLVR is bounded by the base
  model" pass@k-saturation claim was **refuted 0-3** by counter-evidence (ProRL arXiv:2505.24864;
  boundary-aware Curriculum RL; staged-RL grokking taking a base model 0%→100% pass@k on a hard family,
  arXiv:2509.21016). The **knowledge/extraction** side supports persistence though — superficial alignment
  (LIMA, arXiv:2305.11206), Physics-of-LLMs 3.1 extraction failures (arXiv:2309.14316), Gekhman
  new-knowledge-resists-SFT (arXiv:2405.05904). Provisional read: **knowledge gaps persist; reasoning-strategy
  gaps RL can sometimes fix** — which maps onto the H1/H2 split.
- **Cost lesson:** the workflow overran the session token quota (109 agents / ~4.0M tokens) on top of an
  already-heavy day — do NOT launch a large agent fan-out without checking headroom / explicit consent first.

## 2026-07-13
- No completeness work today (session idle after the July-12 reverse-filter results).

## 2026-07-12
- **Reverse-filter results — NEGATIVE: the 1.4B-vs-Qwen uncertainty gap surfaces KNOWLEDGE, not reasoning.**
  Full pipeline ran overnight (`scripts/score_uncertainty.py`, sharded 4 nodes): the 1.4B scored **114,416** DCLM
  docs (base-NLL of the continuation's first sentence), kept the top-30k most-uncertain (NLL 3.59..11.72), then
  Qwen-72B re-scored those (mean NLL 1.4B **4.21** / Qwen **3.09**). **Gold** (1.4B>3.5 & Qwen<2.0) = **2,271**
  docs (955 at 1.4B>4.0 & Qwen<2.0). **But the gold is mostly not reasoning:** the top-8 by gap are all
  memorized/arbitrary (a botanical species name `F. natalensis Hochst.`, a Stripe API event, Turkish/Indonesian
  UI strings, company names, EXIF metadata); a broad sample across the gap range was ~1/9 genuine reasoning
  (a clean one: compromised PC → compromised phone → phone infects the next PC). Estimate **~10–15%** of gold is
  real multi-step reasoning; the rest is knowledge / domain-jargon / foreign-language / boilerplate.
- **Mechanism (the takeaway):** a weak-vs-strong (1.4B-vs-72B) perplexity gap is dominated by what the strong
  model **memorized** — specific facts, names, code identifiers, other languages — not by reasoning ability.
  Reasoning gaps are smaller/subtler and don't rise to the top of a gap ranking. So the two-model gap is a
  **knowledge detector, not a reasoning detector**; it does NOT cleanly replace judgment for finding reasoning
  docs. Gold data in gitignored `data/rf_gold_candidates.jsonl`.
- Fixed a sharding bug in `scripts/score_uncertainty.py`: with `--ids-file`, the shard split was skipped so all
  4 shards redundantly scored the full 30k (4× Qwen waste). Now shards split even with `--ids-file`.
- **Open next step (Dongwei's call):** targeted agent reasoning-categorization on the 2,271 gold (cheap vs
  judging 114k), heuristic-clean-then-judge, or accept perplexity-gap ≠ reasoning and rethink the signal.

## 2026-07-11
- **Winogrande scoring refinements + ground-truth-perplexity view (Dongwei methodology pushes)** →
  `scripts/winogrande_score.py` now has `--score blank` (score option+suffix, blank included) and
  `--score blanktoken` (score ONLY the answer token). Blank-included re-score of all 9 judges: same picture as
  suffix-only, drops ~0.05 bigger (complete −0.65..−0.88, placebo ≈0, base≈instruct). Clean
  ground-truth-continuation ppl table (every setup scores the SAME correct span; base is a genuine cold
  prediction — it gets idx4 wrong): base>principle>full>complete monotone on every judge, complete ≈ halves
  base, placebo ≈ base.
- **Blank-token-only finding + CORRECTION (9 judges).** Scoring just the answer word: on our 1.4B `principle`
  (leak-free) barely moves it (−0.10 ppl, −0.025 acc) — but that does NOT generalize. On all 8 capable judges
  `principle` lowers the answer-token ppl −0.2..−0.9 AND raises pick-accuracy +0.05..+0.17 (up to +0.17 on the
  72B). My earlier "reasoning barely helps the answer token" was a 1.4B-only artifact; corrected 9/9. full/
  complete help most (−0.8..−1.3), partly by naming the answer.
- **Per-token example docs** → `docs/WINOGRANDE_PERTOKEN.md` (per-token NLL of option+suffix; help concentrates
  on the reasoning-loaded token, e.g. `easier` −4.0, not filler) and `docs/DCLM_PERTOKEN.md`
  (base/complete/incomplete/placebo per-token on real DCLM docs — you can watch complete≈incomplete
  (completeness null), placebo wreck it (appendix `treated` 0.04→9.2), complete≪placebo).
- **Target-awareness investigation + `docs/GENERATION_PROMPTS.md`** (Dongwei: document the generators). Found the
  exact DCLM rationale prompt (workflow `complete-reasoning-mine`): agents saw `{context, continuation}` and
  extracted a verbatim target span, so they SAW the target — but were told "from the CONTEXT only; never copy the
  target's words," imperfectly followed ("Latakia" bleeds in). Winogrande generator saw the full sentence
  (continuation) + the answer, fully target-aware by design (principle rung is text-level answer-free). All 3
  generators documented; clean comparisons noted (complete−incomplete; principle-vs-full/complete).
- **Length confound ruled out on DCLM:** complete is 1.69× longer than incomplete (+58 tok) yet ties it, and
  corr(length-gap, ppl-gap)=+0.04 ≈ 0 → completeness-null is NOT a length artifact.
- **Reverse-filter designed + launched** (Dongwei: pick high-reasoning-value docs objectively, not by agent
  judgment) → `scripts/score_uncertainty.py` (sharded + ids-file). Score base-NLL of the continuation's first
  sentence with a WEAK (1.4B) and STRONG (Qwen-72B) judge; the GAP categorizes — 1.4B-high + Qwen-low =
  reasoning/knowledge-fillable (gold), both-high = arbitrary, both-low = trivial. Measured: 1.4B 49 docs/s, Qwen
  4 docs/s; raw pool `dclm_1500m` = 1.25M docs ≈ 1.5B tokens (~1/20 of the ~30B training set; length-filter
  keeps ~114k usable). Pipeline (1.4B full scan → top-30k → Qwen → gold) launched ~00:11; results land in the
  2026-07-12 entry.

## 2026-07-10
- **Winogrande as a reasoning-perplexity probe (Dongwei: separate model-issue from dataset-issue)** →
  `scripts/winogrande_score.py`. base/principle/full/complete/placebo partial-scoring (NLL of the shared
  post-blank suffix under each option; accuracy + continuation perplexity). Natural-prose rationales
  (principle = leak-free rule / full = terse binding / complete = full multi-step chain), 200 val examples,
  Claude-generated. Scored on **9 judges** (1.4B, OLMo-2-1B, Llama-3.1-8B, Qwen3.5-35B base+instruct,
  GLM-4.5-Air, Llama-3.1-70B, Qwen2.5-72B base+instruct): **adding a rationale LOWERS the true continuation's
  perplexity on EVERY judge (complete−base −0.60 to −0.84), monotonic in completeness (principle<full<complete),
  placebo ≈0 (−0.01 to −0.16 — no format penalty), accuracy +0.035 → +0.215;** base≈instruct on both pairs.
  This is the clean reasoning-specific drop that never showed on DCLM — on a reasoning-determined, self-contained
  target. (Mistral-7B failed on missing `sentencepiece`; Qwen3.5-2B cache had no weights.)
- **DCLM prose re-test — was the numbered-list format masking the effect?** Rewrote the 44 DCLM numbered
  rationales as NATURAL PROSE (content + the incomplete-gap preserved, style-only) → `data/complete_dataset_prose.jsonl`;
  added `--dataset` / `--insert {mid,prepend}` to `scripts/perplexity_complete.py`. Scored 9 judges (mid insert):
  **net drop (complete−base) appears ONLY for the 2 smallest models (1.4B −0.10, OLMo −0.13); every judge ≥8B is
  ≈0 to +0.19.** Content benefit real everywhere (complete−placebo −0.22 to −0.67); **completeness NULL on all 9
  (−0.01 to +0.04)**; insertion penalty (placebo−base) large everywhere (+0.37 to +0.71).
- **Topology isolation (1.4B, prepend vs mid)** → `docs/PREPEND_VS_MID.md`. Prepending the rationale before the
  context (context→target flow intact) collapses the insertion penalty (placebo−base +0.444 → +0.068) but dilutes
  the content (complete−placebo −0.546 → −0.197); net stays small. On a continuation target you can't place the
  rationale both near the target AND without breaking coherence.
- **Synthesis:** content benefit ~equal on DCLM and Winogrande (~−0.6) → **DCLM IS reasoning-rich; the DCLM null
  is STRUCTURAL, not a content deficit.** Winogrande wins because (a) self-contained sentence → no insertion
  penalty (vs DCLM mid-continuation +0.4–0.7), and (b) base perplexity stays high even for big judges
  (Qwen2.5-72B base ppl **18.6** on Winogrande vs **6.6** on DCLM) → headroom for the rationale. Completeness is
  never the active ingredient; presence/relevance of reasoning is (complete≫placebo, complete≈incomplete — the
  model fills deleted middle steps itself).
- **Correction:** the earlier "insertion penalty grows monotonically with capability" (from 3 judges) is NOT
  supported by 9 (large everywhere but noisy; Qwen2.5-72B has one of the smallest, +0.46). n=41 per DCLM judge.
- Set up the nightly completeness-commit cron (CronCreate `fe5ca089`, 06:48 UTC = 23:48 PDT; session-only,
  7-day expiry). This entry is its first run (fired 00:19 PDT, ~31 min late — session was busy at 23:48).

## 2026-07-09
- **`docs/JUDGE_CALIBRATION.md`** — 6 judges (base+instruct across DCLM/Llama/Qwen, 1.4B–72B): rationale-vs-base positive everywhere, content effect real (complete–placebo), completeness null; base≈instruct. GLM-4.5-Air (110B) added as a 5th family (same pattern; marked partial, n<41).
- **`docs/PERTOKEN_EXAMPLES.md`** + `scripts/pertoken.py` + `scripts/winogrande_score.py` — full per-token perplexity calc (context + rationale + continuation + per-token NLL + mean + diff) on the DCLM-1.4B judge; `fix_mistral_regex` warning flagged as a false positive (llama tokenizer, would break number tokenization).

## 2026-07-08
- **Completeness test (Dongwei: raise the bar to COMPLETE multi-step reasoning)** → `docs/COMPLETENESS_RESULTS.md`.
  Mined DCLM (1.5M-doc raw pool) for real docs whose continuation needs a ≥3-step chain; wrote a COMPLETE
  rationale + a gap-broken INCOMPLETE one (1-2 load-bearing middle steps deleted). Score the real target under
  1.4B (n=44): **complete−placebo −0.698 (relevant reasoning helps, placebo hurts +0.745), but complete−incomplete
  +0.004 (17/41) — COMPLETENESS makes ~no difference; the model fills the deleted step.** So on zero-shot ppl the
  active ingredient is relevance/presence, NOT gap-freeness; completeness is a TRAINING question. Wave-2 mining
  (→~88 docs) hit the account token limit (reset 9:30 UTC) — dataset stayed at 44. 72B confirm running.
- **Perplexity drop on ORIGINAL data (no Q&A)** → `docs/CONCLUSION_RESULTS.md`. Selected real DCLM docs whose
  continuation IS the doc's own conclusion (strong marker re-split; only ~1% qualify — 30/3000, 22 kept after
  agent-confirming genuine argument→conclusion). Score the **real conclusion** under DCLM-1.4B base:
  **real−base −0.055 (68% drop), placebo−base +0.399, real−placebo −0.454, real<placebo 22/22.** An irrelevant
  rationale *raises* the conclusion's perplexity; the correct reasoning lowers it. First clean, reasoning-specific,
  non-priming drop — on real text. Caveats: n=22, rare docs (~1%), rationale saw the conclusion.
  - **72B-base confirms** the reasoning-specificity: real−placebo −0.353, placebo hurts on 100% of docs. (real−base
    goes to +0.085 on the strong judge — the raw net-drop is format-dependent, but the reasoning signal is robust.)
  - **Marker-free confirmation** (Dongwei's point: filtering "thus/therefore" selects already-explicit docs). On
    real DCLM docs with NO marker, where the target follows from *implicit* reasoning: **real−placebo −0.463,
    real<placebo 14/15, placebo hurts +0.448** — essentially identical. So the drop is the rationale filling a
    genuine implicit gap, not signposting. → `docs/IMPLICIT_RESULTS.md`
- Judge calibration (Q from Dongwei): reasoning-specific gain on Q&A probes vanishes/reverses as judge strengthens
  (1.4B −0.024 → 7B −0.171 → 72B-base −0.015/strict +0.238; Qwen3-32B +0.142) — Q&A drops were noise; the
  conclusion result is the real signal. `scripts/perplexity_judges.py`.
- Cron switched back to CronCreate (Dongwei's choice) — accepts the 7-day expiry / session-only.

## 2026-07-07
- Reorganized the folder into `docs/ scripts/ data/` (mirroring `code_ladder/`); this log flipped to newest-first.
- Nightly experiment-log bot moved from a system crontab to **CronCreate** (12:07 AM local; session-only, 7-day expiry).
- Ran the **perplexity-drop hunt** → `PERPLEXITY_HUNT.md` (full data in `docs/PROBE_RESULTS.md`):
  - Base DCLM-1.4B judge + insertion/target sweep (500 docs): mean still +, ~12% of docs drop.
  - Independent R/N split (100 docs, 19 R) + Claude rationales: **R ≈ N (+0.09 both)** — reasoning-dependent
    docs show NO continuation-perplexity drop; the 12%≈warrant match was coincidence (R/N label is even
    *anti*-correlated with actual drops: R 11% vs N 15%).
  - Probe target (reasoning-determined answers), leakage-guarded: raw drop −0.358 — **but placebo (another
    doc's rationale) −0.333 too** → mostly format/priming, real−placebo ≈ −0.02 mean.
  - Per-probe / strict specific-answer probes: a *genuine* reasoning-specific drop survives only where the
    answer is a specific non-stated consequence (id444 "spoilers never deployed" real −1.58 / placebo −0.71),
    but only ~5/19 R docs can yield such a probe.
  - **FINAL: no clean zero-shot "+rationale lowers perplexity."** Only unconfounded test = training (needs sign-off).
- Natural-prose insertion (`--style natural`, Qwen-32B-instruct judge): format penalty halves (+0.30→+0.13)
  but content shrinks (−0.12→−0.02) → still +0.11 worse than base. → `data/completeness_v2_natural_results.jsonl`
- Warrant audit written up (recomputed from the workflow journal, not memory). `LOGBOOK.md` → `EXPERIMENT_LOG.md`.

## 2026-07-06
- 30-agent **rationale-warrant audit** (`wf_0494a236`): **11.6%** of DCLM warrants a rationale (347/3000).
  → `docs/RATIONALE_WARRANT_AUDIT.md`
- Thread copied out of `experiments/data_efficiency/` (Suhas's) into its own reasoning-pretraining home.
- **Claude-as-teacher rationales** vs Qwen2.5-32B on the 5 warranting docs — Claude wins 5/5 on
  continuation-NLL. → `docs/CLAUDE_VS_QWEN_RATIONALES.md`, `data/dclm_aug_claude.jsonl`
- **Controlled completeness-check v2** (Qwen-32B judge, placebo + memorization controls). Bracketed insertion:
  format penalty **+0.30** dominates, content −0.12, teacher −0.23 (Claude>Qwen), no memorization (`noctx` 3.07).
  → `data/completeness_v2_results.jsonl`

## 2026-07-05
- Generated **3,544** reasoning-augmented DCLM docs (544 @ Qwen2.5-7B + 3,000 @ 32B), local / no API / no egress.
  → `docs/OVERNIGHT_RESULTS.md` + `data/`
- Zero-shot completeness-check (continuation-NLL), 7B + 32B: **NEGATIVE and diagnostic** (format shift, not
  reasoning value). → `docs/OVERNIGHT_RESULTS.md`, `docs/SURPRISING_CASES.md`

## 2026-07-04
- Thread created (overnight autonomous session, Claude Fable 5); 5 research agents dispatched.
- Workstreams W1–W5 → `docs/CODE_REASONING_LIT.md`, `docs/PRETRAIN_AUGMENTATION_LIT.md`, `docs/ENTAILMENT_LIT.md`,
  `docs/STOPPING_RULES.md`, `docs/DATASETS.md`. Synthesis → `docs/LITERATURE_REVIEW.md`; prototype → `docs/PROTOTYPE.md`; plan
  (Stage-0 gate + open decisions D1–D8) → `docs/EXPERIMENT_PLAN.md`.
- Data reality check → Stage-0 resized to a ~3B matched-budget gate. NO launch until Dongwei rules on D1–D8.
