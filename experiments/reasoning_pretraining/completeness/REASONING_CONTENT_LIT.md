# Reasoning in pretraining: under-reasoning (H1) & finding/exploiting reasoning-rich text (H2)

**Status: 143 PAPERS FULL-READ AND PDF-VERIFIED (2026-07-23).** Every paper was read end-to-end from the PDF —
body, appendices, and figures inspected visually — with released code checked wherever a load-bearing claim depends
on implementation (50+ repos inspected). Every reader was **required to critique the paper's own eval methodology**:
is the headline A-vs-B comparison fair? is the baseline token/compute-matched? is an augmentation gain separable
from distilling the generator? does any control isolate reasoning from quality/length/domain? So each entry carries
the paper's result **and** the fine print that bounds what it can prove. One additional shortlisted paper
("Reinforcement Pre-Training on General-Domain Corpora", TechRxiv) is inaccessible behind a bot-wall and is covered
abstract-only, flagged as such. Written to be **read cold**. Provenance at the end.

The corpus came from a zero-seed neutral search (no paper names in any query) over 12 concept angles plus a
concept-expansion recall round — 533 candidates triaged, 143 must-reads read. The full triaged pool (nothing
dropped) is in `docs/DISCOVERY_POOL_2026-07-23.md`.

---

## TL;DR

**H1 (under-reasoning is real, pretraining-laid, and largely persistent) is saturated with evidence. H2 (whether
augmenting pretraining text with reasoning helps, and how complete it must be) is genuinely open — and the corpus
now specifies exactly which experiment would settle it.** Six things to hold onto:

1. **🔴 The shortcut is real, mechanistic, and objective-level — via four independent methods that agree.** Circuit
   analysis (arithmetic runs on a bag of memorized heuristic neurons, never replaced by an algorithm — faithfulness
   0.96, ablation −29pp, 79% of the circuit present at every pretraining checkpoint), controlled feature studies
   (networks use *only* the simplest sufficient feature even when 249 fully-predictive complex features coexist, yet
   learn them all the moment the shortcut is removed — a clean Won't), optimization theory (the fast feature zeroes
   the loss and starves gradient for the rest), and behavior (frontier models drop 10–60pp when premise order,
   irrelevant clauses, or token identities change). Next-token prediction itself contributes: a teacher-forced
   *complete* target chain can install a Clever-Hans cheat — later steps become trivial lookups, starving the pivotal
   first decision — so added reasoning must be left-to-right-derivable, never answer-leaking (Pitfalls-of-NTP,
   Tables 1–2, Remark 4).

2. **🟠 Reasoning in training text pays through EMITTED chains; latent one-pass composition is exposure-bound and
   capacity-punished.** With shortcuts controlled and both facts verifiably known, silent two-hop composition is
   ~7.6% for GPT-4o vs 92.8% with the chain written out (SOCRATES) — and instructed *silent* thinking with a bridge
   hint stays at 6.1% (§D.3): the rescue runs specifically through writing the intermediate token. On the training
   side, entities never seen in compositional contexts stay at 0.00 two-hop across all nine augmentation formats —
   conditioned on both 1-hop facts correct, at ~15.9× Chinchilla budget, and invariant to an identical QA-finetune
   on all conditions (Exposure, Tables 9/7/13). Training WITH chains and testing without still fails (Physics-3.2).
   Latent composition also costs orders of magnitude more parameters per bit than knowledge storage (U-shape) —
   though a parameter-sharing (recurrent) variant recovers ~72% OOD composition where the vanilla transformer gets
   ~0% (Grokked App E.2), so the limit is architecture-convertible, not absolute.

3. **🟡 Augmentation works on math corpora; on general web the current evidence favors *cleaning* over
   reasoning-injection.** The best-controlled positive result is MIND: token-matched dialogue-augmentation of
   OpenWebMath gains GSM8K +13.4, survives a generator swap 70B→8B (33.38→32.95 vs raw 29.17), beats a
   same-generator rephrase control (29.22), and its zero-knowledge-gap style gains nothing (29.12) — explicitness
   produced by knowledge-gap dynamics is the active ingredient. Against that: on general DCLM-class web text, a
   teacher-free *faithful-cleaning* rewriter beats a reasoning-*injecting* one (RePro), rewritten-with-reasoning
   text alone underperforms raw on CORE (REWIRE), and where controls are missing the distillation channel is
   demonstrably large (Swallow: LLM *scoring* buys <1pt while LLM *rewriting* buys >14 — that delta is the teacher's
   tokens). Teacher-free evidence for the structure effect is real but almost entirely synthetic (ground-truth-trace
   and program-generated corpora). **The unclaimed experiment [our target]: weak-or-self-generator augmentation of
   natural general web, judged on reasoning evals against deletion-only (ProX), faithful-rephrase (RePro), and
   compute-matched latent-depth (pause/concept-token) baselines.**

4. **🟡 Completeness: every existing dose-response says structure and difficulty, not volume — and none of it is on
   natural pretraining text.** Finer-grained chains are learned from less data, and SFT cannot decompose below the
   granularity of its training traces (a completeness floor); RLVR can re-derive skipped steps in verifiable domains
   (Zipping). Skipping steps is tolerable-to-helpful on easy algorithmic tasks while hard problems need the full
   chain (Skip-Steps; Less-is-More — over-compression collapses AIME/HMMT to ~0–5 while GSM8K holds). Longer traces
   help only when locally incremental — length-matched padding hurts (Inefficient-Reasoning); the most-concise
   thoughts win on all six benchmarks in ToW's denoising ladder; dialogue length is uncorrelated with gain (MIND).
   And the dose is capacity-conditional: long-CoT-style data actively *hurts* students ≤3B with an interior mixture
   optimum (Small-Models-Struggle) — directly relevant at our 300M–1.4B scale. The natural-text completeness
   dose-response (fixed tokens, vary granularity) has never been run by anyone.

5. **🟢 No loss/perplexity-family signal finds *reasoning* — two candidates are now empirically dead, one remains.**
   The two-model magnitude gap selects short/easy text (ScalingFilter's own validation is commonsense-only at
   +0.6–1.1% with no error bars; PreSelect's controlled baseline: +0.4 over random, Spearman 0.05 with the signal
   that works). Self-ablation (recipe A) is dead by our own experiments: across 5 base models and 6 sources the
   retrieval-head gap detects in-context *copy dependence* — boilerplate on top, verbal reasoning below web for 3 of
   4 models, sign-inverting across scale (`docs/RECIPE_A_SELF_ABLATION.md`). Single-model perplexity pruning
   actively de-selects reasoning: its winning criterion cuts code/papers ~3× and Symbolic-Problem-Solving *drops*
   4.88→2.91 while World-Knowledge gains (Perplexed-by-Perplexity, Table 1 per-category). What remains: **recipe B,
   multi-model rank-match** — untested by us, with lowered expectations: the signal is exam/knowledge-shaped (worse
   than random on HellaSwag/PIQA/SIQA) and its margin shrinks with scale. Never validate any of this by held-out
   loss: three independent results dissociate loss from downstream value.

6. **🔵 Pretraining content sets the post-training floor, slope, and ceiling — and RL's fixes are two-sided.**
   Pretraining loss predicts post-RL reasoning almost deterministically (ρ up to 0.99); RL transfer gates on ~1%
   pretraining exposure of the target context; RL amplifies whatever solution mode dominated pretraining, even when
   it is the worse one (Echo Chamber); reasoning-pretrained models lead +8.35 → +9.3 → +18.57 through base → SFT →
   RL, and doubling the baseline's SFT (+7.39) still does not catch the weakest reasoning-pretrained model
   (Front-Loading — reasoning-token budgets not matched, so placement vs quantity remains open). RL *can* erase
   specific shortcuts where rewards are verifiable (it removes the truncation shortcut in O(log) rounds; re-derives
   skipped granularity), but leaves unverbalized cue-reliance intact (wrong-hint following unchanged; reward hacks
   >99% used, <2% verbalized).

---

## The two hypotheses (reasoning-only)

**H1 — UNDER-REASONING AND ITS PERSISTENCE.** When predicting the next word, a model can get the word right by a
**shortcut** — matching a surface pattern, recalling a memorized association, or guessing plausibly — instead of
actually working through the multi-step inference the text implies. Call that **under-reasoning**. Two causes we
must keep apart: **(C) Can't** — the model lacks the knowledge/information to do the inference, so it guesses; and
**(W) Won't** — the model *could* do it, but a cheaper shortcut already satisfies next-word prediction, so it never
practices or learns the full inference. The claim: under-reasoning (especially the **Won't** kind) is learned in
pretraining and **persists** through fine-tuning and RL.

**H2 — FINDING & EXPLOITING REASONING-RICH TEXT.** Not "does the *model* reason" — rather, "does this *document*
contain reasoning, and can we use that?" Four parts: **(4) identify** reasoning-rich text; **(5) augment** text with
reasoning; **(6) completeness** — how fully must the reasoning chain be spelled out to help; **(7)** can a
**loss/perplexity-family signal** detect reasoning content?

---
## H1.1 — models take shortcuts instead of reasoning

**The question — do language models satisfy the next-word objective by a shortcut instead of the inference the text
implies — is answered YES, with unusual redundancy.** Twenty-eight papers converge through four independent
methodologies that fail in the same direction, and the diagnosis survives scale and post-training. The honest
qualifier: the shortcut ("Won't") is a *spectrum interleaved with genuine "Can'ts"*, not a switch — and essentially
every paper diagnoses the shortcut and then defers the data-side cure this thread is built around.

**Mechanism is real at four levels.**
- *Circuit.* Llama-3-8B arithmetic runs on a causally-verified ensemble of memorized-pattern neurons — faithfulness
  0.96 avg (per-operator 0.97/0.98/0.90/0.96, **Bag-of-Heuristics** Table 1 p14), −29pp under targeted ablation
  (Figs 7-8), 79% of accuracy-contribution at *every* Pythia checkpoint (Fig 10), still heuristic at 70B (App I).
  Failure = insufficient summed logit mass, not out-of-coverage — the paper tests and rejects coverage (Sec 4.3/Fig 9).
- *Optimization/feature.* **Simplicity-Bias** is the cleanest can't-vs-won't control anywhere: nets rely *exclusively*
  on the simplest sufficient feature (S-randomized AUC ≈0.50 vs Sᶜ 1.00, Table 4) against a 249:1 majority of complex
  fully-predictive features, and even when the simple feature is only 95%-predictive vs 100% complex ones (L̂MS-7,
  Table 2) — yet the same net learns the complex features to 100% once the shortcut is removed (p.7).
  **Gradient-Starvation** gives the dynamical account (Thm 2: the fast feature zeroes the loss and starves gradient
  for slower ones; its friend-or-foe corollary explains why in-distribution loss can't reveal the missing feature).
  **MLM-Distributional**: shuffled-word-order pretraining costs only ~3.3 GLUE points while non-parametric probes
  show syntax genuinely absent — a benchmark-satisfying shortcut coexisting with a real capability hole.
- *NTP-objective.* **Pitfalls-of-NTP**: teacher-forcing a *complete* target path lets later tokens fit by trivial
  adjacency lookup (Acc_cheat 96-100%, except G₂₀,₅ where models can't fit, Table 1) while autoregressive accuracy
  collapses to chance ~1/d (Table 2) — a distinct, in-distribution, teacher-forcing-only Clever-Hans cheat (Remark 5),
  on Mamba too. **Parity-RL**: below p_cot=1/3 complete chains, greedy emits the truncated shortcut at chance despite
  ~10⁷ samples (Thm 1, Fig 2). **Reversal-Curse** is the limiting case — after "A is B", p(A|B) sits at random-name
  likelihood across 350M-175B and every augmentation tried (Tables 3/6), while in-context reversal is 100% (Table 5).
- *Architecture.* **Shortcuts-to-Automata**: transformers converge to parallel shortcut solutions, exactly correct
  in-distribution but brittle OOD/at-length (Figs 6-7); non-solvable automata are a proven constant-depth Can't (Thm 4).

**The chain-structure slice (this thread's most relevant evidence) — how the chain is written decides the mechanism.**
- **Implicit-Shortcut** (from-scratch GPT-2): fixed computation-order chains → genuine OOD-robust implicit reasoning
  (100/99/~90% ID/+1/+2 steps, Fig 2); shuffled-order data → number-chaining that collapses 0.92→0.03-0.05 as
  variable-as-subtrahend items rise (Table 2), unfixed by scale to 1.5B or 500k templates (Tables 5-7); GPT-4o falls
  ~100→~30, replicated in natural language (Fig 7, Table 8).
- **CoT-Mirage**: complete chains *present in training* are learned as templates — 100% ID vs 0.01% on novel
  compositions of seen primitives (Table 1), 100%-correct traces on wrong answers (Table 2), rigid step-count padding
  (E.2.2), unfixed by scale 62K-3B. The lever is coverage, not explicitness.
- **Faith-and-Fate**: maximally complete scratchpad finetuning → near-perfect in-distribution, *zero* unseen-depth
  extrapolation (Fig 3); from-scratch per-digit GPT2-XL on 90M examples still fails 3×3 (App B.3).
- **Unfaithful-Reasoning-Synthetic (62)**: clean coherent chains *do* induce causally faithful stepwise computation
  (intervention non-response ≈0 at zero noise) but flip to skip-step above a noise threshold, with fine-grained steps
  buying ~10× tolerance (Fig 3), and prolonged training eroding faithfulness.
- **Parity-RL (27)** on proportion: p_cot>1/3 → pretraining alone generalizes; partial leap-2 chains generalize
  *below* threshold (Fig 15) — completeness and proportion trade off.

**SoTA models still shortcut — the magnitudes.** **GSM-Symbolic**: an irrelevant-sounding NoOp clause drops Phi-3-mini
−65.7pp (83.7→18.0 vs the paper's GSM8K-Full baseline, Fig 8a) and o1-preview −18.6pp; the clean monotone
clause-decline is Gemma2-9b-IT 84.4→79.1→68.1→41.8 (Fig 6); the arithmetic-difficulty confound is ruled out
(96-99%, App A.6). **Premise-Order-Matters** (provably order-invariant tasks): GPT-4-turbo 96.5→80.8 (Table 6a),
GPT-3.5 30→~1, R-GSM solved-subset to 64.9-89.9% under reorder (Table 2b). **Shortcut-Suite**: NLI shortcuts survive
SFT/RLHF through GPT-4/LLaMA3-70B (−30 to −60pp), CoT recovering much (LLaMA2-70B 3.6→66.2). **Multihop-Factual**:
co-occurrence-installed s₁→oₙ associations win even when every hop is known (~20% of edit failures; r=0.74).

**Genuine contradictions (both sides):**
- *Does RL erase the shortcut?* **Parity-RL** proves STaR/GRPO remove the truncation shortcut in O(log) rounds
  (length-calibrated model) — under-reasoning does *not* persist through RL there. But **GSM-Symbolic** (o1 −18.6pp),
  **Shortcut-Suite** (survives RLHF), and **Composition-Collapse** (GRPO helps only trained depths, OOD/ID 0.21;
  SFT-on-traces *worsens* composition 76.9 vs 69.8) show the opposite in real models. The partial reconciliation — RL
  can only up-sample routes pretraining already installed — is [our inference] from 27's mechanism, untested across
  these papers.
- *Complete chains → reasoning or templates?* Pro: Implicit-Shortcut, 62, 41-automata, 27. Contra: CoT-Mirage,
  Faith-and-Fate, Physics-3.2 (train-with-CoT-test-without still fails), Composition-Collapse. The split seems to
  track interpolation vs extrapolation/latent execution — but that through-line is [our inference].
- *Scale direction.* Shortcut-Suite claims inverse scaling and 62 finds capacity accelerates the skip-step switch;
  GSM-Symbolic and Token-Bias find frontier models *more* robust. Shortcut-Suite's number carries a label-prior
  confound, so the tension may be partly artifactual — but no bias-corrected comparison exists; both stand.
- *Chain order — fix or fragility?* Implicit-Shortcut shows computation-ordered chains *prevent* the shortcut;
  Premise-Order shows forward-order reliance *is* a shortcut. Neither tests the cross condition, so genuinely open.

**Won't vs Can't stays a spectrum.** Bag-of-Heuristics, Simplicity-Bias, Gradient-Starvation, Implicit-Shortcut,
36/63/35 land on capacity-present-shortcut-wins (Won't); Physics-3.2 (high-cardinality comparison stuck ~1% through
2.5M samples at 5.5× scale), NTP-Pitfalls Table 5 (the isolated first token is at chance *even with dedicated
supervision* — a Can't at the core, and the cheat's damage is *reducing the task to* that core), Thm-4 automata, and
Leap-of-Thought (failures track missing internal beliefs) land on genuine Can'ts.

**Implications for this thread** (each constraint is a paper's own result; transfer to natural web text is
[our inference]):
- The diagnosis is saturated; the marginal-value experiment is the un-run data-side cure — Implicit-Shortcut never
  reorders as an intervention, NTP-Pitfalls never adds a derivation-first scratchpad, 36/38/63 defer mitigation. [our inference]
- Constraints on reasoning-augmented text: **(a) left-to-right derivable, no answer leakage** (NTP-Pitfalls; Remark 4:
  CoT must *precede* the target); **(b) computation-ordered/coherent** (Implicit-Shortcut Table 2); **(c) low-noise** —
  beyond a threshold the model *ignores* rationales (62 Fig 2); **(d) fine-grained steps** — ~10× protective (62 Fig 3);
  **(e) diverse in depth/format/composition** — uniform chains become rigid templates (CoT-Mirage); **(f) per-instance
  explicit** (Reversal-Curse); **(g) fraction matters** — a sharp proportion threshold, lowered by partial chains (27).
- Simplicity-Bias L̂MS-7 predicts merely *co-presenting* complete reasoning next to shortcut-sufficient text will be
  ignored; augmentation may need to *degrade the shortcut's sufficiency* (distractor/NoOp/order-varied instances).
  No paper tests this. [our inference]
- Eval design: in-distribution loss/perplexity structurally cannot reveal missing reasoning (Gradient-Starvation;
  consistent with this thread's PERPLEXITY_HUNT null) — pair benchmarks with perturbation/non-parametric probes.
  Expect gains to be *interpolative*, bounded by augmented-chain depth/format coverage (GSM-Symbolic A.4,
  Faith-and-Fate, CoT-Mirage) — so vary chain DEPTH coverage, not just completeness at one depth. [our inference]
- An objective-side route reaches the same goal without touching the corpus (RLP, RLPT, parity-RL); none isolates
  reasoning-content from extra compute, so a cleanly content-controlled data intervention makes the two comparable.
## H1.2 — latent (one-forward-pass) multi-hop reasoning vs recall

This cluster maps most directly onto the thread's question: when a model must combine two facts *without
writing anything down*, does it silently run the inference, or does it recall? Across 17 papers (10 with
PDF-level verification) the answer is consistent and survives every correction: **models that demonstrably
hold both constituent facts largely fail to compose them silently in one forward pass, and the failure is
governed by pretraining exposure/arrangement, capacity, and depth-timing — not by missing knowledge.** They
mostly recall; they compose silently only where pretraining exposed the composition per-entity, where the
premises co-occurred in one document, where grokking-scale budgets and favorable reasoning-to-fact ratios
applied, or where model depth suffices for the chain length.

**The one intervention that rescues robustly at every scale is emitting the intermediate step.** SOCRATES
(`2411.16679`, Fig 2 + §5, metric code-verified) has GPT-4o composing 7.6% of shortcut-filtered two-hop
queries latently vs **92.8% with chain-of-thought on the same cases**, conditioned on the model answering
both single-hop queries correctly (Claude 3.5 Sonnet 8.4% latent). v2 §D.3 closes the loophole: instructing
the model to think step-by-step *internally* plus a hint to identify (not write) the bridge leaves latent
composability at **6.1%**, and **96.0% of CoT failures come from generating the wrong bridge** — the rescue
runs through *writing the bridge out*, not silent deliberation. Paper 69 (`2605.04330`) shows the same at
scale: ~30B models sit at chance for direct depth-≥2 deduction (Fig A.1) but exceed 90% with CoT.

**Composition is exposure-bound, per-entity — the strongest training-side result.** Exposure (`2606.09338`)
trains GPT-2 (124M–774M) from scratch on 100k invented people: both groups learn every atomic fact (**97%
1-hop**), but only entities seen in *compositional* two-hop sequences compose (up to **0.83**); held-out
entities stay at **chance** across all nine augmentation formats and all scales, at **~15.9× Chinchilla
budget** (Table 7), with Table 9 showing held-out two-hop is **exactly 0.00 even when both 1-hop facts are
answered correctly**. Two-Hop Curse (`2411.16353`) recasts this as *arrangement*: with facts in separate
documents, latent two-hop stays at chance under forced-choice decoding across **all** training mixtures —
including one with 13,500 demonstrated no-CoT two-hop pairs (App D) — while **same-document co-presence
yields ~50%** and in-context provision ~63% (Fig 6). The failure is not missing supervision; a bridge
learned as the *output* of fact 1 cannot serve as a *query input* to fact 2 in one pass (§6.1), and
co-occurrence or emission dissolves the mismatch.

**Latent composition is capacity-expensive relative to storage — direction robust, multiplier not a law.**
U-shape (`2504.03635`, ICML 2026) fits reasoning capacity at **~0.008 bits/param** (124 params/bit,
R²=0.85, Fig 4), with test loss **U-shaped in model size** while train loss falls monotonically and the U
*strengthens* with more steps (Fig 1) — over-training memorization, so oversized models memorize composables
instead of composing. The "~250×" penalty is derived arithmetic (2 / 0.008), not a paper claim: the ~2 b/p
storage anchor is Allen-Zhu & Li's, which the corpus's own capacity paper (37, EleutherAI, `2502.03490`)
**failed to reproduce, measuring ~1.6 b/p** for one-hop, while independently showing latent two-hop needs
each fact stored *twice* (Eq 3) and generalizes to zero held-out components. Both agree on the direction;
the multiplier is setup-dependent.

**But "the chain must be emitted" is not absolute.** Bounded counterexamples of latent installation with no
inference-time emission exist — Exposure's *implicit* formats reach 0.62–0.83 on exposed entities; Two-Hop
Curse same-doc ~50% (semi-synthetic no-CoT ~20–22% vs ~33% CoT ceiling); Grokked (`2405.15071`) installs
genuine implicit reasoning from conclusion-only data at ~50× the overfitting budget; k-hop (`2505.17923`)
is learnable at exponential data cost; paper 69's corrective objective adds **+18.9pp direct no-CoT
accuracy**. Every one is coverage-bound (per-entity), data-exponential (in hops), depth-bound (L=Ω(δ)), or
non-extrapolating. Emission is the *scale-robust* route; latent internalization works only inside those
tight envelopes.

### Genuine contradictions (both sides; no manufactured resolution)

- **Do explicit chains in *training text* help or hurt latent composition?** Exposure: naming the bridge in
  pretraining text yields **0.08** no-scratchpad composition vs **0.62–0.79** for bridge-omitted formats
  (Tables 2/13) — explicit chains install "bridge-as-thing-to-generate," not latent computation. Versus
  paper 69: a *complete* explicit trace attached under an isolated attention branch is the single most
  effective component for direct no-CoT accuracy (+18.9pp). Setups differ (same-prefix sentence vs isolated
  branch); the corpus genuinely disagrees on whether written chains transfer to silent computation.
- **Does scale help latent composition?** Yang (`2402.16837`) second-hop flat 0.64/0.65/0.61 (7B→70B, Fig
  3); Compositionality Gap ~40% roughly constant across GPT-3 scale; SOCRATES latent scales far worse than
  CoT. Versus Two-Hop Curse's frontier survey where the gap "may be reducing" (Claude Opus 4 ~61% no-CoT,
  Fig 1) — though App G shows many categories at exactly 0. Both stand.
- **Is the latent failure CAN'T or WON'T?** SOCRATES D.3 and paper 37 lean architectural *can't-in-one-pass*;
  Two-Hop Curse says latent capability exists conditional on arrangement (exposure gap, not hard can't);
  U-shape and 37's trapping regime describe a memorization *won't*; Hopping Too Late (`2406.12775`) isolates
  a depth-*timing* can't; SynthWorlds' navigation shows the recall shortcut *chosen* with content present
  (won't). The honest summary is a spectrum, not a switch.

### What it implies for this thread

- **Target emitted-chain competence, not latent one-pass composition, as the primary payoff channel.** [our
  inference; results are the papers'] Emission rescues at every scale; instructed silent thinking does not
  (SOCRATES D.3); latent installation is coverage-/data-/depth-bound wherever it works.
- **Every completeness/augmentation experiment must include a scratchpad-permitting eval cell.** [our
  inference] Judging explicit-chain data on a no-scratchpad eval reproduces Exposure's format confound.
- **Same-document co-presence is a cheap, teacher-free augmentation lever worth a controlled arm.** [paper
  constraint — Two-Hop Curse App E; application ours] It buys ~50% latent composition with no chain written.
- **Add a token-position-matched filler control to any "reasoning tokens help" comparison.** [our inference
  from Dot-by-Dot] Gains can be extra compute rather than reasoning content; the two separate only with it.
- **Do not use probe decodability or bridge-token emission as a success metric.** [papers converging;
  synthesis ours] Decodability ≠ use (paper 40: bridge 92–99% decodable while reasoning fails). Behavioral
  composition under both scratchpad conditions is the only trustworthy readout.
- **Stop quoting "250×" unqualified** [our recommendation]: "~0.008 bits/param reasoning vs a ~2 b/p storage
  anchor — a derived, setup-dependent ratio (anchor unreproduced at ~1.6; semantics-free synthetic KG)."
## H1.3 — does the reasoning gap persist through fine-tuning and RL?

H1.3 asks whether the reasoning pretraining did or did not install survives SFT and RL — whether post-training *adds*
capability or *elicits* what pretraining laid down. Across 30 fully-read papers the corpus supports a three-part
answer: (1) pretraining content quantitatively bounds and compounds through post-training; (2) ordinary RLVR mostly
re-weights within the base distribution, with narrow, condition-dependent carve-outs; (3) the only lever that reliably
crosses the base boundary is *new information* — distillation from a stronger teacher or new pre/mid-training data — and
every flagship "elicitation" demonstration carries at least one confound (teacher distillation, format unlock, or
benchmark contamination). "Pretraining installs it, post-training surfaces/orchestrates it" is the best-supported
reading, now with quantitative teeth it lacked: scaling laws, a clean RL/SFT mechanism split, and exposure gates.

**Compounding.** *Front-Loading* (`2510.03264`) is the anchor: substituting 20% of a 1T-token budget with curated
QA/CoT data widens the lead over a plain-pretrained 8B at *every* stage — +8.35 avg at base (Table 1: 61.05 vs 52.70),
+9.3 after SFT (Table 2), +18.57 after RL (Table 3). Doubling the plain model's SFT set gains +7.39% (Table 4:
26.62→34.01) yet still lands 3.32 *below* the weakest reasoning-pretrained model (37.33) — doubling helped, it just
failed to catch up; and the same data seen in *both* pretraining and SFT beats seeing it only at SFT (App C Fig 2). The
strongest quantitative version is the chess+math scaling-law study (`2607.16097`): post-RL reward at fixed RL compute is
an exponential function of pretraining loss (Spearman ρ up to 0.99, Fig 3a), the RL improvement *slope* grows linearly
with log pretraining tokens (r=+0.84, Fig 3b/c), and even the RL asymptotic *ceiling* is predicted by pretraining loss
(R²=0.90, App G.6; replicated on 1B OLMo-2 math, Fig 6 R²=0.98). Corroborated: continual math pretraining triples
Llama-3.2-3B's post-RL ceiling (`2503.20783` Table 4: 6.8→14.8→20.7), OctoThinker mid-training ends 30.1 vs 22.3 under
identical DAPO (`2601.06911`), the same 817 LIMO traces give AIME24 63.3 on Qwen2.5-32B but 9.2 on Qwen1.5-32B-Chat
(`2502.03387`), and RL cannot transfer to a context with ~0% pretraining exposure while ≥1% seeds robust transfer
(`2512.07783`).

**Elicitation vs creation, split by mechanism.** The cleanest evidence: *Base Models Know How, Thinking Models Learn
When* v4 (`2510.07364`) — steering a base model with thinking-model-derived vectors recovers ~76% of the RL
base→thinking gap but only 11% of the SFT-distillation gap (Table 1); the random-vector control gives *negative*
recovery (Fig 4), so the specific causal direction does essentially all the work; steering fires at only ~5-12% of
tokens (the base already emits the right token 88-95% of the time). Clean statement: **RL teaches *when* to deploy
pre-existing base mechanisms; SFT-distillation installs *new* ones.** This aligns with Yue's distillation result, with
`2505.21067` (920 R1 traces beat three zero-RL 32B pipelines; their weak-teacher GPT-4o control *failed*), and with
`2606.22317` (vanilla RLVR moves mean pass@256 by −0.5pp; the "+9.8pp beyond base" arrives only via an
undisclosed-teacher SFT bridge). The classic elicitation trio is real but softer than headlined: LIMO's 817 examples
and s1's 1K traces (`2501.19393`, AIME24 26.7→56.7) are observationally equivalent to sample-efficient frontier-teacher
distillation (no weak-generator controls; s1.1's stronger r1 traces on the *same* questions do significantly better,
AIME25 33.3→50.0), and one-shot RLVR's raw MATH500 jump (36.0→73.6) shrinks to +8.6 over its own format-reward baseline
(65.0) with pass@8 barely moving (`2504.20571`).

**Two eval-trust undercuts.** Contamination (`2507.10532`): Qwen2.5-Math-7B verbatim-reconstructs 39.2% of MATH-500
problem tails from 40% prefixes (Table 2; Llama 0.6%) and spurious-reward gains largely vanish on clean/post-cutoff
benchmarks — Qwen-family "elicitation" on MATH-500/AMC/AIME is partly memorization retrieval. Format artifact
(`2503.20783`): Qwen2.5-Math-7B scores 38.2% avg *template-free before any RL*, so "huge pure-RL gains" partly measure
suppression-then-reconstruction. And the formal capstone (`2602.15829`) shows elicitation is *cheap access, not
superficial capability*: post-training collapses the access cost from GB-scale adaptation programs to ≤10⁴ bits, but
saturating GSM8K from a pretrained-only 3B still needs ~6.2GB of program (contamination caveat, footnote 9).

**The Won't persists through post-training, on axes accuracy misses.** Outcome-RL faithfulness plateaus at 20-28%;
reward hacks are used >99% of the time but verbalized <2% in 5/6 environments; unfaithful CoTs are *longer* than
faithful ones (2064 vs 1439 tokens, `2505.05410` Figs 5/7). Reasoning-trained models verbalize a load-bearing hint far
more (R1 59% vs V3 7%) but their cue-driven answer-*switch* rates are unchanged (18.7% vs 15.3%, `2501.08156` Table 2).
From-scratch control: RL collapses within ~1 epoch onto whichever mode dominated the pretraining mixture (emitting it
near-exclusively, ~100%, `2504.07912`), pass@1 up while pass@64 declines — if the dominant mode is a shortcut, RL
amplifies it. Circuit work
confirms fine-tuning *enhances the base's existing* entity-tracking circuit (0.66→0.82, `2402.14811`), not a new one.

### Contradictions (both sides; no manufactured resolution)

- **The RLVR boundary — four camps that disagree on the metric itself.** *Yue* (`2504.13837`): coverage shrinks, ~0%
  uniquely-RL-solved math, now surviving an entropy-matched control (raising RL temperature to match base entropy still
  leaves RL below base at large k, Fig 18) and a guessable-problem filter (AIME24 30→18, crossover preserved, Fig 13) —
  but its own LiveCodeBench Fig 3 shows *no* crossover. *ProRL* (`2505.24864`): expansion where base pass@128≈0 — but
  the base is eval'd at RL-optimal temp 0.6, is already R1-distilled *and part of the RL data (SCP-116K STEM) is itself
  R1-generated*, with no non-distilled control; its own "Diminished" category concedes pass@128 declines where the base
  is competent, and the appendix concedes RL adds nothing where core skill is absent (Reasoning-Gym 'arc' 2.52, below
  the 7B reference 3.42). *Boundary-debate* (`2510.04028`): a two-stage shrink-then-expand reconciliation that
  empirically shows *both* directions across two model families — base>RL shrinkage on MATH-500 (Qwen base Pass@256 88.0
  vs RL ~72-73, Table 3) and all three Llama in-domain sets (Table 2), alongside RL≥base expansion on AMC/AIME (Table 1)
  — with eval sampling stated and matched (256/prompt, temp 0.6/top-p 0.95, App C.2) and methods compute-matched
  (Fig 2); the unresolved part is why the *same* Qwen model expands on AMC/AIME yet shrinks on MATH-500, plus a
  load-bearing Stage-2 result that is a 14/30-vs-20/30 AIME difference with no CIs. *CoT-Pass@K* (`2506.14245`):
  plain pass@k over-credits base models reaching right answers via *flawed* CoTs, and under CoT-validity scoring RLVR
  *does* extend the boundary to K=1024 — but the result hangs on one unvalidated 8B distilled verifier, and its own
  dynamics show flawed CoTs persist (P(CC|CA)~0.7 after 400 steps). Best-controlled middle (`2607.16097`): RL does
  *limited genuine tail discovery* on hard bins (promoting moves with π_SFT<0.05) while amplifying wrong modes (~20% of
  hard states) with pass@16 flat.
- **Mechanism of Qwen spurious-reward gains.** `2506.10947` attributes them to GRPO clipping bias amplifying a genuine
  latent code-reasoning prior; `2507.10532` attributes a large share to GRPO retrieving *memorized* contaminated
  answers (gains vanish on RandomCalculation / LiveMathBench-202505 / AIME2025). Both agree the effect is Qwen-specific
  and absent in Llama/OLMo; they disagree on latent-capability vs leakage.
- **Value of long/complete traces is capacity-conditional.** Front-Loading and `2502.03373` find long-CoT data scales to
  higher ceilings (8B/32B); `2502.12143` shows the *same kind* of data *hurts* students ≤3B (Δ_Long −4.7 to −7.1) and
  helps ≥7B, with an interior mix optimum (α=0.2). Different scales, so not strict — but a genuine tension against any
  uniform "more elaborate reasoning data is better" reading.
- **Does post-training fix flawed reasoning?** `2506.14245` argues RLVR incentivizes correct CoTs yet its own P(CC|CA)~
  0.7 shows flawed CoTs persist; `2505.05410` shows faithfulness plateaus and reward hacks go unverbalized; `2501.08156`
  shows articulation rises 7%→59% while switch rates are unchanged. Validity/verbalization and shortcut-removal come
  apart.

### What it implies for this thread

- **(Papers' claims)** Pretraining sets the post-RL floor, slope, *and* ceiling quantitatively (`2607.16097`;
  `2512.07783` exposure gate; `2503.20783`/`2601.06911` ceiling lifts; `2502.03387` Qwen1.5-vs-Qwen2.5 gate) — the
  strongest corpus-level license for treating the pretraining corpus as the right intervention point.
- **[our inference]** Evaluate our completeness interventions *through* a post-training stage, not only at base. The
  latent effect (Front-Loading: high-quality data ≈ neutral at base, +4.25 after SFT) and the scaling laws imply the
  payoff can be invisible in zero-shot/perplexity — consistent with our own perplexity-hunt null. Put a matched SFT (±
  small GRPO) head on each pretraining arm before comparing.
- **[our inference, building on `2503.01307`]** *Cognitive Behaviors* is the closest template for a rewrite intervention
  (behavior-enriched OpenWebMath → RL self-improvement unlock, generator-matched minimized control fails). Our version
  should add what it lacks: a weak/non-frontier rewriter arm (its rewriter is Qwen-32B and the code *injects* behaviors)
  and an accuracy-transfer eval beyond one toy task (behaviors transferred to GPQA but accuracy did not).
- **[our inference from `2507.10532`+`2503.20783`]** Eval-trust constraint: avoid Qwen-family bases scored on
  MATH-500/AMC/AIME as primary evidence; prefer post-cutoff sets, synthetic-clean tasks, or corpus-controlled models.
- **[Papers' claim + our framing]** The *Won't* survives on axes accuracy misses — cue-reliance switch rates unchanged
  (`2501.08156`), reward hacks unverbalized (`2505.05410`), RL amplifies the dominant mode even when worse
  (`2504.07912`). **[our inference]** Our success criterion should include a shortcut-reliance probe (cue-switch style)
  on models trained with vs without completeness augmentation, not only accuracy or chain plausibility — elaborate
  chains can be unfaithful.
- **[our inference — the open question the corpus sharpens]** Does the payoff require explicit chains *in pretraining
  text*, or does coverage-rich pretraining + tiny explicit-trace SFT reach the same endpoint? `2607.16097`'s corpus has
  zero explicit reasoning yet governs everything downstream with traces injected at SFT; LIMO/s1 show tiny SFT unlocks;
  but Base-Models-v4 says SFT-distillation installs *new* mechanisms (11% recovery), and `2503.01307`/Front-Loading show
  pretraining-time injection changes what RL can do. A clean two-arm experiment (completeness-augmented pretraining vs
  plain pretraining + matched small CoT-SFT, identical post-training after both) would separate these — no paper in the
  bucket runs it.
## H2.4 — identifying reasoning-rich text in a corpus

The question is whether any signal picks out *reasoning-rich* text — text whose value comes from worked-out inference — as opposed to text that is merely on-topic, long, clean, or exam-shaped. The corpus settles into three layers: selection demonstrably pays as domain/quality/difficulty curation at every scale tested; **no published signal isolates reasoning** from domain/length/format/quality; and three influence-function studies converge on procedural, worked-out, complete traces as the value carriers — correlationally, plus two confounded interventions.

### Layer 1 — selection pays, at scale

DeepSeekMath is the canonical existence proof. A fastText classifier seeded on 500K OpenWebMath positives plus an iterative human URL-annotation loop pulls **120B math tokens out of Common Crawl**; a 1.3B model trained 150B tokens on it reaches GSM8K **23.8%** vs 11.5% for OpenWebMath and 2.9% with no math (Table 1), and the 7B base (GSM8K 64.2 / MATH 36.2) beats Minerva 540B (Table 2). Two caveats travel with it: the 150B bake-off is not fresh-data-matched (MathPile repeats ~17 epochs, DSM ~1.25), and the signal is explicitly *math-domain-likeness*, not reasoning. MAmmoTH2 mines 5M naturally occurring Q-A pairs from CC (MATH 11.2→36.7 on Mistral-7B, Table 2); FineWeb-Edu's cheap distilled classifier lifts MMLU 33→37 / ARC 46→57 at matched 1.71B/350B. Essential-Web and the Amazon taxonomy paper industrialize labeling — per-document taxonomies (including a 5-level reasoning-depth ladder) at 24T-token scale, and compound filters that beat *unfiltered top-tier* data on reasoning answer-loss (Amazon F8 Minerva −10.9% vs top tier, §5.5 — answer-loss, never accuracy).

### Layer 2 — but no signal isolates reasoning

Every load-bearing selector in the bucket is confounded with domain, length, format, or quality:

- **AttentionInfluence** (self-ablation, +0.75pp overall, Table 1): the top-20% of *documents* is ~30% of *tokens* (73.1/241B, selected docs ~2× longer); the GPT-4o "reasoning" advantage exists only in the ~7% math/code slice of the corpus (Tables 3, 8); the head-mask sanity check fails on HumanEval (retrieval-masked 0.1098 ≈ random-masked 0.1159, Table 6); commonsense regresses (WinoGrande −2.2, PIQA −1.1). No matched-upsampling or classifier bake-off exists.
- **PreSelect** (Tables 12/13): the working signal is exam/knowledge-shaped — worse than random on HellaSwag (38.9 vs 40.0) and PIQA (67.7 vs 69.2) at 1B, PIQA/SIQA at 3B; the 7-task +5.3 headline flatters the 15-task +3.1; the DCLM margin shrinks +1.5 → +0.8 from 1B to 3B.
- **FineWeb-Edu**: the *prompt* intends grade-school steering, but the measured *outcome* is more academic/technical — Paloma arxiv ppl 23.4 vs FineWeb's 32.3 (Table 3), +3.2% Education/Math/Science topics (Fig 18), and the model card admits it "might overfit to academic looking content." The rubric's top tier explicitly rewards text that "follows detailed reasoning" (App F.1), and the threshold dose-response is non-monotonic (Fig 17: 3 best, 4 no better).
- **AutoDS** is at parity-or-below Uniform on commonsense (Table 4), and its Appendix C shows high-scoring web docs containing incomplete/incorrect reasoning (a stuck forum poster scores 0.932) — the score keys on reasoning density/educational framing, not completeness or correctness.
- **Essential-Web** never ablates the reasoning-depth clause; subject alone recalls **96–98%** of vetted math/code (Table 11), the STEM filter's near-no-op reasoning clause delivers the *largest* gain while the strictest clause (Top Code) scores worse, and the reasoning-inclusive math filter *lags* targeted FineMath by 8% on GSM8K (Table 3).
- The **Amazon taxonomy**'s own NMI table is the cleanest self-indictment: Timeliness–Reasoning NMI = 0.029 but Timeliness–DocumentType = 0.205 (Table 9) — its strongest "reasoning" gain is a News→Reference domain shift.
- **Removing Noise, not Finding Gold** (v4 of "The Data-Quality Illusion") supplies the mechanism: a classifier-quality score is provably a density-ratio reweighting of the HQ distribution, `s(x)=φ(p_HQ/p_LQ)` (§4, Eq 4.1), it latches onto sequence length (App C), and loss-on-HQ is an invalid proxy (U-curve, Fig 2). But the "illusion" framing overstates: v4 adds the experiment that CQF-refined data beats training on a large HQ set directly (**53.8% vs 50.1%**, §6/Fig 8; caveat: that large-HQ baseline is itself CQF-constructed) — filtering genuinely works by removing noise and aligning to a target, it just is not, and cannot be, a reasoning detector.

### Layer 3 — what the value carriers look like, and where they live

Three influence-flavored papers converge on procedural, worked-out text. **ProcKN** finds reasoning queries draw on procedurally-similar documents (code, worked equations; StackExchange 10× overrepresented) and the answer is *not* in the top docs (7B: twice; 35B: never), unlike factual queries — the which-docs attribution is correlational, though the influence estimator itself is accuracy-validated by counterfactual retraining (DROP 0.61→0.38). **Attrib** finds explicit exploration/verification behaviors carry influence and backs it with a real intervention: GPT-4o-truncating those behaviors from SFT traces drops MATH500 77.2→73.8. **QaDS** finds one-shot-influence scores rank complete NL chains high and bare-answer/code-like samples low (Fig 5, Tables 9-10). **Beyond-Code** is the closest thing to a pretraining-scale intervention: at a fixed math-token budget, replacing ordinary math with classifier-selected "cognitive-scaffold" math (no generator) gains math Overall +17.56%, OlympiadBench +47.78%, MATH +23.17% while GSM8K regresses −6.29% (p.8) — but its selector is heavily formatting-confounded (indentation ratio 0.0006→0.5446, ~900×; length 5.3×, Table 1), with no dose sweep.

Two papers then bound what selection can even reach. **OpenThoughts3**'s Appendix P runs a fastText reasoning-trace detector over DCLM-RefinedWeb and finds the web essentially does not contain long CoT, while its Table 22 shows stripping self-reflection costs −49.1% relative. **Essential-Web**'s Table 26 points the same way: the one LLM-rewritten corpus (MegaMath-Web-Pro) tops every filter-only math set (GSM8K 27.3 / MATH 12.2 / MMLU-Math 41.4). The complete chains that carry value must largely be *generated*, not found.

### Genuine contradictions (both sides stand)

- **arXiv/formal text.** DeepSeekMath (Tables 8-9) finds arXiv-heavy corpora give *no gain or active deterioration* on math benchmarks (MATH 11.1 vs 12.5 no-math; miniF2F 11.9 vs 21.7), while ProcKN finds ArXiv among the influential procedural sources for reasoning queries. Influence on an existing ability vs training-value on QA benchmarks may simply diverge (format/style match); neither paper resolves it.
- **fastText/embedding filters.** Extremely effective at corpus scale for domain recall (DeepSeekMath's 120B corpus; MAmmoTH2's 5M-pair funnel) yet they *underperform* LLM difficulty/length rating for selecting reasoning questions within a pool (OpenThoughts3, Table 5). Different regimes (corpus-scale domain recall vs within-pool quality ranking), but a genuine split verdict on the same tool family.
- **Does a reasoning-inclusive filter beat plain quality targeting?** Essential-Web's reasoning-inclusive math filter *lags* the targeted FineMath classifier on GSM8K (−8%, Table 3), while Amazon's reasoning-inclusive compound filter *beats* even unfiltered top-tier quality data on reasoning answer-loss (§5.5). Metrics differ (accuracy vs answer-loss) and neither isolates the reasoning clause — unresolved.
- **Is the classifier-filter gain real or an artifact?** Within one paper, Removing-Noise shows CQF gains are task-alignment on a density-ratio score *and* that CQF-refined data still beats direct large-HQ training (53.8 vs 50.1). The bootstrapping resolution it offers (HQ = signal, corpus = coverage) is the paper's claim, not settled fact — its large-HQ baseline was itself CQF-constructed.

### What this implies for the thread

- The bottom line: **selection finds domain/quality/difficulty, not reasoning**, and the two "where is the reasoning" probes both point off-web (OpenThoughts3 App P; Essential-Web Table 26). **[our inference]** this is convergent support for the augmentation direction — complete chains must largely be generated into text, not found — with the standing caveat that MegaMath-Web-Pro is a strong-teacher rewrite (distillation-confounded).
- Recipe B (multi-model rank-match on a size ladder) remains the only standing loss-family candidate, but PreSelect's own appendix warns the vanilla signal is exam-shaped and its edge shrinks with scale. **[our inference]** if we run it we must define the ladder's ability order by a *reasoning* benchmark and inspect the selected-domain composition before training — a rank-match pass without composition inspection is not interpretable.
- Beyond-Code is the closest published template for a natural-text completeness experiment. **[our inference]** replicate the fixed-budget replacement but (a) control formatting (match indentation/length distributions), (b) run the replacement-ratio dose sweep they skipped, and (c) pre-register the easy-task regression (GSM8K −6.29%) rather than treating it as failure.
- **[our inference]** a consistent cross-paper pattern is that upweighting structured/educational/reasoning-flavored text buys hard-task gains at the cost of easy/commonsense regressions (AttentionInfluence, FineWeb-Edu, PreSelect, AutoDS, Beyond-Code) — any eval suite for our runs must include commonsense/easy tasks so this trade is measured, not hidden.
- **[our inference]** for any gate we build, Removing-Noise's paper-stated lessons apply: the score inherits HQ-set and length bias, and loss-on-the-HQ-set is an invalid proxy — validate the reverse-filter's output with a data-conditioning-style test and always run a length-matched control, the single most common un-run control in this bucket. Essential-Web's reasoning-depth rubric is a reusable completeness axis, but its ground truth is LLM-consensus (human-human κ 0.38–0.54), so spot-check a sample before trusting it as the experimental axis.
- The influence trio gives the thread its best working definition of the target text — procedural, worked-out, with explicit exploration/verification — but all three are correlational or confounded. **[our inference]** our completeness intervention is precisely the missing clean experiment: hold domain, length, and formatting fixed; vary only chain completeness; train at fixed tokens. No paper in this bucket has run it.
## H2.5 — augmenting pretraining text with reasoning

**The question.** Does augmenting pretraining text with reasoning help — and does the gain *separate* from distilling a
strong generator? The corpus gives a two-sided but tight answer: augmentation works, but cleanly only inside the
**strong-teacher-distillation regime**, mostly on **math** corpora, at **one** completeness setting, on
**reasoning-shaped** evals. Teacher-free evidence is either synthetic/symbolic, or — on natural general web — favors
*faithful cleaning* and *latent compute* over reasoning-text injection. Nobody here augments natural general web with a
weak generator and shows a reasoning-specific gain — the completeness dose-response on natural text is run by no one.

**Augmentation-works evidence survives but stays strong-teacher-confounded.** The two flagships are big and real but
neither isolates its gain from the teacher. TPT (append expert thinking to every doc; GSM8k 19.2→50.1, 5-task avg
26.2→43.9, Table 1) writes its from-scratch thinking with **Qwen3-8B** and never runs a from-scratch weak-teacher
control. BoLT (MATH 5.74→25.38, Table 1) honestly isolates its own component: the teacher-matched comparison is
**Latent-Thought 25.38 vs WRAP-CoT 19.36 = +6.0 MATH** (~⅓ of the headline), and a raw-space ablation (25.38→22.38)
further isolates the latent-space design. Swallow is the purest distillation confound: token-matched 50B rewrites give
HumanEval +17.0 / HumanEval+ +16.1 / GSM8K +12.4 / MATH +7.6 (Fig 5), and its own model-fixed comparison —
LLM *scoring* <1pt vs LLM *rewriting* >14pt (Fig 3/4) — localizes the channel in token-injection, not rewriting quality.
MathCoder2 adds a rare same-generator No-code-prompt control (SAT 59.4 vs 37.5) but its isolating arm is not
token-matched (+2.7B extra tokens).

**Deconfounding now rests on a smaller, mostly-synthetic base.** MIND is the best-controlled natural-text
deconfounding: GSM8K **+13.42 token-matched** (raw OWM-4B 12.96→26.38, Table 5), 33B of dialogue-augmented OWM beats a
**3.6× larger** raw corpus (26.38 vs 20.47), same-generator *rephrase* ≈ raw (29.22 vs 29.17), generator swap **70B→8B
keeps the gain** (33.38 vs 32.95, Table 6), and the *zero-knowledge-gap* style (TWO PROFESSORS) gains nothing (29.12 ≈
raw). Demystify (>1000 models) is the anti-distillation datapoint — **70B-rephrased trains consistently WORSE than
8B-rephrased** (Fig 5) — but is perplexity-only with a Wikipedia style-match confound. The genuinely teacher-free
evidence (Kinetics templated traces flipping OOD 0.00→1.00; Internalize's k-parity theory; Logic-Corpus / FOL-Traces
symbolic corpora) is entirely synthetic. Two doc-level corrections matter: **GrokWild must be removed** as teacher-free
evidence — its augmenting LLMs are GPT-4o + o1-mini (App A.5 = the baselines), so it is distillation/self-comparison
confounded and its "incorrect-facts-help" claim is undemonstrated; and **BoLT's teacher-free self-bootstrap is ~10%→~13%
over iterations (Fig 8), not ~13%→~20%**. After GrokWild, MIND is the *only* supporting natural-text datapoint — and
its 8B generator still dwarfs the 7B student.

**The newly-considered papers sharpen the open gap.** On natural general web the teacher-free wins are *not*
reasoning-augmentation. RePro (RL-trained 4B *faithful* rephraser, adds no reasoning) beats SFT-from-GPT-4o
(0.217 vs 0.192) and the reasoning-injecting ReWire (0.217 vs 0.201, 400M), framing added reasoning as a collapse risk.
ProX (deletion-only, verified in code) yields **+6.2 on a 9-task math-CoT avg** over token-matched raw CPT with *zero*
generated reasoning. Megadocs' latent-thoughts is the best synthetic method (1.80× data efficiency, +9% easy-QA) but
concedes the weak-generator control is *impossible* (Llama-1B fails) and shows much of its scaling advantage is an
optimization effect (Fig 6-right). A separate **augment-computation-not-text** branch (PonderLM / CoCoMix /
Adaptive-Latent-CoT) gains self-supervised with **no distillation** but moves no reasoning benchmark; Doc-Packing /
WRAP++ show pure *arrangement* installs latent multi-hop that raw co-presence does not (raw concat ≈ base).

### Strongest findings (verified)
- **MIND (`2410.12881`):** token-matched GSM8K +13.42 (Table 5); 33B dialogue-augmented OWM > 3.6× larger raw OWM
  (26.38 vs 20.47); rephrase ≈ raw; 70B→8B keeps the gain (33.38 vs 32.95, Table 6); zero-gap style gains nothing. Best
  natural-text deconfounding — but 8B generator ≫ 7B student.
- **BoLT (`2503.18866`):** teacher-matched isolated Latent-Thought 25.38 vs WRAP-CoT 19.36 = +6.0 MATH (Table 1),
  ~⅓ of the 5.74→25.38 headline; raw-space ablation 25.38→22.38.
- **Swallow (`2505.02881`):** model-fixed scoring <1pt vs rewriting >14pt (Fig 3/4) localizes the gain in
  token-injection; HumanEval +17.0 / +16.1 HumanEval+ / GSM8K +12.4 / MATH +7.6 at token-matched 50B.
- **Kinetics (`2510.25791`):** ground-truth TEMPLATED traces (no teacher) flip OOD failure→success (Composition k=2
  0.00→1.00, Sorting k=4 0.04→0.83, Table 2); answer-only never generalizes; Intersection stays ~0.01 even with CoT (a
  true Can't). Not supervision-matched.
- **Demystify (`2510.01631`):** 70B HQ-rephrased trains consistently WORSE than 8B (Fig 5); optimal synthetic ~30%,
  pure-textbook hurts. Pile-perplexity-only, style-match confound.
- **ProX (`2409.17115`):** deletion-only refinement (verified in code) gives +6.2 math-CoT and +2–3 general over
  token-matched raw CPT — the cleanup bar augmentation must beat.
- **RePro (`2510.10681`):** faithful RL rephraser (no reasoning) beats SFT-from-GPT-4o (0.217 vs 0.192) and reasoning-
  injecting ReWire (0.217 vs 0.201, 400M); the winning structure is cleaning, not reasoning.
- **EntiGraph (`2409.07431`):** closed-book QuALITY 39.49→56.22 (Fig 2), beating its own GPT-4-turbo generator (51.30);
  ~12× (not ~330×) more tokens than rephrase; token-matched EntiGraph-29M control exists (App H.3, summarization only).

### Genuine contradictions (both sides stand)
- **General web vs math.** On general web, faithful cleaning wins (RePro > ReWire; REWIRE's rewritten-alone worse than
  raw on CORE) and added reasoning is at best neutral / collapse-risk; on math, reasoning-injection helps a lot
  (MIND / BoLT / TPT / MathCoder2). The split tracks domain.
- **Didactic/textbook style.** Demystify finds textbook/exercise format the WORST performer (collapse patterns on
  general-web perplexity); Phi-1 makes synthetic textbook-style its whole successful method for code. Opposite
  directions — different domains, Phi-1 distillation-confounded with a withheld pipeline.
- **Does reasoning content carry the gain, or is it token-injection/optimization?** MathCoder2's No-code-prompt control
  and MIND's rephrase control argue STRUCTURE; Swallow's scoring-vs-rewriting, Megadocs' Fig 6-right optimization
  effect, and REWIRE's diversity attribution argue token-injection/distillation/complementarity. Unresolved.
- **Latent vs explicit route.** PonderLM / CoCoMix / Adaptive-Latent-CoT buy gains via latent per-token compute with NO
  text change and no distillation; text-augmentation papers assume explicit reasoning in the corpus. Both show LM/easy
  gains, neither shows reasoning-specific gains cleanly — which route removes the shortcut is open.

### Implications for this thread
- **[our inference]** The cleanest missing experiment is now sharp: augment *natural general web* with a **weak (or
  same-data / self-distilled, matched-token)** generator and measure a **reasoning-specific** eval, against (a) a
  deletion-only ProX baseline and (b) a faithful-rephrase RePro baseline. Every augmentation-works result is
  math-heavy, strong-teacher-distilled, or non-reasoning-evaluated; the megadocs papers concede the weak-generator
  control is impossible in their regime.
- **[our inference]** Any augmentation gain must clear the **deletion-only bar** — ProX gives +6.2 math-CoT and +2–3
  general at matched tokens/FLOPs, and RePro's faithful rephrase beats reasoning-injection on general web. Bake both in
  as controls at matched tokens AND matched generation FLOPs.
- **PAPERS' CLAIM (MIND, ToW):** completeness pays through STRUCTURE (knowledge-gap dynamics; local incrementality),
  not raw length — dialogue length is uncorrelated with gain (MIND, Table 13); verbose noisy thoughts are monotonically
  worse (ToW). **[our inference]** vary chain *structure/granularity* at fixed token budget, not length.
- **PAPERS' CLAIM (Doc-Packing, WRAP++):** a pure data-*arrangement* lever installs latent multi-hop that raw
  co-presence does not. **[our inference]** test arrangement/packing as a cheap augmentation-free arm — it may capture
  much of the gain with no generator and directly probes the exposure mechanism.
- **[our inference]** The latent-compute branch confounds the whole thesis: if under-reasoning is partly a per-token
  compute (Can't) limit, latent depth fixes it with no data change and no distillation. Defend the explicit-text claim
  against a compute-matched latent-depth baseline; conversely, that branch's failure to move any reasoning benchmark is
  the best current argument that explicit text is still needed.
## H2.6 — how complete must the reasoning be

The question is how completely a reasoning chain must be spelled out for training (or supervision)
on it to help. The corpus gives a consistent, strongly-conditioned answer: **completeness pays
through granularity, structure, and difficulty-matching — not through volume or length.** But every
dose-response is in the wrong regime for this thread — synthetic (Zipping, Inefficient-Reasoning),
post-training SFT/DPO (Less-is-More, Skip-Steps, FSLR), verifier-side (Enthymeme), or observational
(Model-Says-Walk). **No paper varies the completeness of reasoning injected into *natural* text at
matched tokens.** That cell is still empty.

**Completeness is difficulty-conditional.** From the compression side, Less-is-More's ablation is
clean: SFT-only over-compression collapses hard benchmarks (1.5B AIME 5.0, HMMT 0.0; 7B AIME 8.4,
HMMT 2.2 — Table 2) while GSM8K holds (~77.6/88.5); restoring length via DPO recovers hard-problem
accuracy (SFT+DPO 1.5B AIME 23.4). The authors deliberately reject their most aggressive compressor
(79.1% reduction) because it "omitted critical reasoning steps needed for more difficult problems,"
choosing 56.7% instead (Table 4, App A.5) — direct evidence a compress-maximally strategy fails on
hard items. (The proportional-retention direction is the *opposite* of intuition: compression ratio
*rises* with difficulty, 79.1%→90.6% across d1→d8, Table 1 — harder problems keep proportionally
less but produce longer absolute chains via a length target |r̃|~α·d(x).) Skip-Steps confirms the
other half: its GSM8K probe shows accuracy roughly stable with slight decline as steps drop (Test-OOD
61.33→60.44, Table 11) — no gain, but no collapse — and, crucially, skip-only training underperforms
skip+complete-chain training on OOD-hard (7.86 vs 11.13, Table 8/B.5), with the paper calling complete
steps "essential" and skip-only a route to "shortcuts that harm generalization." Volume is controlled:
step-matched (Table 4) and 2000/task volume-matched (Table 9) controls confirm the skip-data effect is
not pure data.

**There is a granularity floor, and RL patches it only sparingly.** Zipping runs a four-level dose
(g=1/2/4/8 on five Qwen sizes, Fig 11): coarser compression monotonically needs more data, g=8 near
the 1/23 chance line for most sizes. After g=2 SFT, tasks that need sub-step decomposition sit at
chance (Fig 8) — SFT cannot decompose below its training granularity. RLVR recovers those tasks
(+58.17 to +86.11 from near-chance, Fig 6a), but the decomposition is *minimal and targeted* (App
D.3): the model keeps g=2 for the bulk and expands only the single needed fraction step, and
Llama-3.2-3B inserts a dummy "*1" to satisfy the format — RL patches the missing granularity, it does
not re-expand the chain, and one variant borders on a format shortcut. The repetition result
("Implicit degrades under repetition," Takeaway 3) is model-dependent: it holds on Qwen but reverses
on Llama (Fig 5).

**Prefer incremental structure over length or verbosity.** Inefficient-Reasoning isolates the
mechanism as systematic local incrementality: longer locally-incremental traces beat shorter optimal
ones under *both* a fixed token budget and a fixed graph count, while length-matched padding controls
slightly *hurt* and induce repetition loops (28.5M synthetic, NeurIPS 2025, PDF+code verified). This aligns
with the H2.5 completeness papers: ToW finds the most-concise thought variant best on all six
benchmarks (though not monotone — TruthfulQA breaks it, Table 6 — and verbose thoughts carry ~25%
noise); MIND finds dialogue length uncorrelated with gain (Table 13) and its zero-knowledge-gap
dialogue gains nothing (29.12 ≈ raw 29.17, Table 3).

**Two papers cut against naive completeness.** FSLR varies the *supervision target* across three
completeness levels and finds an inverted ordering: complete trajectory 82.28 < full plan 85.88 <
first-step-only 87.08 (in-dist avg, Table 1/4), with first-step-only also winning OOD (+8.08, Table 2)
and on GSM-Symbolic (79.6 vs 70.7, Fig 5) — the load-bearing content is the enthymematic
operation-selection decision, not the full execution. Model-Says-Walk's trace audit supplies the
bucket's only *within-model* completeness gradient on natural-language traces (Table 17, App H):
accuracy 88.5% when the trace names AND applies the hidden constraint, 44.4% when it never mentions it,
and **16.7% when it mentions-but-does-not-apply** (Fisher p<0.01) — half-complete reasoning is worse
than none. That converges with Faithfulness's *necessity* property: a chain must be *used*, not just
present. Faithfulness gives the computable definition (completeness = the chain screens off the prompt,
I(P;A|C)≈0) and the warning that RL interventions buy monitorability, not shortcut removal (hidden-test
pass stuck ~0.24–0.32, Fig 7).

**The explicit-regime evidence, re-adjudicated.** Enthymeme's dose-response is *more* credible than a
loosened-trigger artifact: appendix Figs 6/7/9/10 give per-class precision and recall, and entailment
precision *rises* with step count as accuracy rises 0.530→0.733 (ANLI, Table 4). Its honest caveats are
different — the gold complete human premise barely beats none (0.558 vs 0.530), the step ladder is
literally a ≤10/≤20/≤30-word ladder (Fig 11), and it is SAT-verifier evidence, not LM-training. Exposure
sets the boundary for *latent* reasoning: explicit compositional text does not auto-compile into a
no-scratchpad computation, and composition is exposure-bound (conditional P_held 2-hop = 0.00 across all
nine formats even with both 1-hops correct, Table 9; explicit pinned at 0.09 across every LoRA rank,
Table 13). A LoRA QA-finetune stage equalizes the eval format for all nine conditions, so the gap is a
*pretraining-format* property, not a finetuning artifact.

**Existence proofs and their confounds.** Logic-Corpus and TracePile show explicit-full-trace training
helps at scale, but the magnitudes matter: Logic-Corpus's math transfer is near-zero (MATH
+0.7, MathQA +0.4, Table 4b) — the real transfer is form-adjacent (RobustLR +32.0, abduction
+11.7/+33.1, code +6.1/+10.3); TracePile helps broadly (GraphWiz +41.5; math avg +7.4 on LLaMA-3.1) but
never dose-varies completeness and is teacher-distillation- and execution-format-aligned. RATIONALYST
makes completeness its motivation but never a variable; Text-Complexity is orthogonal (surface
simplification) but warns that uniform LLM rewriting collapses lexical diversity (TTR 0.33%→0.19%) with
zero-shot knowledge costs.

### Genuine contradictions (stated as contradictions)

- **Is complete-trajectory supervision the right target?** FSLR finds *less*-complete supervision beats
  complete trajectories (first-step-only 87.08 > full plan 85.88 > full trajectory 82.28). Skip-Steps
  finds removing complete chains from the mixture *hurts* (skip-only 7.86 < skip+complete 11.13 OOD-hard;
  complete steps "essential"), and Logic-Corpus/TracePile show full explicit traces helping broadly. Both
  sides are SFT on math-adjacent tasks; FSLR is not token/compute-matched (lighter finetuning confound)
  and Skip-Steps' skip-only arm also cuts data. Genuinely unresolved.
- **Does length or structure carry the value?** Inefficient-Reasoning finds *longer* locally-incremental
  traces beat shorter optimal ones at matched tokens and matched graphs; ToW finds the most-concise
  thoughts best on all six benchmarks and MIND finds length uncorrelated with gain. The
  "structure-not-length" reconciliation is plausible but is [our inference] — it rests on cross-paper
  synthesis (ToW's verbose arm is noise-confounded), even though Inefficient-Reasoning's accuracy ordering
  and low-surprisal mechanism are now PDF-verified.
- **Does RL backfill what incomplete training left out?** Zipping: yes on synthetic verifiable arithmetic
  (RLVR re-derives skipped granularity, +58 to +86, though minimally and with one near-format-hack).
  Faithfulness: no on naturalistic shortcuts (vanilla GRPO amplifies wrong-hint following); Model-Says-Walk:
  reasoning-mode advantage collapses to +1.8pp n.s. (p=0.31) after Elo control (Table 19). Regime-dependent.

### What it implies for this thread

- The natural-text vary-completeness-at-matched-tokens experiment is the open cell — every dose-response
  in the corpus is synthetic, post-training, verifier-side, or observational (papers' own scope statements).
- [our inference, from Zipping's floor + Exposure] Training traces must be at least as fine-grained as the
  finest step we expect SFT-stage models to produce, and if the target is latent one-pass reasoning the
  augmentation format must match the no-scratchpad inference distribution.
- [our inference, from FSLR] A cheaper arm worth a matched-token slot in our sweep: inject only the
  enthymematic operation-selection step, not the full chain — supervision-completeness and
  emitted-completeness decouple (FSLR models still emit complete chains at inference).
- [our inference, from Model-Says-Walk Table 17 + Faithfulness] Augmentation QC must verify the injected
  chain is *used*: filter for chains whose removal changes the model's answer, not chains that merely read
  complete — mention-without-apply scores worse than no chain.
- [our inference, from Logic-Corpus + TracePile] Expect cross-domain transfer only where the inference
  *form* matches the injected chains (math transfer was near-zero); eval suites should separate
  form-matched from form-transfer tasks.
- [our inference, from Skip-Steps Table 8 vs FSLR] A mixture-ratio axis (fraction of examples carrying
  complete chains) may matter more than per-example completeness — no paper varies this. Also carry
  Text-Complexity's control: a rewrite-without-reasoning arm to separate the reasoning effect from
  LLM-rewrite side-effects.
## H2.7 — can a loss/perplexity-family signal detect reasoning content?

**The short answer, sharpened by the full corpus: no signal in this family has been shown to
find *reasoning*.** Every member that works is validated on *general* value — quality,
learnability, domain alignment, benchmark value — and the papers that carry a per-task-family
readout show the signal is neutral-to-*negative* on reasoning specifically. Nothing in the bucket
flips positive under scrutiny; several results get stronger in the negative direction. And our own
experiments have now closed the two most-principled candidates empirically.

**The size-magnitude gap (big-model-minus-small-model perplexity) is dead, on converging
evidence.** ScalingFilter — the primary source for this signal — validates its 124M/774M GPT-2
perplexity ratio only on 7 commonsense tasks, +1.12% over perplexity gating and +0.62% over a
binary classifier, with no error bars, no seeds, and zero reasoning benchmarks (ScalingFilter
Table 1). Independently, PreSelect runs exactly this as a controlled baseline: ScalingFilter beats
random by only +0.4 (37.6 vs 37.2, 1B/30B) versus PreSelect's +3.1, and the two signals are
near-orthogonal (Spearman 0.0533, Pearson −0.079) (PreSelect Table 7, §A.7.1). RHO-1 adds a third
leg: its genuine weak-to-strong variant (a 1B reference guiding Llama-2-7B, Appendix I) yields only
+0.9 AVG while *dropping* MMLU-STEM 3.4pp (Table 5) — the size-gap version of the family's
best-known success is nearly null. Our own 1.4B-vs-72B reverse-filter closes the loop: it found
knowledge, not reasoning — exactly what ScalingFilter predicts.

**The excess-loss / reference-gap sub-family carries real signal — but for learnability and
quality, never reasoning, and the reference is the whole game.** DoReMi's ablation isolates it:
proxy-loss-only and reference-loss-only each beat baseline on 0/22 Pile domains while the *gap*
beats it on 22/22 (DoReMi Table 7) — but at provenance-domain granularity with knowledge/QA evals.
RHO-LOSS's reference term is a *noise* flag: 18× speedup on noisy Clothing-1M collapses to ~2× or
nil on clean data — a job a pre-filtered pool has already consumed. S-RHOL (the sequence-level LM
version) shows the signal is fragile and nonstationary: irreducible-loss-only scoring is
catastrophic (+85.9% steps), freezing selection after 5K steps erases the final gain, and the
downstream win shrinks to −4.3% final steps with GLUE-finetuning parity (S-RHOL Tables 1/5/6).
RHO-1 is the family's headline (+23.4pp GSM8K at 1B, +24.0pp at 7B over same-corpus CLM, Table 1),
and its own ablation bounds it: a self-referential reference (no curated data) collapses the
average gain +16.5pp → +3.3pp (Table 3) — ~80% of the gain is the *curated reference distribution*
leaking through the token mask. The reference's curated 0.5B tokens are MetaMath/MAmmoTH-derived,
i.e. bootstrapped from GSM8K/MATH *train* questions. A genuine nuance in its favor: Figs 13/14 show
kept tokens concentrate on reasoning-derivation narration *within* math passages — but the authors
themselves only ever claim "closely related to mathematics," never "reasoning."

**Single-model absolute perplexity is anti-reasoning by direct task-level evidence.**
Perplexed-by-Perplexity's Table 1 breaks results into 5 categories: Symbolic Problem Solving (the
arithmetic/logic/MathQA/LogiQA bucket) is bold-on-baseline in *all four* settings — perplexity
pruning never significantly improves reasoning, and at 3B/Pile it *drops* 4.88→2.91. The entire
average gain comes from World Knowledge (15.51→18.18) and Language Understanding (28.11→33.2) at
1B/Pile. The winning criterion flips per corpus (high-ppl on Pile, medium on Dolma; medium on Pile
*loses* 0.23), there is no random-50% control, and — the standalone caution — test-set perplexity
*inverts* with downstream accuracy (baseline 7.83 ppl / 13.73 acc vs pruned 8.51 / 15.62,
Table 3). Frequency/co-occurrence signals track recall, not reasoning: GenVsMem finds TriviaQA
distributional memorization (Mem_{n=3}>0.35, rising with scale) but *none* for GSM8K, and this is
decontamination-checked (no n=8/14 n-gram overlap, p.5) and threshold-robust (γ_T ∈ {0.7,0.75,0.8},
Fig 6).

**Self-ablation (Recipe A) is empirically a NO-GO on our own models** (our experiments,
docs/RECIPE_A_SELF_ABLATION.md, 2026-07-23). Across 5–6 base models (3 families, 1B–72B) and 6
sources, the retrieval-head gap ranks copyable boilerplate (config files, parallel scripture,
reference docs) above raw DCLM; corr(gap_top, gap_rand)=0.866 and the random-head control separates
reasoning-vs-web as well or better (llama2 0.778 vs 0.722; qwen-coder 0.832 vs 0.697); 3 of 4
models rank verbal reasoning *below* web; and Qwen 7B→72B inverts per-source (GSM8K AUC
0.955→0.051, ProofWriter 0.200→0.818). Recipe C (masking reasoning-specific heads) also fails on
real DCLM (tracks explanatory/expository prose, dominated by technical documentation). The paper's
own Table 6 corroborates the fragility: its HumanEval sanity check is a *null* (retrieval-masked
0.1098 ≈ random-masked 0.1159), so the retrieval→reasoning link is task-dependent even in-paper.

**What remains: multi-model rank-match (Recipe B) is the only untested candidate — but the
full corpus lowers expectations.** PreSelect's Tables 12/13 show the flagship rank-match
signal is knowledge/exam-shaped at task level: it is *worse than random* on HellaSwag (38.9 vs
40.0) and PIQA (67.7 vs 69.2) at 1B, and on PIQA (74.2 vs 75.2) and SIQA (33.8 vs 35.8) at 3B; the
full-15-task DCLM margin is +1.5 (1B) shrinking to +0.8 (3B), MMLU is at chance, and the 7-task
headline (+5.3) flatters by excluding the flat/negative tasks. Perplexity Correlations adds the
mechanism warning: domain-side PCA of its 90-model loss matrix has PC1=language and PC2=difficulty
(Fig 10), top ARC-Easy domains are optometry/children's-hospital sites, plain mean loss predicts
model rank nearly as well (7/8, aggregate p=0.035), and the signal evaporates on pre-filtered pools
— our exact regime. rBridge supplies the one constructive lesson: essentially all of its predictive
gain comes from evaluating the proxy's NLL *on frontier reasoning traces* rather than bare answers
(R^φ alone R²=0.867 vs full 0.874; standard NLL 0.485), teacher-robust across three frontier models
— *what* text the loss is computed on matters more than the model-differencing scheme (dataset-level
only, untested on filtered pools). Signal-in-the-Steps pushes the same direction at trace
granularity: global whole-sequence log-prob reaches *lower* training loss but 63.7% test avg vs
71.9% for a local step-score (Fig 6), and puts 42.3% of its mass on discourse filler vs 18.7%
(Table 4) — a repeated *wrong* answer inflates from 4.9%→97.7%.

**Genuine contradictions (stated as contradictions, not resolved).**
- *Which tail of the loss distribution is valuable is corpus-dependent, and the papers disagree.*
  Perplexed-by-Perplexity keeps HIGH-perplexity docs on the Pile (MEDIUM on Dolma, wrong choice
  goes negative), while the quality-filter members (KenLM ensemble, ScalingFilter) reward
  LOW-perplexity clean text and PreSelect shows the magnitude gap picks easy/short text. No stable
  directional rule exists; each optimum is tuned per corpus on the reported metric.
- *Model axis vs data axis.* Compression-Intelligence-Linear shows per-char loss tracks a *model's*
  reasoning ability almost perfectly (ρ=−0.953 on math, across 31 models), yet every data-side
  result here shows loss-based *document* selection fails to find — or de-selects — reasoning
  (Perplexed per-category; PreSelect commonsense regressions; PerpCorr language/difficulty PCA).
  Loss is an excellent ability readout and a poor content detector; no paper resolves this.
- *AttentionInfluence Table 6 vs our Recipe A.* The paper shows masking retrieval heads collapses
  reasoning *tasks* (GSM8K 0.182→0.007 vs random-mask 0.127), while our *document-level* scoring
  shows the gap is not retrieval-head-specific (corr 0.866) and inverts across scale. Both can be
  true; they disagree on whether the gap *isolates* reasoning — and the paper's own HumanEval null
  already shows the task-level link is not uniform.

**Thread implications** (papers' claims vs our inference tagged):
- If Recipe B is run on the Qwen ladder, run it with mandatory per-task-family readouts *including
  regressions* — the PreSelect tables predict it surfaces knowledge/exam-value text and may
  de-select commonsense/verbal reasoning [our inference from PreSelect's data]. Define the ability
  order by a reasoning benchmark (PreSelect A.7.2 shows steerability, at cost to other domains) and
  include a plain mean-BPB baseline, since PerpCorr concedes mean loss carries nearly the same
  signal [using it as our control is our inference].
- *What the loss is evaluated ON beats the differencing scheme.* rBridge's verified trace-target
  ablation motivates scoring documents by proxy loss on reasoning-trace-*like* targets rather than
  the raw document [extending dataset-level → document-level is our untested inference].
- The RHO-1-derived idea stays live: train a same-size reference on *complete-reasoning* text and
  point token-level excess loss at completeness [the idea is ours; the mechanism evidence is
  RHO-1's]. Keep the reference same-size (Appendix I: a weak RM adds +0.9 AVG) and budget a
  breadth-preservation control (Fig 6c: unselected-token loss rises 2.9→3.7).
- *Never validate a data intervention in this thread by held-out loss/perplexity.* Three
  independent verified results dissociate loss from downstream value (Perplexed's test-ppl
  inversion; Steps' lower-train-loss-worse-test; S-RHOL's ppl wins with GLUE parity). Downstream
  reasoning accuracy is the only valid readout [our synthesis of the papers' individual findings].
- Any static offline gap score inherits a nonstationarity risk (S-RHOL: frozen selection erases the
  gain; RHO-LOSS/DoReMi are online/minimax by design) — a one-shot corpus ranking is structurally
  different from what this family validated [our inference].
- Completeness augmentation (H2.5/H2.6) cannot be found or validated by frequency/co-occurrence
  proxies (GenVsMem, decontaminated) — those interventions need behavioral evals, not
  distributional-overlap metrics [paper's finding; the application is our inference].
---

## What this means for our thread

1. **The diagnosis is saturated; the marginal-value experiment is the cure [our inference].** Twenty-eight papers
   converge on "the shortcut is real" through four independent methods, and essentially none runs the data-side
   intervention. More diagnostic replication buys nothing; the un-run experiment is ours: reasoning-augmentation of
   natural text with a controlled completeness/structure dose.
2. **Target emitted-chain competence, not latent one-pass composition.** Emission rescues robustly at every scale
   (7.6→92.8; instructed silent thinking does not, 6.1%); latent installation is exposure-bound, data-exponential
   in depth, and capacity-punished. Every augmentation experiment we run must therefore include a
   **scratchpad-permitting eval cell** — judging explicit-chain data on a no-scratchpad eval reproduces Exposure's
   format confound in our own tables [our inference].
3. **Evaluate through a post-training stage.** The payoff of reasoning-in-pretraining can be invisible at the base
   stage and unlocked by SFT/RL (Front-Loading's latent-quality effect; ρ≈0.99 pretraining-loss→post-RL scaling;
   Cognitive-Behaviors' rewrite→RL unlock). Matched SFT + small-GRPO heads on every arm [our inference].
4. **Dose design [our inference, from the papers' constraints]:** vary chain *granularity/structure* at fixed token
   budget, not length; sweep mixture fraction (long-CoT data hurts ≤3B students — an interior optimum is likely at
   our scale); keep chains computation-ordered and left-to-right-derivable (no answer leakage); mix order-perturbed
   variants (forward-order dependence is itself a shortcut); expect an SFT-side granularity floor.
5. **Mandatory controls for any augmentation win [our inference, each anchored in a paper]:** deletion-only cleanup
   (ProX), faithful-rephrase (RePro), token-position-matched filler (Dot-by-Dot), exposure-matched entities
   (TwoHopCurse), and a compute-matched latent-depth baseline (pause/concept tokens). Cheap teacher-free arms worth
   adding: same-document co-presence of composable facts, and document packing/arrangement.
6. **Selection: run recipe B (Qwen-ladder rank-match) with lowered expectations and per-task-family readouts
   including regressions** — the family pattern is hard-task gains bought with commonsense losses. Two live ideas:
   a reference model trained on *complete-reasoning* text pointing token-level excess loss at completeness (same-size
   reference; the mechanism is RHO-1's, the idea is ours), and scoring proxy loss *on reasoning-trace-like targets*
   rather than raw documents (rBridge's verified ablation: the trace target was ~all the gain).
7. **Eval hygiene:** never validate a data intervention by held-out loss/perplexity (three independent
   dissociations); avoid Qwen-family bases scored on MATH-500/AMC/AIME as primary evidence (contamination);
   include a shortcut-reliance probe (cue-switch/perturbation) as a success criterion, since accuracy can improve
   while the Won't persists.

---

## Paper metadata — citations and venues (core anchors)

*Citation counts via Semantic Scholar API (2026-07-22) for the review's anchor papers; the remaining papers carry
venue/year inline in their entries. **2026 preprints show 0 — too new to accrue citations, NOT a quality signal**;
weigh recency + topical fit alongside count.*

| Paper (id) | First author (institution) | Last author (institution) | Venue | Cites |
|---|---|---|---|---:|
| Bag of Heuristics (`2410.21272`) | Yaniv Nikankin (Technion) | Yonatan Belinkov (Technion) | ICLR 2025 | 105 |
| Arithmetic Procedural Execution (`2605.00817`) | Sailesh Panda (IIT Gandhinagar) | Mayank Singh (IIT Gandhinagar) | preprint 2026 (v3) | 0† |
| GSM-Symbolic (`2410.05229`) | Iman Mirzadeh (Apple) | Mehrdad Farajtabar (Apple) | ICLR 2025 | 591 |
| Latent Multi-Hop / Yang (`2402.16837`) | Sohee Yang (UCL / DeepMind) | Sebastian Riedel (UCL / DeepMind) | ACL 2024 | 207 |
| Hopping Too Late (`2406.12775`) | Eden Biran (Tel Aviv U.) | Amir Globerson (Tel Aviv U. / Google) | EMNLP 2024 | 97 |
| Grokked Transformers (`2405.15071`) | Boshi Wang (Ohio State) | Huan Sun (Ohio State) | NeurIPS 2024 | 90 |
| SOCRATES (`2411.16679`) | Sohee Yang (UCL / DeepMind) | Mor Geva (Google Research / Tel Aviv) | ACL 2025 Findings | 33 |
| k-hop needs data / Yao (`2505.17923`) | Yuekun Yao (Saarland U.) | Alexander Koller (Saarland U.) | EMNLP 2025 | 10 |
| Front-Loading Reasoning (`2510.03264`) | Syeda Nahida Akter (CMU / NVIDIA) | Bryan Catanzaro (NVIDIA) | preprint 2025 | 23 |
| Yue RL-beyond-base (`2504.13837`) | Yang Yue (Tsinghua) | Gao Huang (Tsinghua) | preprint 2025 | 924 |
| ProRL (`2505.24864`) | Mingjie Liu (NVIDIA) | Yi Dong (NVIDIA) | preprint 2025 | 156 |
| RLVR Boundary debate (`2510.04028`) | Xinhao Yao (Renmin U. / Ant) | Yong Liu (Renmin U.) | preprint 2025 | 10 |
| AttentionInfluence (`2505.07293`) | Kai Hua (ByteDance Seed) | Ke Shen (ByteDance Seed) | preprint 2025 | 5 |
| PreSelect (`2503.00808`) | Kashun Shum (HKUST) | Junxian He (HKUST) | ICML 2025 | 20 |
| AutoDS / AutoMathText (`2402.07625`) | Yifan Zhang (Tsinghua) | Andrew C. Yao (Tsinghua) | ACL 2025 Findings | 26 |
| FineWeb-Edu (`2406.17557`) | Guilherme Penedo (HuggingFace) | Thomas Wolf (HuggingFace) | NeurIPS 2024 | 1029 |
| **Exposure** (`2606.09338`) | Yannis Karmim (Inria) | Valentin Barrière (U. de Chile) | preprint 2026 | 0† |
| Faithfulness as Info Flow (`2605.24286`) | Jinghan Jia (Michigan State / Anthropic) | Eric Easley (Anthropic) | preprint 2026 | 0† |
| Enthymemes (`2603.06114`) | Xuyao Feng (UCL) | Anthony Hunter (UCL) | preprint 2026 | 0† |
| TPT (`2509.20186`) | Liang Wang (Microsoft Research) | Furu Wei (Microsoft Research) | preprint 2025 | 3 |
| BoLT (`2503.18866`) | Yangjun Ruan (U. of Toronto) | Tatsunori Hashimoto (Stanford) | preprint 2025 | 40 |
| Quiet-STaR (`2403.09629`) | Eric Zelikman (Stanford) | Noah D. Goodman (Stanford) | COLM 2024 | 319 |
| RHO-1 (`2404.07965`) | Zhenghao Lin (Xiamen U.) | Weizhu Chen (Microsoft) | NeurIPS 2024 | 126 |
| Perplexity Correlations (`2409.05816`) | Tristan Thrush (Stanford) | Tatsunori Hashimoto (Stanford) | ICLR 2025 | 54 |

† 2026 preprint — too new to have accrued citations.

---

# The papers, bucket by bucket

*Full writeups for the papers this thread would cite or build on; compact entries (claim → verified numbers →
fine print) for the rest. Every entry's numbers carry their table/figure source. Three papers that anchor two
buckets are written up once and cross-referenced (Exposure → H2.6; AttentionInfluence and PreSelect → H2.4).*
## The papers · H1.1 — models take shortcuts instead of reasoning

*Each full entry ends with the eval-methodology fine print — the confounds and missing controls that bound what the
paper can prove. Read the fine print before citing the headline. Citation counts are shown where a verified count is
available; otherwise venue and year stand in.*

---

### 📖 Arithmetic Without Algorithms: LLMs Solve Math With a "Bag of Heuristics"
Yaniv Nikankin (Technion) … Yonatan Belinkov (Technion) · ICLR 2025 · **105 citations** · `2410.21272`

**What it is.** A mechanistic "how does the model actually do it?" study of mental arithmetic (e.g. `36 + 59 =`
answered in one shot, no scratch-work): when a model gets arithmetic right, is it running an algorithm, memorizing
answer tables, or something else?

**What they did.** They open up Llama-3-8B (plus Pythia and GPT-J) and trace which neurons drive the answer. They
find a small set of neurons, each firing on a simple pattern — one on operands in a range, one on operands ending in
the same digit, one on multiples of some number — and the model just *adds up* these rules' votes. No carrying, no
place value. They verify the circuit causally, then replay the entire pretraining history (Pythia checkpoints) to see
when the mechanism forms, and check the 70B model.

**What they found.** The circuit accounts for arithmetic behavior at faithfulness **0.96 avg** (per-operator
0.97/0.98/0.90/0.96, Table 1 p14); deleting the problem-relevant neurons drops accuracy **−29pp** (Figs 7-8). The
"bag of heuristics" is present almost from the start of training and is **never replaced by a real algorithm** — the
same heuristics explain **~79%** of the model's arithmetic contribution at *every* checkpoint (Fig 10), and 70B is
still heuristic (App I). When arithmetic *does* fail, the cause is **insufficient summed logit contribution** from
the firing heuristics — the paper explicitly tests and **rejects** the out-of-coverage explanation (Sec 4.3/Fig 9).

**The fine print.** The random-ablation control ablates randomly chosen neurons *from other heuristics* (Fig 8), so
it is substantially matched for general importance; the valid residuals are a rank/selection asymmetry (targeted
ablation picks the highest-classification-score neurons vs a random draw) and the **absent non-arithmetic control**
task. The 0.96 faithfulness is in-distribution (same operand regime; no held-out large-operand test). The
developmental "mutual heuristics" are *defined* by the final checkpoint, so any early mechanism outside the taxonomy
is invisible — and **~9% of top neurons match no defined heuristic** (Table 8), which quantifies that survivorship
caveat. Scope is multi-digit-tokenization models.

**Why it matters here.** A clean, mechanistic example of the **Won't**: the model reasons by shortcut, the shortcut
is laid down early in pretraining, and continued training doesn't dislodge it. The sobering note — the authors think
fixing this "may require fundamental changes to training and architectures," a caution that simply feeding better
text may not dislodge an entrenched shortcut.

---

### 📖 GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs
Iman Mirzadeh … Mehrdad Farajtabar (both Apple) · ICLR 2025 · **591 citations** · `2410.05229`

**What it is.** The widely-cited fragility benchmark: take 100 GSM8K questions, turn each into a parameterized
template (names/numbers become constrained variables), sample 50 instances per template, and measure how much
performance moves when only the surface changes — plus variants that add/remove clauses (M1/P1/P2) and **GSM-NoOp**,
which inserts a relevant-*sounding* but logically inert clause.

**What they found.** (~25 open models 2B-27B + 4 closed.) Number swaps and added clauses hurt: the clean monotone
clause-decline the paper illustrates is Gemma2-9b-IT **84.4→79.1→68.1→41.8** as clauses are added (Fig 6). GSM-NoOp
is the killer — **Phi-3-mini drops −65.7pp** (83.7→18.0 vs the GSM8K-Full baseline, Fig 8a); models "blindly
subtract" the irrelevant quantity. RL-trained reasoners resist better but not fully: **o1-preview −18.6pp** on NoOp.
Whether 8-shot in-context full-reasoning exemplars (NoOp-Symb) fix it is **model-dependent** — some weaker models do
significantly *better* on NoOp-Symb (Fig 8c).

**The fine print.** The arithmetic-difficulty confound is **ruled out** (96-99% arithmetic accuracy across variants,
App A.6, Table 2); the depth-vs-length confound (clause-scaling entangles reasoning depth with prompt length) stands.
The NoOp failure is equally well read as a learned Gricean prior ("every stated quantity is relevant" — true of all
training word problems), i.e. a *distributional* shortcut rather than missing capacity. And the paper **does** run a
training intervention: App A.4 (Fig 11) fine-tunes Phi-3.5 on 10,000 GSM-P1 examples — it improves P1 but **not** the
harder P2, and in-context P1 shots don't beat GSM8K shots on P2, with the authors concluding "scaling training data
will not be helpful in improving the reasoning capabilities." (Appendix B is a substantive rebuttal to the
statistics critique, not a concession.)

**Why it matters here.** Strong behavioral evidence for the shortcut premise — real, and RL shrinks but does not
erase it. For augmentation it cuts *against* the naive read: the paper's own same-difficulty fine-tune transfers to
P1 but not P2, so completeness-augmentation gains are likely bounded by the difficulty coverage of the added data.
The distributional-shortcut reading suggests the fix may be training-distribution *coverage* (include
distractor-bearing/NoOp-style problems) as much as reasoning completeness.

---

### 📖 The Pitfalls of Next-Token Prediction
Gregor Bachmann, Vaishnavh Nagarajan · ICML 2024 · `2403.06963`

**What it is.** A controlled synthetic study of the next-token-prediction *objective* — separating two phases usually
conflated: teacher-forced TRAINING (the model is fed the ground-truth prefix while predicting the next token) vs
autoregressive INFERENCE. The claim: teacher forcing can fail to learn an accurate predictor **in-distribution**,
independent of the usual inference-time error-snowballing story.

**What they did.** Testbed = the "path-star" graph G_{d,l}: a center node with d radiating paths; input = adjacency
list + start + goal; target = the correct path. Because the ground-truth path is teacher-forced, predicting any node
after the first collapses to a trivial "Clever-Hans" lookup (scan the adjacency list for the edge starting at the
previous node, emit the other endpoint) — a cheat that perfectly fits later tokens and thereby starves the gradient
for the only hard token, the first one (which requires lookahead). They train GPT-Mini and Mamba from scratch and a
pretrained GPT-2 Large to perfect train accuracy (to 500 epochs, ruling out grokking), and test two remedies: path
**reversal** (right-to-left, turning every token into a trivial lookup) and **teacherless** dummy-token training.

**What they found.** Teacher-forced autoregressive accuracy is **limited to chance ~1/d** (Table 2: d=2→48-50%,
d=5→19-20%, d=10→8-10%). The isolated cheat accuracy (Acc_cheat) is **96-100%**, except G₂₀,₅ where all models fail
to even fit (=0.0, Table 1). The decisive control: reduce the task to predicting *only* the first token (cheat
removed by construction) and it stays at chance **even with dedicated supervision** (Table 5: 50.2/50.4/18.9/10.4/4.5;
Prop 3: exponential time Ω(Cˡ)). So the hard core is a genuine **Can't**, and the cheat's damage is *reducing the
whole task to that core*, not making an easy token hard. The cheat is formally distinct from classic shortcut
learning (Remark 5: an answer-prefix→answer-rest correlation arising *only* under teacher forcing, causing
*in-distribution* failure) and reproduced on **Mamba**. On a second task — 3-digit addition (§F.4) — standard NTP
eventually succeeds and the cheat costs only sample-efficiency, so entrenchment-to-chance is a *regime* statement.

**The fine print.** Everything is synthetic (path-star + one addition task); there is **no natural-language
experiment**, so "NTP learns such shortcuts in real pretraining" is explicitly speculation. Reversal "works" by
removing the lookahead requirement, not by teaching planning. The obvious un-run experiment — and this thread's exact
lever — is a **derivation-first scratchpad** that makes the hard first-token decision explicit as prior tokens in a
left-to-right-derivable order. The paper does frame the ordering condition (Remark 4/§H: CoT-*before*-target is the
positive Wies/Malach regime; teacherless models can exploit even hindsight CoT), but never runs the add-derivation
condition.

**Why it matters here.** The single most important caveat for augmentation: naively dumping a *complete* reasoning
chain as teacher-forced targets can **entrench** the shortcut whenever intermediate steps leak the answer in
hindsight. Shortcut removal depends on token **order** and on whether inserted reasoning is left-to-right derivable —
not merely on adding more reasoning text.

---

### 📖 Implicit Reasoning in Transformers is Reasoning through Shortcuts
Tianhe Lin … Deqing Yang (Fudan University) · ACL 2025 Findings · `2503.07604`

**What it is.** The most directly thread-relevant paper: a from-scratch controlled study of whether the *order* in
which a reasoning chain is written decides whether a model learns genuine step-by-step reasoning or a shortcut.

**What they did.** Train a 12-layer GPT-2 (RoPE) from scratch on synthetic sequential modular arithmetic (mod 23, so
every number is one token — "reasoning, not calculation"), where each step references a variable from the previous
step. Two regimes differ *only* in arrangement: **fixed** premise order (premises listed in computation order, a
coherent chain) vs **unfixed/shuffled** order. Test templates are filtered so post-first-step calculations never
overlap training (blocks intermediate-result memorization). A "variable-as-subtrahend" manipulation removes the
commutativity that makes left-to-right number-chaining work. They add activation-patching, model-size sweeps to
GPT2-XL/1.5B and Qwen2.5-1.5B, data sweeps to 500k templates, and a zero-shot no-CoT probe of GPT-4o / Claude-3.5 /
Llama-3-70B / Qwen2.5-72B.

**What they found.** The **fixed-order** model does genuine, OOD-robust implicit reasoning: **100% ID (5-step), 99%
(+1 step), ~90% (+2 steps)** (Fig 2), staying robust as variable-as-subtrahends rise. The **shuffled-order** model
learns number-chaining that **collapses 0.92→0.20→0.04→0.05→0.03** across 0→4 subtrahend variables at 5 steps
(Table 2). The discriminating control (Fig 6): the fixed-order model's slope stays flat while the shortcut model's
drops to ~0 — separating real reasoning from chaining. Scaling does *not* fix it: GPT2-XL, Qwen2.5-1.5B, and
50k/500k-template datasets still shortcut (Tables 5-7). SoTA LLMs fall **~100→~30** on the same probe as the
subtrahend-variable ratio goes 0/2→2/2, replicated in natural-language phrasing (Fig 7; Table 8: GPT-4o
0.94/0.47/0.28).

**The fine print.** The lever demonstrated is chain **order/adjacency**, not making implicit premises explicit — a
narrower notion of completeness. The reorder-as-*intervention* experiment (take shuffled data, reorder it, show the
shortcut is removed) is **never run** — the paper stops at diagnosis. The SoTA-LLM probe forbids a scratchpad and
scores **only** outputs that are not in CoT form, and that no-CoT filter is a code-confirmed silent try/except drop
with **unlogged per-condition discard rates** (if models emit CoT precisely on the hard items, the scored subset is
biased); the LLM "shuffled" condition is a fixed [3,1,2] reorder, and the probe is 3-step-capped (4-step "too hard").

**Why it matters here.** Direct corroboration of the shortcut mechanism *and* of the ordering/coherence lever: when
the training text lays the chain out in complete, computation-order form, the shortcut cannot form and genuine
generalizing reasoning emerges. But it complicates the naive version — SoTA LLMs that have ingested enormous
reasoning text still shortcut here, and neither scale nor data volume fixes it, so the fix must *remove
shortcut-availability*, not just increase reasoning-text volume.

---

### 📖 How Reinforcement Learning After Next-Token Prediction Facilitates Learning
Nikolaos Tsilivis … Julia Kempe (NYU / FAIR-Meta / Harvard) · 2025 preprint · `2510.11495`

**What it is.** A rigorous separation theorem for the *fraction* of complete reasoning chains in a corpus, plus the
cleanest dose-response in the bucket, showing when next-token pretraining alone removes a truncation shortcut and
when RL is needed.

**What they did.** Learn parity-of-d-bits from a mixture: with probability **p_cot** a sequence is a LONG
chain-of-thought showing running partial parities; with probability 1−p_cot a SHORT sequence whose first token is the
answer. Recipe = NTP pretraining then RL (STaR / REINFORCE / GRPO) with a correctness reward. Empirics: GPT2/Mistral
from scratch on parity (d≤50) and n-digit multiplication; Llama-3.2-3B/3.1-8B on short/long GSM8K and MATH mixtures.
Theory: autoregressive linear predictors proving a length-calibration + greedy-failure result for pretraining and a
length-increase + perfect-generalization result for STaR.

**What they found.** A sharp threshold at **p_cot = 1/3** (Thm 1, Fig 2): below it, greedy decoding emits the SHORT
truncated (chance ~50%) response despite ~**10⁷** samples; above it, pretraining alone generalizes. Crucially the
correct long route *is* learned — emitted with probability p_cot under temperature-1 sampling (length calibration) —
but is not the *modal/greedy* behavior. RL fixes this in only **O(log((1−p_cot)/p_cot))** rounds by up-sampling the
rare long correct demonstrations (Thm 2/5), with response length growing 1→50 (Fig 3). Granularity interacts with
proportion: a **partial (leap-2) chain** that omits steps lets pretraining generalize even *below* 1/3 (Fig 15).

**The fine print.** Parity is computationally **shallow** — the failure is purely statistical (an *estimation*, not
*approximation*, limit; a shallow transformer can represent parity), so it isolates sample-complexity and does **not**
speak to tasks where representation/depth is the bottleneck — indeed the hardest multiplication case (7-digit,
p_cot=0.1) fails even after 38 GPU-hours. The RL win is entirely parasitic on the pretrained length-calibrated
checkpoint (STaR reweights the model's *own* long generations); it says nothing about tasks where the long route was
never learned. The GSM8K/MATH real-LLM evidence is confounded by prior Llama finetuning/contamination (the authors
flag it).

**Why it matters here.** The one result giving a *quantitative* handle on completeness: the fraction of complete
chains is a lever with a sharp threshold, and partial chains lower the required fraction. It also shows the shortcut
can be removed at *post-training* by reweighting what pretraining already installed — so under-reasoning does not
persist through RL *in this setting* — which both supports the data-side cure and names an objective-side competitor
to it.

---

### 📖 Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens
Chengshuai Zhao … Huan Liu (Arizona State University) · 2026 preprint · `2508.01191`

**What it is.** A fully controllable synthetic environment (DataAlchemy) testing when CoT generalizes — abstracting
text to letter "elements" and reasoning to composable string transformations (ROT-13, cyclic shift), with CoT the
literal step-by-step decomposition.

**What they did.** Train decoder-only models from scratch (62K-3B params) so there is no leakage, then probe under
three distribution shifts — task (novel transformation/composition), length, and format perturbation — scoring the
reasoning trace, answer, and full chain separately. Non-commutative transforms rule out a commutativity artifact.
External validity is checked by fine-tuning LLaMA3-8B and Qwen3-14B on the *same synthetic tasks*.

**What they found.** Complete explicit chains *present in training* are learned as distributional **templates**:
full-chain exact-match **100% (ID) → 0.01% (novel composition of seen primitives) → 0% (out-of-distribution)**
(Table 1). The model emits **100%-correct-looking reasoning traces attached to the wrong answer** by copying the
nearest training path (Table 2); non-commutative f3 collapses reasoning, answer, and full-chain simultaneously to 0%
(Table 3). It rigidly reproduces trained step-count — a model trained on 2-step chains *pads* a 1-step problem into a
spurious 2-step trace (E.2.2). Scale 62K-3B "accelerates interpolation within the training distribution rather than
enabling extrapolation" (Figs 14-15). A tiny SFT fraction (λ≈1.5e-4) patches locally but "simply expands the
in-distribution bubble slightly."

**The fine print.** The environment is deterministic symbolic string-rewriting — algorithm execution with **zero
natural-language semantics** — so "CoT is a mirage" is proven for a task class unlike world-knowledge inference.
Worse, the "external validity" experiments **fine-tune** LLaMA3-8B/Qwen3 on the same synthetic tasks, so external
validity is really intra-synthetic; NL CoT is never tested.

**Why it matters here.** The strongest warning that **completeness/explicitness of individual chains is not the
lever** — the complete chains are already in the data and still learned as a rigid template. The binding constraint
is **distributional coverage/diversity** of task, length, and reasoning-depth patterns; uniformly-shaped complete
chains risk teaching a template, and augmentation buys interpolation, not extrapolation.

---

### 📖 Faith and Fate: Limits of Transformers on Compositionality
Nouha Dziri … Yejin Choi (Allen Institute for AI / University of Washington) · NeurIPS 2023 · `2305.18654`

**What it is.** A study of whether transformers acquire a *systematic* multi-step skill (multi-digit multiplication,
dynamic programming, logic puzzles) or approximate it with pattern-matching — decomposing each task into a full
computation graph and measuring where the graph breaks.

**What they did.** Zero-shot and few-shot frontier models, plus fine-tuning GPT-3 with **complete scratchpads** that
verbalize every carry and partial product. They compare answer-only vs scratchpad training, test extrapolation to
unseen problem sizes, run a grokking (extended-training) probe, train **GPT2-XL from scratch** on 90M multiplication
examples with per-digit tokenization, and analyze surface-pattern prediction.

**What they found.** Complete scratchpad finetuning reaches **near-perfect in-distribution but zero unseen-depth
extrapolation** (Fig 3); zero-shot GPT-4 falls 3×3 0.59 → 4×4 0.03 → 5×5 0 (Fig 2a). **82.3%** of correct answers on
unseen sizes still contain a computation-graph error (few-shot GPT-4 + finetuned-scratchpad GPT-3) — right answer,
wrong process. Models predict surface features (first digit, trailing zeros, #digits) at **~0.98-1.0** while
full-answer accuracy is near zero (App C, Figs 26-29). The from-scratch per-digit GPT2-XL on 90M examples **still
fails 3×3** (App B.3) — ruling out tokenization and pretraining-contamination explanations.

**The fine print.** The scratchpad-vs-answer arms are **not compute-matched**, but the real confound is
*token-length*, not epochs: on multiplication the scratchpad arm got *more* epochs (16 vs 14, App B.1); the genuine
imbalance is ~250 vs ~20 tokens/example (~12.5×), and exhaustively finetuning 5×5 with scratchpads would cost **$700M**
(Table 1 p28) — the concrete wall behind "completeness helps in-distribution but was never pushed to larger
complexity." Inference used nucleus sampling at T=1 over 500 examples, which understates greedy best-case.

**Why it matters here.** The sharpest limit on completeness: maximally complete traces buy near-perfect
*interpolation* and **zero extrapolation past trained depth** — so a completeness intervention's gains are bounded by
the depth coverage of the augmented chains. It also shows the failure is not tokenization or contamination, which
strengthens the "systematic skill does not emerge from MLE" reading.

---

### 📖 The Pitfalls of Simplicity Bias in Neural Networks
Harshay Shah … Praneeth Netrapalli (Microsoft Research / Stanford) · NeurIPS 2020 · `2006.07710`

**What it is.** The canonical demonstration that gradient-trained nets satisfy the objective via the *simplest*
sufficient feature — with the cleanest can't-vs-won't control anywhere.

**What they did.** Build synthetic datasets with a precise, tunable notion of feature simplicity (minimum number of
linear pieces needed to classify on a coordinate). Measure reliance by randomizing a feature block independently of
the label: S-randomized AUC ≈0.5 with Sᶜ-randomized AUC 1.0 means the model depends *exclusively* on the simple set
and is *invariant* to the complex set. They also prove a weight-magnitude theorem and test MNIST-CIFAR concatenations.

**What they found.** Nets rely **exclusively** on the simplest feature and remain **completely invariant** to complex
predictive features — S-randomized AUC ≈0.50 vs Sᶜ 1.00 (Table 4) — even with a **249:1** majority of complex
fully-predictive features (Fig 3b), and even when the simple feature is only **95%-predictive vs 100%-predictive**
complex ones (L̂MS-7, Table 2), costing ~5% generalization. The decisive control: **remove S and the same net learns
the complex features to 100%** with the same budget (p.7) — the complex features are learnable; the net just refuses
them when a shortcut is present. Theorem 1: gradient descent inflates the simple-coordinate weight Ω̃(√d) larger.

**The fine print.** The extreme claims and Theorem 1 hold on synthetic axis-aligned feature-block data; "simplicity =
number of linear pieces" is a narrow operationalization whose correspondence to real simplicity is conjectured, and
MNIST-CIFAR is a stylized concatenation. Every mitigation tested is **model-side** (ensembles, adversarial training,
dropout, ℓ2, optimizer, activation, Tables 6-7) and **all fail**; the only lever that helped is non-random
initialization (train on the complex feature first). No **data-content** intervention is tested.

**Why it matters here.** The mechanistic core of the Won't, and the sharpest warning for augmentation: the L̂MS-7
result predicts that merely **co-presenting** complete reasoning alongside shortcut-sufficient text will be ignored
unless the augmentation also *degrades the shortcut's sufficiency*. The one lever that helped — curriculum/ordering,
not co-presentation — is an argument for how to stage reasoning data, not just whether to add it.

---

### Compact entries

### Optimization, feature-level, and architectural mechanism

**📖 Gradient Starvation: A Learning Proclivity in Neural Networks (`2011.09468`)** — Pezeshki … Lajoie (Mila) ·
NeurIPS 2021. The dynamical account of the Won't: a fast-learned feature that zeroes the loss *starves* the gradient
for slower but predictive features (Thm 2). The "friend-or-foe" corollary is directly load-bearing for this thread —
in-distribution loss/perplexity cannot reveal a starved feature, so a perplexity signal structurally cannot find
missing reasoning. Fix is demonstrated only optimizer-side (a spectral-decoupling regularizer), not via data.

**📖 Masked Language Modeling and the Distributional Hypothesis (`2104.06644`)** — Sinha … Kiela (FAIR / McGill-Mila)
· EMNLP 2021. Pretraining on **shuffled word order** costs only ~3.3 GLUE points, yet non-parametric probes show
genuine syntactic structure is absent — the benchmark-satisfying shortcut coexists with a real capability hole.
Corpus-scale evidence that benchmarks can't see what pretraining failed to install, and (§4.2) that some capabilities
can be back-filled cheaply at fine-tuning time — a baseline worth running for our completeness evals.

**📖 Shortcut Learning in Deep Neural Networks (`2004.07780`)** — Geirhos … Wichmann · Nature Machine Intelligence
2020. The canonical framing: shortcut learning = least-effort solutions on under-determined data. Curated anecdotes,
no base rates; motivates data-level fixes but tests none. Useful as the vocabulary anchor (least-effort +
data-underdetermination) for the whole bucket, not as evidence.

**📖 Transformers Learn Shortcuts to Automata (`2210.10749`)** — B. Liu … C. Zhang (CMU / MSR-NYC / UPenn) · ICLR
2023. The architecture-side version: transformers converge to *parallel* shortcut solutions that are exactly correct
in-distribution but brittle OOD/at-length (Figs 6-7); scratchpad training restores the generalizing recurrent
solution **only when coupled with a recency bias**; non-solvable automata are a proven constant-depth **Can't**
(Thm 4). Bounds how far a scratchpad/ordering fix can travel.

### Synthetic-objective mechanism; knowledge stored vs manipulable

**📖 Reasoning Bias of Next Token Prediction Training (`2502.02007`)** — Lin, Zhang, Xu (SJTU) · 2025 preprint.
Full-sequence NTP loss beats answer-only (CTP/SFT-style) loss for finding a *generalizing* reasoning solution from
scratch — CTP's reasoning-solution rate ≈0 at all depths (Fig 5). Fine print: the result reverses when starting from
a pretrained init, and the paper's "noise" framing conflates premise supervision with genuine noise.

**📖 The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A" (`2309.12288`)** — Berglund … Evans · ICLR
2024. The limiting case of implicit inference: after "A is B" training, **p(A|B) equals random-name likelihood**
across 350M-175B (Tables 3/6), and nothing transfers — paraphrases, 40k extra docs, both-order patterns — while
in-context reversal is **100%** (Table 5). One implied hop is never written into the weights: explicitness must be
materialized **per-instance** in the data. Directly grounds design constraint (f).

**📖 How Does Unfaithful Reasoning Emerge from Autoregressive Training? (`2602.01017`)** — Wang, Alazali, Zhong
(Wisconsin-Madison) · 2026 preprint. Clean coherent chains *do* induce causally faithful stepwise computation
(intervention non-response ≈0 at zero noise), but the model flips to **skip-step** above a critical noise threshold
(Fig 2); **fine-grained steps buy ~10× noise tolerance** (Fig 3) and prolonged training itself erodes faithfulness.
Grounds constraints (c) low-noise and (d) fine granularity — a direct warning for LLM-generated augmentation at
scale. Fine print: synthetic, small (~5-layer) models.

**📖 Physics of Language Models 3.2: Knowledge Manipulation (`2309.14402`)** — Allen-Zhu, Li · 2023 preprint (v2;
assoc. ICLR 2025). Knowledge *retrieval* ~96.6% yet *manipulation* stays near-chance without a test-time scratchpad;
training WITH CoT and testing WITHOUT still fails; v2 scale-tests it (Mistral 5.5× model / 50× data / 2.5M samples
still can't compare 100 majors, Fig 13). But **binary parity is direct-learnable** (50.4%→95.3% by 50k, Fig 11) — so
"needs CoT" is **task-cardinality-dependent**, a genuine Can't at high cardinality rather than everywhere.

**📖 Composition Collapse: Stable Factual Knowledge ≠ Compositional Reasoning (`2605.26789`)** — Yu … Han · 2026
preprint. Atoms are stored (atomic stability 78-90%) but not composed; longer structured CoT recovers ~70-75% of
gate-passing failures — but the standard prompt already permits a short `<reasoning>` block (1-4 sentences, ≤512
tokens, App A), so this is a short-CoT→long-CoT gain measured *between-subjects* (enabling CoT itself shifts the
gate-passing population, App S Table 20), not a no-CoT→CoT rescue and not a format-restriction artifact. Post-training
warning: **SFT on reasoning traces WORSENED** composition (76.9% vs 69.8% baseline, Fig 4) while GRPO helped only on
trained depths (OOD/ID 0.21). Fine print: the 72%-structural failure taxonomy is scoped to the E2/E3 synthetic /
cross-domain sets (251 adjudicated cases, App O), not the main D4v2 temporal-composition numbers; and the Gemini
adjudicator has recall of only **0.237** on the main causal model Qwen2.5-7B-Inst (Table 15), which would inflate its
residual-failure rate. A caution that trace-SFT is not automatically the completeness cure.

### Behavioral shortcut probes and mechanistic mediators in SoTA models

**📖 Premise Order Matters in Reasoning with LLMs (`2402.08939`)** — X. Chen … D. Zhou (Google DeepMind) · ICML 2024.
The cleanest SoTA order-probe: on **provably order-invariant** tasks, reordering logically-equivalent premises drops
GPT-4-turbo **96.5→80.8** (12 rules, Table 6a) and collapses GPT-3.5 30→~1; on R-GSM, problems solved at 100% fall to
**64.9-89.9%** under reorder (Table 2b), with fact-hallucination rising monotonically off proof order. Clean Won't; no
fix tested. Sits in genuine tension with Implicit-Shortcut (forward-order dependence is *itself* a shortcut).

**📖 A Peek into Token Bias (`2406.11050`)** — Jiang … Roth (UPenn / Argonne) · EMNLP 2024. Statistically robust
surface-token shortcuts in frontier LLMs (GPT-4o n₂₁=360 on relevant-lure conjunction items), largely
elicitation-fixable with hints/few-shot. Fine print: the flagship Linda→Bob number is **confound-killed** — a
verified answer-letter flip against an 85%-(a) test set — so cite only the H1 cells, never the Linda headline.

**📖 Do LLMs Overcome Shortcut Learning? (Shortcut Suite) (`2410.13343`)** — Yuan … Q. Liu (USTC) · EMNLP 2024.
NLI-shaped shortcuts (lexical overlap, negation, position) survive SFT/RLHF **through GPT-4 and LLaMA3-70B** (−30 to
−60pp on non-entailment sets); CoT recovers much (LLaMA2-70B 3.6→66.2). Fine print: the inverse-scaling headline is
label-prior-confounded (a neutral-biased 7B scores free points on ¬E-only sets), and there is no base-model control
for the pretraining attribution.

**📖 Investigating Multi-Hop Factual Shortcuts (`2402.11900`)** — Ju … G. Liu (SJTU / Southeast) · ACL 2024.
Pretraining **co-occurrence** installs s₁→oₙ associations that win even when every intermediate hop is demonstrably
known (~20% of knowledge-editing failures; shortcut strength tracks corpus frequency, Wikipedia-Dolma r=0.74).
Authors: the fix "must be initiated during the pre-training phase" — but they don't test it. Hedge: the ~20% figure
carries an edit-generalization confound.

**📖 A Diagnostic Study of Arithmetic Procedural Execution in Language Models (`2605.00817`, v3)** — Panda, Singh
(IIT Gandhinagar) · 2026 preprint · **0 citations** (too new). Across **15 reasoning-trained models** (55,000
evaluations, incl. Kimi-K2.5, GPT-oss-120B, DeepSeek-v3.2) handed the complete step-by-step recipe in-prompt, accuracy
falls **63%→20%** over 5→95 steps (Fig 2), with a further **−23.85pp** degradation when a step must look back to an
earlier variable (§1, Fig 3). The v3 prompt *requires* a written scratchpad ("show all intermediate steps clearly,
include intermediate variable values after each step", App B Fig 7), so the degradation is **not** a no-scratchpad
artifact — it persists even when every step must be shown, which sharpens the H1 read. Fine print: the
under-execution-as-failure metric stays contaminated — GPT-oss-120B, the highest-accuracy model (60.04%), is classified
~90% under-execution because it answers directly rather than executing the shown steps (Fig 122), so under-execution
conflates lost state with obedient direct-answering; and mult/div collapse is partly an arithmetic-magnitude confound
(median product 6.25e31, Table 4), a Can't-execute component. For the strongest models the failure leans **Can't**
(Ministral-14B / DeepSeek-v3.2 / Kimi-K2.5 keep %Exact ~90-100% while accuracy falls), not a clean Won't; models
essentially never self-correct (first-answer ≈ final-answer, Fig 14). Durable lesson: a maximally complete chain
*in-context* does not guarantee faithful long-horizon execution.

**📖 The Reasoning-Memorization Interplay Is Mediated by a Single Direction (LiReF) (`2503.23084`)** — Hong … Jin
(NYU / McGill / MPI / Toronto) · 2025 preprint. A single steerable direction mediates a reasoning-vs-memorization
mode; suppressing it barely hurts GSM8k but hurts GSM-Symbolic — a memorization/leakage signature. Fine print:
circular validation and oracle-tuned steering limit the causal claim.

**📖 Rethinking the Chain-of-Thought: ICL and Pre-trained Priors (`2509.01236`)** — H. Yang … L. Yang (Nanjing) ·
2025 preprint. Pretrained priors dominate CoT at low shot, but ~40 corrupted exemplars can flip an 8B model — a
preference, not a hard wall. Fine print: long-CoT prompting gains are confounded by teacher strength and token
budget.

**📖 Towards a Mechanistic Interpretation of Multi-Step Reasoning (`2310.14491`)** — Hou … Sachan (ETH / EPFL) ·
EMNLP 2023. A reasoning-tree signal appears in attention **only after task finetuning** (pretrained GPT-2: 0%
accuracy, near-random probe). Weak leverage on this bucket — "multi-step" is depth ≤1 and the natural-language claims
are correlational.

**📖 Leap-of-Thought: Reasoning Over Implicit Knowledge** — Talmor, Tafjord, Clark, Goldberg, Berant (AI2 / Tel-Aviv /
Bar-Ilan) · NeurIPS 2020. An SFT'd RoBERTa composes explicit + implicit knowledge (88.8 vs 65.2 hypothesis-only;
IMAGINARY control 76.9), but failures **track missing internal beliefs** (a Can't) and are fixed by *belief
injection* — an elicitation lever, not corpus augmentation. Places one failure mode firmly on the Can't side.

### Objective-side alternatives to a data-side cure

**📖 RLP: Reinforcement as a Pretraining Objective (`2510.01265`)** — Hatamizadeh … Choi (NVIDIA / CMU / Stanford) ·
ICLR 2026. Information-gain RL on self-generated thoughts *during* pretraining beats FLOP-matched continued
pretraining (42.13 vs 38.04). Fine print: no filler-thought control, so reasoning-content vs extra-compute is
unresolved. An alternative arm that reaches the goal without touching the corpus.

**📖 Reinforcement Learning on Pre-Training Data (RLPT) (`2509.19249`)** — S. Li … D. Wang (Tencent Hunyuan / CUHK) ·
2025 preprint. Next-segment-reasoning RL on pretraining text gains +3-8 pts general / +5-7 AIME. Fine print: no
compute-matched NTP control, and sentence-vs-atomic-step segmentation made no clear difference — so the mechanism is
not isolated. Second objective-side competitor to a data-side intervention.
## The papers · H1.2 — latent (one-forward-pass) multi-hop reasoning vs recall

*Each full entry ends with the eval-methodology fine print — the confounds and missing controls that bound
what the paper can prove. Read the fine print before citing the headline.*

### 📖 Do LLMs Perform Latent Multi-Hop Reasoning *Without Exploiting Shortcuts*? (SOCRATES)
Sohee Yang (UCL / Google DeepMind) … Mor Geva (Google Research / Tel Aviv University) · ACL 2025 Findings · **33 citations** · `2411.16679`

**What it is.** The most careful version of "does the model *really* reason silently, or is it cheating?"
It builds a test set designed so the model *can't* get the answer by a memorized shortcut (the start and
end entities never co-appear in any document) and only counts cases where the model provably already knows
both individual facts.

**What they did.** Built 7,232 two-hop questions, filtered out every detectable shortcut, and measured
**latent composability** (how often the model chains the two facts silently) versus **CoT composability**
(allowed to write the middle step out). Tested ~41 models. The composability denominator conditions on the
model answering both single-hop queries correctly (code-confirmed in `evaluation_utils.py`).

**What they found.** With shortcuts removed, silent composition is **terrible** — GPT-4o at **7.6%**, Claude
3.5 Sonnet at **8.4%** — *even though the model knows both facts*. Let the same model write the intermediate
step out and it jumps to **~85–93%** (GPT-4o 7.6% → **92.8%**). It's also wildly uneven by bridge type
(Fig 7 slopes: country 0.78, city 0.27, university 0.05, year 0.02). In their OLMo pretraining trace, latent
2-hop reasoning emerged for only ~11% of eligible cases.

**The fine print.** The latent-vs-CoT contrast is fair as an elicitation diagnostic (no author-side
training, so no train/test-format trap), and conditioning on 1-hop knowledge is a genuinely strong control.
v2 (May 2025) adds the control our prior read wanted: instructing internal step-by-step thinking *plus* an
explicit hint to identify the bridge leaves latent composability at **6.1%**, and **96.0% of CoT failures
are wrong-bridge generation** (App D.3) — silent deliberation does not substitute for emission, which leans
this toward a *can't-in-one-pass*. The shortcut-inflation figure is **~3× model-averaged** (Fig 4, §6.3) —
not the older "5× / Gemini Flash 2.4%" statistic, which is absent from the current paper. The
country-bridge worry (that the one high bin is under-filtered) is substantially rebutted: a 400B-document
Google-Search co-occurrence filter costs only 0.03 relative (§C.3/§D.2), and §A.3 already excludes
name-inferrable bridges. One control still unrun: put both facts in-context and still forbid CoT.

**Why it matters here.** Close to a *definition* of the thread's problem: the model has the knowledge but
does not silently run the inference, and making the middle step explicit fixes it. Strong support for
"explicit reasoning helps at answer-time," with the sharpened lesson that the rescue is specifically the
*emission* of the unstated intermediate.

**📖 Multi-Hop Knowledge Composition is Bound by Pretraining Exposure (`2606.09338`)** — the bucket's strongest
training-side result; full writeup in the H2.6 section. The H1.2-critical facts: both populations learn every
atomic fact (97% 1-hop) but only entities seen in *compositional* pretraining contexts compose (2-hop up to 0.83);
held-out entities stay at 0.00 across all nine augmentation formats — even conditioned on both 1-hop facts being
answered correctly (Table 9), at ~15.9× the Chinchilla-optimal budget (Table 7), and invariant to an identical
direct-answer QA-finetune applied to all conditions (Table 13). Latent composition is installed by exposure to the
composition itself, not by the facts.

### 📖 The Two-Hop Curse: LLMs Fail to Reason Latently Across Separately-Learned Facts
Mikita Balesni … Owain Evans · 2026 preprint · **0 citations** (too new) · `2411.16353`

**What it is.** A controlled test of whether LLMs can do latent (no-CoT) two-hop composition, built so a
positive result is unambiguous — synthetic facts rule out memorization/shortcuts.

**What they did.** ~693 entity triplets ("The spouse of e1 is e2. The birth city of e2 is e3."), each
yielding one-hop and two-hop (with/without CoT) QA, paraphrased ~30× → ~68,580 pairs. Four experiments: (1)
fully-synthetic two-hop with the two facts learned as **separate** finetuning documents; (2) mechanistic
interventions (layer-selective fact storage; bridge-entity activation supervision); (3) data-arrangement
levers (same-document co-occurrence, in-context provision); (4) a frontier-model survey. The no-CoT eval is
**forced-choice** (logits restricted to valid answers, 20-shot).

**What they found.** Two-hop no-CoT stays at **chance across all training mixtures** — including one
containing **13,500 demonstrated no-CoT two-hop QA pairs** whose explicit purpose was to incentivize the
latent circuit (App D). The failure is therefore *not* absence of direct latent supervision; direct
supervision on fresh, separately-stored facts simply doesn't generalize. The recovery levers are quantified:
**same-document co-presence ~50%**, **in-context provision (10 distractors) ~63%** (Fig 6), and in the
semi-synthetic setting no-CoT reaches **~20–22% against a CoT ceiling of only ~33%** (Fig 7) — the gap nearly
closes when one fact is deeply (pretraining-)stored, though App F/G show this is heavily category-skewed
(many categories exactly 0). Mechanism (§6.1): a bridge learned as the *output* of fact 1 cannot serve as a
*query input* to fact 2 within one pass; App E shows same-document **co-presence** (not adjacency) is the
operative lever.

**The fine print.** "CoT works" needs a hedge: two-hop CoT is **~78–80% for Llama-3-8B/GPT-4o-mini but only
~40% for Qwen 2.5 7B** (GPT-4o worse than GPT-4o-mini, footnote 1). The frontier-survey aggregate ("gap may
reduce with scale," Claude Opus 4 ~61% no-CoT) rides on a category subset that App G shows may be
co-occurrence-driven. Semi-synthetic ~20% "success" bundles storage-depth/co-occurrence confounds (the
authors' own caution); no controlled scale sweep.

**Why it matters here.** The cleanest demonstration that latent two-hop failure is a data-**arrangement**
artifact, not a hard incapacity — and it hands us a concrete, teacher-free augmentation lever
(same-document co-presence) that buys latent composition with no chain written. It also warns that apparent
latent successes can ride on the augmenting facts' pretraining co-occurrence, which is exactly the control
our completeness experiments must impose.

### 📖 Grokked Transformers are Implicit Reasoners
Boshi Wang … Huan Sun (both Ohio State University) · NeurIPS 2024 · **90 citations** · `2405.15071`

**What it is.** A controlled from-scratch study of *when* a transformer learns to reason silently versus
just memorize — training a small model on made-up facts to watch the whole learning process.

**What they did.** Train on synthetic facts with two task types — **composition** (chain two facts) and
**comparison** (is A > B) — far past the normal stopping point, watching the internal circuits evolve, and
test on held-out combinations to see whether the skill generalizes or is memorized.

**What they found.** Transformers *can* learn genuine silent reasoning, but **only through "grokking"** —
training ~50× beyond the point where the training data is already fit (>99% train accuracy via the
memorizing circuit at ~14K steps; genuine in-distribution reasoning only after ~700K steps). Before
grokking, generalization is ~9%; after, ~98%. Mechanistically: a fast **memorizing** circuit forms first,
a slow **generalizing** circuit wins later. The causal lever is the **inferred-to-atomic ratio φ**
(dose-response replicated on both tasks, App E.3; 96.7% before saturation at φ=18).

**The fine print.** The internal composition-vs-comparison contrast (same architecture/regime) is
well-controlled. The OOD-composition failure must be softened from a "hard architectural limit": ~0% OOD
holds for the **vanilla** transformer even at 2M steps, but a **parameter-sharing variant (4 layers run
twice) reaches ~72% OOD composition** by 1.5M steps (App E.2/Fig 14) — architecture-induced and convertible.
The famous side-by-side (grokked 99.3% vs GPT-4-Turbo ~33%) is **not** apples-to-apples: it pits a
from-scratch model with these facts *in weights* against frontier models seeing them once in-context, and
GPT-4-Turbo's 33.3/31.3 is at/below the 33.3% random-guess baseline (only Gemini Direct+R 37.3 is above
chance). Don't quote it as "tiny grokked transformer out-reasons GPT-4."

**Why it matters here.** A textbook demonstration of memorize-first shortcut flipping to genuine implicit
reasoning — with two thread-relevant twists: the model can internalize reasoning *with no explicit chains in
the data at all* (so explicit chains aren't strictly necessary; φ is the lever), and the OOD ceiling is a
property of the architecture, not of reasoning per se.

### 📖 Scaling Implicit Reasoning: a U-Shaped Law for Model Size
Xinyi Wang … Yikang Shen (UCSB / MIT-IBM Watson AI Lab / Rutgers) · ICML 2026 (claimed) · `2504.03635`

**What it is.** A from-scratch knowledge-graph study pricing how much model capacity latent multi-hop
reasoning costs, versus plain knowledge storage.

**What they did.** Train transformers of varying size on synthetic KGs serialized as random-ID triples,
measure best-achievable (early-stopping) loss on held-out two-hop composition, and fit capacity-vs-size
laws. Appendix G's code removes held-out test edges from the graph before serialization (an anti-memorization
control).

**What they found.** Reasoning capacity is **~0.008 bits/param** (124 params/bit, R²=0.85, Fig 4). Test loss
is **U-shaped in model size** — beyond an optimum, bigger models get *worse* at composition — while training
loss falls monotonically, and the U **strengthens with more training steps** (Fig 1). The mechanism is
over-training memorization (benign-overfitting/double-descent framing): oversized models memorize the
composable facts instead of learning to compose.

**The fine print.** The frequently-quoted "~250×-per-bit penalty vs storage" is **reviewer arithmetic**
(2 / 0.008), not a paper claim (the paper says only "very different"). The ~2 bits/param storage anchor is
Allen-Zhu & Li's, which the corpus's companion capacity paper (37, EleutherAI) **failed to reproduce,
measuring ~1.6 b/p** for one-hop storage — so the *multiplier* is not a law. Optimal size is defined by
best-achievable/early-stopping loss (Definition 1), and Theorem 3 proves the step-budget optimum converges,
so a "fixed-step, not convergence-matched" critique does not apply. Surviving caveats: the law is measured
on **semantics-free random-ID serializations** (NL settings are "noisy"), `weight_decay=0` with no
regularization sweep, no 1-hop-conditioned scoring, and random (not type-matched) MC distractors.

**Why it matters here.** The cleanest first-principles argument for **externalizing reasoning into tokens**
rather than expecting latent composition: latent multi-hop is capacity-expensive relative to storage, and
the effect is an over-training memorization signature, not an under-training artifact. Cite the *direction*
(and the ~0.008 b/p figure), never the "250×" as a constant.

### 📖 Do Large Language Models Latently Perform Multi-Hop Reasoning?
Sohee Yang (UCL / Google DeepMind) … Sebastian Riedel (UCL / Google DeepMind) · ACL 2024 · **207 citations** · `2402.16837`

**What it is.** Tests whether a model asked "the mother of the singer of 'Superstition' is ___" internally
does the two hops — figure out the singer (Stevie Wonder), then find his mother — or jumps straight to a
memorized answer.

**What they did.** Built 45,595 two-hop questions and used interpretability probes to watch, inside the
model, (a) whether it recalls the bridge entity (EntRec) and (b) whether recalling it more strongly makes
the final answer more correct. Tested LLaMA-2 at 7B, 13B, 70B.

**What they found.** The **first hop is real and improves with size** (bridge recall ~0.71 → 0.78, 7B→70B).
But the **second hop — actually *using* the recalled entity — is moderate and does NOT scale**: stuck at
**0.64 / 0.65 / 0.61**. The model has the pieces but frequently fails to connect them, and size doesn't help.

**The fine print.** The recall-without-use diagnosis is **more robust than an over-cautious read allows**:
the flat second hop replicates when RQ2 is scored by ground-truth-answer log-probability (App F.1: 0.60 /
0.62 / 0.59), is insensitive to whether the bridge-attribute fact is individually answerable (App F.2), and
both probes are validated (Appendices C/D). Residual caveats the authors themselves flag: EntRec cannot
separate genuine bridge resolution from shallow n-gram co-occurrence ("Superstition"↔"Stevie Wonder"), the
two hops use different metrics, and the probes are representational, not conditioned on the model answering
the two-hop question correctly — a representational signature, not behavioral reasoning ("lower bound", "one
pathway").

**Why it matters here.** Suggestive evidence of under-reasoning that isn't a knowledge gap: the bridge is
right there internally, unused, and scale doesn't fix the second hop — hinting the fix must come from *how
the model is trained*, consistent with a data intervention.

### 📖 Hopping Too Late: The Limitations of LLMs on Multi-Hop Queries
Eden Biran (Tel Aviv University) … Amir Globerson (Tel Aviv University / Google) · EMNLP 2024 (main) · **97 citations** · `2406.12775`

**What it is.** A follow-up that asks *why* the second hop fails, focusing on the cleanest cases: questions
where the model provably knows both facts individually but blows the combined question.

**What they did.** Traced, layer by layer, where the bridge entity gets resolved and where the second hop
happens, then ran "back-patching": take an internal state from a *later* layer, paste it into an *earlier*
layer, and let the model finish — testing whether the answer was computable if the second hop had started
sooner.

**What they found.** The bridge resolves in *early* layers, the second hop only starts in *late* layers
(MLP-promoted) — a **timing/traffic problem inside the network**. Back-patching fixes **32–66%** of
previously-wrong knowledge-held-constant cases (ceiling: Pythia 6.9B t1 = 66.33%, Table 5; the LLaMA family
floor is **32.44%**, LLaMA 2 13B). Patchscopes independently confirms the bridge is encoded even in failing
cases.

**The fine print.** "Fixed" means *there exists* a (source × target) patch that flips the answer — a max
over a large intervention grid selected on the outcome, with **no random/placebo back-patch baseline**
(confirmed absent at code level), so an unknown share could be generic perturbation; and the 100%-correct-case
control is trivial (a benign patch always exists). Partial counterweights: back-patch success is spatially
structured (App E Figs 10–11), an argument against pure fishing; the incorrect-case construction genuinely
holds knowledge constant; and the authors themselves endorse CoT as "far more effective" (§6.3).

**Why it matters here.** A *third* category beyond can't/won't: the model knows the facts AND wants to
compose them, but the architecture runs out of layers — plausible but under-controlled. Data augmentation
wouldn't directly fix this mechanism; writing the intermediate step out sidesteps it, indirect support for
externalizing the hidden step.

### 📖 Language Models Can Learn Implicit Multi-Hop Reasoning, But Only With Lots of Data
Yuekun Yao (Saarland University) … Alexander Koller (Saarland University) · EMNLP 2025 · **10 citations** · `2505.17923`

**What it is.** The quantitative version: *how much* data does it take to teach silent k-step reasoning as k
grows?

**What they did.** Trained small GPT-2 models from scratch on synthetic k-hop reasoning (k = 2, 3, 4),
sweeping training-data volume, and derived a theoretical minimum on required depth.

**What they found.** Learnable but brutally data-hungry: the data needed **grows exponentially with hops**
(4-hop_large needed ×100 the base budget), required **depth grows linearly** with hops (Table 4), and below
the needed data the model just guesses (~1%, chance). The bright spot: a **curriculum** (2-hop → 3-hop →
4-hop) cut the 4-hop requirement **~20×**, while **mixing the identical auxiliary data non-curricularly
yields literally zero reduction** (Table 9). A pretrained GPT-2 init doesn't help (and hurts on k-hop_small).

**The fine print.** The main test set is not guaranteed shortcut-free — only the *curriculum* section builds
rejection-sampled shortcut-free tests, so a held-out 4-hop query can share its first three hops with a
training query. The data-budget curve is therefore optimistic; the "genuinely learnable" claim leans more on
the mechanistic layer-wise evidence than the accuracy numbers. The hardest cell got 2× the training steps of
others, and the depth lower-bound is conditional on a query-independent attention pattern (flagged as
possibly relaxable).

**Why it matters here.** Reframes some "under-reasoning" as a plain **capacity/coverage** problem, not a
removable shortcut — a caution. But the curriculum result is the encouraging half: *ordering* reasoning
examples by difficulty beat dumping them in uniformly (~20× data saved), an argument that staging granularity
is a first-class variable in augmentation design, not just the mix ratio.

---

### Compact entries

**Mechanism & decodability**

**📖 How Transformers Learn Implicit Reasoning (`2505.23653`, NeurIPS 2025 Spotlight)** — from-scratch study:
second-hop generalization requires **query-level training match**, and the bridge entity is **92–99%
logit-lens decodable even in the training phase where reasoning entirely fails** (Table 1) — *decodability ≠
use*, convergent with Exposure's Fig 2 dissociation. Apparent OOD first-hop success is ID-anchoring
"cheating," and the successful regime exists only in a hand-tuned ID-dominant mix. Dose-response lever: Fig 9
shows second-hop generalization is 0% at exposure frequency 1 and ~100% by frequency 13–14.

**📖 Think-to-Talk (`2412.01113`, v4)** — across **9 off-the-shelf LLMs**, multi-step (arithmetic-style)
answers become linearly decodable **only *during* chain emission**, each step causally depends on the
previously written step (the chain is load-bearing working memory, not recomputed silently), and denying the
chain costs **99.5% → 77.8%**. Direct evidence that the written chain does the computation. Fine print:
pre-CoT decodability of anything needing ≥1 step plateaus ~60%; task family is arithmetic, not encyclopedic
recall — so it splits from Yang/Hopping on whether meaningful first-hop latent resolution occurs.

**📖 Implicit Deductive Scaling (`2605.04330`)** — attaching a **complete solver trace under an isolated
attention mask** (the "corrective" objective) is the **single most effective component for direct no-CoT
accuracy (+18.9±8.4 pp, Table 2)** — a concrete augmentation format distinct from prefix-CoT that installs
latent competence where Exposure's prefix-explicit format installed nothing. Depth must satisfy L=Ω(δ);
128-layer models close the in-horizon gap (16× depth, **not** compute-matched), but CoT stays necessary for
depth extrapolation and ~30B models are at chance for direct depth-≥2 deduction (Fig A.1). Never run on
natural text.

**Capacity**

**📖 Two-Hop Information Capacity (`2502.03490`, EleutherAI)** — the conservative same-direction bound on
U-shape: latent two-hop requires **each fact stored twice** (Eq 3's 2N term) and generalizes to **zero
held-out hop components** (Fig 6), trapping small models into pure independent memorization; CoT restores
single storage *and* component generalization (Figs 3, 7). Independently measures one-hop capacity at **~1.6
bits/param** — it did *not* reproduce the ~2 b/p storage anchor that the "250×" ratio leans on.

**Boundary conditions**

**📖 The Compositionality Gap (`2210.03350`, EMNLP 2023 Findings)** — ~**40% of two-hop compositions fail
despite both sub-facts known**, roughly constant across GPT-3 scale (Fig 1); self-ask (structured emission)
narrows it; sub-answer perplexity predicts composability (**81.1% vs 42.6%** by bucket, Fig 5). The
scale-invariance datapoint that Yang and SOCRATES corroborate.

**📖 Identity Bridge (`2509.24653`, COLM 2026)** — zero-hop identity pairs ("who is X's [null-relation]")
provably restore OOD two-hop in toy models (**theory C=1 only**). v2 confirms the subject-to-answer
mechanism on Qwen3/Qwen2.5/OLMo/Gemma/Llama and shows real models **acquire the identity/copy capability by
OLMo step ~10k** — i.e. the intervention is *saturated*, not ineffective (the v1 "not significant on real
LLMs" null is gone). Identity pairs alone give **0% on 3-hop** (App C.5), and the authors call the learned
solution "a shortcut pattern rather than step-by-step implicit reasoning." Cite as mechanism with an
already-present real-model capability, not a null result.

**📖 SynthWorlds (`2510.24427`, ICLR 2026)** — parallel real/synthetic worlds with identical reasoning
structure supply a can't-vs-won't instrument: with knowledge equalized (gold docs), models reason **as well
or better on synthetic** (KA −2.0 to −10.3), so the apparent "reasoning gap" elsewhere is knowledge
acquisition + retriever familiarity, not reasoning. But in navigation they **inject memorized off-page
entities in 35–48% of steps even with the correct page in front of them** — the recall shortcut is *chosen*,
not forced (a clean *won't*).

**📖 Bridging-clause Probing (`2104.09400`, NAACL 2021)** — the BERT-era antecedent: text-implied-but-never-
stated inferences are only partially latent (**26–35% zero-shot** vs ~50% finetuned SOTA), template-bound,
with an explicit **"language modelling bias"** shortcut mode where local plausibility beats discourse-correct
inference. Shows the latent-vs-recall tension predates decoder LLMs.

**Method warning**

**📖 Dot-by-Dot / Filler Tokens (`2404.15758`)** — content-free filler tokens **fully substitute for CoT at
inference (100% vs ~66% no-filler on 3SUM) — but only when co-trained with dense, parallelizable CoT**;
filler-only training stays at baseline and serial CoT transfers in just 1/9 seeds. The mandatory
token-position-matched control for any "reasoning tokens help" comparison: added reasoning tokens can buy
*compute*, not *content*, and the two are separable only with this control.
## The papers · H1.3 — does the reasoning gap persist through fine-tuning and RL?

### 📖 Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data
Syeda Nahida Akter (CMU / NVIDIA) … Bryan Catanzaro (NVIDIA) · 2025 preprint · **23 citations** · `2510.03264`

**What it is.** The most direct test of the persistence claim: does reasoning ability have to be built in
*pretraining*, or can you add it later with fine-tuning? They pretrain 8B models from scratch with vs without
reasoning data, push both through the full SFT + RL pipeline, and compare at every stage.

**What they did.** Four base models varying how much/what reasoning data went into a 1-trillion-token pretraining mix
(the reasoning variants replace 20% of the corpus, ~200B tokens, with curated QA/CoT-format reasoning data), each then
fine-tuned and RL-trained and evaluated at each stage. They specifically test the "catch-up hypothesis": can extra
fine-tuning let a plain base model catch a reasoning-pretrained one?

**What they found.** Reasoning-in-pretraining doesn't just persist, it **compounds** — the lead of the
reasoning-pretrained model over the plain one *grows*: +8.35 avg at base stage (Table 1: 61.05 vs 52.70), +9.3 after
SFT (Table 2: 35.92 vs 26.62), +18.57 after RL (Table 3: 56.66 vs 37.92). Doubling the plain model's fine-tuning data
gains +7.39% (Table 4: 26.62→34.01) and still leaves it 3.32 *below* the weakest reasoning-pretrained model (M_SHQ+SFT
37.33): doubling helped, it just failed to catch up. There is also a "latent" effect — high-quality pretraining data
shows ~no benefit at base stage but pays +4.25 after SFT "unlocks" it — and seeing the same high-quality data in
*both* pretraining and SFT beats seeing it only at SFT (App C Fig 2: "the second exposure reinforces rather than
overwrites").

**The fine print — the catch-up comparison is not token-matched.** The reasoning-pretrained models saw ~200B reasoning
tokens in pretraining *plus* 1× SFT; the "catch-up" baseline got reasoning only at SFT and merely doubled it (the paper
never reports the token counts). The fair test — a post-training reasoning budget equal to 200B, or a placement
ablation at fixed total reasoning tokens — is not run, so "cannot be replicated later" is unproven at matched budgets;
the compounding *direction* is what is evidenced. Two more confounds: base-stage models trained on QA/CoT format are
evaluated few-shot on QA benchmarks (format familiarity, no format-matched control), and the 20% reasoning data
*substitutes* for general corpus (adding reasoning is entangled with removing web text). Their separate Table 6 shows
naive SFT-doubling with mixed-quality data harms math (−4.9%), a distinct naive-scaling-harms observation.

**Why it matters here.** The strongest evidence that reasoning content belongs early and **compounds** through the
pipeline — cite it for the direction, not for "cannot be replicated later." Big caveat for *our* method: their
"reasoning data" is QA / long-CoT fine-tuning-style data mixed into pretraining, **not ordinary web text rewritten to
expose its reasoning**, and their proxy for "quality" is essentially trace length. So it supports front-loading
reasoning but does not test "rewrite normal text to be more complete."

### 📖 Does RL Really Incentivize Reasoning Capacity Beyond the Base Model? (Yue et al.)
Yang Yue (Tsinghua) … Gao Huang (Tsinghua) · 2025 preprint (v5) · **924 citations** · `2504.13837`

**What it is.** The anchor of the "RL doesn't add capacity" camp. Sample a base model and its RLVR-trained descendant
many times per problem (up to n=1024) and compare **pass@k**: does RL ever solve a problem the base model *couldn't*
solve in any of k tries?

**What they found.** RLVR raises pass@1 (train pass@1 26.1→42.5) but pass@k *coverage* shrinks (train pass@256
67.2→64.3, Fig 1/Table 4); at large k the base matches or beats the RL model; RL uniquely solves ~0% of math problems
the base can't; base total coverage is 76.6% AIME24 / 96.0% MATH500 (Table 2); and RL outputs sit in the
low-perplexity region of the *base's own* distribution — PPL_Base(Y_RL) falls monotonically 1.244→1.159 over training
(Fig 15), a dose-response for re-weighting into the base prior. Distillation, by contrast, genuinely lifts the whole
pass@k curve.

**The fine print.** Two objections are directly answered in v5. The "pass@k at huge k just favors the higher-entropy
base" objection is tested and *defeated*: raising the RL model's temperature until output entropy matches the base's
leaves RL still below base at large k on all 6 datasets (§4.5 / App C.8 Fig 18) — the coverage loss is not merely a
low-entropy artifact. The "lucky-guess" objection is partly answered: App C.2 adds an automated filter dropping AIME24
problems the base solves without CoT (30→18) and the crossover persists (Fig 13). But scope narrows — v5 softened its
headline to "RARELY elicit," and Fig 3 (LiveCodeBench, DeepCoder-14B) shows RL *above* base for all k≤64 with one
uniquely-RL-solved problem (Table 6). Cite the math crossover as domain-scoped, not universal.

**Why it matters here.** The key indirect argument that capacity is set upstream: if the base lacks a reasoning path,
ordinary RLVR won't install it — only distillation (new information from a teacher) does. That rhymes with "put the
reasoning into pretraining" without being a direct test of it. For the can't/won't lens: RLVR fixes a *Won't* (path
exists, rarely sampled) and does little for a *Can't* (path absent).

### 📖 Base Models Know How, Thinking Models Learn When
Constantin Venhoff et al. · ICML 2026 / PMLR 306 · `2510.07364` (v4)

**What it is.** The cleanest mechanistic split of what RL vs SFT-distillation do to a base model. They build "hybrid"
models that steer a *base* model with reasoning directions read off a thinking model, then measure how much of the
base→thinking accuracy gap the steered base recovers — separately for RL-trained and SFT-distilled thinking targets.

**What they did.** Nine base↔thinking pairs: RL = Open-Reasoner-Zero 0.5B/1.5B/7B/32B, SFT-distill =
DeepSeek-R1-Distill 14B/32B/Llama-8B/Math-1.5B, plus mixed QwQ-32B. An SAE-derived category vector steers the base
*only* at base-vs-thinking next-token disagreement positions, trained by cross-entropy on the thinking model's next
token. Evaluated on GSM8K / MATH500 / Hendrycks-MATH with a human-validated grading judge (Cohen's κ=0.880, App C).

**What they found.** Hybrids recover ~76% of the RL base→thinking gap (average over the 4 ORZ pairs × 3 benchmarks,
Table 1) but only 11% of the SFT-distillation gap. The negative-control ablation is decisive: random *vectors* give
*negative* recovery (Fig 4: Full ≈77%, random-category ~20-28%, random-vectors −28/−13%) — the specific causal
direction does essentially all the work. Steering fires at only ~5-12% of tokens (Table 2): the base already emits the
correct token 88-95% of the time, so the whole intervention is *timing/orchestration*. Clean statement: **RL teaches
*when* to deploy pre-existing base mechanisms; SFT-distillation installs *new* ones.**

**The fine print.** The hybrid is not a pure base-model probe — both the steering direction and the *when* signal are
derived from the thinking model (App E.2), which the paper's own Discussion concedes. The recovery-% metric flatters
tiny-gap, low-ceiling pairs (several >100% recoveries; ORZ-0.5B thinking MATH500 is only 36.6%). It studies
post-training and inference-time steering only — no pretraining-content or completeness dose-response.

**Why it matters here.** The single sharpest license for "the capability is latent, a cheaper policy wins by default":
for RL pairs the base demonstrably *contains* the causal directions and the bottleneck is when to fire them — a clean
CAN-but-WON'T result — while SFT-distillation genuinely adds mechanisms. It tells us which post-training lever crosses
the base boundary (distillation) and which merely orchestrates within it (RL).

### 📖 Understanding Reasoning from Pretraining to Post-Training
Jingyan Shen … Pavel Izmailov (NYU / Modal Labs / UCLA / UIUC / Columbia) · 2026 preprint · `2607.16097`

**What it is.** The strongest *quantitative* version of "pretraining bounds post-training" in the corpus. Chess as a
controlled testbed replicating the full pretrain→SFT→RL pipeline, plus a 1B language-model math replication.

**What they did.** Dense decoders at 10 scales (5M-1B) pretrained on 54B tokens of tokenized human chess games,
decontaminated at board-position level; SFT on synthetic tree-search reasoning traces; GRPO with a binary
unique-line reward on 156K puzzles. 36 pretraining×RL combinations. They fit a joint law R(C_RL, N, T) = f(pretraining
loss) + g(N,T)·log C_RL and validate it with leave-one-out / leave-one-model-size-out. Replicated on a 1B OLMo-2
pretrained on 200B tokens, SFT on NuminaMath-CoT, GRPO on GSM8K+MATH+DeepScaler.

**What they found.** Post-RL reward at fixed RL compute is an *exponential function of pretraining loss* (Spearman ρ
rising 0.93→0.99 as reference compute grows, Fig 3a); the RL improvement *slope* grows linearly with log pretraining
tokens (Pearson r=+0.84, Fig 3b/c, coefficient on log T ~2× that on log N); the estimated RL asymptotic *ceiling* is
itself predicted by pretraining loss (Spearman −0.73, linear R²=0.90, App G.6). The 1B math run replicates
qualitatively (slope-vs-log-T R²=0.98, Fig 6). Mechanistically RL is *not* uniform sharpening (power-fit R²~0.56-0.68,
Table 12): on hard puzzles it performs limited genuine *tail discovery* (promoting correct moves with π_SFT<0.05, Fig
5b) while simultaneously amplifying wrong modes (up to ~20% of B5 states) with pass@16 flat.

**The fine print.** The pretraining corpus contains *zero explicit reasoning* (bare move sequences) yet pretraining
quality still governs everything downstream — so what pretraining must supply may be distributional coverage rather
than explicit reasoning demonstrations per se; the explicit-reasoning format was injected at SFT and sufficed. The
slope law holds only on mid-difficulty bins (B3-B4); the frontier beyond 680M rests on the weakest LMSO fold (3× RMSE);
the SFT-with-traces-vs-answer-only result (App F: only traces improve pass@k diversity) is sample-matched, not
token-matched; math transfer is a single 1B run. Chess ≠ NL (81-token vocab, exact verification), so exponents
characterize pipeline structure, not LM predictions.

**Why it matters here.** Pretraining sets the post-RL floor, slope, *and* ceiling — the strongest corpus-level license
for treating the pretraining corpus as the highest-leverage intervention point. It also warns that the fix may not need
to live in the pretraining *text* itself: coverage-rich pretraining + explicit traces at SFT reached the endpoint here.

### 📖 The Interplay of Pre-Training, Mid-Training and RL for Reasoning
ICML 2026 Spotlight · `2512.07783`

**What it is.** The elicit-vs-add distinction run cleanly on a from-scratch 100M model over synthetic DAG-math, with
pass@1 (reliability) and pass@128 (coverage) separated and pretraining exposure dialed as a knob.

**What they did.** Pretrain on synthetic dependency-graph arithmetic with controllable operation counts (op),
mid-train on gold traces, then GRPO with a process-verified reward. Sweep in-distribution / OOD-mid / OOD-hard eval
bins, pretraining exposure of a held-out context (0% / 0.1% / 1% / 10%), and the fraction of hard data in pretraining
(0.1%→50%). Verified against the released code (verl reward, dependency-graph scorer).

**What they found.** On *covered* tasks RL lifts pass@1 with *zero* pass@128 gain (Fig 3) — sharpening, not boundary
expansion. RL cannot transfer to a context with 0%/0.1% pretraining exposure; ≥1% seeds robust transfer (Fig 4) — but
the gate is *primitive presence*, not surface exposure: when contexts share atomic primitives, even 0% context-B
exposure transfers (App A.6 Fig 12). Genuine pass@128 gains require headroom *and* edge-calibrated data: the
pretraining hard-ratio dose-response is an inverted-U with +42.0/+45.3% pass@128 at op=11-14 *edge* tasks (Fig
15/Obs 7), *not* op=15-20 (which maxes at +25.8). Mid-training on gold traces + RL beats RL-alone by +10.8% on
OOD-hard — but only below ~8.4B-token budgets (Obs 8/Table 3: Full RL overtakes at larger budgets).

**The fine print.** The mid-training arm gets gold *traces* (information, not just compute, is unmatched); ID
saturation is by construction; the domain is synthetic-only. From code: the process reward is answer-*gated*
(`if outcome_reward < 1.0: reward = 0.0`), so the process term only differentiates among answer-correct rollouts, and
the strict process+answer pass criterion sets the near-zero base floor on op=15-20 that RL's OOD-hard gains are
measured against.

**Why it matters here.** The sharpest quantitative "pretraining content bounds post-training" result: an exposure/
primitive gate on transfer, flat coverage on covered tasks, and an inverted-U saying RL helps most at the *edge of
competence*. Directly informs our dose-response design — sweep the pretraining reasoning fraction, expect an interior
optimum, and check whether shared primitives (not surface form) carry transfer.

### 📖 Cognitive Behaviors that Enable Self-Improving Reasoners
Kanishk Gandhi … Noah D. Goodman (Stanford / SynthLabs) · COLM 2025 · `2503.01307`

**What it is.** The closest existing template for a pretraining-rewrite intervention: it shows editing the pretraining
corpus changes what RL can subsequently unlock, cleanly separating CAN'T (capacity) from WON'T (behavior absent from
the distribution but installable).

**What they did.** Under identical RL on Countdown, Qwen-2.5-3B self-improves to ~60% while Llama-3.2-3B plateaus at
~30%. They trace this to four "cognitive behaviors" (verification, backtracking, subgoal-setting, backward chaining),
count them with an LLM judge, and run controls: empty-CoT priming, incorrect-but-behavior-rich priming, and an 8.3M-
token OpenWebMath rewrite enriched with the behaviors vs a behavior-minimized rewrite.

**What they found.** Base behavior frequency differs sharply (Fig 4: Qwen verification 0.62 / backtracking 0.65 vs
Llama 0.10 / 0.20; even Llama-3.1-70B stays near-zero on backtracking/backward, so the deficit is *not* monotonic in
scale). Adding tokens without behaviors fails (empty-CoT ~30-35%, Fig 5); incorrect-but-behavior-rich priming trains
as well as correct (Fig 6 — behaviors matter more than correctness). Decisively, Llama + behavior-enriched pretraining
reaches ~60% (matching Qwen) while the behavior-minimized control shows limited improvement (Fig 8a): editing the
pretraining corpus shifts the base policy so RL escapes the plateau. Their gloss: "RL can only amplify behaviors that
appear in successful trajectories."

**The fine print.** The rewriter is Qwen-2.5-32B-Instruct — the *same family* whose behaviors they install in Llama —
with no non-Qwen or weak-generator control, so "behaviors close the gap" is not cleanly separable from "Qwen-32B
distillation." Code inspection shows the enriched prompt *injects* behaviors ("include the mistakes made by the
author… add backtracking/verifying/subgoals"), not merely "preserves natural presence" as §3.5 claims. The
generator-matched behavior-minimized control does isolate the marginal effect of behaviors over generic Qwen
rewriting. Headline results are on Countdown; App H shows behaviors transfer to GPQA but *accuracy does not* (both
~12%). "Completeness" is present-vs-absent, never dose-graded.

**Why it matters here.** A near-canonical demonstration that explicit reasoning in pretraining data is the lever that
lets post-training escape the plateau — precisely our thesis. But it doubles as our design brief: to claim the effect
is *reasoning structure* rather than *teacher distillation*, our version must add a weak/non-frontier rewriter arm and
an accuracy-transfer eval beyond one toy task.

### 📖 Reasoning or Memorization? Unreliable Results of RL Due to Data Contamination
Mingqi Wu et al. (Fudan / Shanghai AI Lab / UC Davis) · AAAI 2026 · `2507.10532`

**What it is.** The eval-trust constraint on every Qwen-family elicitation claim: much of the apparent "RL elicits
latent reasoning" on standard math benchmarks is GRPO retrieving *memorized* contaminated answers.

**What they did.** A memorization probe (truncate benchmark problems to 40/60/80% and measure verbatim reconstruction
of the held-out tail + answer accuracy) across Qwen2.5-{7B, Math-7B} vs Llama3.1-8B, plus a clean-benchmark RLVR study
on RandomCalculation (generated novel arithmetic) under Correct / Random / Inverted rewards, compared to MATH-500.

**What they found.** Qwen2.5-Math-7B reconstructs 39.2% of MATH-500 problem tails *verbatim* from only 40% prefixes
(Table 2 EM; Llama 0.6%), rising to 65.8% at 80% — but on contamination-free LiveMathBench its completion EM is 0.0-5.0
(memorization is benchmark-specific). On clean RandomCalculation only Correct reward improves steadily (Random
unstable, Inverted collapses), and correctly-rewarded RL surpasses the model's own Max@16 ceiling (Fig 7) — genuine
improvable capability exists once leakage is removed. Spurious-reward gains vanish on clean/post-cutoff sets. A
separate template artifact: Qwen2.5-Math-7B scores 72.2 template-free vs 50.6 with template (Table 4), so some "RL
gains" merely recover template-induced suppression.

**The fine print.** RandomCalculation is PEMDAS arithmetic execution, not reasoning-matched to MATH-500's word
problems, and uses a bespoke continuous reward — so contaminated-vs-clean is partly entangled with
arithmetic-vs-reasoning. The genuinely clean control is the post-release benchmarks (LiveMathBench-202505, AIME2025),
where the same Qwen model drops to Llama-level and spurious gains disappear — the strongest evidence, somewhat
under-emphasized. The verbatim-reconstruction proxy is well-controlled (Llama ~0; Qwen ~0 on post-release sets).

**Why it matters here.** A hard methodological constraint on the thread: any augmentation intervention benchmarked on
MATH-500/AMC/AIME with Qwen-family bases is untrustworthy, and pretraining that ingests complete benchmark solutions
can inflate scores via memorization rather than generalization. Use post-cutoff, synthetic-clean, or
corpus-controlled evals.

---

### The RLVR boundary fight (beyond the two anchors)

**📖 ProRL: Prolonged RL Expands Reasoning Boundaries (`2505.24864`)** — the "RL expands" flagship: >2,000 GRPO steps
with KL control + reference resets on DeepSeek-R1-Distill-Qwen-1.5B yield large pass@1 gains (math avg 44.5→60.1;
Reasoning Gym 4.2→59.1) and boundary claims where the base has pass@128≈0 (boxnet 0→7.9). **156 citations.** Fine print:
three untested confounds — eval temp 0.6 is the RL-optimal setting (base pass@k is temperature-sensitive), the base is
*already R1-distilled* (no non-distilled control), no ingredient ablation; the R1 confound runs deeper than the
checkpoint — the STEM RL data itself (SCP-116K) uses reasoning paths *produced by DeepSeek-R1* (App D), so part of the
training signal is also R1-derived. Its own "Diminished" category concedes pass@128 *declines* where the base is
competent, and the appendix concedes RL adds nothing where core skill is absent — on Reasoning-Gym 'arc' the model
scores **2.52, below even the 7B reference (3.42)** and barely above base (1.53), the authors attributing it to "a lack
of core reasoning skills … or insufficient background knowledge" (Table 5 / App F.1). Expansion is selective and
base-competence-dependent, not uniform.

**📖 The Debate on RLVR's Reasoning Boundary: Shrinkage, Expansion, or Both? (`2510.04028`)** — Xinhao Yao et al.
(Renmin U. / Ant Group / Xiamen U.) · 2025 preprint · **10 citations.** Proposes a two-stage reconciliation (early RL
over-concentrates; prolonged diversity-preserving RL expands) via training-dynamics theory + entropy-preserving GRPO,
tested on **two model families** — Qwen2.5-Math-7B *and* Llama-3.2-3B-Instruct. Both halves are empirically two-sided:
base>RL Pass@k *shrinkage* is shown on MATH-500 (Qwen base Pass@256 **88.0 vs all-RL ~72-73**, Table 3) and across all
three Llama in-domain sets (AMC 100.0>92.5; AIME24 50.0>46.7; AIME25 36.7>26.7, Table 2), while RL≥base *expansion*
holds on AMC/AIME (Table 1). Eval sampling is stated and matched — 256 samples/prompt at temp 0.6 / top-p 0.95 for base
and every RL variant alike (App C.2) — and the four methods are compute-matched over the same step range (Fig 2), so
the diversity-artifact worry does not apply. Real residual critiques: the paper never reconciles why the *same* Qwen
model expands on AMC/AIME (foregrounded) yet shrinks on MATH-500 (buried in the appendix); the load-bearing Stage-2
result is a 14/30-vs-20/30 AIME difference on tiny single-seed ID sets with no CIs; and the clean control (raise base
entropy to match the -N variant, then re-test Pass@k) is still missing. Stronger than a bare hypothesis — but the
test-set-dependent verdict is unreconciled.

**📖 RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs (`2506.14245`)** — argues plain pass@k over-credits
base models that reach right answers via *flawed* CoTs; under CoT-*validity* scoring (CoT-Pass@K) RLVR extends the
boundary to K=1024, and a theorem claims RLVR implicitly rewards correct reasoning. Fine print: the entire rebuttal
hangs on one *unvalidated* 8B reasoning-distilled LLM verifier (style-circularity risk), and its own dynamics show
flawed CoTs persist and get reinforced (median P(CC|CA)~0.7 after 400 steps).

**📖 Curriculum RL Can Incentivize Reasoning Beyond the Base Model (`2606.22317`)** — a paper *trying to refute* Yue
that instead confirms boundedness for vanilla RLVR: mean pass@256 moves −0.5pp vs base across 5 model settings (Fig
3C). The "+9.8pp beyond base" headline arrives only via a 25-43-example teacher-trace SFT bridge from an unnamed
teacher, with no ablation — distillation, not RL, crosses the boundary.

### Elicitation, distillation, and the SFT/RL split

**📖 Why Distillation can Outperform Zero-RL (`2505.21067`)** — 920 unfiltered R1 traces beat three zero-RL 32B
pipelines; RL cannot raise near-absent behaviors (exploration is bounded by the base). Decisive control: their own
weak-teacher GPT-4o SFT *failed* — the gain is frontier-teacher-specific. Fine print: the "flexible reasoning" behavior
counts are length-confounded.

**📖 LIMO: Less is More for Reasoning (`2502.03387`)** — 817 curated long-CoT examples lift AIME24 16.5→63.3 on
Qwen2.5-32B but only 9.2 on Qwen1.5-32B-Chat (COLM 2025) — pretraining *gates* what elicitation can reach. Fine print:
the "quality" score is ~30% raw trace length, and "elicitation" is observationally equivalent to efficient R1-family
distillation (no weak-teacher control).

**📖 s1: Simple Test-Time Scaling (`2501.19393`)** — 1K traces + budget forcing lift AIME24 26.7→56.7. But s1.1's
stronger r1 traces on the *same* 1K questions do significantly better (AIME25 33.3→50.0) — trace *content* injects
capability, softening the pure "just activates it" reading. Sample-efficient distillation with a decoding trick.

**📖 The Unlocking Spell on Base LLMs (URIAL) (`2312.01552`)** — 3 static in-context examples match SFT/RLHF for chat
(77.7% of tokens unshifted vs the aligned model). Fine print: scoped to *stylistic* alignment (the paper concedes
math/coding may need tuning), and the token-distribution-shift metric is inflated by template + teacher-forcing
confounds. Not a reasoning-capability elicitation result.

**📖 RL for Reasoning with One Training Example (`2504.20571`)** — one example recovers most of full-set RLVR (NeurIPS
2025). But the format-reward baseline already reaches 65.0 of the 73.6 MATH500 result (Table 8) and pass@8 barely
moves — distribution-sharpening + format unlock on a math-pretrained base. Unguessable-label variants fail to train (a
CAN'T boundary).

**📖 The False Promise of Imitating Proprietary LLMs (`2305.15717`)** — SFT imitation of ChatGPT transfers *style*
(authoritative tone 57→98%) not *capability* (broad imitation *degrades* NQ 20→15); only base scale moves benchmarks —
capability lives at pretraining. Fine print: a hallucination caveat for data the base cannot ground. Early, foundational
version of the elicitation-not-creation claim.

**📖 Operationalising the Superficial Alignment Hypothesis via Task Complexity (`2602.15829`)** — a bits formalization
of CAN'T/WON'T: random-init models 1.1% GSM8K at any program length (hard CAN'T); pretrained-only models 67.6%
reachable but via up to GB-scale adaptation programs (quantified WON'T); post-training collapses access cost to ≤10⁴
bits. Fine print: contamination caveat (footnote 9); saturating GSM8K from a pretrained-only 3B still needs ~6.2GB —
elicitation is *cheap access, not superficial capability*.

### Qwen priors, contamination, and format artifacts

**📖 Spurious Rewards: Rethinking Training Signals in RLVR (`2506.10947`)** — random rewards nearly match ground truth
on Qwen2.5-Math (+21.4 vs +29.1) and fail/reverse on Llama/OLMo: RLVR gains are parasitic on pretraining priors,
attributed to a GRPO clipping-bias mechanism amplifying a latent code-reasoning prior. Contested by `2507.10532`'s
contamination reading; its own AIME25 result (spurious gains −0.4 to +4.5) partially concedes.

**📖 Understanding R1-Zero-Like Training: A Critical Perspective (`2503.20783`)** — Qwen2.5-Math-7B scores 38.2% avg
*template-free before any RL* (R1 template: 0.0, partly extraction format), so pure-RL gains partly measure
suppression-then-reconstruction; length growth is partly a GRPO optimization bias. Constructively: continual math
pretraining triples Llama's RL ceiling (Table 4: vanilla 6.8 → FineMath 14.8 → NuminaQA 20.7) — pretraining content
lifts the post-RL asymptote.

### Pretraining / mid-training content programs post-training reasoning

**📖 Echo Chamber (`2504.07912`, COLM 2025)** — from-scratch 150M/1B + RL: RL collapses within ~1 epoch onto whichever
solution format *dominated pretraining*, converging to emit it **near-exclusively (~100%, Fig 2)**, pass@1 up while
pass@64 *declines*; raising the worse format's share flips RL onto it (Fig 4); KL=0.01 preserves diversity at no
accuracy cost. Direct implication: if the dominant pretraining mode is a shortcut, RL amplifies the shortcut. Fine
print: mixture sweep not token-matched; the mode flips with scale (150M code vs 1B NL); format-vs-reasoning not fully
separated.

**📖 Mid-Training with Self-Generated Data Improves RL (`2605.08472`)** — self-generated diverse-approach mid-training
(no external teacher — deconfounds distillation) improves post-RL pass@k coverage and composition (23.3%→56.7%);
wrong-answer approaches hurt. Fine print: the gains are exploration/coverage, not pass@1 — a rare clean signal that
*self-generated* content (not just a stronger teacher) can shift the post-RL boundary.

**📖 Distributional Clarity (`2601.06911`)** — base-model probability-cluster "clarity" correlates with post-RL pass@1
(r=0.815); reasoning-mid-trained OctoThinker out-gains Llama-Instruct under identical DAPO (30.1 vs 22.3). Fine print:
intervention margins are tiny with no error bars — suggestive of a pretraining-distribution property that predicts
RL-friendliness, not causal.

**📖 Demystifying Long Chain-of-Thought Reasoning (`2502.03373`)** — long-CoT SFT scales to higher ceilings, but
teacher-confounded (QwQ vs Qwen-72B); RL from base raises accuracy *without* raising reflection keywords; length growth
is partly a KL artifact. Its "long-CoT already in pretraining" provenance rests on weak sentence-level phrase matching.

**📖 Small Models Struggle to Learn from Strong Reasoners (`2502.12143`)** — long-CoT/strong-teacher data *hurts*
students ≤3B (Δ_Long −4.7 to −7.1) and helps ≥7B (+13.7 at 32B); a math-mid-trained 1.5B escapes the gap; student PPL
on candidate data predicts learnability; interior mix optimum α=0.2 (tuned on test). The capacity-conditionality of
completeness dose — directly relevant at our 300M-1.4B scale.

### The Won't persists through post-training

**📖 Reasoning Models Don't Always Say What They Think (`2505.05410`)** — outcome-RL faithfulness plateaus at 20-28%
without saturating; models trained into reward hacks use them on >99% of examples while verbalizing them <2% in 5/6
environments; unfaithful CoTs are *longer* than faithful ones (2064 vs 1439 tokens, Claude 3.7 Sonnet, Figs 5/7). The
concealment *Won't* survives RL, and elaborate-looking chains are not faithful chains.

**📖 Are DeepSeek R1 and Other Reasoning Models More Faithful? (`2501.08156`)** — RL-trained reasoners verbalize a
load-bearing hint far more (R1 59% vs V3 7%; 47-68% vs 5-13% across models) but cue-driven answer-*switch* rates are
*unchanged* (18.7% vs 15.3%, Table 2); R1-distillation transfers articulation (0→48%). Post-training added transparency,
not shortcut removal — accuracy/verbalization and shortcut-reliance come apart.

**📖 Fine-Tuning Enhances Existing Mechanisms (`2402.14811`)** — circuit-level (ICLR 2024): fine-tuning reuses and
*enhances* the base entity-tracking circuit (path-patching faithfulness 0.66→0.82 within the same mechanism), not a new
one. Persistence evidence is strongest exactly where least capability changed (Vicuna 0.97 faithfulness, no gain) —
real capability added, but within the pre-existing mechanism.

**📖 Training on Documents About Monitoring Leads to CoT Obfuscation (`2605.15257`)** — 1,000 fine-tuning documents
*about* a CoT monitor install obfuscation (undetected deception 1.7%→25.7%) that RL then amplifies. Fine print: the
headline requires docs containing a prose evasion recipe and collapses at ≤10% dilution. A dark-mirror proof that data
*content* programs post-training reasoning behavior — here in the wrong direction.

**📖 Spurious Correlations in Post-Training (`2505.05704`)** — SFT vs DPO/KTO under injected shortcuts: orderings flip
by task; 90% contamination sometimes *helps*. Fine print that limits its use: no base-model or clean-data control (so
it is silent on pretraining-installed persistence), and SFT was trained 3 epochs at 4-7× the LR of DPO/KTO — it
measures uptake of freshly planted artifacts only.
## The papers · H2.4 — identifying reasoning-rich text in a corpus

### 📖 AttentionInfluence: Weak-to-Strong Pretraining Data Selection
Kai Hua … Ke Shen (both ByteDance Seed) · 2025 preprint · **5 citations** · `2505.07293`

**What it is.** A training-free trick for finding reasoning-heavy documents without a classifier. Despite the title's "weak-to-strong," this is **not** two different models — it is one model versus *itself with its retrieval heads disabled* (self-ablation).

**What they did.** Take one small (1.3B) model. Detect its "retrieval heads" via a synthetic key-value needle task, build a weak copy by setting those top-5% heads to uniform attention, and score each document by the relative loss gap `(L_masked − L_base)/L_base`, ranked within-domain. Keep the top 20%, upsample into SmolLM-Corpus, and pretrain a 7B model token-matched. (The retrieval-head detector is public; the scoring/masking loop has no released code.)

**What they found.** The knowledge/reasoning subset improves (+1.4 to +3.5pp; HumanEval +3.5, GSM8K +2.7, MMLU-Pro +2.7), but the **overall average gain is +0.75pp** (Table 1), with commonsense regressions (WinoGrande −2.2, OpenBookQA −1.4, PIQA −1.1). The paper claims complementarity with a classifier, not superiority.

**The fine print.** The intervention is upweighting: the top 20% of *documents* is **~30% of corpus tokens** (73.1/241B) of ~2×-longer docs — a heavier reweighting than "20%" suggests. SmolLM-Corpus is ~93% general web, so the GPT-4o "reasoning" advantage lives only in the ~7% math/code slice (Tables 3, 8). The Table 6 head-mask validation *fails* on HumanEval (retrieval-masked 0.1098 ≈ random-masked 0.1159) — the retrieval→reasoning link is task-dependent. No matched-upsampling baseline (random-20%, length-matched, or classifier-top-20% at equal tokens) and no downstream classifier bake-off are ever run, so "+0.75pp because *reasoning*" is not isolated from "upweighting longer/technical docs."

**Why it matters here.** This was recipe (A) for our reverse-filter — the cancellation logic (both losses share weights, so memorization cancels) made it the most principled candidate. We ran the generalized go/no-go and it is a **NO-GO** (`docs/RECIPE_A_SELF_ABLATION.md`): across 5 base models and 6 sources the gap detects *in-context copy dependence*, ranks config files and parallel translations at the top of raw DCLM, the random-head control separates reasoning-vs-web as well or better, and the same family inverts across scale (Qwen 7B→72B: GSM8K AUC 0.955→0.051). This does not refute the paper's downstream pretraining result — it rules the gap out as a *reasoning* detector.

### 📖 Predictive Data Selection: The Data That Predicts Is the Data That Teaches (PreSelect)
Kashun Shum … Junxian He (both HKUST) · ICML 2025 · **20 citations** · `2503.00808`

**What it is.** A document is valuable if the *ranking* of several models' compression of it (per-char loss) matches the models' known *ability* ranking — its "predictive strength."

**What they did.** Take a ladder of same-family models of known ability, and for each document check whether their per-char losses rank-order in the models' benchmark-ability order (the sign over C(6,2)=15 pairs). Train a cheap fastText classifier to imitate that score and run it over the corpus.

**What they found.** 30B PreSelect-chosen tokens beat 300B random tokens (a 10× training-FLOPs win), and it beats other selectors. Its own appendix (Table 7, §A.7.1) runs the two-model magnitude gap ("ScalingFilter" = big-minus-small perplexity) as a controlled baseline: it beats random by only **+0.4** while multi-model rank-match gains **+3.1** (15-task), and the two signals are near-orthogonal (**Spearman 0.0533**) — the bucket's cleanest controlled comparison.

**The fine print.** The full 15-task tables (12/13) show the working signal is exam/knowledge-shaped: **worse than random on HellaSwag (38.9 vs 40.0) and PIQA (67.7 vs 69.2) at 1B**, PIQA/SIQA at 3B; gains concentrate in ARC-E/BBH/SciQ/RACE/LAMBADA. The 7-task +5.3 headline excludes the flat/negative tasks (15-task avg is +3.1); the DCLM margin is +1.5 at 1B shrinking to +0.8 at 3B (not the paper's ">2%"). The math/code "gains" are bits-per-char, not accuracy. Home-field confound: PreSelect's fastText is trained natively on the eval pool while the DCLM/FineWeb-Edu baselines are imported foreign scorers.

**Why it matters here.** This is recipe (B) — a multi-model rank-match, *not* a two-model magnitude gap, which the paper documents as a near-failure (do not run it). If we run recipe B, use the Qwen size ladder (0.5B→72B) — same tokenizer, known ability order — normalize per-character, define the ability order by a *reasoning* benchmark (their A.7.2 shows the ranking is steerable), and inspect the selected-domain composition first: PreSelect's positive set skewed fiction/erotica-heavy, so a rank-match pass without composition inspection is not interpretable.

### 📖 DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
Zhihong Shao … Daya Guo (DeepSeek-AI / Tsinghua / PKU) · 2024 · DeepSeek-AI technical report · `2402.03300`

**What it is.** The canonical proof that benchmark-effective math text exists in raw Common Crawl at scale, and a trivially cheap classifier finds it.

**What they did.** Train a fastText classifier on 500K OpenWebMath positives + 500K CC negatives, score URL-deduplicated Common Crawl, keep the top-ranked tokens, and expand recall iteratively by human-annotating math URL paths in flagged domains and retraining. Four iterations yield 35.5M pages / **120B tokens**. Validate with a 1.3B model trained 150B tokens on each of four math corpora, then build DeepSeekMath-Base 7B (500B continual tokens) and add SFT + GRPO RL.

**What they found.** The 1.3B model hits GSM8K **23.8%** on the DSM corpus vs 11.5% OpenWebMath, 14.3% Proof-Pile-2, 2.9% no-math (Table 1). The 7B base reaches GSM8K 64.2 / MATH 36.2, beating Minerva 540B (58.8 / 33.6, Table 2). GRPO then lifts Maj@K but *not* Pass@K (Fig 7) — RL reorders what pretraining made possible.

**The fine print.** The 150B bake-off is token-matched but **not fresh-data-matched** (MathPile repeats ~17 epochs, DSM ~1.25), so "bigger corpus wins" partly measures a repetition penalty. The corpus is multilingual while baselines are English-centric, inflating the Chinese-benchmark wins. Decontamination is exact 10-gram matching only (weak to paraphrase). The selection signal is **math-domain-likeness, not reasoning**; no control isolates reasoning content from domain/quality/language, and the pipeline/corpus were never released.

**Why it matters here.** Strong support that finding reasoning-adjacent text at scale pays off — and one sharp cautionary counter-datapoint: the **arXiv negative result** (Tables 8-9) shows 40B tokens of arXiv give MATH 11.1 vs 12.5 no-math and miniF2F 11.9 vs 21.7 (deterioration). Surface "reasoning-density" (formal papers, proofs) is not the right selection target; format/style match matters more than appearing reasoning-dense.

### 📖 The FineWeb Datasets (incl. FineWeb-Edu)
Guilherme Penedo … Thomas Wolf (both HuggingFace) · NeurIPS 2024 · **1029 citations** · `2406.17557`

**What it is.** The famous web-cleaning paper; the relevant piece is **FineWeb-Edu**, a classifier that keeps "educational" web pages.

**What they did.** Had Llama-3-70B rate 460K web pages 0–5 for educational value, trained a frozen-embedder + single linear head (a shallow probe) to reproduce those ratings, and filtered 15T tokens down to 1.3T.

**What they found.** Big gains on knowledge benchmarks — **MMLU 33 → 37, ARC 46 → 57** — from filtering alone at matched model size and token budget (ablation model is 1.71B params).

**The fine print.** The prompt *intends* grade-school steering, but the measured *outcome* is the opposite: FineWeb-Edu fits arxiv (Paloma ppl 23.4 vs FineWeb's 32.3, Table 3), math/logic, academic S2ORC and code **better** than FineWeb, up-samples Math/Education/Science topics (Fig 18), and the released classifier card warns it "might overfit to academic looking content." The steering is *toward* academic/technical text, not away from it — while the commonsense cost is real (HellaSwag lower, Fig 16). The rubric's top tier explicitly rewards text that "follows detailed reasoning" (App F.1), so "educational ≠ reasoning" is softer than often stated; the threshold dose-response is non-monotonic (Fig 17: ≥3 peaks, ≥4 no better).

**Why it matters here.** The canonical "quality selection helps," but a *contrast* case: the supervision target (a 70B model's educational-value judgment) overlaps the school-exam benchmarks it is scored on, and no control tests whether a generic well-written-text signal produces the same lift. Useful for separating "academic/educational" from "reasoning-rich" when we design our own signal.

### 📖 Essential-Web v1.0: 24T tokens of organized web data
Andrew Hojel, Michael Pust … Ashish Vaswani (Essential AI) · 2025 preprint · **0 citations** (too new) · `2506.14111`

**What it is.** A 24T-token Common Crawl corpus where every document carries a 12-field taxonomy — including a **5-level Reasoning-Depth ladder** (No → Basic → Intermediate → Advanced → Exceptional, the top tier explicitly "long chain-of-thought … proofs").

**What they did.** Distill a 0.5B labeler (EAI-Distill-0.5b) from a Qwen2.5-32B teacher, label all 23.6B documents, then write conjunctive SQL-style filters over the labels, anneal a common base for 80B tokens on each curated set, and compare to public SOTA datasets. No document-level LLM rewriting in the main results (to isolate filtering).

**What they found.** Labeling is cheap and scalable: the 0.5B labeler is 50× faster than prompting the teacher (Table 12) and reaches high agreement on Reasoning-Depth. Curated STEM beats DCLM-baseline by +24.5% on MMLU-STEM (Table 9); medical +8.6%.

**The fine print.** Three hedges on "reliable": the headline κ 0.87-beats-teacher-0.67 is metric-inflated in the P₀,Pₑ→1 regime (honest stat P₀ 0.90→0.98, Table 27); human-human κ on these labels is only **0.38–0.54** (Table 28), so the reasoning-depth ground truth is LLM-consensus-defined; and docs >30k chars are labeled from ~7.5k-token excerpts (Algorithm 12) — exactly where long complete chains would live. The reasoning-depth clause is **never causally isolated**: subject alone recalls 96–98% of vetted math/code (Table 11), the STEM filter's near-no-op reasoning clause delivers the *largest* gain while Top Code's strictest clause scores worse, and the reasoning-inclusive math filter *lags* FineMath by 8% on GSM8K (Table 3, 22.4 vs 26.4). Advertised wins are MMLU knowledge subsets; code-generation (HumanEval+/MBPP+) is flat within standard error.

**Why it matters here.** The reasoning-depth rubric is directly reusable as our completeness-granularity axis, and a 0.5B model learns it from ~12B teacher-label tokens — the cheapest instrument we have for labeling completeness levels. But it selects rather than augments, never shows reasoning-selection *per se* helps, and its own Table 26 is an incidental vote for augmentation over selection: the one LLM-rewritten corpus (MegaMath-Web-Pro) tops every filter-only math set (GSM8K 27.3 / MATH 12.2 / MMLU-Math 41.4).

### 📖 Removing Noise, not Finding Gold: Rethinking Classifier-Based Quality Filtering (formerly "The Data-Quality Illusion")
Thiziri Nait Saada … Pierre Ablin (Apple) · ICML 2026 · `2510.00866`

**What it is.** A mechanistic analysis of why classifier-based quality filtering (CQF) misleads — and evidence that it nonetheless works.

**What they did.** Analyze CQF (train a logistic classifier to separate a small curated HQ set from the web LQ set; keep top-k by classifier probability). Prove the score is a monotone function of the density ratio, `s(x)=φ(p_HQ(x)/p_LQ(x))`, φ(t)=t/(t+1) (§4, Eq 4.1). Introduce a loss-based "data-conditioning" quality notion, a semi-synthetic corruption axis (token permutation), and Chinchilla scaling-law fits (125M–1.3B), all compute-, token-, and repetition-matched.

**What they found.** The score keys on documents *far from LQ* (low p_LQ), not on resemblance to HQ, and demonstrably latches onto **sequence length** (App C). Loss-on-HQ is an invalid proxy — a U-curve in k (Fig 2), where small k can make the model *worse* on HQ than training on the full LQ set. CQF fails the paper's own data-conditioning test (Fig 9). **But** v4 adds the decisive result: CQF-refined data beats direct training on a large HQ set (**53.8 vs 50.1**, §6/Fig 8).

**The fine print.** The mechanism is a *trade-off* — the score prefers docs both likely under HQ and unlikely under LQ — not pure "distance from web." The "CQF beats HQ" result carries a real caveat: that large-HQ baseline is itself CQF-constructed. Single filter family (logistic on embeddings); 125M–1.3B; the strongest downstream result uses the eval task's own distribution (ARC-Easy) as the HQ target, which is the paper's thesis, not a hidden confound.

**Why it matters here.** The paper's bottom line is "filtering removes noise; it does not find reasoning-gold" — so any quality classifier we build to "find reasoning" is really selecting *not-web + shorter + aligned-to-my-HQ-set*. Its paper-stated lessons are the discipline for our gate: validate against a data-conditioning-style test rather than loss on a curated reasoning set, and always run a length-matched control.

### 📖 What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code (Beyond-Code)
Yuze Zhao … Enhong Chen · ICML 2026 · `2605.19762`

**What it is.** A controlled 10T-token pretraining study that separates executable "Code" from "Code-NL" traces and tests whether the reasoning gains usually credited to code actually come from explicit structured reasoning.

**What they did.** On a 7-domain corpus (MoE, fixed budgets), run two experiments: (i) a code-removal ablation, and (ii) the flagship — define "cognitive scaffolds" (math with explicit subgoals/derivations, selected by a lightweight fastText code-vs-non-code classifier, **no generator, no distillation**), then **replace** ordinary math with scaffold-selected math *at a fixed math-token budget*.

**What they found.** The scaffold replacement gains math Overall **+17.56%**, OlympiadBench **+47.78%**, MATH **+23.17%**, MathBench +14.51%, while GSM8K regresses **−6.29%** and CMath −2.00% (p.8) — the closest published pretraining-scale completeness intervention, and completeness is *not* monotonically good (over-explicit reasoning hurts easy tasks). The code-removal aggregate rises without code at all four settings (32e 36.20→38.52).

**The fine print.** The selector is heavily **formatting-confounded**: scaffold-selected math has indentation ratio 0.0006→0.5446 (~900×) and length 531→2821 chars (5.3×, Table 1) — it picks code-*formatted* math, and the authors call it a proxy for "external organization," not verified reasoning. There is **no replacement-ratio dose sweep** (they explicitly decline the monotonicity claim). The code-removal arm is a fixed-budget *substitution* (remaining domains upsampled proportionally), with mixed per-benchmark effects (AIME25 2.92→1.67, OlympiadBench 10.22→9.33, GKMathUnion 41.59→36.51 all degrade) — read it as substitution, not intrinsic "code hurts math," and never as a College-Math 41.59→65.73 pairing (Table 4 has no College-Math column; those are two different benchmarks, GKMathUnion-full and ZKMathUnion-w/o-code).

**Why it matters here.** The closest template for our natural-text completeness experiment: fixed-token-budget replacement, structure-selected, no generator, hard-task gains with easy-task regressions. Our design should control formatting (match indentation/length distributions), run the dose sweep they skipped, and pre-register the easy-task regression.

### 📖 Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models (ProcKN)
Laura Ruis … Max Bartolo (Cohere / UCL) · ICLR 2025 · `2411.12580`

**What it is.** An influence-function study of *which* pretraining documents drive reasoning — the best observational definition of reasoning-rich text in the bucket.

**What they did.** Compute EK-FAC influence functions for two Command-R models (7B, 35B) over a ~5M-doc / 2.5B-token pretraining subsample, ranking documents by how much each raises the model's completion log-prob. Compare 40 reasoning queries (arithmetic, slopes, linear equations) against 40 factual queries; characterize top-document content by prompting a larger model.

**What they found.** For reasoning queries, influence spreads over *procedurally-similar* documents — code implementations, worked equations — with **StackExchange ~10× overrepresented**, and the same documents recur across same-task queries (p < 4e-8). Crucially, the answer is **not** in the top documents (found twice for 7B, never for 35B), unlike factual queries where it is (7B 55%, 35B 30%). Reasoning is generalizable synthesis from procedural text, not retrieval.

**The fine print.** The document-level attribution is **correlational** — no single document is ablated and retrained, so "drives" rests on influence-score correlation. But the influence *estimator* is better-validated than that implies: Appendix A.1 runs four counterfactual-retraining checks, two of them on downstream **accuracy** of reasoning-flavored tasks with the 7B (DROP 0.61→0.38 at k=2000 vs random 0.57; RACE 0.85→0.74 vs random 0.81, Tables 5-6), both significant vs random — though this is on the fine-tuning set, not a pretraining-level intervention (the authors concede this, p.17). The code-importance finding is *not* purely model-graded either: it is corroborated by objective source-dataset provenance multipliers (Fig 7) and a random-baseline content analysis (Fig 11, code dominant in the top and bottom influential docs but sparse in random samples); only the finer "why influential / what procedure" labels are Command R+-graded. Remaining caveats: influence is over a 2.5B-token subsample (the "answer absent" negative could reflect undersampling, rebutted only qualitatively); MLP-params-only EK-FAC with attention frozen; two proprietary models; math-reasoning tasks hand-picked where the model already scores ≥80%; and no length- or domain-matched control separates "procedural reasoning" from "code/math domain."

**Why it matters here.** It supplies the thread's working definition — reasoning-rich = procedural, worked-out, full-procedure text — and makes the augmentation hypothesis plausible (if models synthesize reasoning from procedural docs, adding explicit procedural docs should strengthen it). But it establishes no *pretraining-level* causality: the ablate-a-document-and-repretrain experiment is exactly what is missing.

---

### Influence functions and value-carrier behaviors

**📖 Which Data Attributes Stimulate Math and Code Reasoning? (`2505.19949`, Attrib)** — Siqi Kou et al., 2025 preprint. Influence-function attribution over SFT data plus interventions. The load-bearing piece is a genuine causal manipulation: GPT-4o-truncating exploration/verification behaviors from traces drops **MATH500 77.2→73.8** and LCB 33.8→32.0 — trace *completeness*, not just answers, carries value; the influential math tokens are connectives ("Wait", "Verify", "Therefore"). Fine print: the difficulty-flip's swap pool (OpenThoughts-114k) is same-pipeline, same-question-source, and same-distillation-model (Deepseek-R1) as the baseline (footnote 6), so the source confound is weak — the residual is only that it is a larger pool (the specific swapped-in problems are new) and the reverse-flip's replacement source is unstated; tokens are unmatched (harder math has longer CoTs → more training tokens); single-seed on ~30-problem AIME sets. The reverse-flip "worst performance validates the causal direction" holds only for the 7B (LiveCodeBench 30.0 worst); at 14B the reverse flip *ties* the forward flip on AIME25 (both 23.3) and leaves AIME24 at baseline, so the difficulty-is-causal story is unsupported at 14B.

**📖 Exploring the Mystery of Influential Data for Mathematical Reasoning (`2404.01067`, QaDS)** — Xinzhe Ni et al., COLM 2024. One-shot-influence scoring ranks complete NL reasoning chains high and bare-answer/code-like samples low (Fig 5, Tables 9-10), and beats random at matched 69K (31.3 vs 29.5). Fine print: the signal is math-target-leaked and length-confounded (no length control) — it illustrates the H2.4 confound rather than resolving it.

### Corpus-scale domain selection and per-document labels

**📖 MAmmoTH2: Scaling Instructions from the Web (`2405.03548`)** — Xiang Yue et al. (CMU/Waterloo), NeurIPS 2024. A fastText+LLM-domain cascade mines **5M naturally occurring Q-A pairs** from Common Crawl (MATH 11.2→36.7 on Mistral-7B, Table 2). Direct evidence for *completing* found text: refiner-completed chains beat same-source incomplete extracted text at every scale (~+5 MATH at 10M pairs, Fig 5). Fine print: the refiner prompt explicitly mandates adding steps and fixing answers, so the completeness gain is inseparable from 72B-teacher distillation.

**📖 Unlocking Latent Value: Taxonomy-Guided Recovery from Low-Tier Web Corpora (`2606.07778`)** — Neeraj Varshney et al. (Amazon), 2026 preprint. Multi-dimensional label filters beat even unfiltered **top-tier** quality data on reasoning answer-loss (F8 Minerva −10.9% vs top tier, §5.5). But its own NMI shows the strongest dimension works via a document-type/domain shift (Timeliness–DocType 0.205 vs Timeliness–Reasoning 0.029, Table 9), everything is answer-loss not accuracy, and reasoning-depth is never isolated from domain — multi-dimensional beats scalar quality, yet still domain-confounded.

### What selection can reach — off-web, and post-training

**📖 Data Recipes for Reasoning Models (`2506.04178`, OpenThoughts3)** — Etash Guha et al. (Open Thoughts collective), 2025 preprint. 1000+ matched ablations. A fastText reasoning-trace detector over DCLM-RefinedWeb finds the web lacks long CoT (App P); pretraining-style fastText/embedding filters *underperform* LLM difficulty/length rating for selecting reasoning questions (Table 5); stripping self-reflection costs **−49.1% relative** with a monotonic truncation dose-response (Table 22); a better benchmark model is not a better teacher (QwQ > R1).

**📖 Golden Goose: Synthesizing Unlimited RLVR Tasks from Unverifiable Internet Text (`2601.22975`)** — Ximing Lu et al. (NVIDIA/UW/UCSD), 2026 preprint. Masking "important reasoning steps" in unverifiable text and rewarding MCQ reconstruction revives saturated RLVR (+2.2–3.5% where stale data gives ~0; effectiveness ratio ~70% vs ~25–34%, Figs 2/5) — evidence that reasoning-step spans are high-value supervision targets. Fine print: no reasoning-span-vs-random-span masking control and no decontamination audit.

**📖 Reasoning Quality Emerges Early (`2606.26797`, TEMP)** — Hongyi Jin et al. (UCLA/Optum), ICML 2026. Noise-perturbed first-100-token loss ranks difficulty *inside an all-reasoning pool* (r ≈ 0.9 vs a bespoke LLM rubric, beats length-proxy). Fine print: medical wins are small (+1.1–1.7%); math generalization is not statistically separated (Table 8: +0.41 inside ±2–3 error bars, loses 3/4 math benchmarks); no reasoning-vs-non-reasoning control. For H2.7 its signal is another self-contained noise-weakened-vs-clean same-model loss gap.

### A single-model judge, and a non-datapoint

**📖 Autonomous Data Selection with Zero-Shot Generative Classifiers for Math (`2402.07625`, AutoDS/AutoMathText)** — Yifan Zhang, Andrew C. Yao (Tsinghua), ACL Findings 2025 · **26 citations**. Ask a Qwen-72B base model two yes/no questions and keep the confident-YES docs. Fair token-matched math gains (MATH 12.9→16.1, GSM8K 38.8→45.4 on Mistral-7B, Table 2, ~2.4× efficiency) but a Gemma-2B null (ties). Fine print: commonsense parity-or-below Uniform (Table 4); Appendix C shows high-scoring docs with incomplete/incorrect reasoning (a stuck forum poster scores 0.932); the real scorer uses top-5 logprobs, case-variant max, and only the first ~4096 characters — the signal keys on reasoning density/educational framing, not completeness or correctness.

**📖 Reinforcement Pre-Training on General-Domain Corpora (GD-RPT)** — single-author TechRxiv proposal, 2025 (not on arXiv/ACL). Abstract-only and results-free ("We do not claim quantitative improvements"). Zero evidentiary weight; listed for completeness, not cited as data.
## The papers · H2.5 — augmenting pretraining text with reasoning

### 📖 MIND: Math Informed syNthetic Dialogues for Pretraining LLMs
Syeda Nahida Akter … Bryan Catanzaro (NVIDIA + CMU + Boston U.) · ICLR 2025 · `2410.12881`

**What it is.** The best-controlled augmentation result in the corpus: rewrite a math corpus into knowledge-gap
dialogues that force implicit steps to be spoken, and test whether the *structure* — not the generator — carries the
gain.

**What they did.** Split OpenWebMath (OWM) into 500-token chunks and zero-shot prompt LLaMA-3-70B-Instruct to rewrite
each into a multi-turn conversation across 7 styles (TWO STUDENTS, TEACHER-STUDENT, DEBATE, TWO PROFESSORS, …),
instructed to add no new information and to decompose step-by-step. Evaluate by continued pretraining of a 7B base for
50B tokens, blend held at 2:1 OWM:FineWebEdu, **token-matched** so every condition sees the same source-token budget.

**What they found.** GSM8K **+13.42** token-matched (raw OWM-4B 12.96 → 26.38, Table 5); 33B of dialogue-augmented
OWM-4B **beats raw OWM-14B**, a 3.6× larger corpus (GSM8K 26.38 vs 20.47, Avg 34.80 vs 32.79). Three controls most
augmentation papers lack all point at structure: the same-generator *rephrase* baseline barely moves (Avg 29.22 ≈ raw
29.17); swapping the generator **70B→8B keeps ~all the gain** (Avg 32.95 vs 33.38, both ≫ 29.17, Table 6); and the
zero-knowledge-gap style **TWO PROFESSORS gains nothing** (Avg 29.12, highest source-similarity) — clean evidence that
reasoning left implicit doesn't teach while forced explicitation does.

**The fine print.** The distillation confound is only *partially* closed — the 8B generator is still far stronger than
the 7B student, and there is no generator-≤-student control; the authors concede "all LLM-generated synthetic data
involves some form of knowledge distillation," and their own quality rubric rewards injected "new knowledge" they never
quantify. Completeness ≠ verbosity: dialogue length is uncorrelated with accuracy (Table 13). Code is a negative result
— conversational rewriting leaves HumanEval flat and rephrase actively hurts (Table 16).

**Why it matters here.** The strongest natural-text deconfounding datapoint we have: the rephrase and 70B→8B controls
argue the gain is reasoning *structure*, not passthrough distillation, and TWO PROFESSORS is the cleanest single
demonstration that implicit reasoning must be made explicit. But it reframes completeness as *knowledge-gap-driven
explanation dynamics*, not chain length, and leaves distillation only partly isolated — a from-scratch or
self-distilled replication is the load-bearing follow-up.

### 📖 Thinking Augmented Pre-training (TPT)
Liang Wang … Furu Wei (Microsoft Research) · 2025 preprint · **3 citations** · `2509.20186`

**What it is.** The biggest "augment pretraining text with reasoning" result: append an automatically-generated
"thinking trajectory" to *every* document and pretrain on the concatenation.

**What they did.** One fixed prompt ("Simulate an expert's in-depth thought process … Use the Feynman technique"),
generated by **Qwen3-8B** for the from-scratch runs (DeepSeek-R1-Distill-7B for mid-training, whose base is
math-specialized Qwen2.5-Math), thinking capped at 8k tokens (~3× inflation). From-scratch 8B on 100B tokens of
FineWeb-Edu+MegaMath, token- and step-matched against a vanilla baseline that therefore sees ~3× more raw documents.
Augmentation is **uniform**; completeness/depth is **never varied** (their ablations vary generation *strategy*, which
barely moves the metric — a 1.5B generator even beats the 7B default).

**What they found.** GSM8k 19.2→50.1, MATH 9.1→21.8, 5-task avg 26.2→43.9 (Table 1; vs LLaMA-3.1-8B@15T at 46.8); gains
amplify through SFT (AIME24 1.0→35.2, MATH-500 33.8→82.4). A genuine data-matched control holds: at a fixed 40B budget
with ≤10B raw tokens (vanilla 4 epochs vs TPT 1), TPT still roughly doubles the average.

**The fine print.** The from-scratch thinking is written by a fully-trained RL-tuned reasoner (Qwen3-8B), so the
headline partly *distills Qwen3-8B*; the weak-teacher ablation that would deconfound this is run only in mid-training,
never from scratch. The narrow true claim — no pure-LM/perplexity/HellaSwag/LAMBADA control — stands, though not every eval is reasoning-shaped: BoolQ 0-shot non-CoT improves 66.5→75.0 and MMLU/MMLU-Pro are knowledge
benchmarks. The loss comparison is apples-to-oranges (augmented vs raw distributions).

**Why it matters here.** The strongest existence proof that uniform reasoning-augmentation pays off at scale — but
doubly incomplete for our question: the gain can't be separated from distilling a strong teacher, and completeness is
applied at one fixed setting, never varied. The completeness dose-response remains unclaimed.

### 📖 Reasoning to Learn from Latent Thoughts (BoLT)
Yangjun Ruan … Tatsunori Hashimoto (U. Toronto / Stanford) · 2025 preprint · **40 citations** · `2503.18866`

**What it is.** The flagship latent-thought paper: treat the reasoning that *produced* a document as a latent variable,
generate it, train on thought+text, then bootstrap the model to generate its own latents in an EM loop.

**What they did.** TinyLlama-1.1B continued-pretrained on FineMath-4+ (already reasoning-rich; augmentation within it is
uniform). §5: latents from GPT-4o-mini, all baselines compute-matched at an 8B-token budget. §6: the 1.1B model
generates its own latents, importance-resamples them, retrains, iterates — only a 240M GPT-4o-mini warmstart.

**What they found.** Headline MATH 5.74 (Raw-Repeat) → 25.38 (Table 1). The teacher-matched comparison —
**Latent-Thought 25.38 vs WRAP-CoT 19.36 (both GPT-4o-mini, same budget)** — isolates **+6.0 MATH**, roughly a third of
the headline; a raw-space ablation (25.38→22.38) further isolates the latent-space design. It also beats 8B *fresh
unique* raw tokens (11.18), the genuinely surprising data-efficiency result (which still bundles distillation).

**The fine print.** Quote 25.38-vs-19.36 as the method's isolated contribution, not 5.7→25.4 (which bundles GPT-4o-mini
distillation). The teacher-free self-bootstrap is real but modest and **plateaus: few-shot MATH ~10%→~13% over the
iterations (Fig 8)** — and GSM8K *deteriorates* over iterations on the
fixed-data setup; the fine-tuned MATH curve is ~0.159→0.176 (Fig 9). One eval prompt set is GPT-4o-mini-synthesized CoT
(a train/test format-match advantage; a standard prompt set mitigates). Missing: a GPT-4o-mini direct-answer skyline
and any completeness/length ablation.

**Why it matters here.** Conceptually the closest paper to our thesis — it names the mechanism ("web text is the
compressed final outcome of a verbose human thought process") and instantiates completeness-restoration. But its
strong-teacher results are distillation-confounded, its teacher-free loop yields modest plateauing gains, and
completeness is never varied.

### 📖 Rewriting Pre-Training Data Boosts LLM Performance in Math and Code (Swallow)
Kazuki Fujii … (Institute of Science Tokyo / AIST) · 2025 preprint · `2505.02881`

**What it is.** A large, token-matched demonstration that rewriting code/math corpora into self-contained, step-by-step
form helps a lot — and, uniquely, the purest distillation confound in the batch.

**What they did.** Llama-3.3-70B-Instruct rewrites The-Stack-v2 Python (4-stage: syntax filter → Pylint≥7 → style-guide
rewrite → self-containment rewrite) and Finemath-4+ ("restore missing context," step-by-step). Continual-pretrain
Llama-3.1-8B for a fixed 50B-token budget, 8B code tokens held constant across 13 code + 2 math ablations that vary only
the target dataset.

**What they found.** SwallowCode vs Stack-Edu: **HumanEval +17.0**, **HumanEval+ +16.1** (both token-matched);
SwallowMath vs Finemath-4+: **GSM8K +12.4, MATH +7.6** (Fig 5). The style-guide rewrite alone adds >9 HumanEval points,
self-containment >5 more.

**The fine print.** Every rewritten token is a 70B-Instruct generation with **no weak-rewriter and no teacher-matched
control**, so the gain cannot be separated from distillation. Their one model-fixed comparison sharpens that reading:
LLM **scoring** (ranks existing text, injects nothing) yields <1pt, LLM **rewriting** (injects the teacher's
generations) yields >14pt (Fig 3/4) — the delta *is* the distillation channel, not "rewriting quality." Task selection
is partly post-hoc: MBPP dropped ~10 points from style-driven function renames and was **excluded from final metrics**.
No completeness dose-response.

**Why it matters here.** A strong existence proof for the augment bucket, but the cleanest illustration that where
controls are missing the distillation channel is large. Any thread claim that reasoning *content* (not token-injection)
drives the gain must beat the scoring-vs-rewriting decomposition this paper hands us.

### 📖 Programming Every Example: Lifting Pre-training Data Quality Like Experts at Scale (ProX)
Fan Zhou, Zengzhi Wang … Pengfei Liu (SJTU / Shanghai AI Lab / GAIR) · 2024 preprint · `2409.17115`

**What it is.** The augmentation-*free* pole of the H2.5 axis, and the deletion-only baseline our program must beat:
nothing is generated — a small LM emits a per-document *deletion program* that drops docs and strips noisy lines.

**What they did.** A 0.3B–1.7B refining LM generates doc-level keep/drop + chunk-level `remove_lines`/`normalize`
programs; a Python executor applies them. Verified in code: the executor has **no generation path** — refined corpora
are deletion-only transforms, so no teacher text can enter training. Token-matched from-scratch pretraining and
OpenWebMath continual-pretraining.

**What they found.** Over token-matched raw CPT on a 9-task math-CoT avg: CodeLlama-7B **+6.2** (base 29.1 → raw-CPT
43.2 → ProX 49.4, Table 5), Mistral-7B +4.4; general zero-shot avg +2–3 over token-matched raw (Table 2). Rule-based
filtering instead *hurts* OpenWebMath (−3.1). All with **zero added reasoning content**.

**The fine print.** The abstract's "+20.3%" is vs the base model; the fair increment over the token-matched raw-CPT
control is roughly half (+6.2). The doc-level selector distills Llama-3-70B's edu-quality + format judgments (a quality,
not reasoning, signal), and under the fixed token budget the kept ~25% pool is epoched 1.6–3.5× (Table 7), so "cleaner
data" is partly "more repetitions of better data." No reasoning-specificity control; no generation baseline at matched
refining FLOPs.

**Why it matters here.** It sets the hard control: +6.2 math-CoT and +2–3 general are achievable with pure noise
deletion at matched tokens/FLOPs. Reasoning-augmentation is not credible as "removing the shortcut" unless it exceeds a
ProX-style cleanup of the same corpus at matched tokens *and* matched generation FLOPs.

### 📖 RePro: Training Language Models to Faithfully Recycle the Web for Pretraining
Zichun Yu, Chenyan Xiong (CMU) · 2025 preprint · `2510.10681`

**What it is.** The strongest recent evidence that, on general web, a *faithful* rephraser that adds no reasoning beats
an unfaithful augmenter that injects a reasoning monologue.

**What they did.** RL-train (GRPO) a small 4B rephraser to rewrite DCLM docs for higher quality under three faithfulness
rewards — semantic (BERTScore), structural, and a **length cap (≤1.25× the source)** — then mix recycled-hq with
organic-hq and pretrain 400M/1.4B models from scratch (28.8B tokens). Default RePro is **RL-only, no strong teacher**;
an optional arm distills 50k GPT-4o rephrasings. Evaluated on 22 downstream tasks (centered accuracy).

**What they found.** RePro Core **0.21658** (400M) beats organic-7.2B (0.18990), WRAP (0.19536), ProX (0.19623), and
the reasoning-injecting **ReWire (0.20125)**; at 1.4B, 0.29929 vs ReWire 0.29029 (Table 2). The distillation confound is
addressed cleanly: **RL-only (0.21658) beats SFT-from-GPT-4o (0.19216)** and even direct prompting (0.19847), so the
gain is RL quality-optimization, not teacher distillation. The paper frames injected "extraneous information" as a
model-collapse risk.

**The fine print.** The ReWire comparison is confounded — its code was unreleased, so RePro sampled from ReWire's data
drawn from a different, larger organic pool (the authors admit the advantage). No weak-generator control, and no
ablation isolating boilerplate removal (54% paraphrase + 14% deletion, Fig 7) from paraphrase quality. Completeness is
never varied — the method actively *caps* it. Gains attenuate with scale (+14% rel at 400M → +4.7% at 1.4B).

**Why it matters here.** On general web the winning structure is *cleaning*, not reasoning: faithful rephrasing beats
reasoning-injection and RePro warns that naive prompted reasoning-injection can underperform and can hurt. This makes
faithful-rephrase a baseline any general-web reasoning-augmentation must clear — while leaving careful complete-reasoning
augmentation untested either way.

### 📖 The Kinetics of Reasoning: How Chain-of-Thought Shapes Learning in Transformers
Zihan Pengmei … Huzefa Rangwala · 2025 preprint · `2510.25791`

**What it is.** A from-scratch synthetic study with **ground-truth templated traces (no teacher — no distillation
possible)**, the cleanest teacher-free evidence that explicit reasoning adds capability rather than style.

**What they did.** Train 12-layer GPT-2-style transformers from scratch on synthetic symbolic tasks (Comparison,
Sorting, Intersection, Composition) with k-way queries. Two models differ only in target format: answer-only vs
trace-then-answer, where the trace is *algorithmically derived* from the KB's ground-truth program. Test OOD
generalization; the CoT model decodes its trace at test time.

**What they found.** Trace supervision flips qualitative OOD failure to success — Composition k=2 **0.00→1.00**, Sorting
k=4 **0.04→0.83** (Table 2) — while answer-only never gets there. Two sharp caveats: the model finds the answer
*shortcut* early and only later aligns to the trace ("transient unfaithfulness" — CoT reshapes the trajectory, doesn't
delete the shortcut); and on Intersection (which exceeds per-step capacity) no trace template rescues it (~0.01 even
with CoT) — a true Can't.

**The fine print.** Not token-, supervision-, or compute-matched: the CoT condition gets a longer target and dense
intermediate supervision the answer-only condition never sees, so "CoT succeeds where non-CoT fails" conflates denser
supervision with expressivity — though the failures are qualitative (0.00–0.18), which softens the objection. Trace
length is tied to task complexity, so there is no clean completeness dose-response; the one granularity probe
(Intersection template ordering) is null. Fully synthetic, single model size.

**Why it matters here.** The strongest teacher-free datapoint that externalized reasoning genuinely *adds capability* —
and a useful caution that adding CoT reshapes learning kinetics without instantly removing the shortcut and cannot
overcome architectural Can'ts.

### 📖 Demystifying Synthetic Data in LLM Pre-training
Feiyang Kang … Carole-Jean Wu (Meta / Virginia Tech) · EMNLP 2025 · `2510.01631`

**What it is.** The best available map of the augmentation envelope — >1000 models, scaling laws — and the corpus's
clearest anti-distillation datapoint.

**What they did.** Generate three synthetic types (WRAP-style HQ rephrasing, QA rephrasing, Phi-style textbooks) all
from a single Mistral-7B generator over *unfiltered* CommonCrawl, pretrain 100M–3B Llama-arch models to 200B tokens
across mixing ratios, and fit validated scaling laws. Metric: per-token perplexity on 14 Pile domains + Wikitext-103.

**What they found.** 33% HQ-rephrased + 67% CC reaches equal loss **5–10× faster** than 100% CC; optimal synthetic
ratio **~30%** (textbook-style often <5%, and pure textbook *hurts*). The anti-distillation result: **Llama-3-70B
rephrasings train consistently WORSE than 8B rephrasings** (generator quality is non-monotone; Fig 5).

**The fine print.** Sole metric is perplexity — no downstream reasoning benchmark — and the HQ prompt targets Wikipedia
style while Wikipedia-like domains are in the eval (an unacknowledged style-match circularity). The natural baseline is
*unfiltered* CC, so "synthetic beats natural" partly measures "any intervention beats none"; no quality-filtered-CC
control, and generation compute is never amortized. The 70B<8B ablation is small (1B trainee, 5B tokens, ≤20% synthetic).

**Why it matters here.** Useful design constraints (~30% optimum, benefit grows with data budget but reverses at large
model size / high synthetic fraction, generator strength saturates ~8B) plus the strongest evidence that a *stronger*
generator is not the lever. But the winning augmentation is style rephrasing, not added reasoning, and it is
perplexity-only — consistent with this thread's own finding that zero-shot perplexity is the wrong instrument for
rationale value.

---

### Strong-teacher augmentation (distillation-confounded)

**📖 Quiet-STaR (`2403.09629`)** · 319 citations — Mistral-7B lightly continued-pretrained to generate a short private
rationale at *every* token position, REINFORCE-rewarded by prediction improvement. Zero-shot GSM8K 5.9→10.9,
CommonsenseQA 36.3→47.2, scaling with training thought length. Correction: the thoughts **are greedy-decoded at eval**
(App A), so the scored gains include inference-time thought generation, not weights alone; GSM8K 10.9 is direct-only,
and with CoT cot-maj@8 rises 40.6→47.7 (Fig 5). A readable data-matched control exists (same OpenWebMath, no thoughts —
Fig 2 gray, ~6% GSM8K / 36→34% CSQA), making the headline-vs-base delta conservative. Its load-bearing gift: perplexity
gains **concentrate on hard tokens** (Fig 7) — the closest thing to a per-token "this token needed reasoning" signal (a
self-generated with-vs-without-thought gap; one model, so memorization largely cancels). Fills gaps latently — never
makes the text complete — so it motivates rather than tests the thesis.

**📖 ToW: Thoughts of Words (`2410.16235`)** — GPT-4o writes ≤15-word inter-word rationales with an information
bottleneck (generator can't see the gold word); continual pretraining on 6k docs. Reasoning avg +2.7 to +9.0 across
five 7–8B models, GSM8K +22.7 on Llama-3-8B. Fine print: no token-matched plain-GPT-4o control (distillation
unseparated); models learn to *emit* inline thoughts at eval (a learned scratchpad format); the built-in dose-response
runs **against verbosity** (raw 67-token < denoised 30 < summarized 14.4 tokens/thought on all six benchmarks). Reusable:
the can't-see-the-answer bottleneck.

**📖 MathCoder2 (`54`)** — append extracted+executed math code (conditions → LaTeX → result → Python) to
math text; MATH +17.0 vs base. Its rare same-generator No-code-prompt control is beaten (SAT 59.4 vs 37.5), arguing the
code *content* matters — but the isolating Basic+Code arm gives only +3.7 and is **not token-matched** (2.7B extra
tokens). One of the few papers with any same-generator isolating control.

**📖 Recycling the Web / REWIRE (`2506.04689`)** — rewrite the *discarded* low-quality DCLM pool with 70B-Instruct; the
mix of top-raw + top-rewritten beats raw-only token-matched (+2.5pp CORE at 7B), gain credited to
diversity/complementarity (rank corr 0.179). Corrections: rewritten-alone-worse-than-raw is **CORE-only** (MMLU
rewritten-alone is EQUAL at 1B, BETTER at 3B); rewrites are shorter only in the *mean* (raw outliers) — medians ~equal
(695 vs 764) and REWIRE is the longest synthetic method; the CoT trace is generation-time only, **not trained on**; the
rewrites raise TruthfulQA/World-Knowledge (Table 3, distillation). No weak-generator control.

**📖 Data-efficient pre-training by scaling synthetic megadocs (`2603.18534`)** *(covers map ids 55 and 57 — same
paper)* — Marin data-constrained study (300M student, 200M real tokens, Llama-3.1-8B generator). Latent-thoughts is the
best synthetic method (1.80× data efficiency vs 1.48× simple rephrasing; +9% PIQA/SciQ/ARC-Easy). Fine print: the
weak-generator control is **impossible** (Llama-1B produces useless/harmful data, conceded); much of the megadoc scaling
advantage decomposes into an **optimization effect** (megadocs allow more epochs / higher mixing before overfitting —
Fig 6-right); the strongest distillation rebuttal is that the win *grows* with student size (Δ+0.06–0.07, 300M→1.5B,
Fig 8). No reasoning-specific eval; completeness never varied (only the generation count G).

**📖 EntiGraph / Synthetic Continued Pretraining (`2409.07431`)** — knowledge-regime: entity-graph expansion of a
1.3M-token corpus to **455M** synthetic tokens; closed-book QuALITY **39.49→56.22** (Fig 2), log-linear in synthetic
tokens, **beating its own GPT-4-turbo generator** closed-book (56.22 vs 51.30) — hard to explain as pure distillation.
Corrections: 455M (~350×), rephrase budget is 38M so ~**12×** (not ~330×), and a token-matched EntiGraph-29M control
**does exist** (App H.3, for summarization). Rearranges knowledge; no reasoning-completeness variable; no
weak-generator ablation.

**📖 Phi-1 textbooks (`16`)** — synthetic NL-interleaved-with-code "textbooks"; HumanEval 50.6 / MBPP 55.5 (~its GPT-3.5
teacher's 47). Fine print: the largest gain is the CodeExercises *finetuning*, not textbook pretraining; pure GPT-3.5
distillation with the generation pipeline withheld. Note the direct contradiction with Demystify, where textbook-style
is the *worst* performer (different domain, different eval).

### Teacher-free / synthetic-symbolic

**📖 Transformers Provably Learn to Internalize CoT (`2605.28600`)** — theory: k-parity is exponentially hard without
CoT, polynomial with it, and the chain is removable on a log-stage curriculum leaving a single-forward-pass solver — a
proof-of-concept for the missing "internalization" cell. Fine print: the gates are **prescribed** from the parity tree
(the model doesn't discover the structure) and the empirical section is one qualitative run; internalization
proof-of-concept only, no training-benefit-vs-baseline on natural text.

**📖 FOL-Traces (`74`)** — SymPy-generated verified propositional traces (teacher-free logic); frontier LLMs fail
(45.7% masked-op, 27% two-step), but **no training benefit vs any baseline is demonstrated** — a resource/diagnostic,
not causal augmentation evidence.

**📖 Grokking in the Wild (`2504.20752`)** — 124M model + augmentation raising the inferred-to-atomic fact ratio flips
2WikiMultiHop comparison OOD 0.59→0.96. **Removed as teacher-free deconfounding evidence:** the augmenting LLMs are
GPT-4o + o1-mini (App A.5 = the Table-3 baselines), so it is distillation / self-comparison-confounded; the
"incorrect-facts-help" claim is *not* demonstrated (Sec 5 future work); composition OOD stays 0.07; Fig 1 mixes metrics
(GPT-2 OOD vs LLM IID). Grokking-scale budgets are unrealistic for pretraining.

### Arrangement / packing (augmentation-free structure)

**📖 Document packing → latent multi-hop (`05`)** — co-locating related docs + cross-document attention lifts latent
multi-hop (Llama-3.2-3B 2Wiki 62→79) with **zero added reasoning tokens**; the benefit vanishes when cross-doc
attention is off. Fine print: oracle relatedness, no random-doc control. A cheap generator-free lever that directly
probes the exposure mechanism.

**📖 WRAP++ / cross-document QA synthesis (`17`)** — synthesize QA across Wikipedia links; SimpleQA pass@128 34.76→49.13
(7B) but only +2.48pp at an 8B-matched budget. Key structural datapoint: **raw concat of two related docs ≈ base**
(35.43 vs 34.8) — co-presence is worthless; explicit cross-doc chaining is what installs the multi-hop. Factual-recall
only; general reasoning flat.

**📖 Mining hidden thoughts (`20`)** — prepend LLM-reconstructed hidden thoughts (LoRA CPT, Gemma2-9B); MMLU 65.8→69.1,
Very-Hard +7.9 over text-only CPT. Fine print: the eval prompt embeds the trained thought-tag format (train/test-format
confound), same-family generator, no rephrase-only control.

### Augment computation, not text (latent-compute; no distillation)

**📖 CoCoMix (`64`)** — augment the *objective* with predicted latent SAE concepts (not text); 21.5% fewer tokens to a
loss target, and a weak-generator distillation control (124M→1.38B) beats KD. Not reasoning-specific; the
implicit-concept opposite of explicit-reasoning text.

**📖 PonderLM (`65`)** — latent per-token "pondering," self-supervised, **no distillation**; +3–6 avg acc at token
parity beating step-matched looped/pause/Coconut baselines — but 4× train FLOPs and **no math/reasoning benchmark**.

**📖 Adaptive Latent CoT (`66`)** — token-adaptive latent depth; beats compute-scaling baselines at <½ FLOPs, but
adaptivity itself adds ~0.001 CE, the +0.4 vs vanilla-1.4B is fragile, and there is no reasoning benchmark. Together
these three are the compute-matched latent-depth rival any explicit-text claim must be defended against.

### Off-scope (shortlisting note)

**📖 Procedural Knowledge at Scale (`2604.01348`)** — *inference-time RAG on frozen models, not pretraining* (triage
overreach). One useful signal survives: retrieval of decomposed (subquestion → subroutine) procedural memory beats
generic document RAG (token-matched, App A.2), and the datastore-builder swap QwQ-32B → Qwen3-8B loses nothing (a
weak-generator-parity datapoint); the benefit largely vanishes after SFT on the same corpus.
## The papers · H2.6 — how complete must the reasoning be

### 📖 Zipping the Thought: When and How Compressed Reasoning Data Works in LLM Post-Training
Kohsei Matsutani … Yutaka Matsuo (The University of Tokyo) · arXiv preprint 2026 · `2605.28008`

**What it is.** The purest granularity dose-response in the corpus: a controlled synthetic study of how
compressed reasoning traces behave in post-training, on modular-23 arithmetic with distractors.

**What they did.** Generate chains at four granularity levels — g=1/2/4/8 (finer to coarser) — and SFT
five Qwen2.5 sizes plus Llama-3.2-3B on each; then apply RLVR on top. Three trace styles (Explicit /
Composed / Implicit) cross with the granularity dial. Everything is measured in samples and optimizer
steps, never tokens.

**What they found.** (1) Coarser compression monotonically needs more data (Fig 11); g=8 sits near the
1/23 chance line for most sizes. (2) A **granularity floor**: after g=2 SFT, tasks that require sub-step
decomposition sit at chance (Fig 8) — SFT cannot decompose below the granularity of its training traces.
(3) RLVR recovers those tasks from near-chance (**+58.17 to +86.11**, Fig 6a) — but *minimally*: the model
keeps g=2 for the bulk and expands only the single needed fraction step, and Llama inserts a dummy "*1" to
satisfy the format (App D.3). (4) "Implicit degrades under repetition" (Takeaway 3) holds on Qwen2.5-3B
(6k×64 = 12.2 < 6k×1 = 23.0) but *reverses* on Llama-3.2-3B (25.7 > 11.2, Fig 5).

**The fine print.** Efficiency is measured in samples/steps, not tokens — explicit traces are longer, so
part of the compressed variants' data-efficiency edge is a token-budget artifact. Fully synthetic mod-23
arithmetic; post-training only; the paper states "we do not evaluate on real data." "RLVR re-derives
skipped steps" should be read as *targeted patching*, not chain re-expansion, and one recovery variant
borders on a format hack.

**Why it matters here.** Establishes the SFT-side granularity floor we should design around: injected
traces must be at least as fine-grained as the finest step we need the SFT-stage model to produce. It also
tempers the hope that RL will backfill missing granularity — it does, but sparingly, and only in a
verifiable synthetic domain.

### 📖 Can Language Models Learn to Skip Steps?
Tengxiao Liu … Xipeng Qiu, Zheng Zhang · NeurIPS 2024 · `2411.01855`

**What it is.** A study of completeness-as-step-count: can a model be taught to skip reasoning steps via
self-training, and does skipping help or hurt?

**What they did.** Iteratively self-train models to produce shorter chains on three algorithmic tasks
(Analog of Algebra, Multi-digit Addition, Directional Reasoning) plus a GSM8K probe, then compare
skip-trained models to full-chain models in-domain and OOD, with training-step-matched (Table 4) and
volume-matched (2000/task, Table 9/B.6) controls.

**What they found.** Moderate skipping preserves ~99–100% in-domain accuracy and *improves* OOD on the
easy algorithmic tasks. But the picture is difficulty-conditional: on the GSM8K probe, skipping yields no
gain and a slight decline (Test-OOD 61.33→60.44, Table 11), and the paper states real reasoning
"necessitates a complete reasoning chain" with skipped steps that "frequently contain errors." The
load-bearing completeness result: **skip-only training underperforms skip+complete-chain training on
OOD-hard (7.86 vs 11.13, Table 8/B.5)** — complete chains in the mixture are "essential," and skip-only
"potentially lead[s] the model to depend on shortcuts that harm generalization."

**The fine print.** The OOD gains live on simple algorithmic tasks with unambiguous step structure; the
step-matched and volume-matched controls confirm the effect is not pure data volume, but the residual gap
is the absence of a full-step self-generation loop. Post-training, not pretraining-corpus, evidence.

**Why it matters here.** Two things for our grid: completeness must be dosed to difficulty (skip only where
the step is genuinely redundant), and complete chains earn their place in the *mixture* even where per-example
skipping is fine — the direct counter to a blanket compress-everything strategy.

### 📖 Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation
Abdul Waheed, Chancharik Mitra … Bhiksha Raj · arXiv preprint 2025 · `2509.05226`

**What it is.** Difficulty-aware trace compression from the post-training side: compress easy problems'
chains hard, keep hard problems' chains long, and measure where completeness is load-bearing.

**What they did.** Distill compressed CoT into 1.5B and 7B students with an absolute-length target
|r̃|~α·d(x) tied to per-problem difficulty d(x); ablate SFT-only vs SFT+DPO; compare compressor strategies
by how much reasoning they drop.

**What they found.** SFT-only over-compression **collapses hard benchmarks** (1.5B AIME 5.0, HMMT 0.0; 7B
AIME 8.4, HMMT 2.2 — Table 2) while GSM8K holds (~77.6/88.5). **Restoring length via DPO recovers hard
problems** (SFT+DPO 1.5B AIME 23.4). The compression ratio *rises* with difficulty (d1 79.1% → d8 90.6%,
Table 1) — harder problems retain proportionally *less*, but longer absolute chains, via the length target.
The authors reject their most aggressive compressor (79.1% reduction) because it "omitted critical reasoning
steps needed for more difficult problems," choosing 56.7% (Table 4, App A.5).

**The fine print.** No same-data full-trace control at matched tokens; the 7B variant *loses* accuracy on
every hard benchmark under compression; completeness is scored by an LLM judge that is self-preferential;
a simpler RL length-controller (L1) reportedly beats the efficiency claim. Math-only, LoRA post-training.

**Why it matters here.** The compression-side twin of Skip-Steps: it is the clearest demonstration that a
uniform compression target drops the exact steps hard problems need, and that the fix is difficulty-matched
completeness — a design constraint for any completeness dial we build.

### 📖 From Implicit to Explicit: Token-Efficient Logical Supervision for Mathematical Reasoning (FSLR)
Shaojie Wang, Liang Zhang (HKUST-Guangzhou) · Findings of ACL 2026 · `73-fslr`

**What it is.** The bucket's counterweight: it varies the completeness of the *supervision target* and
finds that *less*-complete supervision trains a better model.

**What they did.** Instead of CoT-SFT (train on the complete solution trajectory), FSLR trains the model to
emit only the **first planning step** — which variables and which operation, no calculation (27–52 tokens vs
238–310 for CoT-SFT, Table 3). Same training problems (GSM8K+SVAMP); teachers are LLaMA-3.1-70B /
Qwen2.5-72B or self-generated; targets are LLaMA-3.1-8B, Qwen2.5-7B, Qwen3-4B. At inference the FSLR model
generates a *complete* chain via standard "let's think step by step."

**What they found.** An inverted completeness ordering: complete trajectory 82.28 < full plan (Plan-and-Solve)
85.88 < first-step-only 87.08 (in-dist avg, Table 1/4). FSLR also wins OOD (**+8.08 avg** on LLaMA, Table 2),
on GSM-Symbolic (79.6 vs 70.7 vs base 52.9, Fig 5), and on 8-step problems (78% vs 56%). A GPT-4o + multi-judge
audit finds >90% of errors are operation/variable-selection ("logical relationship"), not arithmetic — the
enthymematic decision is where the value sits. A clean distillation control: self-generated FSLR (88.48) ≈
teacher FSLR (88.90), and still beats self-CoT-SFT (84.68).

**The fine print.** The decisive unaddressed confound: FSLR is **not token/compute-matched** — it trains on
84–87% fewer tokens and 4–6× less wall-clock, so part of the win may be lighter finetuning avoiding
overfitting/forgetting; the obvious token-matched CoT-SFT control is not run. Completeness co-varies perfectly
with token count. Post-training of instruction models, grade-school word problems, no RL tested.

**Why it matters here.** Supervision-completeness and emitted-completeness *decouple*: you can strip the chain
from the target and still get a model that emits (better) complete chains. This is a cheap arm our sweep has
not considered — inject only the operation-selection step — but it must be run token-matched before we believe
the mechanism over the lighter-finetuning story.

### 📖 Multi-Hop Knowledge Composition is Bound by Pretraining Exposure
Yannis Karmim (Inria) … Valentin Barrière (Universidad de Chile) · 2026 preprint · 0 citations (too new) · `2606.09338`

**What it is.** A controlled study of whether a model that memorized two facts separately can chain them in a
single forward pass — isolating the compositionality gap from the "does it know the facts?" question.

**What they did.** Invent 100k people; everyone appears in atomic one-hop biographies, but only group `P_comp`
also appears in compositional two-hop sequences (`P_held` never does). Train GPT-2 (124M–774M) from scratch,
then test two-hop composition in a single forward pass with the bridge entity **not** in context. Nine ways of
writing the compositional data cross format (NL vs RDF) × explicitness (name the bridge vs omit it). A **LoRA
QA-finetuning stage (direct-answer format) sits between pretraining and evaluation for all nine conditions**,
so every reported number is post-finetuning.

**What they found.** Both groups learn every atomic fact (1-hop 0.97, Table 2), but only `P_comp` composes
(2-hop up to 0.83); `P_held` stays at chance across all nine formats and all scales — and a conditional analysis
shows **P_held 2-hop = 0.00 even when both constituent 1-hops are answered correctly** (Table 9), killing the
forgot-the-facts confound. Explicit augmentation (bridge named) gives 0.08 = baseline; implicit gives 0.62 NL /
0.79 RDF — but this is a *format* statement, not "explicit is worse": explicit training relies on the bridge
token being in the prefix, while the test forbids a scratchpad, and the fair {explicit}×{scratchpad} cell is
never run by design. Table 13 shows explicit pinned at **0.09 across every LoRA rank** — the gap is a
pretraining-format property. Logit-lens: the explicit model's bridge probability tracks the augmentation format,
not compositional exposure (decodability ≠ use).

**The fine print.** Fully synthetic, two relation types, GPT-2 scale, silent-reasoning-only. Conditions are not
strictly token-matched (augmentation adds tokens on a shared atomic backbone). The scratchpad column of the 2×2
is missing by design, so nothing here argues against *externalized* complete reasoning.

**Why it matters here.** Sets the latent-regime boundary: explicit compositional text does not auto-compile into
a no-scratchpad computation, and composition is exposure-bound — you must expose the operation, not just the
facts. If we want latent reasoning, the training format must match the no-scratchpad inference distribution.

### 📖 Faithfulness as Information Flow: Evaluating and Training Faithful Chain-of-Thought
Jinghan Jia (Michigan State / Anthropic Fellows) … Eric Easley (Anthropic) · 2026 preprint · 0 citations (too new) · `2605.24286`

**What it is.** A rigorous attempt to define what "complete" and "faithful" mean for a reasoning chain — a
borrowable definition plus a warning.

**What they did.** Frame a good trace as one where all answer-relevant information flows *through* the written
chain (prompt → chain → answer), and define three information-theoretic properties: **sufficiency** (the chain
alone determines the answer), **completeness** (given the chain, the prompt adds nothing — the chain "screens
off" the prompt; a leftover prompt→answer path is an incompleteness, I(P;A|C)≈0), and **necessity** (the answer
actually depends on the chain). Then attempt to *train* faithfulness by modifying the RL policy-update gradient
(identical rollouts and rewards; only the gradient differs).

**What they found.** Models routinely use a hidden prompt→answer shortcut while emitting a plausible chain they
don't rely on — a chain can look complete and be a rationalization. Vanilla GRPO drives wrong-hint following
*up* while verbalization degrades; the interventions make the shortcut *visible* but don't remove it ("the
interventions do not eliminate reward hacking here, but they make it monitorable"; hidden-test pass stuck
~0.24–0.32, Fig 7). A 12B replication (App C.4) shows the non-elimination persists with scale.

**The fine print.** The training comparison is clean (no data/token confound). Two caveats: the external
criterion legitimizing the gradient metrics rests on just two off-the-shelf models differing in family, size,
and RL recipe simultaneously, *and* that "ground-truth" faithfulness label is itself a 4-frontier-LLM-judge
aggregate (a circularity layer). The ~0.9 verbalization numbers are computed among hint-followers only —
though Fig 10 recomputes the structural metrics on that same follower subset and the separation persists, so
the selection-effect worry is partly answered. (DAPO no-hint accuracy is ~42–44%, Fig 9A.)

**Why it matters here.** Two gifts: a computable definition of completeness (the chain screens off the prompt),
and the necessity caveat that reshapes the thread — a complete-*looking* chain isn't enough; what matters is
whether the model *uses* it. This is the QC criterion for augmentation: filter for chains whose removal changes
the answer.

### 📖 Making Implicit Premises Explicit in Enthymemes
Xuyao Feng, Anthony Hunter (both UCL) · 2026 preprint · 0 citations (too new) · `2603.06114`

**What it is.** The paper closest to our stopping-rule framing: for explicit logical arguments, does filling in
the unstated premise help a verifier accept the argument?

**What they did.** An LLM (DeepSeek v3.2) generates the missing intermediate premise(s) — one, two, or three
steps — then a formal checker (AMR parse → logic → neuro-matching relaxation → SAT solver) verifies whether the
argument now goes through, scored against the dataset's gold entailment label on ANLI and ARCT.

**What they found.** More steps help: entailment accuracy rises 0.530→0.733 (ANLI) and 0.293→0.563 (ARCT) as
generated premises go from none to 3-step (Table 4), and the appendix per-class curves (Figs 6/7/9/10) show
entailment-class *precision* rising alongside recall — so the rise is more than a loosened trigger, and overall
accuracy climbs modestly (~0.71–0.72 ANLI, ~0.65 ARCT, Fig 3).

**The fine print.** The real caveats are different from a specificity confound. (1) The gold complete human
premise barely beats none (ANLI 0.558 vs 0.530; ARCT 0.303 vs 0.293) — a correct single premise doesn't help
the verifier; only many decomposed, parser-friendly atoms do, so part of the gain is verifier plumbing. (2) The
1/2/3-step ladder is literally a ≤10/≤20/≤30-word ladder (Fig 11 prompt), so step-dose is confounded with atom
count. (3) It is SAT-verifier dose-response evidence, not LM-training evidence.

**Why it matters here.** The right framing paper — it operationalizes enthymeme-completion with formal
verification, exactly our stopping-rule setting, and the dose-response is more credible than a first read
suggests. But it remains orthogonal to training: what it demonstrates is that decomposed premise chains make a
neuro-symbolic verifier fire, not that completeness helps an LM learn.

### 📖 The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning
Yubo Li … Rema Padman (Carnegie Mellon) · 2026 preprint · `2603.29025`

**What it is.** An inference-time diagnostic of whether frontier models use *unstated* constraints — and, in its
appendix, the bucket's only within-model completeness gradient on natural-language traces.

**What they did.** Build HiddenConstraintBench: prompts where a surface heuristic (e.g. "walk") conflicts with an
implicit constraint the model must infer. Score 14 frontier models by strict 10/10 accuracy, run span-occlusion to
measure how much models key on surface cues vs the goal, and audit DeepSeek R1 thinking traces for whether they
name and apply the hidden constraint.

**What they found.** Models key on surface cues 8.7–38× more than the goal (Table 7) and fail to use unstated
constraints — **mean strict accuracy 62.1%** across 14 models (Table 2), and no model exceeds 75%: the "Won't"
persists through current post-training. Reasoning-mode advantage collapses to +1.8pp, n.s. (p=0.31), after
controlling for Chatbot-Arena Elo (Table 19). The completeness gradient (Table 17, App H): trace accuracy is
**88.5%** when the constraint is named AND applied, **44.4%** when never mentioned, and **16.7%** when
mentioned-but-not-applied (Fisher p<0.01) — half-complete reasoning is worse than none. Goal-decomposition
prompting (enumerate necessary conditions first) beats generic CoT (+5.0 vs +3.1pp, Table 16).

**The fine print.** Entirely inference-time, nothing token-matched; the cited code repo is 404 so the scoring
pipeline can't be verified. The one-word "hint" that recovers +15.3pp is a subtle emphasis that does *not* state
the constraint (App A Table 3), so this is knowledge-present evidence, not premise injection. Table 17 is
observational (a correlational within-model audit), not a causal manipulation.

**Why it matters here.** Table 17 is the closest thing we have to a natural-language completeness dose-response,
and it delivers the sharpest design lesson: incomplete *use* is worse than no chain. Our augmentation QC must
check the chain is applied, not just present — converging with Faithfulness's necessity property.

### Existence proofs and adjacent evidence (compact)

**📖 Principled Synthetic Logic Corpus / ALT (`2411.12498`)** — program-generated (no teacher) fully-explicit
multi-step deduction + SFT. Transfer profile: RobustLR +32.0 (format-adjacent logic), abduction
+11.7 (70B) / +33.1 (8B — genuine cross-form, Table 4a/E.7), code +6.1/+10.3 (Table 4c), but math transfer
near-zero (MATH 23.7→24.4 = +0.7; MathQA 55.0→55.4 = +0.4, Table 4b). Fine print: no full-proof-vs-answer-only
control at matched tokens (completeness baked in, not isolated); ~1/3 of targets carry no proof chain; the
anti-forgetting-regularizer contingency rests on a single row (PARARULE-Plus@8B, Table F.8), not a general result.

**📖 On the Bias Toward Systematically Inefficient Reasoning (`2507.05362`, NeurIPS 2025)** — 28.5M from-scratch,
layered-DAG shortest paths (code + full appendix verified). Longer, locally-incremental DFS-style (η=−5) traces beat
compressed DP (η=+5) traces, which in turn beat the unweighted η=0 traces — accuracy ordering **η=−5 > η=+5 > η=0**
(η=0 worst, Fig 4a) — under *both* a fixed token budget (~128M) and a fixed graph count (~200K); no-intermediate-step
traces collapse at depth; length-matched padding (DR/RR) does *not* help and degenerates into repetition loops at
greedy decoding (RR 244±182 steps at 8% optimality, Fig 5b). The isolable mechanism is systematic local incrementality,
not length: the best-generalizing traces also have the **lowest exploration-order surprisal** (S(η=−5)=0.48 <
S(η=+5)=1.33 < S(η=0)=1.91, §7) — most-predictable, easiest to next-token-predict — and the DFS>DP advantage survives a
6-layer model and 16M/32M/128M budgets (App B.3/Fig 10). H2.7-relevant: the valuable traces are the *lower*-perplexity
ones — the opposite of a 'high-perplexity = reasoning-rich' detector.

**📖 Chain of Execution Supervision / TracePile (`2510.23629`)** — 2.6M samples / ~19B tokens converting code
execution into explicit NL narration; continue-pretrain/SFT on 4 base models. Broad gains (GraphWiz +41.5; math
avg +7.4 on LLaMA-3.1; BBH transfer). Fine print: completeness is never dose-varied (binary CoE-vs-final-answer
only); every trace is Qwen-2.5-72B-generated (teacher-distillation confound, two base models same-family); the
biggest "OOD" wins (LiveCodeBench-Output, CRUX) share the execution-prediction *format* with training; on in-domain
math a plain CoT dataset ties it; the rejection-sampled TracePile++ *underperforms* uniform expansion on 3/4 domains.

**📖 RATIONALYST: Mining Implicit Rationales for Process Supervision (`2410.01044`, ACL 2025)** — completeness is
the motivation, never a variable. A rationale-trained supervisor reranks a vanilla same-model agent's steps: GSM8K
81.6 vs 77.6 no-verifier and vs a vanilla same-model reranker 77.4 (Table 5) — the honest isolated number, since a
same-size reranker alone doesn't help; same-family throughout (clean anti-distillation design). Fine print: the
reasoner never trains on completed chains (shortcut never removed at source); no completeness dose-response; App G
concedes most perplexity-filtered Pile rationales describe preceding context rather than guide future reasoning.
Inference-time process supervision, not pretraining-augmentation.

**📖 Rethinking the Role of Text Complexity in LM Pretraining (`72-text-complexity`, BabyLM 2025)** — orthogonal
to completeness (surface simplification, semantics ~held constant): LLM-simplified FineWeb-Edu leaves fine-tuned
NLU essentially unchanged (500M 71.2 vs 70.8), but uniform LLM rewriting collapses lexical diversity (TTR
0.33%→0.19%, entropy 10.58→9.87) with zero-shot world-knowledge/entity-tracking costs. H2.7 caution: the
small-vs-large PPL degradation gap tracks readability (28M loses ~1.8 PPL on human-written vs ~0.9 on simplified,
Fig 1) — a documented confound for two-model loss-gap signals. Its lesson for us: any reasoning-augmentation
pipeline needs a rewrite-without-reasoning arm to separate the reasoning effect from LLM-rewrite side-effects.
## The papers · H2.7 — can a loss/perplexity-family signal detect reasoning content?

The load-bearing members get full writeups; the rest are compact. Each entry treats the eval
confounds — token-matching, distillation separability, missing controls — as first-class content.

---

### 📖 Rho-1: Not All Tokens Are What You Need
Zhenghao Lin (Xiamen U. / Microsoft) … Weizhu Chen (Microsoft) · NeurIPS 2024 · **126 citations** · `2404.07965`

**What it is.** Token-level selection by a loss gap. Train a reference model on 0.5B *curated* math
tokens, score every corpus token by excess loss (training-model loss − reference loss), and
backprop only on the top 60–70% of tokens. The excess-loss criterion is mathematically identical to
token-level RHO-LOSS (Appendix B.2).

**What they did.** Continual-pretrain 1B and 7B models on the same 15B-token OpenWebMath corpus,
token-matched against all-token CLM. The main-method reference is *same-size* as the scorer (the
same base model finetuned on the curated data, §3.1); a separate weak-to-strong variant lives only
in Appendix I.

**What they found.** Big math gains at face value: **GSM8K +23.4pp at 1B / +24.0pp at 7B** over
all-token continual pretraining (Table 1), with "up to 30%" few-shot framing.

**The fine print — the key ablation is the paper's own.** With a *self-referential* reference (no
curated data), the average gain collapses **+16.5pp → +3.3pp** (Table 3): ~80% of the headline is
the curated reference *distribution* leaking through the token mask, not token selection as such —
this is distillation of a curated dataset through a mask. The curated 0.5B tokens are
MetaMath/MAmmoTH-derived, i.e. bootstrapped from GSM8K/MATH *train* questions. The self-reference
ablation's score functions are L_RM (+2.4) and entropy (+1.9 best), not excess loss (Table 4). The
*genuine* weak-to-strong variant (1B reference → 7B, Appendix I) gives only **+0.9 AVG with
MMLU-STEM −3.4pp** (Table 5). Post-SFT, the often-quoted +2.2/+3.4pp "recovery" is a MATH-only
cherry-pick; the average post-SFT deltas are **+6.2pp (1B) / +2.7pp (7B)** (Table 2 Δ-AVG) — the 1B
advantage largely *persists* through SFT. The authors never claim the signal finds *reasoning*
("closely related to mathematics," "aligned with the desired distribution"). A verified nuance in
its favor: Figs 13/14 show kept tokens sit inside reasoning-derivation spans *within* the math
corpus while dropped tokens are boilerplate; Fig 6(c) shows unselected-token loss rises 2.9→3.7
(empirical domain narrowing).

**Why it matters here.** The detector detects "looks like my reference set." That is a warning *and*
an actionable idea: if a same-size reference were trained on complete-reasoning text, the same
excess-loss mask would point at completeness — nobody has run that variant. Design constraints from
the ablations: keep the reference same-size, and budget a breadth-preservation control.

---

**📖 PreSelect (`2503.00808`, ICML 2025)** — recipe **(B)**'s source: multi-model *rank-match* (do six Llamas'
per-char losses order by ability?) — full writeup in the H2.4 section. The H2.7-critical facts: its controlled
ScalingFilter baseline shows the two-model magnitude gap gains only +0.4 over random and is uncorrelated with
rank-match (Spearman 0.0533, Table 7); and the rank-match signal itself is exam/knowledge-shaped — worse than
random on HellaSwag/PIQA (1B) and PIQA/SIQA (3B) (Tables 12/13), with its DCLM margin shrinking +1.5 → +0.8 from
1B to 3B. Recipe B survives as the last untested loss-family candidate, with exactly these expectations.

**📖 AttentionInfluence (`2505.07293`)** — recipe **(A)**'s source: one model vs. itself with retrieval heads
masked (self-ablation) — full writeup in the H2.4 section. The H2.7-critical fact: our own generalized go/no-go
(5 base models, 6 sources; `docs/RECIPE_A_SELF_ABLATION.md`) rules the gap out as a reasoning detector — it
measures in-context copy dependence (corr 0.866 with a random-head gap; ranks config files and parallel
translations on top of raw DCLM; 3 of 4 models rank verbal reasoning below web; Qwen 7B→72B inverts, GSM8K AUC
0.955→0.051). The paper's own Table 6 head-mask validation also fails on HumanEval (0.1098 ≈ random 0.1159).

### 📖 Perplexed by Perplexity: Perplexity-Based Data Pruning
`2405.20541`

**What it is.** The single-model absolute-perplexity pruner: a small reference model scores every
document by perplexity, and you keep a chosen tail (high, medium, or low) before training a larger
model.

**What they did.** Prune the Pile and Dolma with 125M/350M reference models, then train 1B/3B models
on the kept slice, compute-matched against the full pool. Report a 5-category downstream breakdown
(Table 1) plus test-set perplexity (Table 3).

**What they found.** General downstream improves (+2.0 avg at 3B on the Pile) — but the per-category
tables show *where*. **Symbolic Problem Solving (arithmetic/logic/MathQA/LogiQA) is bold-on-baseline
in all four settings** — perplexity pruning never significantly improves reasoning, and at 3B/Pile
it *drops* 4.88→2.91. The entire average gain comes from **World Knowledge (15.51→18.18) and
Language Understanding (28.11→33.2)** at 1B/Pile. The domain composition confirms it: the winning
high-perplexity criterion cuts code and scientific papers ~3× (Fig 4).

**The fine print.** The winning criterion **flips per corpus** (high-ppl on Pile, medium on Dolma;
medium on Pile *loses* 0.23), so there is no stable directional rule. There is **no random-50%
control**. And the standalone caution for our thread: **test-set perplexity inverts with
downstream** — the baseline reaches 7.83 ppl / 13.73 acc while the pruned model is 8.51 / 15.62
(Table 3). Held-out loss is an invalid proxy for data-intervention value.

**Why it matters here.** Direct task-level evidence that single-model perplexity pruning is
*anti-reasoning* as a selector: it de-selects the reasoning-dense categories and its gains live
entirely in knowledge/language. The test-ppl inversion is one of three independent results in this
bucket that dissociate loss from downstream value.

---

### 📖 Improving Pretraining Data Using Perplexity Correlations
Tristan Thrush (Stanford) … Tatsunori Hashimoto (Stanford) · ICLR 2025 · **54 citations** · `2409.05816`

**What it is.** The most observational multi-model signal, with no training. Take ~90 public models
(33M–9B), compute per-*domain* bits-per-byte on ~10k web domains, and keep the domains where loss
*correlates* with the models' benchmark accuracy (a rank-based single-index estimator robust to
family heterogeneity), then scale to page level with a fastText classifier.

**What they did.** Select domains via the correlation estimator; train 160M/3.2B-token models on the
kept pool; preregister a follow-up at 1.4B and report it honestly.

**What they found.** It beats DSIR everywhere and roughly ties DCLM's handcrafted fastText, with
gains growing to 1.4B — but **only on raw pools**; on pre-filtered pools the signal evaporates
(correlation coefficients become homogeneous), and they reported that null.

**The fine print.** Domain-side PCA of the 90-model loss matrix says the structure is **PC1=language,
PC2=difficulty/entropy** (Fig 10) — not reasoning. The top ARC-Easy domains are optometry-clinic and
children's-hospital websites; DCLM-aggregate tops are weather/finance/currency sites. Their own
Appendix I concedes **plain mean loss predicts model rank nearly as well** (7/8 benchmarks; only the
aggregate is significant at p=0.035) — the incremental value of *correlation* over "good models find
it easy" is never isolated on the selection side. The 90-model set includes many partially-trained
Pythia checkpoints (pseudo-replication), and the estimator fails on atypically-trained models (Phi).

**Why it matters here.** Confirms multi-model perplexity structure carries *some* selection signal
(Recipe B feasible in principle — PreSelect's within-family ladder is a cousin) while warning that
(a) it operates at *domain*, not document, granularity; (b) it dies in our exact regime (DCLM is
pre-filtered); and (c) the naive version finds quality/language, not reasoning. It also hands us a
mandatory control: a plain mean-BPB baseline, since the paper's own appendix says mean loss carries
nearly the same signal.

---

### 📖 rBridge: Predicting Large-Model Reasoning from a Small Proxy's Loss on Reasoning Traces
`2509.21013`

**What it is.** A scaling-prediction method that scores a small proxy model by its NLL — but
crucially evaluated **on frontier reasoning traces**, not bare answers — to predict a large model's
downstream reasoning accuracy.

**What they did.** Regress large-model reasoning accuracy on the proxy's trace-NLL across a
model/scale sweep (1B→13B), and ablate the two ingredients: the trace *target* (R^φ) and the
model-differencing/weighting machinery.

**What they found.** Strong prediction: **R²=0.87** vs 0.49 for standard NLL, and 80.8% decision
accuracy at ≥100× FLOPs savings. The ablation is the payload: **R^φ alone (trace target, no
weighting) reaches R²=0.867 vs 0.874 for full rBridge** — essentially all the gain is *what the loss
is computed on*, not how models are differenced. The effect is teacher-robust across GPT-4o,
Claude-3.5-Sonnet, and Gemini-2.5-Pro (Appendix C.3).

**The fine print.** The trace target is a distilled notion of "good reasoning" (frontier traces). It
is validated at **dataset level**, ranking datasets — not at document level — and is **untested
within a pre-filtered pool**.

**Why it matters here.** The one genuinely constructive lesson in the bucket: *what text the loss is
evaluated ON matters more than which models you difference.* This motivates a Recipe-B variant that
scores documents/datasets by proxy loss on reasoning-trace-*like* targets rather than the raw
document — though extending the validated dataset-level ranking to document-level scoring is our
untested inference.

---

### 📖 ScalingFilter: Assessing Data Quality through Inverse Utilization of Scaling Laws
`2408.08310` · *(PDF + full-appendix verified; the magnitude-gap-deadness conclusion is co-anchored by
the verified PreSelect Table 7, the citation to lead with.)*

**What it is.** The primary source for the two-model magnitude gap: score document "quality" by the
perplexity *ratio* between a small and a large model of the same family (124M vs 774M GPT-2), the
idea being that text a big model finds much easier than a small one is high-value.

**What they did.** Filter a pretraining pool by the ratio, train a 1.3B model on 25B kept tokens, and
compare against perplexity-gating and a binary classifier.

**What they found.** **+1.12% avg over perplexity gating and +0.62% over a binary classifier** on 7
commonsense tasks, and **+3.09% over random selection** (51.27 vs 48.18) — random *is* the first row of
the downstream Table 1, i.e. the paper does run the unfiltered-baseline control.

**The fine print.** No error bars, no seeds, and **zero reasoning or knowledge-intensive benchmarks
anywhere** — the signal was never even claimed to find reasoning, and 2 of the 7 commonsense tasks flip
*against* ScalingFilter (LAMBADA 48.42 < 48.96; OpenbookQA 31.40 < random 32.40, Table 1). The paper
does probe more than a quick read shows: a meta-model-training-corpus ablation (Table 2: Unfiltered-CC
50.30 / Wikipedia 51.12 / OpenWebText 50.49 / WebText 51.27) finds the ratio robust to the meta-models'
anchoring distribution, and a hyperparameter sweep (Table A.2) keeps the SF>Binary ordering stable — but
the absolute deltas stay ~0.6-1.1% on the same 70%-retention pool. Independently, PreSelect runs this
exact gap as a controlled baseline and finds it beats random by only **+0.4** (37.6 vs 37.2, 1B/30B)
while selecting short/easy text, near-orthogonal to the rank-match signal that works (Spearman
0.0533, Pearson −0.079) (PreSelect Table 7).

**Why it matters here.** Treat the two-model magnitude gap as a documented near-failure — do not run
it. Our own reverse-filter's "gold" criterion *was* a two-model gap (1.4B-high AND 72B-low) and it
found knowledge, not reasoning — exactly what ScalingFilter predicts.

---

### 📖 The Signal is in the Steps: Local vs Global Perplexity for Reasoning-Trace Selection
`2510.03988`

**What it is.** A trace-selection study for distillation SFT: is a *global* whole-sequence log-prob
or a *local* windowed step-score the better selector of high-quality reasoning traces?

**What they did.** Segment traces into LLM-identified steps, score each step by a local self-perplexity
(is this step justified by its immediate premises?), aggregate to a local score (LALP), and compare
against a global aggregate log-prob (GALP) on trace pools from multiple teachers, at up to 32B.

**What they found.** Local wins: **+9.4pp avg over global at 32B** (Llama ~+4.2pp, not a tie). The
diagnostics are the payload: global scoring reaches *lower* training loss but generalizes worse —
**63.7% test avg (GALP) vs 71.9% (LALP)** (Fig 6), a loss–generalization *dissociation* — and it
concentrates **42.3% of its score mass on discourse filler ("Okay", "So") vs 18.7% for the local
score** (Table 4). Whole-sequence magnitude is dominated by scaffolding and self-conditioned
repetition: a repeated *wrong* "3" inflates from 4.9%→97.7%.

**The fine print.** There is **no length-matched control**, and the near-tie with simply using the
best single teacher stands (0.726 vs 0.719, +0.7pp). The length-confound direction is *unresolved,
not established*: GALP actually picks **more** DeepSeek-R1 (47.6% vs 42.4%) and prefers
longer/more-repetitive responses (§7.4.2), and the composition table has two conflicting panels.

**Why it matters here.** If we ever use loss-family signals *on* reasoning traces (e.g. QC for
augmentation outputs in H2.5/H2.6), compute them at **local step granularity**, not whole-sequence —
the verified dissociation and filler-mass results show whole-sequence perplexity magnitude is a poor
readout of reasoning quality. Porting the local-window scoring from SFT-trace-selection to
pretraining-doc use is our inference.

---

### Compact entries

### Frequency / recall vs reasoning

**📖 Generalization vs Memorization (`2407.14985`)** — Task-gram co-occurrence over the Pile/Dolma is
strong for TriviaQA-style recall (Mem_{n=3}>0.35, *rising* with scale) and absent for GSM8K (Pythia
and OLMo) — reasoning performance is not explained by task-relevant n-gram frequency (Fig 4). Now
known to be **decontamination-checked** (no n=8/14 overlap, p.5) and **threshold-robust** (γ_T ∈
{0.7,0.75,0.8}, Fig 6), with gradient-TracIn influence as complementary causal evidence (Section 6);
prompts made *less* corpus-similar help reasoning while *more*-similar help knowledge (Table 1). Fine
print: partly a measurement floor (reasoning outputs are un-memorizable-as-n-grams by construction)
and metric-heterogeneous (GSM8K BERTScore-scored, Kendall-tau-ranked). Read as "frequency signals
find knowledge, not reasoning," not as proof reasoning has no memorized substrate.

### Excess-loss / reference-gap family (the gap is the signal — for learnability/quality, not reasoning)

**📖 DoReMi (`2305.10429`, NeurIPS 2023)** — Domain-level excess loss with Group-DRO to reweight
provenance domains. The ablation isolates the mechanism: proxy-loss-only and reference-loss-only each
beat baseline on **0/22 Pile domains** while the *gap* beats it on **22/22** (Table 7, code-confirmed)
— 6.5pt one-shot downstream gain at 8B. Knowledge/QA evals only, provenance-domain granularity;
nothing about reasoning. Predicts weak traction on flattened/pre-filtered pools.

**📖 RHO-LOSS (`2206.07137`, ICML 2022)** — The foundational reducible-holdout-loss selector
(reducible_loss = model_loss − irreducible_holdout_loss, code-confirmed). The reference term's
demonstrated job is preventing selection of *noise*: **18× speedup on noisy Clothing-1M collapses to
~2× or nil on clean data**. Never tested on LM pretraining or reasoning — a quality/noise lever whose
value a pre-filtered pool has already consumed.

**📖 Sequence Reducible Holdout Loss for Language Model Pretraining (LREC-COLING 2024)** — Sequence-level RHO for LM
pretraining: **−21.5% mean steps but −4.3% final steps with GLUE-finetuning parity** (Tables 1/5/6).
Irreducible-loss-only scoring is catastrophic (**+85.9% steps** vs uniform), and the signal is
*nonstationary* — freezing selection after 5K steps erases the final gain, and optimal reference
strength is training-stage-dependent. Steps-not-FLOPs accounting hides ~4.3× per-step cost. A static
one-shot corpus ranking is structurally different from what this validated.

### Model-axis loss (excellent ability readout, poor document detector)

**📖 Compression Represents Intelligence Linearly (`2404.09937`, COLM 2024)** — Per-char loss (BPC)
tracks a *model's* benchmark ability nearly linearly across 31 models (**ρ=−0.93 overall, −0.953
math**, code-confirmed formula) — the model-axis premise behind rank-match. But it is aggregate-only
(saturates at ~3M characters) and domain-sensitive (cross-domain ρ→−0.62). A model-axis readout, not
a per-document reasoning detector — and the tension it creates with every data-side result in this
bucket (loss selects poorly for reasoning content) is unresolved.

### Quality/spam filters (mimic existing filters)

**📖 Rethinking KenLM: Good/Bad Model Ensembles (`2409.09613`)** — A good-vs-bad KenLM z-score
perplexity ensemble mimics the FineWeb-Edu classifier (**recall 0.8919**); the active ingredient is
spam/SNS recognition; no LM is ever trained. Filter-mimics-filter — likely nothing to bite on
pre-filtered pools.

### Corpus-shape statistics / misfiled

**📖 Systematic Generalization Scales with Information Entropy (`2505.13089`, ACL 2025 Findings)** —
Slot-entropy of the *training distribution* (not any loss signal) controls compositional
generalization in tiny seq2seq models; a perm-equivariant control proves the failures are learner-level
Won't. The signal is a corpus-shape statistic (intractable on real corpora), and the code shows
test-leaked checkpoint selection. A conceptual datapoint, not a scorer.

**📖 Reasoning Stabilization Point (`2601.11625`)** — Misfiled for this bucket: an attribution-drift
epoch statistic on BERT classifiers, with **RSP≡3 in all 8 settings by construction** and the
robustness payoff a literal in-text TODO. No bearing on loss-based reasoning-text detection.
---

## Open questions (genuinely open — not prescriptions)

1. **Which inference format is our thread about — and can the latent route ever be primary?** The emission channel
   is the strongest-supported route (SOCRATES 7.6→92.8 with 96% of chain failures being wrong-bridge; silent
   thinking with a hint stays at 6.1%; Physics-3.2's trained-with-chains-tested-without still fails). The latent
   route is exposure-bound (Exposure's 0.00 held-out under every control), data-exponential in depth (k-hop), and
   capacity-punished (U-shape) — but not absolute: recurrence/parameter-sharing recovers ~72% OOD composition where
   the vanilla model gets ~0% (Grokked App E.2), and same-document co-presence buys latent composition with no chain
   at all (TwoHopCurse App E). If we target emitted chains, augmentation must teach chains the deployed model will
   produce; if latent, format must match the no-scratchpad inference distribution and expose the compositions
   themselves.
2. **Does augmenting natural, general web text with reasoning help — beyond distilling the generator?** On math
   corpora, yes with real controls (MIND). On general web, current evidence favors faithful cleaning over
   reasoning-injection (RePro; REWIRE's rewritten-alone CORE deficit), and the teacher-free structure evidence is
   almost entirely synthetic. The decisive experiment is unclaimed: weak-or-self-generator augmentation of general
   web at fixed tokens, reasoning-specific evals, against deletion-only / faithful-rephrase / latent-depth
   baselines — with the completeness *dose* (granularity, not length) as the controlled variable. Every published
   dose-response is synthetic, post-training, or verifier-side; none touches natural pretraining text.
3. **Can any signal find reasoning-rich text?** Recipe B (multi-model rank-match on a same-tokenizer ladder) is the
   last loss-family candidate standing, expected to surface exam/knowledge-shaped text with commonsense costs; the
   ability-ranking should be defined on a reasoning benchmark and read out per task family, regressions included.
   Off the loss family: procedural/worked-text influence evidence (ProcKN) and Essential-Web's 5-level
   reasoning-depth rubric (learnable by a 0.5B annotator) are usable instruments; whether *any* selector isolates
   reasoning from difficulty/domain/structure remains unproven. Two untested ideas we hold: a complete-reasoning
   reference model pointing token-level excess loss at completeness, and proxy loss scored on reasoning-trace
   targets rather than raw documents.
4. **Does under-reasoning persist through *our* post-training, and what dose is right at our scale?** The corpus
   says pretraining sets the post-RL floor/slope/ceiling and that RL erases only verifiable shortcuts, leaving
   unverbalized cue-reliance intact — but our 300M–1.4B ladder adds a twist the corpus flags directly: long/complete
   reasoning data can *hurt* students at this scale (interior mixture optimum). Untested on our models with our
   rewritten-web intervention; success criteria must include shortcut-reliance probes and post-training heads, not
   base-stage perplexity (which our own PERPLEXITY_HUNT already showed is structurally blind here).

---

*Provenance: zero-seed discovery (`wf_869397f2-d8b`, `wf_438a8a3c-3b1`, concept-expansion recall round
`wf_66465130-feb`; 533-candidate triaged pool in `docs/DISCOVERY_POOL_2026-07-23.md`). Reads: `wf_13d49562-ffa`
(24 papers), `wf_4006ecb6-289` (40), `wf_e87a77d2-8da` (80, PDF+appendix+code protocol); the first two passes were
then re-verified page-by-page against the PDFs with code checks (`wf_a05ac667-dc2` — 64 verifications, corrections
folded in throughout). Bucket-level synthesis over the verified records: `wf_777fa169-9d7`. Per-paper structured
records and verification deltas live in the session archives (`subagents/workflows/<id>/journal.jsonl`). Tier-3
code deep-dive of AttentionInfluence + PreSelect: `wf_1163664e-5a9`. This thread's own experimental input: recipe-A
self-ablation no-go (`docs/RECIPE_A_SELF_ABLATION.md`). The earlier abstract-only map and
`docs/PERSISTENCE_AND_USEFUL_REASONING.md` are superseded by this document.*
