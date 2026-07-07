# Next Steps Strategy

Written 2026-06-03 as a strategic-thinking doc for review when you wake up. Captures where we are, what we know, and three concrete paths forward with my reasoning on each. The counterfactual-probes path (B) is detailed in [`counterfactual_probes.md`](counterfactual_probes.md).

## Where we are

**Research question (H1):** What kinds of pretraining data teach NL/reasoning capability at 1.4B / 30B-token scale?

**Established at this point:**

1. **Matched-token 25% code mix HURTS NL at 1.4B / 3.3B-token / 16-epoch repetition** ([EVALUATION.md §3 headlines](../EVALUATION.md)):
   - paloma_macro: 1.824 vs baseline 1.631 (+0.19 nats worse)
   - dclm_200m_val: 4.596 vs baseline 4.070 (+0.53 nats worse)
   - Per-task NL (arc_easy, sciq, hellaswag, openbookqa_fact, piqa, social_iqa, logiqa): code-mix loses by 1-7pp on each
   - Only boolq and a few tied tasks go the other way.

2. **Same pattern at 1-epoch / 30B-token scale (A5 vs B4 at step-14672, ~50% trained):** A5 (DCLM-only) wins 10 NL benchmarks by 1-3pp each. B4 (code-mix) wins the 2 code benchmarks decisively (HumanEval +3.7pp, MBPP +6.8pp) and ties 3 at floor. Same direction, no scale-induced reversal.

3. **Phi-1.5 dominates every NL subset** at the same parameter count (1.3B). Synthetic NL textbook training pays off massively at our scale. Phi-1 (code-only) is WORSE than our 1.4B baseline on paloma_macro (1.738 vs 1.631) because it was trained almost exclusively on code.

4. **Our 1.4B models cannot do code generation.** bigcode HumanEval = 0.000 across all 4 of our small models (lm-eval-harness gave them spurious 0-4% partial credit; bigcode correctly rejects). Phi-1 = 0.543 bigcode, matches paper 50.6%.

5. **Math is at floor for everyone except phi-1.5** (GSM8K: 0.000-0.014 for our 4, 0.305 for phi-1.5). We don't yet know whether the floor is "models have no arithmetic capability" or "models have arithmetic but can't parse the word problem".

**Not yet established:**
- Whether a Phi-1.5-style synthetic NL textbook recipe replicates at our compute budget.
- Whether our floor-score models have the *underlying* capability circuit for math/reasoning, or just lack it entirely.
- Whether the matched-token negative-NL result on code-mix would reverse at larger scale (the MAI-Thinking-1 tech report — added to [reasoning_curriculum.md](../../papers/reasoning_curriculum.md) — shows **rank non-invariance in data-mixture scaling**: stem-heavy beat code-heavy at 5B-active, then code-heavy overtook stem-heavy at 23B-active. Our 1.4B results may not predict 30B-active behavior.)

---

## Three paths forward

### Path A — Run a third leg: cosmopedia synthetic NL textbook training

**What:** Train a third 1-epoch 1.4B variant on cosmopedia_v2 (Phi-1.5-style synthetic NL textbook data, 27.37B tokens already tokenized at [`phi_1_5/cosmopedia_v2-21b787/`](MODELS.md#cosmopedia_v2-tokenized-locally-not-yet-used)). Same hyperparams as A5/B4 (LR=3e-4, WD=0.1, 30.77B target). Three-way comparison: DCLM (A5) vs DCLM+25%-code (B4) vs cosmopedia-only (C, hypothetical).

**Why:**
- Phi-1.5 is the gold-standard reference for "synthetic NL textbooks at 1.3B → strong NL/reasoning". We have the data and the infra.
- Triangulates the 3 data axes we care about: raw web (DCLM), code-augmented, and synthetic NL textbook.
- If C wins NL benchmarks while losing paloma, that's the Phi pattern reproduced at our scale.
- If C loses too, that's stronger evidence "1.4B + 30B tokens isn't enough for this story regardless of data" — pushing us toward scale.

**Cost:** ~24h on 4 nodes (same as A5/B4). Adds one column to EVALUATION.md.

**Risk:** Phi-1.5 used ~150B trained tokens (5 epochs × 30B unique). We'd be doing 30B (1 epoch). The Phi-1.5 paper attributes a lot of gains to repetition over high-quality data. Our 30B-token result might be weaker than the published 150B Phi-1.5, even with identical data. We need to be honest about this caveat in the comparison.

**Caveat to acknowledge:** the data itself is GPT-3.5-generated content. We can't decompose "is it the textbook style?" from "is it the GPT-3.5 distillation?" — that's a confound Phi-1.5 has too.

### Path B — Counterfactual probes on existing checkpoints

> **Update 2026-06-02 23:00 PDT:** Phase 1 (arithmetic decomposition probe) ran tonight on all 6 models. Result is informative — see [EVALUATION.md §4](EVALUATION.md#4-counterfactual-probes--arithmetic-decomposition-phase-1). Headline: **B4 (25% code mix) has 83%/84% single-digit add/mult, A5 (DCLM only) has 35%/13%**. Code teaches arithmetic at our scale even though benchmark scores stay at floor. phi-1 and phi-1.5 need Phase 2 prompt reformatting to compare fairly. Moving forward with Phase 1 → Phase 2 (CRUXEval + counterfactual MMLU) is now strongly motivated by this result.


**What:** Build a set of cheap evaluations that decompose "model can do X" from "model knows the surface pattern of X", on the 6 checkpoints we already have. No new training. Three probe families:
1. **Arithmetic decomposition** — single-digit / two-digit / multi-digit arithmetic, generated synthetically. Discriminates "lacks arithmetic" from "has arithmetic, can't parse word problem".
2. **CRUXEval-style code execution** — given a Python function, predict its output. Discriminates "writes Python syntax" from "understands Python semantics". Particularly informative for phi-1 vs B4.
3. **Wu-et-al-style counterfactual MMLU** — rewrite MMLU questions to flip a surface feature while preserving reasoning structure. Tests whether scores reflect knowledge or token-pattern memorization.

Full design: [`counterfactual_probes.md`](counterfactual_probes.md).

**Why:** This is the only path that *answers the H1 question we actually care about* — what does the model learn, mechanistically, from each data type. The three-leg ablation (Path A) tells us **which mix wins on benchmarks**; counterfactual probes tell us **why**. Wu et al style probes have been validated in the literature (e.g., Wu, Geiger, Goodman, Manning's "Counterfactual Reasoning" line, CRUXEval) and are tractable to build.

**Cost:** Probe construction = 2-3 days of script writing + one-time dataset generation. Running on 6 checkpoints = a few GPU-hours (probes are small).

**Risk:** If our models are at floor on all probes too, the result is "1.4B + 30B tokens is too small for these capabilities to emerge regardless of data" — informative but not actionable for our scale. We'd then either accept that conclusion and write up, or move toward scale (Path A or beyond).

### Path C — Stop and write up the matched-token negative result

**What:** Accept that we have a clean, controlled negative finding: at 1.4B / 30B-token / 1-epoch scale, matched-token 25% code mix HURTS NL by ~0.2 nats paloma without proportionate code-generation gains (B4 HumanEval = 0.043 vs baseline 0.000). Write this up as a finding. Park further data-mix experiments.

**Why:** The result is solid: matched-token, two compute regimes (3.3B and 30B), same direction. Negative results are publishable, and there's no signal yet that pushing harder will reveal a reversal at our compute budget.

**Cost:** ~1 week of writing.

**Risk:** Misses the chance to actually answer *why* (Path B) or to test whether the result reverses at meaningfully larger scale (which would need outside compute). We'd be reporting "here's what doesn't work" without the mechanistic insight.

---

## My recommendation

**Sequence: do B and C in parallel. Defer A.**

Reasoning:
- **B (counterfactual probes) is the highest-EV next step** because it's the only one that actually answers our research question, not just adds another data point. It also costs almost nothing (no new training). Even a partial result — e.g., "our DCLM-only model can do single-digit addition but not two-digit" — is informative.
- **C (write up) is independent of the new evals.** We can draft the matched-token negative result while probes are being built; if probes give us a mechanistic story, we fold it in; if not, we publish the cleaner negative finding.
- **A (cosmopedia leg) is deferred** because the rank-non-invariance result from MAI-Thinking-1 makes me skeptical about what we'll learn at 1.4B / 30B. If probes (B) tell us "our model has no math circuit at all", running a third 30B-token training won't change that — the path is "go bigger" not "swap data". And if probes tell us "the circuit is there, just unused", we have a story without needing the cosmopedia leg.

If you disagree and want A anyway (e.g., because you want the three-leg table for a paper), it's cheap to start once B4/A5 evals are wrapped — just say the word and I'll prep the run script using the same hyperparameter scaffold as A5.

---

## What's running tonight (no new direction needed)

While you sleep:
1. Wait for A5 final step (~22:23 PDT).
2. Convert both A5 and B4 final-step Levanter → HF checkpoints.
3. Run the full eval suite on each (downstream lm-eval + paloma 16-subset + dclm_200m_val + new evals + bigcode HumanEval).
4. Update EVALUATION.md with final-step numbers replacing or augmenting the s14672 columns.
5. Commit + push so you can review changes when you wake up.

If anything crashes I'll debug and restart per the monitor playbook.
