# Experiment plan — reasoning-completeness transfer (code → text)

**Status: PROPOSAL for Dongwei's review. NOTHING launched.** Per standing rule, no pretraining run
starts until you rule on the open decisions (§5) and greenlight.

## 0. The refined question (honest, post-literature)
The general recipe "augment text with reasoning, train normally" is **already established and works**
(BoLT/TPT/Reasoning CPT, 2025). The **open, contributable** questions (see LITERATURE_REVIEW.md N1–N3):
- **N1** Does *completeness depth* (the stopping rule) matter — shallow single-hop vs deep bounded chain —
  and what is the reasoning-gain vs distribution-shift (perplexity) tradeoff as a function of it?
- **N2** Can reasoning-explicit text act as a **substitute for code**, buying code→reasoning transfer
  WITHOUT code's damage to NL/perplexity — measured on our existing code-budget ladder / §3 suite?
- **N3** (cheap gate) Does *found* reasoning-explicit text (data we already have) transfer like code at all?

## 1. Staging logic (de-risk before spending generation compute)
Generation of completeness-augmented text needs heavy teacher inference (§5 D5). So test the **premise
first, for ~free**, then only build the augmentation if the premise holds:
- **STAGE 0 (launch-ready, ~zero data cost):** does *found* reasoning-explicit text (openthoughts / openwebmath,
  already tokenized) transfer to reasoning like code does, on our ladder, with the code↔NL tradeoff measured?
  → answers N3, partially N2. **Gate:** if found-reasoning-text does NOT beat plain-text on reasoning transfer
  at matched budget, the expensive augmentation (Stage 1) is not worth building — stop and report.
- **STAGE 1 (needs generation, the novel study):** generate completeness-augmented DCLM at 2+ stopping-rule
  depths (A1 shallow, A2 deep — see PROTOTYPE.md), sweep completeness, measure N1. Only if Stage 0 passes.

---

## 2. STAGE 0 — full design (this is the one I'd launch on your OK)

**2a. Hypothesis (falsifiable).** At matched added-token budget over a common base, continuing on
reasoning-explicit text (openthoughts CoT / openwebmath) improves reasoning/math evals **comparably to
continuing on code**, while doing **less damage to NL/perplexity** than code.
- **Confirms:** reasoning-explicit arms reach ≥ ~⅔ of code's reasoning gain over the plain-text control
  AND have dclm/paloma bpb closer to control than code does. → reasoning-explicit text is a viable
  code-substitute; motivates Stage 1.
- **Refutes:** reasoning-explicit arms ≈ plain-text control on reasoning (no transfer). → premise weak; stop.

**2b. Why this hypothesis.** Waheed 2025 (structure not executability), Zhao 2026 (code-text/math-text
traces are the active ingredient, not executable code), Kim 2024 (code = state tracking). If they're right,
reasoning-explicit *text* should carry the transferable structure. openthoughts/openwebmath are the
cheapest realization ("found completeness"). Directly extends our ladder (which already has the code point).

**2c. Arms (matched budget; only the 30% "structured source" varies).** Mirror C5-v6's 70/30 phase-2 recipe:
| Arm | 70% slot | 30% slot | Status |
|---|---|---|---|
| **R-ctrl** | DCLM | DCLM (more) | NEW (or reuse A5 DCLM-only if base matches) |
| **R-code** | DCLM | code+markup | **already have = C5-v6** (reference) |
| **R-openthoughts** | DCLM | openthoughts (CoT traces) | NEW |
| **R-owm** | DCLM | openwebmath | NEW |
→ **2–3 new 1.4B runs** (ctrl, openthoughts, owm); code + DCLM points largely exist.

**2d. Data (exact).** All in `outputs/tokenized/`: `dclm_baseline`; `openthoughts_flat` (or `_filtered`);
`openwebmath`; code = the C5-v6 clean code+markup caches. **Pre-launch: read 10 real samples per source
(standing rule)** — esp. openthoughts (verify it's clean CoT, not degenerate) and confirm token counts /
epochs so no source repeats.

**2e. Hyperparameters (reuse ladder exactly).** 1.4B LLaMA, seq 4096, batch 256, LR 3e-4 cosine→0,
warmup 1%, wd 0.1, max_grad_norm 1.0, β=(0.9,0.95). Phase-2 continued-pretrain (separate cosine, init from
the SAME phase-1 base as C5-v6 so R-code = C5-v6 exactly) OR from-scratch single-phase — **decision D1**.
Budget: 15.39B added tokens (match C5-v6 phase-2) — **decision D3** (or smaller for a faster first pass).

**2f. Eval.** §3 v2 suite (reasoning: arc/hellaswag/winogrande/mmlu/…; math: gsm8k/minerva; code:
humaneval/mbpp; NL/perplexity: dclm/paloma bpb) via `/eval-for-section3`, single-node. **Add multi-step
reasoning evals** (EntailmentBank / ProofWriter / aNLI) — **decision D4** (needs harness wiring, ~modest effort).
The dclm/paloma bpb rows are essential — they measure the distribution-shift/NL-damage side of N2.

**2g. Confirm/refute** = 2a, read off the §3 table + comparison (read-only, no auto-fill), same as the ladder.

---

## 3. STAGE 1 — completeness-augmentation study (sketch; only if Stage 0 passes)
**Hypothesis (N1):** completeness *depth* is a real axis — deeper explicit chains buy more reasoning
transfer up to a point, then hurt NL/perplexity (off-manifold). There's an optimum stopping rule.
**Arms:** same 70/30 recipe, 30% slot = DCLM augmented at different stopping rules (PROTOTYPE.md):
A1 (single-hop enthymeme), A2 (bounded depth-k chain), + depth sweep k∈{1,2,3}; vs Stage-0 winners.
**Needs:** a generation pipeline (teacher LLM writes A1/A2 augmentations for a DCLM slice) — **decisions
D5 (teacher model + cost), D6 (scale), D7 (faithfulness filtering — the verification gap for web text).**
**Reusable blueprints:** TPT (append), BoLT (prefix + EM self-bootstrap to cut teacher dependence),
Reasoning CPT. EntailmentBank `[BECAUSE]/[INFER]` as the A2 target format; abduction (ProofWriter) as the
A1 target.

---

## 4. What's already done (staged for you)
- Literature: LITERATURE_REVIEW.md + 4 thread files (code→reasoning, augmentation recipes, entailment, stopping rules), all with quotes + citation counts + honest caveats.
- Datasets: DATASETS.md (shortlist <150MB); downloaded + inspected QASC + ProofWriter (`data/structure_examples.md`); openthoughts/openwebmath confirmed already tokenized locally. (EntailmentBank HF mirror errored — get from AI2 GitHub tarball if needed.)
- Prototype: PROTOTYPE.md — hand-authored A1/A2/A3 augmentations on your Alice example + real DCLM.
- Design core: STOPPING_RULES.md — 6 implementable stopping rules + recommended combination.

## 5. OPEN DECISIONS — I need your call before launching (do NOT want me to guess these)
- **D1** Stage-0 base: continued-pretrain from the C5-v6 phase-1 base (so R-code == C5-v6 exactly, cleanest
  code reference) — or from-scratch single-phase? *[recommend: continued from C5-v6 phase-1 base — reuses ladder, code point is free]*
- **D2** Which reasoning-explicit sources in Stage 0: openthoughts, openwebmath, both? *[recommend: both]*
- **D3** Stage-0 budget: full 15.39B (match C5-v6) or a smaller/faster first pass (e.g. 5B) to gate quickly? *[recommend: 15.39B for apples-to-apples with C5-v6; or 5B if you want a fast read first]*
- **D4** Add EntailmentBank/ProofWriter/aNLI to the eval suite (needs harness wiring), or Stage-0 on §3 only first? *[recommend: §3 first, add multi-step evals for Stage 1]*
- **D5** Stage-1 teacher for generation: local open model (free compute, we host vLLM) vs API (Claude/GPT — **costs $, needs your approval**). *[recommend: local open model to avoid spend + region issues; decide later]*
- **D6** Stage-1 scale (how many augmented tokens).
- **D7** Stage-1 faithfulness filtering (verification gap — web text has no cheap correctness signal).
- **D8** Goal/framing: publishable N1+N2 study, or internal "does it help our models"? (changes eval breadth + rigor.)

## 6. My recommendation for the FIRST launch (pending your OK)

**⚠ Data reality check (found 2026-07-04, see data/found_completeness_samples.md):** the found-reasoning
sources are SMALL — openthoughts ≈ **0.97B tokens**, local openwebmath ≈ **0.19B** (full HF owm is 14.7B but
only a slice is on disk). 30% of a 15.39B run needs 4.62B/source → **impossible without repeating.** Also
openthoughts is **SFT-chat-formatted** (fixed "Your role as an assistant… systematic long thinking" preamble)
→ style-confound (explicit-reasoning vs chat-format).

**Revised recommendation: Stage 0 as a small-scale ~3B-token matched-budget GATE** (fits our existing
"small" run family: C5-v2-small / C5-v3-small were 3.36B). At ~3B total, 30% slot ≈ 0.9B ≤ openthoughts
(<1 epoch, no repeat). Arms at matched 3B budget: **R-ctrl (DCLM)**, **R-code (DCLM+code)**,
**R-openthoughts (DCLM+openthoughts, preamble stripped)**, **R-owm (DCLM+owm)**. Eval §3 (single-node) +
dclm/paloma bpb. This is fast, cheap, no-repeat, and cleanly gates Stage 1.
- Open sub-decisions: D2 (which sources), whether to **strip openthoughts' chat preamble** (recommend yes,
  to isolate reasoning from format), and whether to **download+tokenize full open-web-math** (27GB) for a
  larger owm arm later.
- If Stage 0 gate passes → build Stage 1 (completeness-augmentation, the novel N1 study).

**Next step on your OK:** decide D1/D2 + preamble-strip → confirm exact tokenized token counts →
write the small-run scripts → present pre-launch spec (train-and-eval Stage -1) → launch. **I will not
launch before you rule on this.**
