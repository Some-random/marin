# Completeness of Reasoning (code → text) — Experiment Log

**Thread:** `experiments/reasoning_pretraining/completeness/`
**Owner:** Dongwei Jiang · **Started:** 2026-07-04
*(Renamed from LOGBOOK.md → EXPERIMENT_LOG.md on 2026-07-07 to match the project convention.)*

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
1. **Transfer mechanism = STRUCTURE not executability** (Waheed 2025 pseudocode recovers the benefit; Zhao
   2026 code-text/math-text traces are the active ingredient; Petty 2024 helps math/structured, HARMS
   syntax/knowledge = our code↔NL tradeoff). → `CODE_REASONING_LIT.md`
2. **"Augment text w/ reasoning, train normally" already works & is active** (BoLT/TPT/Reasoning-CPT 2025).
   Honest novelty = **N1** completeness/stopping-rule as a controlled variable, **N2** reasoning-text as a
   code-substitute on OUR ladder, **N3** cheap "found-completeness" gate. → `PRETRAIN_AUGMENTATION_LIT.md`,
   `LITERATURE_REVIEW.md`
3. **Stopping rule is THE unsolved problem** (no textual method has learned dynamic stopping); 6 implementable
   rules enumerated. → `STOPPING_RULES.md`
4. **DATA CONSTRAINT:** openthoughts ≈0.97B tok (SFT-chat-formatted), local owm ≈0.19B — too small for a 30%
   slot of 15.39B → Stage 0 must be small-scale (~3B gate). → `DATASETS.md`
5. **Found-completeness gate quantified (N3):** only **11.6%** of DCLM docs warrant a rationale (347/3000,
   30-agent audit, tight 6–18% spread); the warrant set ≈ argument+causal doctypes (12.4%); ~7/8 of web text
   is narrative/opinion/fact-list with no latent reasoning → augmentation must be a **filter**, not blanket.
   → `RATIONALE_WARRANT_AUDIT.md`
6. **Zero-shot completeness-check (continuation-NLL) is not a valid metric** — it measures format shift, not
   reasoning value (higher-quality 32B chains scored *worse*). → `SURPRISING_CASES.md`

---

## 2026-07-04
- Thread created (overnight autonomous session, Claude Fable 5); 5 research agents dispatched.
- Workstreams W1–W5 done → `CODE_REASONING_LIT.md`, `PRETRAIN_AUGMENTATION_LIT.md`, `ENTAILMENT_LIT.md`,
  `STOPPING_RULES.md`, `DATASETS.md` (+ QASC/ProofWriter → `data/structure_examples.md`).
- Synthesis → `LITERATURE_REVIEW.md`; prototype (Alice + real DCLM, A1/A2/A3 by hand) → `PROTOTYPE.md`;
  plan (Stage-0 gate + open decisions D1–D8) → `EXPERIMENT_PLAN.md`.
- Data reality check → Stage-0 resized to a ~3B matched-budget gate (R-ctrl / R-code / R-openthoughts /
  R-owm). NO launch until Dongwei rules on D1–D8.

## 2026-07-05
- Generated **3,544** real reasoning-augmented DCLM docs (544 @ Qwen2.5-7B + 3,000 @ 32B), local / no API /
  no egress → `OVERNIGHT_RESULTS.md` + `data/`.
- Zero-shot completeness-check (continuation-NLL) on 7B + 32B: **NEGATIVE and diagnostic** (format shift, not
  reasoning value) → `OVERNIGHT_RESULTS.md`, `SURPRISING_CASES.md`.

## 2026-07-06
- 30-agent **rationale-warrant audit** (workflow `wf_0494a236`): **11.6%** of DCLM warrants a rationale
  (347/3000) → `RATIONALE_WARRANT_AUDIT.md`.
- Thread copied out of `experiments/data_efficiency/` (Suhas's thread) into its own reasoning-pretraining home.
- **Claude-as-teacher rationales** vs Qwen2.5-32B on the 5 warranting docs — Claude wins 5/5 on
  continuation-NLL → `CLAUDE_VS_QWEN_RATIONALES.md`, `data/dclm_aug_claude.jsonl`.
- **Controlled completeness-check v2** (`compute_completeness_v2.py`, Qwen-32B judge, placebo + memorization
  controls). Bracketed insertion: format penalty **+0.30** dominates, content **−0.12** (real rationale beats
  placebo), teacher **−0.23** (Claude>Qwen), memorization ruled out (`noctx` 3.07, not low) →
  `data/completeness_v2_results.jsonl`.

## 2026-07-07
- Warrant audit written up (recomputed from the workflow journal, not memory).
- `LOGBOOK.md` renamed → `EXPERIMENT_LOG.md`.
- **Natural-prose insertion** (`--style natural`): format penalty halves (+0.30→**+0.13**) but content shrinks
  (−0.12→**−0.02**) → `+rationale` still **+0.11 worse** than base. No raw perplexity drop yet; bottleneck
  reframed from format to the **target** (raw web continuation isn't the reasoning's payoff) →
  `data/completeness_v2_natural_results.jsonl`.
- Nightly experiment-log bot set up (CronCreate 12:07 AM local). Ran the **perplexity-drop hunt** → `PERPLEXITY_HUNT.md`:
  - Base DCLM-1.4B judge + insertion/target sweep (500 docs): mean still +, but ~12% of docs drop.
  - Independent R/N split (100 docs, 19 R) + Claude rationales: **R ≈ N (+0.09 both)** — reasoning-dependent
    docs show NO continuation-perplexity drop. The 12%≈warrant match was coincidence. **Continuation-perplexity
    can't show it** (raw next-web-text isn't the reasoning's output) — verified across 4 configs.
  - Probe target (reasoning-determined answers), leakage-guarded: raw drop −0.358 (70.6%). Placebo control
    (another doc's rationale): −0.333 too → **mean is mostly format/priming, not reasoning** (real−placebo −0.024).
  - **But per-probe**, the mean washes out a real split: **8/17 show a genuine reasoning-specific drop**
    (real≪placebo) on **specific answers** (id444 "spoilers never deployed" real −1.66 / placebo −0.00; id458 −1.21/−0.41;
    id89 −0.61/−0.22); vague answers ("worse") are pure priming and drag the mean to 0. `data/probe_placebo_perprobe.jsonl`.
  - Confirmatory strict specific-answer probes (n=8): real −0.406 vs placebo −0.307, reasoning-specific
    −0.099 (4/8). Clean wins only where the answer is a specific non-stated consequence (id444 r−p −0.87,
    id404 −0.49); numbers/terms ("64","malnutrition") still priming-confounded. Only ~5/19 R docs could yield
    a valid strict probe at all.
  - **FINAL: no clean zero-shot "+rationale lowers perplexity."** Raw drops (continuation & probe) are
    format/priming (placebo-matched); the genuine reasoning-specific drop is faint & rare. The only
    unconfounded test is the **training experiment** (needs sign-off; not launched). → `PERPLEXITY_HUNT.md`.
