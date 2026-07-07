# Completeness of Reasoning (code → text) — Experiment Log

**Thread:** `experiments/reasoning_pretraining/completeness/`
**Owner:** Dongwei Jiang · **Started:** 2026-07-04
*Entries are newest-first. Layout: `docs/` (this log + write-ups), `scripts/` (`*.py`), `data/` (jsonl).*

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
   active ingredient; Petty 2024 helps math/structured, HARMS syntax/knowledge). → `CODE_REASONING_LIT.md`
2. **"Augment text w/ reasoning, train normally" already works** (BoLT/TPT/Reasoning-CPT 2025). Honest novelty
   = **N1** completeness/stopping-rule as a controlled variable, **N2** reasoning-text as a code-substitute on
   OUR ladder, **N3** cheap "found-completeness" gate. → `PRETRAIN_AUGMENTATION_LIT.md`, `LITERATURE_REVIEW.md`
3. **Stopping rule is THE unsolved problem**; 6 implementable rules enumerated. → `STOPPING_RULES.md`
4. **Found-completeness gate (N3):** only **11.6%** of DCLM warrants a rationale (347/3000). → `RATIONALE_WARRANT_AUDIT.md`
5. **Zero-shot perplexity does NOT show rationale value** (2026-07-07, thorough): raw drops are format/priming
   artifacts (placebo-matched); the reasoning-specific effect is faint & rare. The real test is training. → `PERPLEXITY_HUNT.md`, `PROBE_RESULTS.md`

---

## 2026-07-07
- Reorganized the folder into `docs/ scripts/ data/` (mirroring `code_ladder/`); this log flipped to newest-first.
- Nightly experiment-log bot moved from a system crontab to **CronCreate** (12:07 AM local; session-only, 7-day expiry).
- Ran the **perplexity-drop hunt** → `PERPLEXITY_HUNT.md` (full data in `PROBE_RESULTS.md`):
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
  → `RATIONALE_WARRANT_AUDIT.md`
- Thread copied out of `experiments/data_efficiency/` (Suhas's) into its own reasoning-pretraining home.
- **Claude-as-teacher rationales** vs Qwen2.5-32B on the 5 warranting docs — Claude wins 5/5 on
  continuation-NLL. → `CLAUDE_VS_QWEN_RATIONALES.md`, `data/dclm_aug_claude.jsonl`
- **Controlled completeness-check v2** (Qwen-32B judge, placebo + memorization controls). Bracketed insertion:
  format penalty **+0.30** dominates, content −0.12, teacher −0.23 (Claude>Qwen), no memorization (`noctx` 3.07).
  → `data/completeness_v2_results.jsonl`

## 2026-07-05
- Generated **3,544** reasoning-augmented DCLM docs (544 @ Qwen2.5-7B + 3,000 @ 32B), local / no API / no egress.
  → `OVERNIGHT_RESULTS.md` + `data/`
- Zero-shot completeness-check (continuation-NLL), 7B + 32B: **NEGATIVE and diagnostic** (format shift, not
  reasoning value). → `OVERNIGHT_RESULTS.md`, `SURPRISING_CASES.md`

## 2026-07-04
- Thread created (overnight autonomous session, Claude Fable 5); 5 research agents dispatched.
- Workstreams W1–W5 → `CODE_REASONING_LIT.md`, `PRETRAIN_AUGMENTATION_LIT.md`, `ENTAILMENT_LIT.md`,
  `STOPPING_RULES.md`, `DATASETS.md`. Synthesis → `LITERATURE_REVIEW.md`; prototype → `PROTOTYPE.md`; plan
  (Stage-0 gate + open decisions D1–D8) → `EXPERIMENT_PLAN.md`.
- Data reality check → Stage-0 resized to a ~3B matched-budget gate. NO launch until Dongwei rules on D1–D8.
