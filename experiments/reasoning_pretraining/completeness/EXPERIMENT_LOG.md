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

---

## 2026-07-08
- **Perplexity drop on ORIGINAL data (no Q&A)** → `docs/CONCLUSION_RESULTS.md`. Selected real DCLM docs whose
  continuation IS the doc's own conclusion (strong marker re-split; only ~1% qualify — 30/3000, 22 kept after
  agent-confirming genuine argument→conclusion). Score the **real conclusion** under DCLM-1.4B base:
  **real−base −0.055 (68% drop), placebo−base +0.399, real−placebo −0.454, real<placebo 22/22.** An irrelevant
  rationale *raises* the conclusion's perplexity; the correct reasoning lowers it. First clean, reasoning-specific,
  non-priming drop — on real text. Confirming on 72B judge; caveats: n=22, rare docs, rationale saw the conclusion.
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
