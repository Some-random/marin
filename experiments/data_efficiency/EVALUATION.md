# Evaluation Reference: Tasks, Taxonomy, and Model Results

Canonical reference for the evaluation suite used in the data-efficiency project. Lives outside `EXPERIMENT_LOG.md` (which is a chronological narrative) so that the eval setup and current-best numbers can be read independently of any particular day's work.

---

## 1. Why this doc exists

Names like "NL reasoning benchmark" hide important mechanistic differences. Before reasoning about why a benchmark moved (or didn't), **read 2-3 actual examples** and classify the task by the cognitive mechanism it tests. The May 26 code-mix probe is the standing example: sciq and piqa are both routinely labeled "NL reasoning", but inspection shows they operate through completely different mechanisms (sciq is passage-grounded extraction; piqa is parametric commonsense). Confusing them produces false positives for "reasoning gains".

This doc:
- Classifies every eval we use by mechanism (§2).
- States which evals give signal at our 1.4B / 3.3B-token scale, and which don't (§3).
- Maintains a canonical results table across the models we compare against (§4–6).

---

## 2. Taxonomy of evaluations by mechanism

### A. Continuous PPL (no task structure, just next-token loss)
- **Paloma macro** (16 subsets, see `experiments/data_efficiency/run_1_4b_25code_alg.py`) — domain-diverse held-out web / forum / code text. Primary continuous signal at our scale. Sensitive to general LM quality and to per-domain coverage.
- **dclm_200m_val** — held-out NL within our training distribution. Sensitive to overfitting on the 209M-token DCLM training slice.
- **opc_algorithmic (loss)** — when used as code training data, final eval loss on the code training slice. Signals code memorization, not generalization.

### B. Passage-grounded reading comprehension (answer is *in the prompt*; model just attends and extracts)
The model does NOT need to know the answer in its weights — the passage contains or strongly implies it.
- **sciq** — every question comes with a `support` paragraph that *literally states the answer* (e.g. "what is X called?" → support: "X is called Y"). 4-way MC. Tests context attention and lexical matching, not science knowledge.
- **boolq** — yes/no question + Wikipedia-style passage that contains the answer. Tests passage-question matching.
- **openbookqa** — provides a relevant fact alongside the question (when used with the fact).

### C. Parametric world knowledge (no passage; the model must recall facts from weights)
- **arc_easy** — 4-way MC science questions, no support. Tests grade-school science recall (photosynthesis, meiosis, safety equipment, ...).
- **arc_challenge** — harder ARC subset; mostly at-random at 1.4B / 3.3B-tokens.
- **mmlu** — 4-way MC across 57 domains, no passage.
- **triviaqa / naturalqs** — generation, no passage. (Not currently in our suite.)

### D. Physical / social commonsense (no passage; intuition from weights)
- **piqa** — 2 alternative everyday-task descriptions; pick the physically plausible one ("paper bedding" vs "jeans bedding" for a guinea pig).
- **social_iqa** — social-situation MC.
- **hellaswag** — sentence-completion plausibility (which next sentence is most natural).

### E. Coreference / logical / linguistic
- **winogrande** — pronoun-resolution pairs (Winograd schema).
- **logiqa** — formal logical-reasoning MC.
- **blimp** — minimal-pair grammaticality judgment. Linguistic, mostly above-random for any model; not used as primary signal.
- **commonsense_qa** — 5-way MC, ConceptNet-derived commonsense reasoning.

### F. Math (multi-step generation)
- **gsm8k_cot** — grade-school math with 8-shot CoT, free generation. At our scale all 1.4B runs score 0%; used as a **looping smoke test** (does the model emit short sensible-shaped responses like `The answer is X.\n\n` vs n-gram loops).
- **gsm8k** (logprob variant) — multiple choice; at-random at our scale.
- **minerva_math** — competition math, free generation.

### G. Code generation
- **HumanEval** — function generation from docstring; pass@1 by running unit tests on the generated code.
- **MBPP** — Python programming problems; pass@1.

---

## 3. What's usable at our 1.4B / 3.3B-token scale

| Eval | At our scale | Why |
|---|---|---|
| Paloma macro PPL | ✅ Primary continuous signal | Always informative; ~0.5 nat gaps detectable |
| dclm_200m_val PPL | ✅ | Sensitive to overfitting |
| arc_easy | ✅ Above-random | Baseline ~0.42 vs random 0.25 — 17pt headroom |
| sciq | ✅ Above-random (passage-grounded) | Baseline ~0.65 vs random 0.25 — 40pt headroom |
| piqa | ✅ Above-random | Baseline ~0.63 vs random 0.50 — 13pt headroom |
| boolq | ⚠️ Barely above-random | Baseline ~0.50, at chance for yes-no; large lift possible |
| gsm8k_cot (qualitative) | ✅ As looping smoke test | exact_match always 0; sample shapes are informative |
| arc_challenge | ❌ at-random | 0.22 vs 0.25 |
| mmlu | ❌ at-random | 0.25 |
| hellaswag | ❌ ~ at-random | 0.31 vs 0.25, ~6pt above |
| winogrande | ❌ at-random | 0.50 |
| openbookqa | ❌ ~ at-random | 0.30 vs 0.25 (with acc_norm) |
| commonsense_qa | ❌ at-random | 0.20 vs 0.20 (5-way) |
| social_iqa | ❌ ~ at-random | 0.37 vs 0.33 |
| logiqa | ❌ at-random | 0.22 vs 0.25 |
| gsm8k (logprob) | ❌ floor | 0 to 0.02 |
| minerva_math | ❌ floor | 0 |
| HumanEval | ❌ floor for our recipes | 0% (phi-1: 49%) |
| MBPP | ❌ floor | 0% |

**Bottom line:** at our scale, the only signal-producing benchmarks are **Paloma + dclm_200m_val + arc_easy + sciq + piqa + boolq + gsm8k_cot looping behavior**. Everything else is logged for completeness but should be treated as noise around the model. The four discrete benchmarks split across mechanisms — sciq+boolq are passage-grounded (B), arc_easy+piqa are parametric knowledge/commonsense (C/D). Always classify deltas by mechanism, not by name.

---

## 4. Models tracked

| Label | HF repo (or local path) | Params | Train tokens | Notes |
|---|---|---|---|---|
| **1.4B baseline** | `1_4b_wd1_6_x16_nocrossblock_hf` (`peach-thunder-100` / `6xx0hu3l`) | 1.4 B | 3.3 B (DCLM-200M, x16 epochs) | wd=1.6, LR=1e-3 cosine, block_cross_doc=False |
| **1.4B code-mix 25%** | `1_4b_25code_alg_hf` (`eager-grass-104` / `p2n84bo3`) | 1.4 B | 3.3 B (75% DCLM + 25% opc_algorithmic) | Same recipe as baseline + code component |
| phi-1 | `microsoft/phi-1` | 1.3 B | ~7 B (filtered Stack + ~1B GPT-3.5 synth Python) | Code-only; HumanEval-tuned |
| phi-1.5 | `microsoft/phi-1_5` | 1.3 B | ~30 B (phi-1 mix + ~20B synthetic NL textbooks) | 5 epochs × 30B unique |

---

## 5. Canonical downstream results

All numbers from our own `lm-eval-harness` pipeline (lm_eval 0.4.11) at the n-shot settings shown. `acc_norm` used where reported; `acc` otherwise. Random column shows chance accuracy.

| Task | n-shot | Random | 1.4B base | 1.4B code25 | phi-1 | phi-1.5 |
|---|---:|---:|---:|---:|---:|---:|
| arc_easy | 25 | 0.25 | 0.401 | 0.416 | 0.378 | **0.805** |
| arc_challenge | 25 | 0.25 | 0.242 | 0.236 | 0.232 | **0.532** |
| sciq | 0 | 0.25 | 0.652 | 0.709 | 0.707 | **0.933** |
| piqa | 0 | 0.50 | 0.634 | 0.619 | 0.562 | **0.766** |
| boolq | 0 | 0.50 | 0.502 | 0.579 | 0.451 | **0.746** |
| hellaswag | 10 | 0.25 | 0.348 | 0.341 | 0.301 | **0.635** |
| winogrande | 5 | 0.50 | 0.504 | 0.500 | 0.498 | **0.710** |
| openbookqa | 0 | 0.25 | 0.302 | 0.288 | 0.248 | **0.482** |
| commonsense_qa | 0 | 0.20 | 0.192 | 0.200 | 0.175 | **0.507** |
| social_iqa | 0 | 0.33 | 0.366 | 0.362 | 0.364 | **0.523** |
| logiqa | 0 | 0.25 | 0.218 | 0.234 | 0.214 | 0.240 |
| mmlu | 5 | 0.25 | 0.252 | 0.249 | 0.248 | **0.422** |
| gsm8k | 5 | 0 | 0.000 | 0.000 | 0.012 | **0.305** |
| gsm8k_cot | 0 | 0 | 0.024 | 0.022 | 0.014 | **0.069** |
| **humaneval** | 0 | 0 | 0.000 | 0.006 | **0.494** | 0.342 |
| mbpp | 0 | 0 | 0.000 | 0.000 | 0.010 | 0.004 |
| minerva_math | 0 | 0 | 0.0002 | 0.0002 | 0.000 | 0.000 |

### Caveat on code-gen numbers

`humaneval` and `mbpp` in our pipeline use `lm-eval-harness`'s scoring path (`pass_at_1,none` for MBPP, `pass@1,none` for HumanEval) with `--confirm_run_unsafe_code` + `HF_ALLOW_CODE_EVAL=1`. The original phi-1 paper reported MBPP 55.5% with the BigCode evaluation framework; our pipeline reports phi-1 MBPP 1.0%. The methodology difference (extraction patterns, n-shot, runner) is substantial. **Treat our code-gen pipeline numbers as conservative lower bounds; do not directly compare to published phi paper numbers.**

---

## 6. Paloma macro PPL: 1.4B baseline vs 1.4B code-mix (per-subset)

All numbers are eval-loss (lower is better) at step 12,799 = end of training (3.34 B tokens).

| Subset | 1.4B base | 1.4B code25 | Δ |
|---|---:|---:|---:|
| paloma 4chan | 3.640 | **3.254** | −0.39 |
| paloma c4_100_domains | 4.252 | **3.890** | −0.36 |
| paloma c4_en | 4.547 | **4.154** | −0.39 |
| paloma dolma-v1_5 | 4.348 | **3.931** | −0.42 |
| paloma dolma_100_programing_languages | 4.049 | **3.370** | **−0.68** (largest NL-subset gain; code-adjacent text) |
| paloma dolma_100_subreddits | 4.585 | **4.191** | −0.39 |
| paloma falcon-refinedweb | 4.665 | **4.265** | −0.40 |
| paloma gab | 6.476 | **5.807** | **−0.67** |
| paloma m2d2_s2orc_unsplit | 4.164 | **3.816** | −0.35 |
| paloma m2d2_wikipedia_unsplit | 4.067 | **3.733** | −0.33 |
| paloma manosphere_meta_sep | 4.569 | **4.183** | −0.39 |
| paloma mc4 | 4.396 | **4.002** | −0.39 |
| paloma ptb | 5.115 | **4.709** | −0.41 |
| paloma redpajama | 4.464 | **3.988** | −0.48 |
| paloma twitterAAE_HELM_fixed | 7.792 | **6.743** | **−1.05** (largest single gain; baseline very high) |
| paloma wikitext_103 | 4.195 | **3.847** | −0.35 |
| **paloma macro (16 subsets)** | **~4.71** | **~4.24** | **−0.47** |
| dclm_200m_val (held-out NL) | 4.070 | **3.733** | −0.34 |
| dclm_200m (training data) | 1.631 | 1.956 | +0.33 *(less memorization, expected with regularization)* |

Phi-1 / phi-1.5 Paloma numbers are not currently in our pipeline (they were evaluated only on the lm-eval-harness suite, not the Levanter Paloma eval).

---

## 7. Generation behavior — gsm8k_cot looping

`exact_match` on gsm8k_cot is 0% for every 1.4B / 3.3B-token model we have. The **shape** of the generation is the signal:
- **Non-looping (good)**: model emits a short response ending in `The answer is X.\n\n`, no n-gram repetition.
- **Looping (bad)**: model locks into n-gram repetition, e.g. `The answer is 4. The answer is 4. The answer is 4. ...` for the full 256 tokens.

| Model | Loops on gsm8k_cot? | Source |
|---|---|---|
| 1.4B baseline (wd=1.6/x16/block=False) | **No** | May 25 (Step 10) |
| 1.4B code-mix 25% | **No** | May 26 |
| 1.4B wd=3.2/x8 (konwoo recipe) | **Yes** | May 23 (Step 1-9) |
| 1.4B wd=1.6/x16/block=True | **No** | May 24 (Step 9) |
| Qwen2-0.5B-base | No | May 23 (Step 2) |
| OLMo-2-0425-1B | No | May 23 (Step 2) |

For the looping investigation chronology and root-cause analysis, see EXPERIMENT_LOG.md May 23–25.

---

## Updating this doc

When a new model is trained or a new eval is added, update this doc with the new row/column. Add a brief follow-up entry in EXPERIMENT_LOG.md ("Wide eval suite re-run on N.M model → see EVALUATION.md table"). The chronology stays in EXPERIMENT_LOG, the canonical reference stays here.
