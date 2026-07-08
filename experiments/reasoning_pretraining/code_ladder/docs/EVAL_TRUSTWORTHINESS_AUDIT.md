# Eval-Trustworthiness Audit (300M–1.4B)

**Date:** 2026-07-06. **Method:** read raw per-example predictions (`resps` / per-choice logprobs / `is_greedy` flags), NOT scores or `filtered_resps`, from `outputs/eval_results/` for a strong model (a5 / c5v6) and a weak model (code_p1_half) across §3 + §3c. **Status:** analysis only — **no eval changes applied; all fixes below are gated on user review.**

## Bottom line

Of ~28 audited tasks, **13 (46%) are noise-floor or degenerate at 300M–1.4B** — they measure sampling noise, first-token position bias, or majority-class collapse, not model quality. Two of the six category means (**both Math means**) are numerically constant; **Aggregate** is ~50% chance-offset; the 14-task **Closed-book NL** mean is carried almost entirely by *two* tasks; and **boolq fabricates a "code helps open-book" effect**. Only ~8 tasks carry trustworthy signal.

## Trust tiers

**Tier A — trust (8).** Discriminating, monotonic with model quality:
| Task | spread (our models) | note |
|---|---|---|
| lambada_openai[0] | 0.05 → 0.52 | exact-match (`acc==is_greedy`); cleanest axis, ~10× spread |
| arc_easy[25] | 0.30 → 0.63 | acc_norm, full-sentence distractors (not surface-freq) |
| sciq[0] | 0.49 → 0.83 | **but** raw `acc` → surface-frequency confounded (see below); it's attend+extract, not knowledge |
| piqa[0] | 0.54 → 0.72 | acc_norm binary minimal-pair |
| storycloze_2018_local[0] | 0.50 → 0.66 | N=1571 → stable ordering |
| copa[0] | 0.51 → 0.76 | real, but N=100 → ±0.05 jitter |
| mbpp[3] | 0.00 → 0.31 | real assert execution, n=500; anchor of Mean Code |
| humaneval[0] (lm-eval) | 0.00 → 0.31 | really executes tests (see bigcode note) |

hellaswag[10] is tier A at 1.4B (→0.50) but **tier C at 300M** (all models 0.267–0.299 = chance).

**Tier B — weak / scale-limited (6):** arc_challenge (raw acc = exactly 0.25; signal only in acc_norm), logiqa (raw acc **0.203, below chance**; acc_norm inflates via length), openbookqa_fact (hugs chance at 300M; gold is an indirect paraphrase, NOT an extractable span despite the `_fact` name), social_iqa (floor = chance 0.333, best +0.08), quac_first_turn (F1 0.07–0.20 but strict EM ≈ 0.007 for every model — F1 is partial word-overlap credit).

**Tier C — noise floor, pinned at chance (9):** winogrande, mmlu, gpqa_diamond, agieval_lsat_ar, gsm8k, gsm8k_cot, minerva_math, gsm_symbolic_main, gsm_noop.

**Tier D — degenerate / actively misleading (4):** commonsense_qa, boolq, wsc, cb.

**Tier E — fixable scoring bug (1 + 1 cell):** bigcode-humaneval (deflated), cb phi-1.5 cell (mis-filled).

## The degenerate collapses (Tier D) — with exact counts

**boolq — the headline anomaly.** Gold is imbalanced: 1237 "no" / 2033 "yes" (N=3270) → the **always-"yes" baseline is 0.622, not 0.50**. Code-heavy models emit "yes" almost unconditionally:
- 300M code_p1_half: "yes" on **3242/3270 (99.1%)** → 0.619
- 600M code_p1_half: **3266/3270** → 0.621
- 1.4B code25b: **3256/3270** → 0.620 — all three land **exactly on the 0.622 majority baseline by collapse.**
- A text model that actually reads (a5sp, balanced 1069 no / 2201 yes) scores **0.535 — BELOW the constant.** The eval *penalizes reading and rewards collapse.*
- Only phi-1.5 (balanced 1411/1859) shows real discrimination → 0.746.
- **Impact:** boolq inflates code-model Open-book. Removing it: code_p1_half Open-book 0.485 → 0.441 (**−0.044**) vs a5 0.551 → 0.539 (**−0.012**). The "code prior helps open-book QA" read was largely this artifact.

**commonsense_qa — letter-frequency collapse.** The prompt lists `A. … B. … … E. …` then `Answer:`, and the model is scored on P(' A')…P(' E') — single **letter** tokens, all equal length (so acc_norm/length is moot). At weak scale it collapses to the highest in-context-frequency letter (' A' / position-0) — the **same mechanism as mmlu**, NOT a text-length effect. (Correction: an earlier audit pass mis-described this as variable-length *text* scoring; verified from the samples it is letter-scored.)
- 300M code_p1_half predicts **choice #0 on ALL 1221/1221 items** → acc 239/1221 = **0.196 = exactly P(gold at position 0)**.
- c5v6 (1.4B) still 85% first-choice: PRED positions {0:1039, 1:8, 2:133, 3:41} — never picks option 4.
- Example: "When drinking booze what can you do to stay busy?" gold=D ("examine thing"), model picks [0] ("reach tentative agreement", lp −1.18 > −1.93).
- Every one of the 46 model-columns sits at 0.184–0.238 clustered on 0.196; only phi-1.5 escapes (0.507).

**mmlu — first-token position bias.** 83% of mass on choice A (mmlu_marketing PRED {0:195, 1:12, 2:18, 3:9}); logprob top1−top2 margin median 0.125 nats (near-uniform). All 30 × 1.4B within 0.238–0.272 (std 0.009, flattest in the suite). Only phi-1.5 clears chance (0.437).

**wsc — two-constant collapse.** Gold 66 no / 38 yes (N=104) → always-no = 0.635, always-yes = 0.365. code_p1_half / code25b / C5-final all predict "yes" (coref) on **104/104** → 0.365; c5v3_half_p2 predicts "no" on 100/104 → 0.635. **No model beats always-no except by being always-no** — even phi-1.5 (0.606) < 0.635. The ubiquitous 0.365/0.635 cells are these two constants.

**cb — class collapse on 56 examples.** Gold 23 entail / 28 contra / 5 neutral (N=56, 1 example = 1.8%). Models collapse to always-entailment → 0.411 (= 23/56, appears in ~9 columns); c5v8r collapsed to neutral → 0.125. Best score (0.411) is below even the trivial always-contradiction majority (0.500).

## The Math floor is REAL (not broken scoring)

The user's core question, answered from raw resps: extraction works, the arithmetic doesn't.
- gsm8k_cot: **88.4% of outputs extract a number** (only 11.6% `[invalid]`), exact_match = 2.0% (52/2638). Janet-eggs (gold 18): model outputs "16−3=11 … The answer is 8" — correct format, wrong math (subtracts 3 repeatedly instead of 3+4).
- **Bounding argument kills broken-scoring:** even if every `[invalid]` were secretly correct, the ceiling is ~13.6% ≪ phi-1.5's 27.2%.
- minerva_math: scored by `math_verify` (no extraction regex to blame); only 10.1% of outputs even contain `\boxed`, 38.3% are literal repetition loops.
- gsm_noop: 117 items → scores quantize to k/117 (0.000 = 0, 0.009 = 1, 0.017 = 2 correct). The observed range is 0–2 correct items = pure noise; one lucky item = +0.009.
- gsm_symbolic re-instantiates gsm8k templates that already floor → carries no signal independent of gsm8k.
- **Latent trend the metric hides:** format-following improves with scale (300M loops into repetition, 22.9% `[invalid]`; 1.4B follows CoT cleanly, 11.6%) — invisible to exact_match.

## The bigcode-vs-lm-eval HumanEval artifact (Tier E)

Both harnesses execute **byte-identical** unit tests — bigcode's `custom_metrics/execute.py` is a vendored copy of HF `code_eval` (which lm-eval loads via `evaluate.load('code_eval')`); both build `candidate + "\n" + test_case`, run the same `check()`, same `estimate_pass_at_k` (byte-diffed). **bigcode is NOT stricter about what constitutes passing** — the common assumption is wrong. The gap is entirely a **prompt-format artifact**:
- **Root cause = a stripped trailing newline.** bigcode's `get_prompt` returns `doc['prompt'].strip()`; lm-eval uses `{{prompt}}` unstripped. The HumanEval prompt ends `…    """\n`; stripped, it ends `…    """`. For a weak/undertrained base model that removed newline flips it from *continuing inside the indented body* to *emitting `\n\n` then a de-indented top-level token that hits a stop word → empty body → auto-fail*. **49% (81/164) of c5v6's bigcode generations are empty stubs.** Same checkpoint, same problem (HumanEval/0): lm-eval = an 810-char working body; bigcode = `\n\n` and nothing.
- c5v6: lm-eval HE **0.213** vs bigcode **0.012** (17.5×). The ratio shrinks with model strength (code25b_clean 1.8×) and **INVERTS for phi-1** (bigcode 0.543 > lm 0.494) — strong models write an indented body regardless of the newline, so the harnesses converge. It is a capability-dependent artifact concentrated on weak models.
- **Secondary, NOT the driver:** bigcode `max_length_generation=512` is total prompt+gen vs lm-eval's 1024 generation-only; bigcode has 8 stop words vs 5. These matter only for long prompts; HumanEval/0's prompt is ~100 tokens, so they don't cause its empty stub.
- **So for our weak base models, lm-eval HE is the more faithful number.** Fix: use lm-eval HE for these models, or re-run bigcode with `strip_prompt=False`.

## Why the MC collapse happens — it's the scoring, not the few-shot examples

The position/letter collapse and length bias are a **loglikelihood-over-surface-forms artifact at weak scale**, NOT a few-shot-diversity problem (tested):
- The worst collapse (**commonsense_qa, 99.3% position-0** on 300m_c5v6) is **0-shot** — no demonstrations exist, so few-shot cannot be the cause.
- Where demos exist (**mmlu 5-shot**), they are **balanced** (A=67/B=69/C=71/D=78 across 57 subjects) and the model's letter bias is **anti-correlated** with them (world_religions demos = 3×B, model picks B only 6/171). If few-shot drove it, the model would over-pick the demo-frequent letter; it does the opposite.
- **arc_easy (25-shot) scores the answer TEXT, not letters** (`doc_to_choice = choices.text`; the dataset's A/B/C/D labels are never shown to the model or scored — unlike mmlu, which scores the letters), yet it still collapses — raw loglikelihood favors the **shortest** continuation (45.6% vs 11.3% longest; chance 25%).
- Severity is **independent of shot count** (0-shot commonsense_qa collapses harder than 25-shot arc).

Mechanism: at weak scale the model can't discriminate answer *content*, so per-choice loglikelihood is set by *surface form* — for single-letter answers, the highest-frequency letter token wins (content-independent, e.g. 300M→'A'/pos-0, 600M→'B'); for text answers, the shortest/most-frequent string wins. **More or more-diverse few-shot would not fix this.** acc_norm helps only the text/length case, not the equal-length letter case (where the model is genuinely uninformative).

## Doc / scoring bugs found
- **cb phi-1.5 cell = 0.464 is mis-filled** with phi-1's value; the real acc from samples is **0.643** (phi-1.5 is the only model that discriminates 3-way).
- **gsm8k[5]** is labeled "logprob" in §1 but the samples are free generation scored by exact_match on `#### N` (same mechanism as gsm8k_cot).
- **humaneval[0] (lm-eval)** is described as "regex-match on the generated function body" — it actually **executes** the unit tests.

## Category-mean distortion

| Category | tier C/D members | verdict |
|---|---|---|
| Math (standard) | 3/3 C | **meaningless** — constant ~0.01; gsm8k[5] & gsm8k_cot double-weight one skill |
| Math (perturbation-robust) | 2/2 C | **meaningless** — constant ~0.005; "drop" is undefined from a floor; 117-item quantization |
| Aggregate | 2/4 C | **half dead weight** — gpqa + agieval are fixed chance offsets diluting bbh + mmlu_pro |
| Closed-book NL (14) | 5/14 C/D | **heavily diluted** — lambada + arc_easy carry ~all variance; ~7 near-constant riders |
| Open-book (4) | 1/4 D | **biased** — boolq inflates code models; sciq is the sole real discriminator |
| Code (3) | 1/3 E | **2/3 trustworthy** — lm-eval HE + mbpp concordant; bigcode HE deflated (lowers Mean Code ~33%) |

## Recommendations (NOT applied — awaiting decision)

- **Drop from category means at this scale:** commonsense_qa, wsc, cb, winogrande, mmlu (Closed-book NL); boolq (Open-book, or report only vs its 0.622 baseline); gpqa_diamond + agieval_lsat_ar (Aggregate).
- **Both Math means:** collapse to a single footnote ("all 300M–1.4B recipes floor, EM 0.000–0.028"); don't compute a perturbation drop from a floor.
- **Code:** report lm-eval HE + mbpp as the code signal; do not average bigcode HE into the same mean.
- **Fixes:** re-score cb phi-1.5 cell (0.464 → 0.643); re-run or drop bigcode HE; report raw acc next to acc_norm for logiqa/arc_challenge; enable `log_samples` for quac; correct the gsm8k "logprob" and humaneval "regex-match" descriptions in §1.
- **Trustworthy headline set (tier A):** lambada_openai, arc_easy, sciq (as attend+extract), piqa, storycloze, copa (±N=100), mbpp, lm-eval humaneval; hellaswag joins at 1.4B.
