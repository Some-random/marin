# Experiment Log: Data Efficiency & Reasoning Pretraining

---

## ⭐ CURRENT GOALS & OVERARCHING HYPOTHESES (READ FIRST)

### Success criterion (revised May 27)

A recipe is judged against the capability it is supposed to teach, **not** uniformly across all evals. Trading tokens away from natural language to add (e.g.) code reduces NL exposure — small NL regressions are expected, not failures. The criterion has three parts:

1. **Target capability** — the capability the recipe is designed to teach (e.g. *reasoning* for the code-mix recipe).
   - Required: **strictly improves**, measured on probes that test the target capability under conditions where surface pattern-matching can't substitute (see H1 probes below).
2. **Substrate capability** — capabilities the model needs to even attempt the task: basic NL fluency, ability to read and write coherent English, absence of generation pathologies (looping, gibberish).
   - Required: **preserved within reason**. Measured by Paloma macro PPL, dclm_200m_val PPL, and generation smoke tests (gsm8k_cot loop check). Floor: if substrate collapses, the model can't be evaluated at all on the target.
3. **Non-target capabilities** — everything else (e.g. knowledge benchmarks, lexical pattern-matching tasks, code metrics for an NL-targeted recipe).
   - Tolerable: **small regressions allowed**, in proportion to the data trade-off. Disqualifying only if a non-target capability collapses or regresses far more than the data shift would explain.

Why three parts, not one uniform gate:
- "No regression on any eval" was too strict. If we swap 25% NL tokens for code tokens, NL knowledge benchmarks may see small drops — that's the price, not failure. The recipe should still be evaluated on whether it taught its target capability.
- The old framing's "reasoning quality" criterion was satisfied by passage-grounded extraction gains (May 26 code-mix: sciq +6pt, boolq +7pt) that were not actually reasoning. **Reasoning gains require a probe that controls for the extraction confound** — see H1 probes.
- Substrate-vs-non-target distinction matters: a model that can't write coherent English is unevaluable on reasoning, regardless of latent capability. Substrate has a floor; non-target capabilities have a budget.

**For the May 26 code-mix recipe specifically** (target capability = reasoning, per the H1 hypothesis it's supposed to test):
- Substrate: preserved (Paloma macro improved, no looping). ✅
- Non-target NL knowledge / lexical: flat or improved across the board. ✅ (better than expected — no trade-off paid)
- **Target (reasoning): undetermined.** Downstream improvement on sciq/boolq was extraction, not reasoning. H1 probe pending.

So the May 26 result has cleared the gate's *substrate* and *non-target* requirements but the *target* requirement is still open. The toy probe defined under H1 below is the test of the target requirement.

### Overarching research hypotheses

H1 and H2 are independent and both required. Without H1 (good reasoning data), H2 has nothing to retain. Without H2 (retention mechanism), H1 gains are erased. **Neither is testable from token-efficiency gains alone — both require dedicated capability probes** (see Probe Plans section).

**H1 — What kind of data teaches reasoning *capability* (not just domain knowledge or extraction skill)?**

The question is what STRUCTURE in pretraining data teaches transferable reasoning skill, separately from teaching domain knowledge or context-extraction skill. Many data types help WITHIN a domain (OWM → SciQ, code → HumanEval) but don't transfer (OWM hurts ARC/PIQA, code hurts NL). Code-mix at our scale appears to help context-grounded extraction (sciq, boolq) without teaching reasoning per se — distinguishing these requires probes that controlly vary "is the answer extractable from surface" vs. "is an algorithm required".

**Candidate experiments under H1** (tests *of* H1, not the hypothesis itself):
- Aryabumi-style code mix at 1.4B (DONE May 26 — passed token-efficiency gate, but mechanism analysis suggests extraction not reasoning; algorithmic-capability probes pending).
- Synthetic structural data (formal languages, procedural data) — does abstract structure transfer to NL reasoning?
- Domain-controlled data (math-only, code-only, mixed) — what's the transfer pattern?

**Probes under H1** — distinguish "improved extraction" from "acquired algorithm". Designed to work at 1.4B / 3.3B-token scale where standard reasoning benchmarks are at-random.
- **Output-probability conditioning probe** (McCoy "Embers of Autoregression" style) — same algorithmic task at two output-frequency tiers. If a recipe improves both equally, algorithmic; if only the high-frequency tier, surface.
- **Counterfactual task variants** (Wu et al. "Reasoning or Reciting" style) — apply the algorithm under a non-default rule (e.g. base-9 arithmetic, mod-7 addition). Surface pattern-matching fails on the counterfactual; genuine algorithmic capability transfers.
- **Reversal Curse** (Berglund) and **SCAN/COGS compositional generalization** as harder optional probes.

**What's been ruled out for H1**: pure OpenThoughts / OWM / code-only — all failed token-efficiency by hurting NL benchmarks at our scale (300M–1.4B).

**H2 — Once a model has reasoning capability, how do we retain & use it through general pretraining?**

Even if H1 is solved (we find data that builds reasoning capability), two failure modes can erase that capability during subsequent general web text training:

- **H2a — Catastrophic forgetting**: web text overwrites the reasoning representations. *Candidate mitigation*: replay — mix a small fraction of reasoning data throughout web text training. **Untested at our scale.**
- **H2b — No training pressure to use reasoning circuits**: even if reasoning circuits exist after phase 1, next-token prediction on standard web text doesn't activate them, so they sit dormant — replay alone doesn't fix this. *Candidate mitigations*: perplexity-filtered web text (train only on documents the reasoning-capable model finds surprising), joint training objectives that tie reasoning eval to web prediction. **Speculative and untested.**

**Probes under H2** — measure capability decay over continued training.
- Run an H1-passing recipe for phase 1, then continue training on standard web text for varying durations; re-run the H1 capability probes at intervals. Decay curve answers H2a.
- Compare decay curves with and without replay (H2a mitigation), with and without perplexity-filtered web text (H2b mitigation).

### Status of work as of May 27

- **May 26 Aryabumi code-mix probe** passed the token-efficiency gate (Paloma macro −0.47 nats, sciq +6.2pt, boolq +7.7pt, no looping regression). **Mechanism inspection** revealed gains came from passage-grounded extraction, not reasoning — the recipe satisfies the gate but its H1 status (does it teach reasoning capability) is undecided.
- **Next step**: H1 capability probes (Embers-style + Wu-style counterfactual) on the May 26 baseline + code-mix checkpoints — see Probe Plans section. This is the first attempt to test H1 directly rather than via downstream-benchmark proxies.
- **Wide eval suite** in progress on both checkpoints (13 logprob task groups + 4 generation tasks, 8-GPU data-parallel via accelerate). Results will populate the Evaluation Taxonomy with actual per-task numbers.
- **Looping investigation (Steps 1–10)** is **methodology cleanup**, complete. wd=1.6/x16/block=False (`peach-thunder-100` / `6xx0hu3l`) is the chosen non-looping baseline that all H1 candidates compare against.

### Evaluation taxonomy — what each eval *actually* tests (read the data, not the name)

Names like "NL reasoning benchmark" hide important mechanistic differences. Below we classify every eval we use by the actual cognitive mechanism a model needs to score above chance, based on inspection of per-example samples. **Always sample the data before reasoning about why scores differ.**

#### A. Continuous PPL (no task structure, just next-token loss)
- **Paloma macro** (16 subsets, see `run_1_4b_25code_alg.py`) — domain-diverse held-out web/forum/code text. Primary continuous signal at our scale. Sensitive to general LM quality and to per-domain coverage.
- **dclm_200m_val** — held-out NL within our training distribution. Sensitive to overfitting on the 209M-token DCLM training slice.
- **opc_algorithmic** (when used as code training data) — final eval loss on the code training slice; signals code memorization, not generalization.

#### B. Passage-grounded reading comprehension (answer is *in the prompt*; model just needs to attend and extract)
The model does NOT need to know the answer in its weights — the passage contains or strongly implies it.
- **sciq** — every question comes with a `support` paragraph that *literally states the answer* (e.g. "what is X called?" → support: "X is called Y"). 4-way MC. Tests context-attention and lexical matching, not science knowledge.
- **boolq** — yes/no question + Wikipedia-style passage that contains the answer. Tests passage-question matching.
- **openbookqa** (if used) — provides a relevant fact alongside the question.

#### C. Parametric world knowledge (no passage; the model must recall facts from weights)
- **arc_easy** — 4-way MC science questions, no support. Tests grade-school science recall (photosynthesis, meiosis, safety equipment, ...). Above-random at our scale.
- **arc_challenge** — harder version, mostly at-random at 1.4B/3.3B.
- **mmlu** — 4-way MC across 57 domains, no passage. At-random at our scale.
- **triviaqa / naturalqs** — generation, no passage. At-random at our scale.

#### D. Physical/social commonsense (no passage; intuition from weights)
- **piqa** — 2 alternative everyday-task descriptions; pick the physically plausible one ("paper bedding" vs "jeans bedding" for a guinea pig). Above-random at our scale.
- **social_iqa** — social-situation MC. At-random at our scale.
- **hellaswag** — sentence-completion plausibility. Mostly at-random at our scale.

#### E. Coreference / logical / linguistic
- **winogrande** — pronoun-resolution pairs (Winograd schema). At-random at our scale.
- **logiqa** — formal logical-reasoning MC. At-random at our scale.
- **blimp** — minimal-pair grammaticality judgment. Linguistic, mostly above-random; not used as primary signal.

#### F. Math (multi-step generation)
- **gsm8k_cot** — grade-school math with 8-shot CoT, free generation. At our scale all 1.4B runs score 0/20. Used as a **looping smoke test**: does the model emit short sensible-shaped responses (`The answer is X.\n\n`) or n-gram loops?
- **gsm8k** (logprob variant) — multiple-choice; at-random at our scale.
- **minerva_math** — competition math, generation. At-random at our scale.

#### G. Code generation
- **HumanEval / MBPP** — function generation. At-random pass@1 at our scale.

#### What's usable as outcome metric at our scale (1.4B, 3.3B tokens)
- **Continuous (always usable)**: Paloma macro, dclm_200m_val
- **Discrete above-random**: arc_easy (D), sciq (B), piqa (D), boolq (B). All others should be logged but treated as noise.
- **Looping smoke test**: gsm8k_cot generation samples (qualitative, not exact_match).

**Key mechanism note (from May 26 code-mix result):** the four above-random benchmarks span two distinct cognitive mechanisms — sciq+boolq are passage-grounded (B), arc_easy+piqa are knowledge-from-weights (C/D). Interventions that change attention/extraction (e.g. code data with input→output structure) can move (B) without moving (C/D). Always classify benchmark deltas by mechanism, not by name.

---

### Historical hypotheses superseded by the H1/H2 framing above

- **Causal bridge** (May 11) — old candidate for H1; Wikipedia-wikilink conditional generation. Shelved.
- **OWM curriculum / OpenThoughts injection** (May 1–10) — tested at 300M–1.4B, failed all three criteria (only SciQ improved, ARC/PIQA degraded).
- **Procedural knowledge / Dyck / NCA** (May 4 + lit review May 17–21) — explored as H1 candidates; not pursued empirically beyond initial 300M procedural-knowledge runs.

---

## May 28: Phi-1 / phi-1.5 four-way comparison + open-data sourcing plan

*Goal: get apples-to-apples reference for what's possible at 1.3B params with the right data, and plan whether to attempt phi-1-style or phi-1.5-style training at our scale.*

### What ran

- Pulled `microsoft/phi-1` (1.3B, ~7B training tokens, code-only) and `microsoft/phi-1_5` (1.3B, ~30B training tokens, code + NL) from HuggingFace.
- Ran both through the SAME lm-eval-harness pipeline + n-shot settings as our 1.4B baseline + code-mix runs (25-shot arc, 10-shot hellaswag, 5-shot winogrande/mmlu/gsm8k, 0-shot rest, gen tasks with `HF_ALLOW_CODE_EVAL=1`).
- Script: `experiments/data_efficiency/run_phi_evals.sh`. Output: `outputs/eval_results/phi_evals_20260527_2257/`.
- 8-GPU data-parallel via accelerate. phi-1: 22:57→23:29 PDT (~32 min). phi-1.5: 23:29→23:59 PDT (~30 min). Total ~62 min wall.

### Full 4-way comparison (all numbers from our pipeline)

Random column shows chance accuracy for that task. `acc_norm` used for arc/hellaswag/openbookqa, `acc` elsewhere.

| Task | n-shot | Random | **1.4B base (3.3 B tok)** | **1.4B code25 (3.3 B tok)** | **phi-1 (7 B tok)** | **phi-1.5 (30 B tok)** |
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

### Side note: leaderboard n-shot reruns on our 1.4B models (May 27 evening)

Before the phi runs we re-ran arc_easy/arc_challenge/hellaswag/winogrande at OpenLLM Leaderboard n-shot counts (25/25/10/5) on both 1.4B checkpoints. The numbers in the table above use these reruns where applicable.

Notable: **arc_easy code-mix-vs-baseline Δ flipped sign with shot count**:
- 0-shot: baseline 0.388 vs code-mix 0.386 → code-mix **−0.2 pt**
- 25-shot: baseline 0.401 vs code-mix 0.416 → code-mix **+1.5 pt**

Code-mix gained +3.0 pt from going 0→25 shot; baseline only +1.3 pt. Consistent with the "code data improves context attention/extraction" story — more in-context examples → bigger ICL gain for the code-mix model. Same direction as sciq/boolq passage-grounded gains, smaller magnitude. arc_challenge/hellaswag/winogrande deltas stayed within noise across n-shot changes.

### Findings

**1. We are slightly BETTER than phi-1 on NL benchmarks** — by 2-5pt on piqa, boolq, hellaswag. This is consistent with phi-1 being a code-only model: its NL ability is no better than ours despite phi-1's "high-quality data" framing, because their training was almost entirely code. The phi-1 paper doesn't report NL benchmarks because they're not the point of that model.

**2. phi-1 destroys us on HumanEval** — 49.4% vs 0.6%. This is the apples-to-apples evidence that **the right code data unlocks real code-generation capability at 1.3B params and 7B training tokens**. We have a similar parameter count and similar token budget; the only difference is data quality (filtered Stack + GPT-3.5 synthetic Python textbooks/exercises vs our unfiltered DCLM + opc_algorithmic Q&A pairs).

**3. phi-1.5 with 9× our tokens lifts EVERY benchmark off the floor**:
  - arc_easy 0.42 → 0.80 (+38pt), arc_challenge 0.24 → 0.53 (+29pt) — both far above random
  - mmlu 0.25 → 0.42 — first time we see a 1.3B class model meaningfully above random
  - gsm8k 0% → 30.5% — explicit "solve via Python emission" capability, possible because the code half of phi-1.5's training is preserved
  - commonsense_qa 0.20 (random) → 0.51 — emerges only at this combo of scale + data
  - humaneval 0.49 (phi-1) → 0.34 (phi-1.5) — small drop from adding NL data; mbpp similar

  This is the empirical answer to "is there a way to get something out of nothing at our scale": **yes, but you need ~10× more training tokens AND the data quality discipline of phi-1.5**, not just one or the other.

**4. MBPP discrepancy** — we measured phi-1 MBPP pass@1 at 1.0% in our pipeline, vs the paper's 55.5%. The Open LLM Leaderboard / lm-eval-harness MBPP scoring uses a specific extraction pattern and 0-shot setup that's likely undercounting. The paper used a different code-eval framework (BigCode evaluation harness). Our gsm8k_cot at 7% for phi-1.5 vs paper's 40.2% (via "Python emission") is similar — different methodologies. **Caveat: our pipeline's code-gen numbers are not directly comparable to published phi numbers; treat as conservative lower bounds.**

### Implication for H1

The May 26 Aryabumi-style code mix gave 0.6% HumanEval. phi-1 gave 49%. Same model size, similar token budget, *different code data*. So the H1 conclusion sharpens:

> **Not all code data is equal.** Off-the-shelf code Q&A (opc_algorithmic) at 25% mix did not transfer to HumanEval. Phi-style filtered+synthetic textbook code at higher mix DID. The H1 hypothesis "code helps reasoning" is still alive but **conditional on data type and curation**.

Whether phi-1.5-style mix would unlock NL reasoning at our scale is the next H1 question. The phi-1.5 result shows it's *possible*, but requires ~30B tokens — about 9× what we trained the May 26 recipe on.

### Open-data sourcing plan (no Microsoft data available)

Microsoft never released phi-1 / phi-1.5 training data. Closest open substitutes (per dataset cards):

**Phi-1-style code mix (~7 B tokens, ~12-22 GB download)**

| Component | Dataset | Tokens | License | Note |
|---|---|---|---|---|
| Filtered Python (educational) | `HuggingFaceTB/smollm-corpus/python-edu` | ~4 B (per SmolLM blog) | ODC-BY | Stack-v2 scored ≥4 by edu classifier; metadata only, content via S3 |
| Synthetic Python exercises | `jinaai/code_exercises` | ~120 M | **CC-BY-NC-SA (NON-COMMERCIAL)** | GPT-3.5-generated, Python only; closest open clone of phi-1's `CodeExercises` |
| (commercial alt) | `nampdn-ai/tiny-codes` | unstated, ~1.6 M rows / 981 MB | MIT | Multi-lang, lower QC |

**Phi-1.5-style code+NL mix (~30 B tokens, ~190 GB download)**

| Component | Dataset | Tokens | License |
|---|---|---|---|
| Phi-1 code base (above) | python-edu + code_exercises | ~7 B | mixed |
| Synthetic NL textbooks | `HuggingFaceTB/smollm-corpus/cosmopedia-v2` | ~28 B | Apache-2.0 |
| (optional add) | `HuggingFaceTB/smollm-corpus/fineweb-edu-dedup` | ~220 B (subsample) | ODC-BY |

License watch:
- jinaai/code_exercises is **non-commercial**; replace with tiny-codes for commercial release.
- `open-phi/textbooks` has **no license listed**; don't use without clarification.
- Microsoft's original phi-1 / phi-1.5 weights themselves are research-license (non-commercial), but we're only running inference on them for comparison — that's fine.

### Compute estimates at our 1.4B / 8× A100-40GB

Our May 26 run: 3.34 B tokens, 7h 40min wall → **~435 M tokens/hour** throughput at 1.4B / bs=64 / seq=4096.

| Target | Unique tokens | Epochs | Total tokens | Wall time |
|---|---|---|---|---|
| Phi-1-scale, 1 epoch | 7 B | 1 | 7 B | ~16 h |
| Phi-1-scale, paper's 8 epochs | 7 B | 8 | 56 B | ~5.4 days |
| Phi-1.5-scale, 1 epoch | 30 B | 1 | 30 B | ~2.9 days |
| Phi-1.5-scale, paper's 5 epochs | 30 B | 5 | 150 B | ~14.4 days |

**Storage**: phi-1 mix fits comfortably on the local FS (~22 GB raw + ~10 GB tokenized). Phi-1.5 mix (~190 GB raw + ~80 GB tokenized) also fits.
- Current `/fsx` usage: 34 TB / 39 TB (87% used), **5.0 TB free**.
- Our existing footprint: `outputs/tokenized` 113 GB, `outputs/raw` 6.6 TB, `checkpoints` 738 GB.
- Phi-1.5 mix would add ~270 GB total — ~5% of remaining free space, fine.

### Recommendations / open decisions for tomorrow

Three paths, in increasing cost:

1. **(0.5-1 day)**: Run the toy reasoning probe (already drafted in chat) on our current models to settle "did Aryabumi 25% code teach algorithmic capability, even tiny?" Doesn't need new data.
2. **(~6 days compute)**: Phi-1-style 1.4B replication. Download `python-edu` (S3 fetch step) + `jinaai/code_exercises` (or tiny-codes for commercial). Train 1.4B for 8 epochs on the ~7B-token mix. Target: HumanEval > 0, confirming "right code data → code capability at our scale" *with open data*.
3. **(~14 days compute, ~190 GB download)**: Phi-1.5-style 1.4B replication. Train on python-edu + code_exercises + cosmopedia-v2 for 5 epochs. Target: reproduce phi-1.5's NL reasoning lift (arc_easy ~0.80, mmlu ~0.42, gsm8k ~0.30) with fully open data. Most informative result but biggest commitment.

Open questions (need user input before proceeding):
- Which path? Toy probe first (cheap, narrow), or jump to phi-1 replication (medium, high-information)?
- For path 2 / 3: any constraint on multi-day GPU occupancy?
- Commercial use needed? (affects jinaai vs tiny-codes choice)
- Free space on `/fsx/users/dongweij/marin/`?

---

## May 27: Wide benchmark suite on baseline vs code-mix — confirms extraction-not-reasoning

*Direct follow-up to the May 26 code-mix probe. The 4-benchmark comparison there left open: does code-mix help on **any** reasoning-flavored task at our scale? Wide eval suite ran today on both checkpoints to answer that.*

### Setup

- Models: `peach-thunder-100` / `1_4b_wd1_6_x16_nocrossblock_hf` (baseline, 0% code) and `eager-grass-104` / `1_4b_25code_alg_hf` (code-mix, 25% opc_algorithmic).
- Script: `experiments/data_efficiency/run_wide_evals.sh` + `run_wide_evals_resume.sh`. Output: `/fsx/users/dongweij/marin/outputs/eval_results/wide_eval_20260527_1343/`.
- Parallelism: `accelerate launch --multi_gpu --num_processes 8` — each rank holds a full 1.4B model copy (fits 40 GB), processes 1/8 of requests. Logprob batch=32/dev, gen batch=8/dev.
- Tasks (17 task groups, mmlu expands to 57 subtasks):
  - Logprob: arc_easy, arc_challenge, sciq, piqa, boolq, hellaswag, winogrande, openbookqa, commonsense_qa, social_iqa, logiqa, mmlu, gsm8k
  - Generation: humaneval, mbpp, gsm8k_cot, minerva_math
- Required env: `HF_ALLOW_CODE_EVAL=1` (separate from `--confirm_run_unsafe_code` flag) for HumanEval/MBPP to run.

### Full comparison (n covers full eval split per task)

| Category | Task | Random | Baseline | Code-mix | Δ | Notes |
|---|---|---:|---:|---:|---:|---|
| **Passage-grounded extraction** | sciq | 0.25 | 0.652 ±0.015 | **0.709 ±0.014** | **+5.7 pt (~3σ)** | answer is in `support` paragraph |
| | boolq | 0.50 | 0.502 ±0.009 | **0.579 ±0.009** | **+7.7 pt (~9σ)** | answer in `passage`; baseline at random |
| **Parametric knowledge / commonsense** | arc_easy | 0.25 | 0.418 ±0.010 | 0.407 ±0.010 | −1.1 pt | non-target |
| | piqa | 0.50 | 0.634 ±0.011 | 0.619 ±0.011 | −1.6 pt | non-target, ~1.5σ |
| **At-random (no signal at our scale)** | arc_challenge | 0.25 | 0.218 ±0.012 | 0.213 ±0.012 | −0.5 pt | both below random |
| | mmlu (57 sub) | 0.25 | 0.252 ±0.004 | 0.249 ±0.004 | −0.3 pt | both at random |
| | hellaswag | 0.25 | 0.307 ±0.005 | 0.312 ±0.005 | +0.5 pt | both slightly above |
| | winogrande | 0.50 | 0.490 ±0.014 | 0.504 ±0.014 | +1.3 pt | both at random |
| | openbookqa | 0.25 | 0.180 ±0.017 | 0.184 ±0.017 | +0.4 pt | both BELOW random |
| | commonsense_qa | 0.20 | 0.192 ±0.011 | 0.200 ±0.011 | +0.8 pt | both at random |
| | social_iqa | 0.33 | 0.366 ±0.011 | 0.362 ±0.011 | −0.5 pt | both ~random |
| | logiqa | 0.25 | 0.218 ±0.016 | 0.234 ±0.017 | +1.5 pt | both at random |
| **Math (floor)** | gsm8k (MC) | 0 | 0.015 ±0.003 | 0.024 ±0.004 | +0.8 pt | both near zero |
| | gsm8k_cot (gen) | 0 | 0.024 ±0.004 | 0.022 ±0.004 | −0.2 pt | both near zero, no looping |
| | minerva_math | 0 | 0.0002 | 0.0002 | 0 | both effectively zero |
| **Code (floor)** | humaneval | 0 | 0.000 ±0 | 0.006 ±0.006 | +0.6 pt | 1/164 problems passed |
| | mbpp | 0 | 0.000 ±0 | 0.000 ±0 | 0 | both zero |

### Applying the 3-part success criterion (target / substrate / non-target)

| Part | Status | Evidence |
|---|---|---|
| **Substrate** (NL fluency, no generation pathology) | ✅ preserved | Paloma macro improved by 0.47 nats across every subset; no looping on gsm8k_cot |
| **Non-target** (NL knowledge, lexical commonsense) | ✅ within budget | arc_easy −1.1pt, piqa −1.6pt; both within ~1.5σ. Small regressions consistent with trading 25% NL tokens for code |
| **Target** (reasoning capability) | ❌ **no signal** | All math benchmarks (gsm8k, gsm8k_cot, minerva_math) at floor for both models. HumanEval +0.6pt = 1/164 problems = noise. All at-random NL benchmarks (arc_challenge, mmlu, hellaswag, logiqa) flat or random-walk |

### Interpretation

The wide eval confirms the May 26 nuance: **code-mix at our scale is a token-efficiency win for passage-grounded extraction, but produces no measurable reasoning capability on any standard reasoning benchmark.** The two large gains (sciq +5.7pt, boolq +7.7pt) come from tasks where the answer is in the prompt; everything that requires generating a multi-step solution or recalling parametric knowledge is at floor.

So the May 26 "Aryabumi-effect reproduces" headline needs a sharper qualifier: it reproduces *the Paloma-PPL and downstream-MC parts* of the Aryabumi effect at our scale, but the "code helps NL reasoning" claim (the most interesting part of the Aryabumi paper) is **not testable here** — every reasoning-flavored benchmark is at-random for both models.

### Open question (handed to the H1 probe design)

Why does every reasoning benchmark floor for both models? Most likely: 1.4B at 3.34B training tokens is severely under-Chinchilla (Chinchilla optimal for 1.4B is ~28 B tokens — we're at ~12% of that). Reference points: Pythia-1.4B (300B tokens), TinyLlama-1.1B (3T tokens), OLMo-1B (4T tokens), Llama-3.2-1B (9T tokens) — every public 1.4B model that scores above-random on reasoning was trained on 100–3000× more tokens than we have here.

We have two options:
1. Build a probe that gives signal even at floor (the toy probe — Embers-style output-prob conditioning + Wu-style counterfactual addition). Doesn't require improving the model.
2. Train longer / on better data so standard benchmarks lift off the floor.

These aren't mutually exclusive but option 1 is much cheaper.

---

## May 26: Aryabumi code-mix probe — result

*Same research thread as the May 25 planning section. Continuation of the active H1 hypothesis.*

### Aryabumi code-mix probe (run `eager-grass-104` / `p2n84bo3`) — RESULT

Training: 1.4B, 12,800 steps × bs=64 × seq=4096 = 3.355B tokens. 75% DCLM (`konwoo/dclm-164k-docs-train`, 209M tokens, ~12 epochs) + 25% opc_algorithmic (Python competitive-programming QA, 943M tokens, ~3.5 epochs over the slice). Hyperparams identical to baseline `peach-thunder-100` (LR=1e-3 cosine, WD=1.6, x16, block_cross_document_attention=False, seed=0).

Started 2026-05-26 ~20:40 PDT (after one earlier crash at step ~3.17k restarted via nohup), finished 2026-05-26 23:00 PDT. Total ~7h40min wall clock. WandB: <https://wandb.ai/dongwei_jiang/dongwei-data-efficiency/runs/p2n84bo3>.

#### Paloma macro PPL (final, step 12799) — strict improvement across every subset

| Subset | Baseline `peach-thunder-100` (0% code) | **`eager-grass-104` (25% code)** | Δ |
|---|---|---|---|
| paloma 4chan | 3.64 | **3.25** | −0.39 |
| paloma c4_100_domains | 4.27 | **3.89** | −0.38 |
| paloma c4_en | 4.55 | **4.15** | −0.40 |
| paloma dolma-v1_5 | 4.35 | **3.93** | −0.42 |
| paloma dolma_100_programing_languages | 4.05 | **3.37** | **−0.68** *(largest NL-subset gain; code training transfers directly to code-adjacent text)* |
| paloma dolma_100_subreddits | 4.59 | **4.19** | −0.40 |
| paloma gab | 6.48 | **5.81** | **−0.67** |
| paloma m2d2_s2orc_unsplit | 4.16 | **3.82** | −0.35 |
| paloma m2d2_wikipedia_unsplit | 4.07 | **3.73** | −0.33 |
| paloma manosphere_meta_sep | 4.57 | **4.18** | −0.39 |
| paloma mc4 | 4.40 | **4.00** | −0.39 |
| paloma ptb | 5.12 | **4.71** | −0.41 |
| paloma redpajama | 4.46 | **3.99** | −0.48 |
| paloma twitterAAE_HELM_fixed | 7.79 | **6.74** | **−1.05** *(largest single-subset gain; baseline was 7.79 → very far from ground truth, easier to move)* |
| paloma falcon-refinedweb | 4.67 | **4.27** | −0.40 |
| paloma wikitext_103 | 4.20 | **3.85** | −0.35 |
| **paloma macro (16 subsets)** | **~4.71** | **~4.24** | **−0.47** |
| dclm_200m_val (held-out) | 4.07 | **3.73** | −0.34 |
| dclm_200m (train data) | 1.63 | 1.96 | +0.33 *(less memorization, expected with regularization)* |
| opc_algorithmic (train data, code-specific) | — | 0.29 | — |

Pattern: code-mix model fits the NL training data *less* (higher train loss on dclm_200m) but generalizes much better (lower loss on every held-out NL subset). Consistent with code acting as a regularizer that prevents over-memorization of the 209M-token DCLM slice.

#### Above-random downstream benchmarks

| Benchmark | Baseline acc ±stderr | **25% code acc ±stderr** | Δ | Significance |
|---|---|---|---|---|
| arc_easy | 0.418 ±0.010 | 0.408 ±0.010 | −0.93 pt | within noise |
| sciq | 0.649 ±0.015 | **0.711 ±0.014** | **+6.20 pt** | ~3σ |
| piqa | 0.633 ±0.011 | 0.621 ±0.011 | −1.20 pt | within noise (1.1σ) |
| boolq | 0.502 ±0.009 | **0.579 ±0.009** | **+7.74 pt** | ~9σ |

#### Looping (gsm8k_cot, limit=20, 8-shot CoT)

Both models score 0/20 exact_match (expected at 3.34B-token scale — neither baseline nor code-mix has math capability). Critically, **the code-mix model does NOT loop**: generations are short (median ~50 chars, no n-gram repetition), preserving the non-looping behavior of the wd=1.6/x16/block=False baseline. Sample 0: `Janet eats 16 eggs per day. ... The answer is 4.\n\n` — terminates cleanly, no repetition.

#### Step 12 confirm/refute criteria — applied

From the May 25 plan:
> **Confirm**: Paloma macro improves AND ≥2 of {arc_easy, sciq, piqa, boolq} strictly improve AND no benchmark falls below baseline-minus-noise.

- ✅ Paloma macro strictly improves (~−0.47 nats, every single subset lower).
- ✅ 2 of 4 benchmarks strictly and significantly improve (sciq +6.2pt ~3σ, boolq +7.7pt ~9σ).
- ✅ No benchmark falls below baseline-minus-noise (arc_easy and piqa regressions are within 1σ).
- ✅ No regression in generation behavior (no looping).

**Result: Aryabumi-style code-mix effect REPRODUCES at our 60×-smaller scale with the open `opc-annealing-corpus/algorithmic_corpus` Python QA subset.** This is a positive H1 finding: 25% code mixed with DCLM text strictly improves NL performance at 1.4B / 3.34B-token scale on the Paloma macro and on 2 of 4 above-random benchmarks, without harming the others or causing generation regressions.

#### Caveats and open questions

- **Scale**: this is 60× fewer tokens than Aryabumi (200B). Effect size could shrink or grow at scale.
- **Data**: we used open `algorithmic_corpus` (Python QA), not Aryabumi's proprietary "Python programming problems formally verified" set. The mechanism that gives both the boost may not be identical.
- **Sciq + boolq, but not arc_easy/piqa — mechanism inspection (read the actual samples):** the split is not random; the two that improved are both **passage-grounded reading comprehension**, the two that didn't are both **knowledge-from-weights**. See the Evaluation taxonomy section above (B vs C/D).
    - sciq sample: question `"Compounds that are capable of accepting electrons, such as o2 or f2, are called what?"` comes with `support: "Oxidants and Reductants Compounds that are capable of accepting electrons, such as O 2 or F2, are called oxidants ..."` — the answer is **literally in the passage**. Task = attend to support, lexical match.
    - boolq sample: question `"is house tax and property tax are same"` with `passage: "Property tax or 'house tax' is a local tax on buildings..."` — passage contains the answer. Task = passage-question matching.
    - arc_easy sample: `"Which statement best explains why photosynthesis is the foundation of most food webs?"` — no support, must recall biology from weights.
    - piqa sample: `"How do I ready a guinea pig cage?"` with `sol1: paper bedding` vs `sol2: jeans bedding` — no support, must have physical intuition in weights.
    - **Interpretation**: 25% code data appears to improve the model's ability to *attend to and extract from provided context*, not its parametric world knowledge or physical commonsense. The +6/+7pt gains on sciq/boolq are a **reading-comprehension / context-attention** effect, not a generic "NL reasoning" effect. This is consistent with code being dense in "given input → produce structured output" patterns (functions operating on arguments, problem statements followed by solutions).
    - This nuance should temper the headline: we should NOT claim "code helps NL reasoning" in general. The honest claim is "code helps passage-grounded extraction tasks; effect on parametric-knowledge tasks is null".
- **What's the active ingredient?**: code itself, or the Q&A structure, or just "more high-quality text"? Would need to compare against a 25% addition of non-code high-quality text (e.g. wikipedia subset) to isolate.
- **Confound vs baseline**: data_seed and total token budget match; data ordering differs (DCLM shuffles in 75% rate + code interleaved). No obvious confound, but ordering effects at 3.34B-token scale haven't been measured here.

#### Files & artifacts

- Training script: `experiments/data_efficiency/run_1_4b_25code_alg.py`
- Tokenization step: `experiments/data_efficiency/code_data_alg.py`
- Tokenized data: `/fsx/users/dongweij/marin/outputs/tokenized/opc_algorithmic-ffc825/` (943M tokens, 5.3M docs)
- Levanter checkpoint: `checkpoints/1_4b_25code_alg/p2n84bo3/step-12799/`
- HF checkpoint: `checkpoints/1_4b_25code_alg_hf/`
- Eval results: `outputs/eval_results/25code_alg_gsm8k/`, `outputs/eval_results/25code_alg_4bench/`, `outputs/eval_results/baseline_nocross_4bench/`
- Training log: `logs/1_4b_25code_alg_20260526_203726.log`

---

## May 25: Cross-doc-attention ablation + WD-vs-epochs ablation + Aryabumi code-mix planning

This section is newest-first within the day. The code-mix experiment design is its own thing (pivot to the active H1 hypothesis); Steps 10 and 11 continue the looping investigation that started May 23.

### Aryabumi-inspired code-mix experiment design — evening discussion

*This is the start of a new research thread (active H1 hypothesis: what data teaches reasoning capability?), separate from the Steps 1–11 looping investigation.*

After the looping investigation closed, attention turned to the active hypothesis (H1 from the header): what data teaches reasoning capability without hurting NL? Aryabumi (2408.10914) is the closest published result — 25% code at 470M/2.8B / 200B-token scale yields +8.2% NL reasoning, +4.2% world knowledge, 12× code. Goal: probe whether this transfers to our 1.4B / 3.34B-token regime with **open-source** code data (Aryabumi's synthetic Python is proprietary).

#### What we figured out today about the data

**Aryabumi paper re-read (with exact quotes from Section 2.1):**
- Web Stack: "We apply quality filters" — only filtering, not verification
- Synthetic Code: "Python programming problems that have been **formally verified**" — verification is the explicit quality marker. The paper "treat[s] this as a high-quality source" specifically because of verification.
- So the implicit mechanism Aryabumi proposes is: **verified code is high-quality code, and high-quality code teaches reasoning**. Synthetic-vs-human-written is NOT the operative axis the paper isolates.

**OpenCodeReasoning is NOT a faithful Aryabumi-synthetic proxy** (verified after reading the OCR README + sampling rows):
- OCR's `solution` field is **human-written competitive programming code** from codeforces/codechef/atcoder/aizu/hackerearth, test-case verified
- Aryabumi's synthetic is **AI-generated and "formally verified"**
- Both are Python ✓, both are problem-solutions ✓, but origin differs (human vs synthetic) and verification standard differs (test-case vs formal)
- Best characterization: OCR-solution is "test-case-verified competitive Python," not "AI-generated formally-verified Python"
- The original `aryabumi_code_synth_solution` naming is misleading; better names: `ocr_solution` or `verified_python_ocr`

**Better open candidate identified: `OpenCoder-LLM/opc-annealing-corpus`** (Huang et al. 2024, arxiv 2411.04905):
- License: odc-by (clean open license)
- Three subsets, each tested in OpenCoder paper ablations:
  - `algorithmic_corpus`: curated algorithmic code from The Stack v2
  - `synthetic_code_snippet`: AI-rewritten code (rewrites of algorithmic_corpus seeds)
  - `synthetic_qa`: AI-generated code Q&A pairs
- **Caveat**: the OpenCoder paper evaluates *code* capabilities (HumanEval/MBPP), NOT general NL reasoning. So while the data is published with effectiveness evidence, that evidence is for code performance, not the NL-reasoning gain we actually care about. Using this data for our experiment is a **novel probe** of whether the same data also helps NL reasoning at our scale.

**Scale honesty**: Aryabumi trained 470M and 2.8B for 200B tokens (~60× more than our 3.34B total). At our scale most NL reasoning benchmarks our 1.4B baseline scores at-or-below random:

| Benchmark | Random | Our 1.4B baseline | Above random? |
|---|---|---|---|
| sciq | 25% | 71.7% | +47 ✓ strong |
| arc_easy | 25% | 43.6% | +19 ✓ usable |
| piqa | 50% | 62.6% | +13 ✓ usable |
| boolq | 50% | ~60% | +10 ✓ usable |
| arc_challenge | 25% | 23.5% (norm) | random ✗ |
| hellaswag | 25% | 26.4% | random ✗ |
| winogrande | 50% | 48.7% | random ✗ |
| mmlu | 25% | ~23% | random ✗ |
| commonsense_qa | 20% | 19.9% | random ✗ |
| social_iqa | 33% | 35.3% | random ✗ |
| logiqa | 25% | 21.2% | random ✗ |
| openbookqa | 25% | ~17% | below random ✗ |

4 of 12 benchmarks have signal at our scale. The Aryabumi-style aggregate (+8.2% averaged across all 11) is **not measurable** in our regime — averaging mostly-noise dilutes any signal. We have to scope the eval accordingly.

#### Refined experiment plan (Aryabumi-inspired probe, scaled to our regime)

**1. Hypothesis.** Mixing 25% high-quality code (`opc-annealing-corpus`) with 75% DCLM during pretraining improves NL performance at our 1.4B / 3.34B-token scale, measured by metrics that have signal at our scale.

**2. Why.** Aryabumi published a +8.2% NL reasoning gain at 470M-2.8B / 200B tokens. Two questions:
- Does the effect direction hold at our 60×-smaller-data regime?
- Does an *open* high-quality code corpus (OpenCoder annealing data) replicate it, even though that data was originally evaluated on code, not NL?

**3. Why this configuration.** Single-variable change from our baseline: replace 25% of the text mix with high-quality code. Hold all other variables (model, recipe, eval set, seed) constant.

**4. Data.**
- Text base: `konwoo/dclm-164k-docs-train` (209M tokens) — same as our existing baseline
- Code source: `OpenCoder-LLM/opc-annealing-corpus`, specifically the `synthetic_code_snippet` subset (closest to Aryabumi's "high-quality synthetic verified")
- Mix: 75% text, 25% code (Aryabumi's optimum), via Levanter `train_weights`

**5. Hyperparameters.** Identical to our `wd=1.6/x16/block=False` non-looping baseline (script `run_1_4b_wd1_6_x16_nocrossblock.py`):
- LR=1e-3 cosine to 0, WD=1.6, min_lr_ratio=0, β₁/β₂=0.9/0.95, warmup=0.01, max_grad_norm=1
- batch=64, seq=4096, num_train_steps=12800, seed=0, data_seed=0
- `block_cross_document_attention=False`, `stop_strategy=restart`

**6. Eval sets** — only metrics that actually have signal at our scale:
- **Paloma macro PPL** (continuous, sensitive — primary signal)
- **dclm_200m_val PPL** (held-out NL)
- **4 above-random benchmarks**: arc_easy, sciq, piqa, boolq (where our baseline scores meaningfully above random)
- **gsm8k_cot generation behavior** (does adding code cause regressions in generation? does it improve or worsen looping?)
- *Not used as outcome metrics* (because too noisy at our scale): hellaswag, winogrande, arc_challenge, mmlu, openbookqa, commonsense_qa, social_iqa, logiqa, HumanEval, MBPP, GSM8K aggregates. These will still be logged for completeness but not the primary signal.

**7. Confirm/refute criteria.**
- **Confirm Aryabumi-style effect at our scale**: Paloma macro improves (strictly lower loss vs baseline) AND ≥2 of {arc_easy, sciq, piqa, boolq} strictly improve AND no benchmark falls below baseline-minus-noise.
- **Refute (null result)**: Paloma macro flat-or-worse OR none of the 4 benchmarks improve. This would suggest the Aryabumi effect doesn't manifest with this data at our scale.
- **Partial**: some benchmarks improve, some hurt — informative, suggests data/scale interaction.

**8. Caveats acknowledged in advance.**
- A null result would NOT refute Aryabumi — our scale is 60× smaller and our code data is open, not his proprietary set.
- A positive result would be a *novel* finding (no published study has shown this open code data improves NL reasoning at 1.4B scale).
- We are NOT testing the headline "+8.2% NL reasoning aggregate" claim — that requires above-random performance on all 11 benchmarks, which we don't have.

#### Status of preparation

**Done today:**
- Downloaded OpenCodeReasoning (5.4 GB total raw, 30 parquets) — kept as a separate "competitive Python verified" data source, may use as a comparison point
- Tokenized 3 code variants: `aryabumi_code_web` (1.35B tokens, multi-language web code), `aryabumi_code_synth_solution` (183M, OCR solutions), `aryabumi_code_synth_full` (5.42B, OCR full)
- Wrote training scripts: `run_1_4b_25code_web.py`, `run_1_4b_25code_synth_full.py`
- Initial plan doc: `ARYABUMI_REPLICATION_PLAN.md` (now somewhat outdated — superseded by this section)

**Open / next**:
- Download `OpenCoder-LLM/opc-annealing-corpus` (synthetic_code_snippet at minimum, plus algorithmic_corpus and synthetic_qa for comparison) — these are now the preferred code sources over OCR
- Tokenize via marin
- Rename existing tokenized dirs to drop misleading `aryabumi_` prefix (use `ocr_*` and `opencoder_*` instead)
- Write training script using opc-annealing-corpus
- Launch comparison vs baseline (`divine-dream-99` / `iue9to5a`, our wd=1.6/x16/block=False text-only)
- Compare on Paloma macro + 4 benchmarks per the refined criteria above

**Decisions still owed to user**:
- Which opc-annealing-corpus subset to use as primary (synthetic_code_snippet vs algorithmic_corpus vs synthetic_qa, or all three concat)
- Whether to also run the OCR comparison as a separate experiment, or skip it (since OCR isn't a faithful Aryabumi proxy anyway)

---

### Step 11: WD-vs-epochs ablation — wd=3.2/x16/block=False (launched and finished May 25 PDT; final checkpoint at 20:57 PDT)

Run name: `fiery-paper-101` / `gm6by3tb`. ~8h on 8× A100-40GB.

**Tests**: holding epochs=16 and flipping WD from 1.6 → 3.2 (matching our konwoo-match baseline's WD but with double the epochs). If this loops, low WD is the looping fix; if it doesn't, extra epochs are sufficient.

**Result: PARTIAL loop — 30% loop rate (12/40 samples).**

Comparison across all 1.4B 16-epoch variants (gsm8k_cot, limit=20):

| Recipe | Loop rate | Median resp len | Max resp len |
|---|---|---|---|
| wd=3.2/x8 (konwoo-match baseline) | 100% | (token budget) | (token budget) |
| **wd=3.2/x16 (this run)** | **30% (12/40)** | 133 chars | 1301 chars |
| wd=1.6/x16 (non-looping) | 0% | 143 chars | 761 chars |

**Attribution: low WD is the dominant lever; more epochs is complementary but insufficient alone.**
- More epochs alone at wd=3.2: 100% → 30% looping (helps but not enough)
- Low WD alone at x16: 30% → 0% looping (fully closes)

**PPL trade-off** (apples-to-apples, paloma + held-out dclm):

| Subset | wd=3.2/x16 | wd=1.6/x16 | Konwoo wd=1.6/x16 |
|---|---|---|---|
| dclm_200m (training data) | 2.09 | 1.63 | — |
| dclm_200m_val (held-out) | 3.67 | 4.07 | — |
| paloma c4_en | 4.09 | 4.55 | 4.26 |
| paloma dolma-v1_5 | 3.89 | 4.35 | 4.14 |
| paloma macro | **4.20** | 4.71 | 4.43 |

Higher WD = less memorization (training loss 2.09 vs 1.63), better OOD generalization (paloma 4.20 vs 4.71, ~0.5 nats better). So **wd=3.2/x16 trades 30% looping for ~0.5 nats better Paloma PPL** vs wd=1.6/x16.

**Conclusion**: in our token-limited regime (1.4B model, 3.34B total training tokens, 209M unique base data), there's a real trade-off along the WD axis between memorization-flavored generalization (lower with high WD) and loop-prone generation (higher with high WD). Neither recipe wins both objectives. The "fix" likely requires moving along a different axis — data composition, training-data scale, or recipe — not just WD/epoch tuning.

**Unanswered question — why does higher WD cause more looping?**

Three plausible mechanisms exist but we have NOT tested any of them:

1. *Representational compression*: high WD prevents the model from representing the diversity of natural text → it learns average continuation patterns → greedy decoding lands in the same most-common phrase repeatedly. Predicts our pattern.
2. *Memorization-driven diversity*: low WD lets the model overfit to specific document continuations from training. At inference, it can recall these varied trajectories rather than producing average loops. Also predicts our pattern.
3. *Standard intuition (FAILS)*: high WD → smaller weights → smaller logits → less peaked softmax → MORE diverse generation. This contradicts our observation, so either it's wrong or another effect dominates.

(1) and (2) both predict the data but we can't distinguish them. (3) is what one would naively predict and is refuted.

**To test**: measure per-position output entropy / argmax probability at decoding time on each of the three checkpoints (wd=3.2/x8, wd=3.2/x16, wd=1.6/x16) on the same prompts. If (1) is right, high WD should have *lower* entropy. If (3) were right, high WD should have *higher* entropy. Cheap follow-up — single inference pass on each model, no training needed.

---

### Step 10: Cross-doc-attention ablation — wd=1.6 / x16 with `block_cross_document_attention=False`

Run script: `experiments/data_efficiency/run_1_4b_wd1_6_x16_nocrossblock.py`. Diff from Step 9: flip `block_cross_document_attention: True → False`. Otherwise identical.

Run name: `peach-thunder-100` / `6xx0hu3l`. Total time ~7h 50min.

**Hypothesis 1 (paloma gap closes): REFUTED.** Final eval losses are within <0.03 nats of the `block=True` version on every subset — well within run-to-run noise. We are still ~0.27 nats worse than konwoo's matching run.

| Subset | block=True | **block=False** | Konwoo |
|---|---|---|---|
| paloma c4_en | 4.554 | 4.547 | **4.264** |
| paloma dolma-v1_5 | 4.355 | 4.348 | **4.141** |
| paloma wikitext_103 | 4.213 | 4.195 | **4.093** |
| paloma 4chan | 3.669 | 3.640 | **3.428** |
| paloma macro | ~4.72 | ~4.71 | **4.43** |

**Hypothesis 2 (looping preserved): CONFIRMED.** 0/40 loops at limit=20, same as the prior wd=1.6/x16 run.

**Unexpected secondary finding: `block_cross_document_attention` DOES affect generation behavior despite not affecting Paloma PPL.**

| gsm8k_cot metric | block=True | block=False | Konwoo |
|---|---|---|---|
| em_strict | 0.0 | 0.0 | 0.0 |
| em_flexible | 0.0 | **0.10** (4/40 correct) | 0.0 |
| median response length | 52 chars | 143 chars | 90 chars |
| max response length | 217 chars | 761 chars | 563 chars |

Setting `block=False` produces longer, more varied responses with marginal math accuracy improvement. Sample 0 illustrates the qualitative difference:
- `block=True`: `16 bucks for 3 chickens\n\n` (terse, terminates fast)
- `block=False`: `$2 per duck egg is 16 - 16 = $6. So the answer is $6. $6 - 16 = $6. The answer is 16 - 16 = $6...` (more attempts, longer, not strictly looping)
- Konwoo: `Janet's ducks take 16 eggs. 3 dollars for 16 eggs is 4. The answer is 4.` (cleanest, also terminates)

So cross-document attention during training produces a more conservative model in generation — fewer continuation attempts, shorter responses. Possibly because the model never learned cross-document continuation patterns during training so it doesn't try to riff after committing to one answer.

**Outstanding 0.27 nat Paloma gap to konwoo — unexplained.** Candidate causes (none isolated):
1. Levanter version drift (konwoo's commit was June 2025; ours is newer with unknown subtle changes to init, optimizer math, kernel selection, mask construction, etc.)
2. Data ordering / sequence packing differences — we use `konwoo/dclm-164k-docs-train` HF parquet directly; he uses cache-with-batch-cap from `dclm_baseline-0206f1` which packs and shuffles differently
3. Hardware-driven numerical differences (his per_device_parallelism=1 vs ours=8)

None of these are hypothesis-relevant for our research questions about reasoning curriculum / data efficiency — they're framework noise.

### Summary of Steps 7-11

| Run | WD | Epochs | Loops? | paloma macro | dclm_200m_val |
|---|---|---|---|---|---|
| Konwoo wd=3.2/x8 | 3.2 | 8 | YES | 3.72 (best) | — |
| Our konwoo-match | 3.2 | 8 | YES | 3.81 | 3.42 |
| Our wd=3.2/x16 | 3.2 | 16 | **30%** | 4.20 | 3.67 |
| Konwoo wd=1.6/x16 | 1.6 | 16 | NO | 4.43 | 4.06 |
| Our wd=1.6/x16 (block=True) | 1.6 | 16 | **NO** | ~4.72 | 4.09 |
| Our wd=1.6/x16 (block=False) | 1.6 | 16 | **NO** | ~4.71 | 4.07 |

**Takeaways:**
1. Low WD is the dominant lever for fixing looping; more epochs is complementary but insufficient alone (Step 11 finding).
2. Our framework faithfully reproduces both behaviors at recipe-specific level.
3. There is a persistent ~0.27 nat absolute-PPL gap to konwoo across all our replications, not closed by `block_cross_document_attention`. Likely framework-version drift, not hypothesis-relevant.
4. Real WD-PPL-generation trade-off: high WD = better PPL + some looping; low WD = no looping + worse PPL. No single recipe wins both at our scale.

---

## May 24: Replication run #2 — wd=1.6 / x16

### Step 9: Replication run #2 — wd=1.6 / x16 (launched May 24, finished May 25)

Hypothesis: a 1.4B trained with WD=1.6, 16 epochs (3.34B total tokens) on the same data does NOT loop. This tests whether the recipe difference alone is sufficient to fix looping in our framework.

Run script: `experiments/data_efficiency/run_1_4b_wd1_6_x16.py`. Diffs from konwoo-match: `weight_decay 3.2 → 1.6`, `num_train_steps 6400 → 12800`. Same data, eval, model, seed, schedule. Save to `checkpoints/1_4b_wd1_6_x16/`.

Run name: `divine-dream-99` / `iue9to5a` (`dongwei_jiang/dongwei-data-efficiency`). Total time ~7h 50min.

**Looping result: HYPOTHESIS CONFIRMED.** gsm8k_cot at limit=2 and limit=20:
- 0/2 and 0/40 samples loop
- Median response length 52 chars (well under the 256-token budget)
- Max 217 chars
- Compare to wd=3.2/x8 (looped): all samples filled the 256-token budget with n-gram repetition
- Konwoo's matching wd=1.6/x16 also 0/40 loops (his median 90 chars)

**But generalization is meaningfully worse**, as predicted by Konwoo's own wandb numbers:

| Model | Paloma macro | dclm_200m_val | gsm8k_cot loops |
|---|---|---|---|
| Konwoo wd=3.2/x8 | 3.72 (best) | — | YES |
| Our konwoo-match wd=3.2/x8 | 3.81 | 3.42 | YES |
| Konwoo wd=1.6/x16 | 4.43 | 4.06 | NO |
| Our wd=1.6/x16 | ~4.72 | 4.09 | **NO** |

**Observed pattern (from 2 recipes, not a general law):** the wd=3.2/x8 recipe (best Paloma PPL we've measured) loops on gsm8k_cot, while wd=1.6/x16 (worse Paloma PPL by ~0.7 nats) does not loop. We cannot conclude PPL and looping are anticorrelated in general — this is two data points along a confounded axis (WD AND epochs changed together). Possible causes: (a) more epochs → more exposure → softer distribution; (b) lower WD → less weight regularization → softer argmax; (c) both interact. **To isolate**, we'd need to ablate `wd=3.2/x16` and `wd=1.6/x8` separately. *(Step 11 the next day did exactly this.)*

Mechanistic intuition (not verified): higher WD + fewer epochs → less peaked softmax → less prone to greedy lockup but better OOD generalization on PPL because weights stay close to prior. Lower WD + more epochs → distribution sharpens around training distribution → memorizes (low PPL on similar data) but pathological under greedy decoding.

**Outstanding gap**: our wd=1.6/x16 paloma_macro (~4.72) is ~0.2-0.3 nats worse than konwoo's (4.43) across every subset. One config difference identified: we trained with `block_cross_document_attention=True` while konwoo's wandb shows `None` which the Levanter code (`LmExample.causal()`) treats as False (no within-sequence attention masking across doc boundaries). *(Step 10 the next day ablated this.)*

---

## May 23: Looping investigation — diagnostic phase

### The observation that started it

Our 1.4B baseline (`8be9dtfq` / super-glade-5, trained on `konwoo/dclm-164k-docs-train` with WD=3.2, 8 epochs, LR=1e-3 cosine) **loops** on gsm8k_cot generation. Greedy generation locks into n-gram repetition almost immediately:

- Sample 0 (Janet's eggs, target 18): `Janet's ducks lay 16 eggs per day. She buys 4 eggs per day. The answer is 4. The answer is 4. The answer is 4...` × ~40 → 738 chars total
- Sample 1 (robe, target 3): `3 + 2 = 5. The answer is 3 + 2 = 5. The answer is 3 + 2 = 5...` × ~22 → 530 chars

Loop is also present at 300M and 600M baseline.

### Step 1: Rule out decoding artifacts

Verified that decoding params at the lm_eval level are identical across all models: `do_sample=False, until=['Q:', '</s>', '<|im_end|>'], max_gen_toks=256`. So it's not a decoding-config mismatch.

### Step 2: Compare to other base models (no instruction tuning)

Tested gsm8k_cot (limit=2, log_samples=True) on:

| Model | Coherent tokens before lock-in | Loops? |
|---|---|---|
| Our 300M baseline | 0 | yes — immediate `She sells 2,000 to 3,000 pounds of duck egg per day` × ∞ |
| Our 600M baseline | 0 | yes — `10 eggs per day` × ∞ |
| **Our 1.4B baseline** | **0** | **yes — `The answer is 4` × ∞** |
| Qwen2-0.5B-base | ~30 | no — produces coherent attempt then `Q:` next-prompt continuation |
| Qwen3-0.6B-Base | ~50 | no — gets robe question correct |
| OLMo-2-0425-1B | ~80 | no — gets eggs question correct (answer `$18`) |
| `konwoo/1_4b4k-209Mx16-wd1.6` (best) | ~25 | **no** — terminates cleanly within 75-189 chars |
| `konwoo/1_4b4k-209Mx8-wd3.20` | partial | **yes on robe** — `2 = 3. 3 = 3.` × ∞, sample 0 happens to terminate short |
| Our `1_4b_konwoo_match` (wd=3.2, x8, post-replication) | 0 | **yes** — `16 - 16 = 16 dollars` × ~13 → 612 chars |

Outputs saved under `/fsx/users/dongweij/marin/outputs/eval_results/base_model_loop_comparison/`.

### Step 3: Verify EOS handling

For our 1.4B and konwoo's 1.4B both: ran free-generation test (`max_new_tokens=300, do_sample=False, eos_token_id=None`). Both emit **0** EOS tokens in 300 generated tokens. So neither model is trained to emit `<|end_of_text|>`. Konwoo's appears to "stop" only because his model is coherent enough to generate the few-shot `Q:` marker, which lm-eval treats as a stop token. Our model never produces `Q:` because it locks into repetition first.

Tokenizer configs and `eos_token_id` are byte-identical between our and konwoo's HF checkpoints.

### Step 4: Identify the recipe difference

Konwoo's runs come from `stanford-mercury/suhas-data-efficiency` WandB project. His HF uploads at `konwoo/*` materialize specific runs from there. Three relevant variants:

| Run | WD | Epochs | Total tokens | Loops on gsm8k_cot? |
|---|---|---|---|---|
| `1_4b4k-209Mx16-wd1.60` (his "best") | 1.6 | 16 | 3.34B | **NO** |
| `1_4b4k-209Mx8-wd3.20` | 3.2 | 8 | 1.67B | **YES** |
| `1_4b4k-209Mx4-lr0.0003` | 0.1 | 4 | 836M | not tested |

Our baseline (`8be9dtfq`) matches the wd=3.20/x8 recipe almost exactly. The non-looping konwoo recipe (wd=1.6/x16) has both **2× more total training tokens** and **half the weight decay**.

### Step 5: Verify data identity

Konwoo's wd=3.20 run draws data from `gs://marin-us-central2/tokenized/dclm_baseline-0206f1/` (the canonical Marin DCLM cache) with `max_train_batches={dclm: 800}` — limits training to 800 batches × 64 × 4096 = 209,715,200 tokens per epoch. With `stop_strategy=restart`, training cycles through these same 209M tokens 8 times (6400 steps total).

Verified our local cache (35.5B tokens, 8 parts) is consistent with the canonical `dclm_baseline-0206f1` naming (source: `mlfoundations/dclm-baseline-1.0 @ a3b142c`, tokenizer: `Llama-3.1-8B`). The cache is enough for the experiment (need only 1.67–3.34B of 35.5B available locally).

Verified konwoo's HF dataset `konwoo/dclm-164k-docs-train` is a real subset of DCLM-baseline-1.0 (sampled docs found in raw `global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst`). Doc count = 164,459 matches the 800-batches × 64-batch slice exactly. Our run uses this HF dataset directly; konwoo's runs use the cache-with-batch-cap mechanism.

### Step 6: Comprehensive eval suite expansion

To make eval losses directly comparable to konwoo's wandb numbers, downloaded and tokenized all 16 Paloma subsets locally:

| Path | Size | Used by |
|---|---|---|
| `/fsx/users/dongweij/marin/outputs/raw/paloma-fc6827/65cd6fc/` | ~1.2 GB | (raw HF download) |
| `/fsx/users/dongweij/marin/outputs/tokenized/paloma/<name>-<hash>/` | ~few MB each | training/eval components |

16 subsets tokenized: 4chan, c4_100_domains, c4_en, dolma-v1_5, dolma_100_programing_languages, dolma_100_subreddits, falcon-refinedweb, gab, m2d2_s2orc_unsplit, m2d2_wikipedia_unsplit, manosphere_meta_sep, mc4, ptb, redpajama, twitterAAE_HELM_fixed, wikitext_103. Cache hash suffixes match konwoo's wandb config exactly (e.g., `c4_en-cf1f79`, `4chan-496ad5`), confirming canonical naming consistency.

Note: `allenai/paloma` is a gated HF dataset. Requesting access via the HF "Request Access" button worked within minutes.

### Step 7: Replication run #1 — konwoo-match (wd=3.2, x8)

Goal: rule out framework / code drift as the cause of our looping. Match konwoo's wd=3.20/x8 config as closely as possible on our Levanter version.

Run name: `icy-snowflake-98` / `4m4o7xvd` (`dongwei_jiang/dongwei-data-efficiency`).
Diffs from our 8be9dtfq baseline:
- `data_seed`: 42 → 0 (matches konwoo)
- `optimizer.min_lr_ratio`: 0.1 → 0.0 (matches konwoo)
- Add 16 Paloma val components (weight=0) so eval losses are apples-to-apples with konwoo
- Otherwise identical: model 1_4b4k, data = `konwoo/dclm-164k-docs-train`, WD=3.2, LR=1e-3 cosine, batch=64, 8 epochs, 6400 steps, seed=0

Total time: ~4h 15min on 8× A100-40GB. Final eval losses (subset):

| Subset | Ours `4m4o7xvd` | Konwoo `1_4b4k-209Mx8-wd3.20` |
|---|---|---|
| eval/loss (overall avg) | 2.88 | 3.53 |
| eval/dclm_200m/loss (held-out, our val) | 3.42 | — |
| eval/dclm_200m/loss (train data, near-memorization) | 2.67 | — |
| paloma/c4_en/loss | 3.78 | 3.61 |
| paloma/dolma-v1_5/loss | 3.61 | 3.52 |
| paloma/dolma_100_subreddits/loss | 3.85 | 3.75 |
| paloma/falcon-refinedweb/loss | 3.87 | 3.70 |
| paloma/c4_100_domains/loss | 3.55 | 3.48 |
| paloma/m2d2_wikipedia_unsplit/loss | 3.41 | 3.36 |
| paloma/macro_loss | 3.81 | 3.72 |

Our losses are 0.05–0.16 nats *higher* than konwoo's wd=3.20 across most Paloma subsets. Plausible drivers (we did NOT isolate): Levanter version drift, specific docs differ (ours uses konwoo's 164k-docs HF upload vs his original cache-batch-cap), `min_lr_ratio` previously different in our baseline (now matched). The `eval/dclm_200m/loss` of 2.67 is on training data so it's a memorization signal, not generalization.

**On gsm8k_cot the konwoo-match model still loops** — same n-gram repetition as our baseline. Confirmed that:
1. Konwoo's own wd=3.20/x8 run also loops (sample 1 locks into `2 = 3. 3 = 3.` × ∞)
2. So our framework is faithfully reproducing the wd=3.20/x8 recipe — and that recipe produces loop-prone models
3. The "non-looping" reference model is `wd=1.6/x16` (different recipe), not anything matching our baseline

### Step 8: Anti-pattern caught

Replication #1 answered a narrower question ("does our framework drift from konwoo's") than the question that matters ("how do we fix looping"). The reference run to replicate, if the goal is to fix looping, is **the one that does NOT exhibit the bug** — i.e. konwoo's wd=1.6/x16, not wd=3.20/x8. New rule added to `CLAUDE.local.md`:

> Critical anti-pattern: replicating a config that already exhibits the bug you want to fix.
> When the user's goal is "fix behavior X", the reference run to match is the one that does NOT exhibit X.


## May 22: Comprehensive Evaluation Suite

### Motivation

Prior experiments only evaluated on 7 benchmarks (ARC-E/C, PIQA, SciQ, HellaSwag, WinoGrande, MMLU). The papers we're comparing against use much broader eval suites — Aryabumi uses 11 NL reasoning + TriviaQA/NQ + HumanEval/MBPP, Petty uses 204 BigBench tasks, Between Circuits uses BLiMP grammaticality. To properly measure our three research objectives, we need benchmarks covering all of them.

### What was added

Expanded from 7 to 28 benchmarks in `experiments/data_efficiency/run_comprehensive_evals.py`, organized by category:

| Category | Benchmarks | Covers objective |
|---|---|---|
| NL Reasoning (12) | ARC-E/C, HellaSwag, PIQA, WinoGrande, OpenBookQA, COPA, BoolQ, SocialIQA, CommonsenseQA, LogiQA, SciQ | (3) General NL |
| World Knowledge (4) | MMLU, TruthfulQA (logprob); TriviaQA, NQ Open (generation) | (3) General NL |
| Math (4) | GSM8K, MathQA (logprob); GSM8K-CoT, Minerva MATH (generation) | (2) Reasoning |
| Code (2) | HumanEval, MBPP (generation) | (2) Reasoning |
| Linguistic (2) | BLiMP (67 subtasks), LAMBADA | (3) General NL |
| BigBench Hard (1→27) | BBH zero-shot (27 subtasks, logprob); BBH CoT few-shot (generation) | (2) Reasoning |
| Reading (2) | RACE (logprob); DROP (generation) | (3) General NL |

### Implementation notes

- **Logprob vs generation**: 20 tasks are logprob-based (multiple choice, fast), 8 require generation (slower). Organized into separate suites (`--suite logprob` vs `--suite generation` vs `--suite all`).
- **`social_iqa`** requires `HF_DATASETS_TRUST_REMOTE_CODE=1` (custom dataset loader).
- **`humaneval`/`mbpp`** require `HF_ALLOW_CODE_EVAL=1` and `confirm_run_unsafe_code=True` (executes model-generated code).
- **`minerva_math`** requires `sympy`, `math_verify`, `antlr4-python3-runtime==4.11` (installed via `pip install lm-eval[math]`).
- **Metric extraction**: Different tasks use different metric keys — `acc,none`, `exact_match,flexible-extract`, `exact_match,get-answer`, `pass@1,create_test`, `pass_at_1,none`, `f1,none`, etc. The script handles all of them via a priority list.
- **Large eval sets**: TriviaQA has ~17K test examples requiring generation — use `limit=N` for smoke testing. For full evals, parallelize across GPUs (8x A100-40GB available).

### Not included

- **COGS, COGS-vf, English Passivization** (Petty et al.): These require finetuning for 10K steps then measuring full-sequence accuracy. Not a standard eval — would need a custom training+eval loop.
- **Full BigBench (204 tasks)**: Available in lm-eval-harness but most would be noise at 300M scale. BBH (27 hard tasks) is the standard subset.

### Verification

All 28 tasks verified working on the 300M DCLM baseline checkpoint. Key results:

| Benchmark | 300M baseline |
|---|---|
| BLiMP (aggregate) | 78.9% |
| BoolQ | 60.9% |
| COPA | 60.0% |
| PIQA | 60.2% |
| BBH zero-shot | 16.1% |
| TruthfulQA MC2 | 44.9% |
| HumanEval | ~0% (expected at 300M/200M tokens) |
| GSM8K-CoT | 0% (expected) |

---

## May 17–21: Deep Literature Review — Curriculum, Formal Languages, and Data Selection

Expanded the paper reading from the May 11 survey into deep dives on the actual mechanics behind each approach. Read full papers (including appendices) and documented detailed notes with Dongwei comments on the underlying theory. Papers added/updated in [papers/reasoning_curriculum.md](../../papers/reasoning_curriculum.md):

### Papers read in full (with Dongwei comments)

1. **Between Circuits and Chomsky** (Hu et al., 2025) — Pre-pretraining on formal languages. Key insight: hierarchy matters, not just Chomsky complexity class. k-Shuffle Dyck (context-sensitive + hierarchical + in C-RASP) gives 33% token efficiency gain at 1B; ww (context-sensitive but non-hierarchical) actively hurts. Mechanistic proof via circuit discovery shows the model reuses the exact same attention heads for English syntax. Commented on: Chomsky hierarchy, C-RASP (what Transformers can express in constant depth), the 2x2 grid of formal languages, and the 3-step mech interp proof (pruning → NL training → targeted ablation).

2. **McCoy & Griffiths — Bayesian Inductive Bias Distillation** (2023) — Meta-learn (MAML) an LSTM on 25K formal languages sampled from a Bayesian prior. The prior-trained LSTM matches Bayesian data efficiency on formal languages and gains ~11% perplexity on low-data natural language. Commented on: how MAML's inner/outer loop works concretely (support set → temporary update → query set → meta-update of initialization), and why the simplicity prior transfers to human language (recursion, concatenation, alternation match NL structure).

3. **Curriculum Learning for LLM Pretraining** (Elgaar & Amiri, 2026) — All curricula share the same 5 latent training phases (proven via HMM with BIC/AIC); curricula only change time spent in each phase. Benefits diminish at 410M+. Commented on: HMM methodology (observable metrics are trace/singular-value statistics, not loss; state space held fixed across curricula), softmax bottleneck (effective rank of hidden state, not vocab size), gradient noise scale.

4. **Beyond Random Sampling** (Zhang et al., 2026) — Most comprehensive CL study for pretraining (200+ models, up to 100B tokens). CL as warmup yields +3.5% sustained improvement. Best metrics: compression ratio, MTLD, Flesch Reading Ease. Perplexity-based ordering hurts late training.

5. **Perplexity Correlations for Data Selection** (Thrush et al., 2025) — Use 90 existing LLMs to compute rank correlation γ_j between per-domain loss and benchmark performance. Select high-correlation domains, train fastText classifier to scale to page-level. Commented on: the γ_j formula, what "domains" means (top-level web addresses in RedPajama V2), two-stage pipeline (domain ranking → page-level classification), and why it beats DSIR.

6. **Open Thoughts** (2025) — Full 72-page paper including appendix. Concentration > diversity for question sources. Self-reflection critical (-49.1% without it). Cross-domain transfer vanishes when in-domain data mixed in. Single-model limitation (all 1000+ ablations on Qwen2.5-7B-Instruct only).

7. **On Code-Induced Reasoning** (Aryabumi et al., 2025) — Java-favors-math claim from abstract only holds for 1/5 models. Code fine-tuning consistently beats NL-only fine-tuning as a baseline.

### Current state

All prior experiments (May 1–11) showed the same result: reasoning data (OpenThoughts, OWM, code) injected during pretraining does not help downstream benchmarks compared to pure DCLM, across 300M–1.4B scales. The literature review has identified several mechanisms that *should* work based on theory, but we haven't found an experiment design that bridges the gap between "formal language pre-pretraining improves data efficiency" and "reasoning data during pretraining improves downstream reasoning."

### Research objectives

Using reasoning/synthetic data in pre-pretraining or pretraining should achieve three things simultaneously:

1. **Data efficiency**: The model reaches the same loss with less general data or less training time
2. **Reasoning quality**: The trained model performs better on reasoning tasks, hallucinates less, reasons more reliably
3. **General NL performance**: Normal natural language benchmarks also improve (not just reasoning — no regression)

These three objectives are the success criteria for any experiment going forward. An approach that only achieves (1) but hurts (2) or (3) is not useful. Aryabumi et al. is the closest to achieving all three in pretraining: 25% code gives +8.2% NL reasoning, +4.2% world knowledge, and 12x code boost — but we haven't replicated this at our scale/data budget. Our prior experiments failed all three — reasoning data injection hurt general benchmarks and didn't help reasoning ones.

---

## May 11: Literature Review & Hypothesis Refinement

After the H1 experiment showed no clear benefit from reasoning data injection, we stepped back to survey the literature on what makes reasoning data effective for pretraining. Reviewed 15+ papers across synthetic data composition, pretraining vs post-training, code and reasoning, abstract reasoning transfer, and data selection/curriculum. Paper notes organized by category in [papers/reasoning_curriculum.md](../../papers/reasoning_curriculum.md) and [papers/causal_bridge.md](../../papers/causal_bridge.md). Downloaded all cited paper PDFs to `papers/`.

The key finding: our results are consistent with the broader literature — pure reasoning data hurts, domain-specific gains (OWM → SciQ) don't transfer, and the diversity of reasoning patterns matters more than any single domain. This led to a revision of the research hypotheses.

### Revised Hypotheses

The original H1/H2/H3 hypotheses (May 5) have been refined based on accumulated experimental evidence across all runs (300M–1.4B, multiple data types and curriculum designs).

#### H1: What Makes Reasoning Data Good for Pretraining?

**The problem:** Not all "reasoning data" is equal. OpenThoughts (long exploratory CoT traces) consistently hurts performance across all scales and curriculum orderings. OpenWebMath shows a SciQ gain (73.2% vs 63.2% baseline) but this saturates with enough general pretraining and does not transfer beyond science domains — consistent with domain knowledge transfer rather than general reasoning capability. Code alone hurts all benchmarks.

**The constraint:** Good reasoning data must teach something that (a) transfers beyond the domain it was trained on, and (b) is not confounded with domain familiarity — i.e., the gain should not disappear when the model sees enough general web text.

**What we know from the literature:**
- Content-free synthetic tasks (Percy's work, arxiv 2206.10139; Procedural Pretraining, arxiv 2601.21725) can close ~65% of the gap to natural pretraining, suggesting structural patterns matter even without semantic content
- Procedural knowledge — data demonstrating how to derive something step by step — is 10x overrepresented in influential pretraining documents for reasoning (Ruis et al., arxiv 2411.12580)
- OpenThoughts fails because its exploratory back-and-forth CoT is the wrong structure for a model starting from scratch with no world knowledge to anchor on

**What we don't know:** Whether real language data with explicit causal structure — as opposed to content-free synthetic tasks — can teach transferable reasoning capability. The causal bridge idea is the most natural candidate: by conditioning generation on two real document endpoints (causally related via Wikipedia wikilinks), the model is forced to construct relational understanding grounded in real-world events. This is neither content-free nor domain-specific — it is structured real language. Whether this teaches transferable reasoning is the core empirical question.

#### H2: How Do We Retain Reasoning Capability Through General Pretraining?

**The problem:** Even if we solve H1 and identify good reasoning data, there are two distinct mechanisms by which the capability could be lost during subsequent general web text training:

**Sub-problem 2a — Catastrophic forgetting:** The model overwrites representations learned from reasoning data when exposed to web-scale text. The May 8 and May 10 H1 experiments are consistent with this — the SciQ gains from OWM disappear after phase 2 DCLM training. Replay (mixing a small fraction of reasoning data throughout web text training) is a standard mitigation but untested here.

**Sub-problem 2b — No training pressure to use reasoning circuits:** Steven Cao's point: even if reasoning circuits exist after phase 1, there is no mechanism during standard next-token prediction on web text that activates or reinforces those circuits. The model is not prompted to reason during web text training, so whatever was built in phase 1 sits dormant. This is a more fundamental problem than forgetting — replay does not solve it, because the problem is not forgetting but never using.

**What we don't know:** Whether there exists a training signal during web text exposure that both retains reasoning circuits and actively uses them. Possible directions include: perplexity-based filtering of web text (only train on documents the reasoning-capable model finds surprising, not documents it can predict via shortcuts), or a joint training objective that ties reasoning evaluation to web text prediction. Both are speculative.

#### The Relationship Between H1 and H2

H1 is the more fundamental bottleneck. Until we have data that demonstrably teaches transferable reasoning (H1 solved), H2 is moot — there is nothing to retain. The causal bridge experiments address H1 first.

### Literature Review

See [papers/reasoning_curriculum.md](../../papers/reasoning_curriculum.md) for paper notes on reasoning, synthetic data, and curriculum. See [papers/causal_bridge.md](../../papers/causal_bridge.md) for causal bridge related papers.

Key takeaways:
1. Pure reasoning data hurts; ~30% mixed with web data is optimal (Kang et al.)
2. Diversity of reasoning patterns matters more than domain specificity (NVIDIA Front-Loading)
3. Relational/combinatorial structure drives quality (EntiGraph)
4. Abstract reasoning from toy domains DOES transfer (Warm Up Before You Train)
5. Pretraining is the ceiling — post-training amplifies but cannot create (Echo Chamber, Front-Loading)

---

## May 10: H1 Revisited — Continuous Cosine LR, OWM+Code Treatment

### Motivation
The May 8 H1 experiment had two problems:
1. **Fresh cosine LR per phase** — LR jumps at phase boundaries, optimizer moments reset
2. **OpenThoughts as treatment** — already conclusively shown to be useless at all scales (300M–1.4B)

This run fixes both: continuous cosine LR across phases 1+2 (via `initialize_from_step`), and uses OWM+Code as treatment data since OWM showed the only positive signal (SciQ 73.2% vs 63.2% baseline).

### Technical Implementation
Added `initialize_from_step` to `TrainLmConfig` in `lib/levanter/src/levanter/main/train_lm.py`:
- Loads weights+optimizer from checkpoint via `initialize_from_checkpoint_path`
- Sets optimizer schedule counter AND `state.step` to specified value
- Enables continuous cosine LR across phases without `load_checkpoint_path` (which OOMs)
- Verified with smoke test: 40-step single run vs 20+20 split has 0.00e+00 max LR difference

### Design
```
Phase 0 (shared):     Train from scratch on 203M DCLM, 4 epochs = 3,096 steps
Phase 1 (1,667 steps / 437M tokens):
  Treatment: OWM (219M) + Code (218M) mixed 50/50
  Control:   Disjoint DCLM (~407M tokens)
Phase 2 (3,052 steps / 800M tokens):
  Both arms: Disjoint DCLM (~778M tokens)
```

LR schedule: Phases 1+2 share one continuous cosine over 4,719 total steps.
- Phase 1: `stop_step=1667`, `num_train_steps=4719`
- Phase 2: `initialize_from_step=1667`, `num_train_steps=4719`

All DCLM data is disjoint across phases (phase 0: 203M, phase 1 control: 407M, phase 2: 778M — downloaded 1.52B total from DCLM baseline).

Model: 300M, batch_size=64, seq_len=4096, LR=3e-3, WD=1.6

### WandB Runs
| Phase | Run ID | Description |
|-------|--------|-------------|
| Phase 0 (pretrain) | hvu9zzrj | 300M on DCLM 200M, 4 epochs |
| Treatment Phase 1 | ja7ty1se | OWM+Code mix, 1667 steps |
| Control Phase 1 | rd5wfmmu | Disjoint DCLM, 1667 steps |
| Treatment Phase 2 | un39dx11 | Disjoint DCLM, 3052 steps (from step 1667) |
| Control Phase 2 | m67nooef | Disjoint DCLM, 3052 steps (from step 1667) |

### Results

| Benchmark | Treatment (OWM+Code) | Control (DCLM only) | Delta |
|-----------|---------------------|---------------------|-------|
| ARC Easy | 35.5% | 36.7% | -1.1% |
| ARC Challenge | 22.3% | 22.5% | -0.3% |
| PIQA | 50.0% | 50.2% | -0.2% |
| SciQ | 74.1% | 74.1% | 0.0% |
| HellaSwag | 27.3% | 27.4% | -0.0% |
| WinoGrande | 50.4% | 51.1% | -0.6% |
| MMLU | 26.7% | 25.3% | **+1.4%** |
| **Macro avg** | **27.6%** | **26.9%** | **+0.7%** |

DCLM val: Treatment 1.198 BPB (3.705 loss) vs Control 1.191 BPB (3.686 loss)

### Analysis
1. **SciQ is flat** (74.1% both arms) — surprising given OWM-only showed 73.2% vs 63.2% DCLM baseline. The control also reaches 74.1%, suggesting phase 0 pretraining (4 epochs of 203M DCLM) already saturates SciQ at this model size.
2. **MMLU is the only treatment win** (+1.4%) — OWM+Code may help with knowledge breadth
3. **Most benchmarks within noise** (0–0.6%) — no clear treatment advantage or disadvantage
4. **DCLM val loss slightly worse for treatment** (3.705 vs 3.686) — expected since treatment saw less DCLM in phase 1
5. **Continuous cosine LR worked correctly** — both arms resumed from step 1667 with matching LR schedules

### Conclusion
**H1 remains unsupported even with proper LR continuity and better treatment data.** Injecting OWM+Code mid-training does not meaningfully help reasoning benchmarks compared to pure DCLM training. The previous SciQ signal from OWM (73.2%) appears to be a domain knowledge effect that saturates with enough general pretraining, not a lasting advantage from procedural knowledge injection.

### Comparison with May 8 H1
| Change | May 8 | May 10 |
|--------|-------|--------|
| LR schedule | Fresh cosine per phase | Continuous cosine (initialize_from_step) |
| Treatment data | OpenThoughts (170M) | OWM+Code (437M) |
| Phase 0 | Paper's 16-epoch ckpt | 4-epoch fresh pretrain |
| DCLM data | Repeated across phases | Disjoint per phase |
| SciQ delta | +1.9% | 0.0% |
| Macro avg delta | -1.3% | +0.7% |

The improved design (continuous LR, better treatment data, disjoint data) eliminated the macro avg deficit but still shows no clear benefit from reasoning data injection.

---

## May 8: H1 Experiment — Reasoning Data in the Middle of Training

### Hypothesis
Model needs language/world knowledge first before reasoning data is useful.
If we inject reasoning data after initial pretraining, the model should perform better
on reasoning benchmarks compared to training on web data only.

### Design
- **Treatment**: Run A (3B DCLM pretrained) → 200M OT → 400M DCLM
- **Control**: Run A (3B DCLM pretrained) → 200M DCLM → 400M DCLM
- Both use `initialize_from_checkpoint_path` with fresh cosine LR schedule per phase
- Phase1: 763 steps (200M tokens), Phase2: 1526 steps (400M tokens)
- Model: 300M, batch_size=64, seq_len=4096, LR=3e-3, WD=1.6

### Fixes Applied
1. **LR schedule counter reset**: `initialize_from_checkpoint_path` now resets optimizer schedule counters (was loading stale counters from source checkpoint, giving wrong LR)
2. **Force checkpoint save**: `LambdaCallback.on_step` now passes `force` parameter (was being dropped, so final checkpoint never saved)
3. **Checkpoint wait**: Trainer now waits for async checkpoint save to complete before returning

### WandB Runs
| Phase | Run ID | Tags |
|-------|--------|------|
| Treatment Phase1 (OT) | 06va0rn2 | h1-v2, treatment, phase1, ot |
| Control Phase1 (DCLM) | ncpocjta | h1-v2, control, phase1, dclm |
| Treatment Phase2 (DCLM) | d47v5z8y | h1-v2, treatment, phase2, dclm |
| Control Phase2 (DCLM) | vothg0mz | h1-v2, control, phase2, dclm |

### Results

| Benchmark | Treatment (OT→DCLM) | Control (DCLM→DCLM) | Diff |
|-----------|---------------------|----------------------|------|
| ARC Easy | 35.0% | 35.0% | 0.0% |
| ARC Challenge | 19.0% | 18.9% | +0.2% |
| PIQA | 48.9% | 49.2% | -0.4% |
| SciQ | 70.9% | 69.0% | **+1.9%** |
| HellaSwag | 26.2% | 26.4% | -0.2% |
| Winogrande | 50.9% | 51.0% | -0.1% |
| MMLU | 25.8% | 26.7% | -1.0% |
| **Macro avg** | **27.0%** | **28.3%** | **-1.3%** |

DCLM val loss: Treatment 3.743 vs Control 3.720

### Conclusion
**H1 is not supported.** Injecting 200M tokens of reasoning data (OpenThoughts) in the
middle of training does not help reasoning benchmarks. The control (pure DCLM) slightly
outperforms on most benchmarks (macro avg -1.3%). Treatment only wins on SciQ (+1.9%),
consistent with H3 (domain-specific knowledge transfer) rather than general reasoning
improvement.

### Caveats
- Each phase gets a fresh cosine LR from max → 0. This means there's a LR jump at the
  phase boundary. Both conditions have the same jump so the comparison is fair, but a
  continuous cosine schedule would be more representative of real training.
- The 200M tokens of OT may not be enough to teach reasoning at 300M model scale.
- Fresh optimizer (Adam moments reset) at each phase means the model "forgets" gradient
  history, which may hurt the treatment more since it switches domains twice.

---

## May 5: Mixed DCLM+OWM Run & Research Hypotheses

### Mixed Run: 80% DCLM + 20% OpenWebMath (300M)

This is an **off-ramp exploration** from the original staged curriculum hypothesis. The original idea was that reasoning-style data (first OpenThoughts, then OpenWebMath) should be staged sequentially — reasoning first, then web data, or vice versa. Sequential curriculum failed in both directions:
- OWM→DCLM: model forgets SciQ gains
- DCLM→OWM: model forgets language/world knowledge

Simultaneous mixing is a fallback to see if we can get OWM's SciQ benefit without losing DCLM's general capabilities.

**Run config:** 300M, LR=3e-3, WD=3.2, 6400 steps, 80% DCLM + 20% OWM mixed throughout training.

| Metric | Mixed 80/20 | DCLM baseline | OWM only |
|---|---|---|---|
| dclm_val | **3.687** | 3.797 | 4.304 |
| ARC Easy | 38.2% | 39.6% | 34.9% |
| PIQA | 58.0% | 60.3% | 48.9% |
| SciQ | **64.5%** | 63.2% | **73.2%** |
| ARC-C | 17.7% | 17.5% | — |
| HellaSwag | 26.6% | 27.4% | — |
| WinoGrande | 52.1% | 50.4% | — |

**Analysis:** The mixed run slightly improves SciQ over DCLM baseline (64.5% vs 63.2%) but ARC Easy and PIQA are flat or slightly down. This supports **H3 (domain-specific knowledge)**: OWM's benefit is concentrated on science benchmarks, not a general reasoning improvement. The dclm_val improvement (3.687 vs 3.797) suggests the model benefits from data diversity for perplexity, but this doesn't translate to broad benchmark gains.

### Original Research Hypotheses (May 5)

We now have a clear empirical pattern: OpenWebMath trains a model that excels at SciQ (73.2% vs 63.2% DCLM baseline) but hurts ARC Easy (34.9% vs 39.6%) and PIQA (48.9% vs 60.3%). Sequential curriculum in either direction loses one set of gains. Three hypotheses explain different aspects of this pattern.

#### H1: Model needs language/world knowledge first before reasoning data is useful

The idea: a model that already understands language and the world can extract more value from procedural math content than a model learning both from scratch.

- **Prediction for DCLM→OWM:** SciQ > 73.2% (language foundation makes reasoning data more useful)
- **Prediction:** ARC Easy/PIQA stay decent (world knowledge partially survives from DCLM phase)
- **How to test:** Vary DCLM phase length before switching to OWM. Run 1600/3200/4800 steps of DCLM, then OWM for the remaining steps (4800/3200/1600). If more DCLM first leads to better SciQ, that supports H1.

#### H2: Catastrophic forgetting — later data overwrites earlier

The idea: whatever the model learns last dominates. Earlier training is largely wasted because the model overwrites those representations.

- **Prediction for DCLM→OWM:** SciQ ≈ 73.2% (same as OWM-only; the DCLM phase is wasted)
- **Prediction:** ARC Easy/PIQA drop to OWM-only levels (~34.9% and ~48.9%)
- **How to test:** Run DCLM→OWM with DCLM replay during phase 2 (10% DCLM + 90% OWM in the second phase). If replay mitigates forgetting (ARC Easy/PIQA stay higher), that confirms H2 as the mechanism.
- **Note:** H1 and H2 can both be true simultaneously — the model may need prior knowledge AND suffer from forgetting.

#### H3: OWM teaches domain-specific science knowledge, not general reasoning

The idea: OWM's SciQ improvement comes from memorizing science facts and math procedures, not from learning transferable reasoning skills.

- **Prediction for mixed run:** SciQ improves but ARC Easy/PIQA stay flat (science knowledge helps science benchmarks only)
- **How to test:** Evaluate OWM-trained models on reasoning benchmarks outside math/science domains. If OWM only helps science-related tasks, it is domain knowledge transfer, not general reasoning improvement.

### Discriminating Experiments

These experiments produce different predictions under each hypothesis, allowing us to distinguish between them:

| Experiment | H1 predicts | H2 predicts | H3 predicts |
|---|---|---|---|
| DCLM→OWM (3200+3200) | SciQ > 73.2% | SciQ ≈ 73.2% | SciQ ≈ 73.2% |
| DCLM→OWM varying lengths | More DCLM → better SciQ | SciQ always ≈ 73.2% | — |
| Mixed run (80/20) | SciQ + ARC + PIQA all improve | — | SciQ up, ARC/PIQA flat |
| OWM + DCLM replay in phase 2 | — | Forgetting mitigated | — |
| OWM model on non-science reasoning | — | — | No improvement (domain-specific) |

The mixed run (80% DCLM + 20% OWM) is already complete and benchmark results will directly test H1 vs H3: if all three benchmarks improve, that favors H1 (general synergy); if only SciQ improves, that favors H3 (domain-specific knowledge).

---

## May 4: Procedural Knowledge Experiments (300M)

### Motivation
Based on "Procedural Knowledge in Pretraining Drives Reasoning" (Ruis et al., arxiv:2411.12580):
- Models learn reasoning from **code and math that demonstrates procedures**, not from explicit CoT traces
- Code on StackExchange is 10x overrepresented in influential documents for reasoning
- The same procedural documents help across different reasoning questions of the same type

This explains why OpenThoughts (explicit CoT) failed — it's the wrong type of reasoning data. We should test procedural knowledge sources: code and math web pages.

### Data
- **DCLM 200M**: 164K web documents, ~200M tokens (baseline)
- **Code Procedural 218M**: ~218M tokens of Python, JavaScript, C, C++ code from The Stack
- **OpenWebMath 219M**: ~219M tokens of math web pages with formulas and procedures
- **OpenThoughts filtered 170M**: ~170M tokens of CoT traces (for comparison)

### Runs (300M, all with LR=3e-3, WD=3.2, 6400 steps)

| Run | Data | ARC Easy | PIQA | SciQ | dclm_val |
|---|---|---|---|---|---|
| Baseline | DCLM 200M | 39.6% | 60.3% | 63.2% | 3.797 |
| Code only | Code 218M (Python/JS/C/C++) | 26.1% | 49.4% | 49.4% | 5.947 |
| **OpenWebMath only** | OWM 219M (math web pages) | 34.9% | 48.9% | **73.2%** | 4.304 |
| OpenThoughts only | OT 170M (CoT traces) | — (not eval'd on easy benchmarks) | — | — | 6.187 |

### Key Findings (Procedural Knowledge)
1. **OpenWebMath beats DCLM on SciQ**: 73.2% vs 63.2% — first reasoning data to beat baseline on ANY benchmark
2. **Code alone doesn't help**: Hurts all benchmarks (ARC Easy 26.1%, PIQA 49.4%, SciQ 49.4%)
3. **OpenThoughts confirmed bad**: Worst dclm_val loss (6.187), no benchmark improvements
4. **Procedural knowledge hypothesis validated**: Math web pages (which show HOW to solve problems) help more than explicit reasoning traces (which show step-by-step solutions)
5. **Sequential curriculum still fails**: When we tried OWM→DCLM sequentially, the model forgot the SciQ gains

### Open Questions
1. **Simultaneous mixing untested**: 80% DCLM + 20% OpenWebMath mixed during training — might preserve both web text quality AND SciQ gains
2. **600M with correct LR**: 600M v2 runs crashed, need restart with LR=1e-3
3. **Code + DCLM mixing**: 80% DCLM + 20% Code — code alone fails but mixed might help
4. **Causal bridges**: The cross-document bridge idea from `causal_bridges_proposal.txt` — still unexplored

---

## May 3: Eval Consolidation, Reference Models & 1.4B Experiments

### Paper's Benchmarks (arc_easy, piqa, sciq)
The paper evaluates on easier benchmarks than what we initially used. Results on these:

**300M models:**

| Model | ARC Easy | PIQA | SciQ |
|---|---|---|---|
| Paper 300M (16ep, WD=1.6) | **43.8%** | **62.5%** | **72.1%** |
| Our 300M A (DCLM baseline) | 39.6% | 60.3% | 63.2% |
| Our 300M C (OT→DCLM) | 32.1% | 54.5% | 50.3% |
| Our 300M D (DCLM→OT) | 37.5% | 57.6% | 58.8% |
| Random | 25% | 50% | 25% |

**600M models (our experiments):**

| Run | ARC Easy | PIQA | SciQ | dclm_val |
|---|---|---|---|---|
| A (DCLM baseline) | **37.3%** | **58.2%** | **58.1%** | 3.789 |
| C (OT→DCLM) | 30.9% | 53.4% | 47.5% | 5.668 |
| D (DCLM→OT) | 34.1% | 56.2% | 47.6% | 4.074 |

Reasoning data hurts all benchmarks at 600M — even the easier ones. DCLM baseline is best.

### Reference: OLMo 1B Models

| Model | Params | Tokens | ARC Easy | ARC-C | PIQA | SciQ |
|---|---|---|---|---|---|---|
| OLMo 1B | 1B | 3T | 63.3% | 28.5% | 75.0% | 86.7% |
| OLMo 1B 0724 | 1B | 3T | 61.1% | 30.5% | 74.7% | 92.7% |
| OLMo 2 1B | 1B | 4T | **72.4%** | **38.7%** | **75.7%** | **95.2%** |

Massive gap between our 300M-600M models (200M tokens) and properly trained 1B models (3-4T tokens).

### Key Finding: PIQA Test Split Has No Labels
PIQA test split returns label=-1 for all examples. Must use validation split for per-example eval. The lm-eval-harness handles this correctly but our manual eval script initially didn't.

### 1.4B Reasoning Experiments (completed ~4:02 AM PST May 4)

#### Runs (1.4B)

| Run | Description | dclm_val | ARC Easy | PIQA | SciQ | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|----------|----------|------|------|-------|-----------|------------|------|
| A (baseline, from earlier) | DCLM 200M, 8ep | **3.413** | 43.6% | 62.6% | 71.7% | 18.5% | 28.3% | 50.0% | 23.2% |
| B | OT only, 6400 steps | 6.211 | 31.3% | 53.6% | 51.5% | 18.8% | 26.2% | 49.9% | 23.0% |
| C | OT→DCLM (3200+3200) | 5.935 | 28.6% | 54.5% | 42.4% | 17.0% | 26.3% | 49.4% | 23.1% |
| D | DCLM→OT (3200+3200) | 4.331 | 32.1% | 57.1% | 44.9% | 17.4% | 26.0% | 50.0% | 23.4% |

#### Key Findings (1.4B)
- Same pattern as 300M/600M — reasoning data hurts both DCLM perplexity AND downstream benchmarks
- Run D (DCLM→OT) best among reasoning runs but still worse than DCLM baseline on all metrics
- 1.4B model shows same U-shape in dclm_val during OT-only training: drops, recovers, plateaus
- dclm_val trajectory for Run B: 12.3 → 7.8 → 9.5 → 6.5 → 6.2 (interesting overfitting then recovery)
- No model size from 300M to 1.4B shows benefit from OpenThoughts reasoning data on any benchmark

#### Cross-Scale Summary (all models, same experiment design)

**dclm_val loss:**
| Run | 300M | 600M | 1.4B |
|-----|------|------|------|
| A (DCLM baseline) | 3.797 | 3.789 | 3.413 |
| B (OT only) | 6.187 | 6.151 | 6.211 |
| C (OT→DCLM) | 5.051 | 5.668 | 5.935 |
| D (DCLM→OT) | 3.906 | 4.074 | 4.331 |

**ARC Easy:**
| Run | 300M | 600M | 1.4B |
|-----|------|------|------|
| A (DCLM baseline) | 39.6% | 37.3% | 43.6% |
| B (OT only) | — (not eval'd) | — (not eval'd) | 31.3% |
| C (OT→DCLM) | 32.1% | 30.9% | 28.6% |
| D (DCLM→OT) | 37.5% | 34.1% | 32.1% |

#### Conclusion (OpenThoughts)
At 200M token data budget with models 300M–1.4B, pretraining on reasoning data (OpenThoughts CoT traces) provides NO benefit over standard web text (DCLM) on any metric — perplexity, ARC, PIQA, SciQ, HellaSwag, WinoGrande, or MMLU. The reasoning data actively hurts performance. This holds regardless of curriculum order (reasoning first or web first).

---

## May 2: Reasoning Data Curriculum Experiments (600M)

**NOTE:** These 600M runs used LR=3e-3 (same as 300M), but the paper specifies LR=1e-3 for 600M. This was fixed in commit `0aa2c60a6` but the runs below have NOT been re-run with the correct LR. Results may be slightly off.

### Hypothesis
Same as 300M experiments but at 600M scale — does larger model show clearer signal from reasoning data?

### Runs (600M)

| Run | Description | Phase 1 | Phase 2 | Steps | dclm_val | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|---------|---------|-------|----------|-------|-----------|------------|------|
| A (baseline) | DCLM only | DCLM 6400 steps | — | 6400 | **3.789** | 0.170 | 0.264 | 0.487 | — |
| B | OT only | OT 6400 steps | — | 6400 | **6.151** | 0.225 | 0.275 | 0.500 | 0.263 |
| C | OT→DCLM | OT 3200 steps | DCLM 3200 steps | 6400 | **5.668** | 0.172 | 0.261 | 0.509 | 0.258 |
| D | DCLM→OT | DCLM 3200 steps | OT 3200 steps | 6400 | **4.074** | 0.177 | 0.262 | 0.493 | 0.252 |

### Key Findings (600M)
- Same pattern as 300M — reasoning data hurts DCLM perplexity, order matters (DCLM first is better)
- Eval harness still near random — 600M not enough to show reasoning gains
- 600M doesn't show improvement from reasoning data on any metric vs 300M

---

## May 1: Reasoning Data Curriculum Experiments (300M)

### Hypothesis
Does mixing reasoning data (OpenThoughts-114k) with web data (DCLM) during pretraining improve perplexity or reasoning benchmarks?

### Data
- **DCLM 200M**: 164K web documents, ~200M tokens
- **OpenThoughts filtered**: 54K reasoning traces (math/code/science CoT), ~170M tokens. Filtered to docs ≤4096 tokens to avoid truncating reasoning chains (53% of original data was >4096 tokens and would lose conclusions).

### Runs (300M)

| Run | Description | Phase 1 | Phase 2 | Steps | dclm_val | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|---------|---------|-------|----------|-------|-----------|------------|------|
| A (baseline) | DCLM only | DCLM 6400 steps | — | 6400 | **3.797** | 0.175 | 0.274 | 0.504 | — |
| B | OT only | OT 6400 steps | — | 6400 | **6.187** | 0.226 | 0.267 | 0.500 | 0.259 |
| C | OT→DCLM | OT 3200 steps | DCLM 3200 steps | 6400 | **5.051** | 0.218 | 0.266 | 0.507 | 0.253 |
| D | DCLM→OT | DCLM 3200 steps | OT 3200 steps | 6400 | **3.906** | 0.214 | 0.272 | 0.505 | 0.269 |

### Key Findings (300M)
- Pure reasoning data pretraining (B) is bad for web text perplexity (6.187 vs 3.797)
- OT first then DCLM (C) doesn't recover — 5.051 still far from baseline
- DCLM first then OT (D) barely hurts perplexity (3.906 vs 3.797) but reasoning benchmarks near random
- All eval harness scores near random chance for 300M — model too small to show reasoning signal
- Model D learned **structure** of reasoning (markdown, numbered steps, "therefore") but not actual reasoning

### Text Generation Samples (300M)
Saved to `outputs/generations/300m_generations.json` and `outputs/generations/300m_runC_benchmark_generations.json`.
Key observation: Models produce fluent-looking but factually wrong text. Model D (DCLM→OT) produces formatted reasoning that is wrong.

---

## Pre-May 2: Paper Replication

### Hypothesis
Replicate "Pre-training Under Infinite Compute" paper results on local 8x A100-40G GPUs.

### Runs

| Run | Model | Data | Tokens | Epochs | WD | LR | Steps | Time | dclm_val | Notes |
|-----|-------|------|--------|--------|-----|-----|-------|------|----------|-------|
| 300M baseline | 300M (seq_len=4k) | dclm_200m | 200M | 8 | 0.1 | 1e-3 | 6400 | ~1.5h | **3.797** | Paper gets 3.785. Match. |
| 1.4B regularized (dclm_200m) | 1.4B (seq_len=4k) | dclm_200m | 200M | 8 | 3.2 | 1e-3 | 6400 | ~3.5h (TE) | **3.413** | Paper single-model best: 3.462. We beat it slightly — likely dclm_200m is a curated subset. |
| 1.4B (dclm_shard73) | 1.4B (seq_len=4k) | dclm shard73 | 655M | ~2.6 | 3.2 | 1e-3 | 6400 | ~4.5h (TE) | **3.309** | More unique tokens → less repetition → lower val loss. |
| 8B (dclm_200m) | 8B | dclm_200m | 200M | 1 | 0.1 | 3e-3 | 6104 | ~5h | **6.897** | 8B on 200M tokens is massively undertrained. |
| 1.4B OpenThoughts (unfiltered) | 1.4B (seq_len=4k) | openthoughts_flat | 795M | ~2.1 | 3.2 | 1e-3 | 6400 | ~4.7h (TE) | **5.647** | Pure reasoning data → bad at web text. Expected. |

### Key Findings (Pre-May 2)
- Successfully replicated paper's single-model results within 0.05 nats
- The 3.174 number we chased was an **ensemble** result, not single-model (paper's best single 1.4B = 3.462)
- `max_train_batches=800` slices a fixed 51,200 sequences — every epoch sees the same data
- Transformer Engine 2.13 works with Levanter after adapting attention code (~30% speedup)
- High weight decay (3.2 vs 0.1) is critical for multi-epoch training

---

### Open Questions / Next Steps

1. **Need DCLM-only baselines with eval harness** for both 300M and 600M to compare properly
2. **Paper's benchmarks are easier** (arc_easy, piqa, sciq) than what we used (arc_challenge, hellaswag, winogrande, mmlu). Paper's 300M model gets 44% arc_easy. Should switch to their benchmarks.
3. **Paper's models are on HuggingFace** (`konwoo/300m4k-*`) — can download and replicate their exact eval numbers
4. **Scale question**: Do we need 1.4B+ to see reasoning data benefits? NVIDIA front-loading paper used 8B.
5. **Data mixing**: Haven't tried simultaneous mixing (80% DCLM + 20% OT) — only sequential curriculum.
6. **The half-baked idea**: Cross-document causal bridges — still unexplored. Requires generating bridges, not just selecting data.

---

### Infrastructure Notes

- **Transformer Engine 2.13**: Required 3 changes to Levanter attention.py (global mesh resource, AttnSoftmaxType, keyword args for fused_attn). ~30% speedup (2.5s → 1.8s/step for 1.4B).
- **Tokenization**: Full DCLM tokenization infeasible (~60 days estimated). Got 8 usable shards (~36B tokens).
- **Bug fixes**: OverflowError in iris backoff, VersionedValue tokenizer bug, GPU support in DataEfficiencyConfig.
- **OpenThoughts truncation**: 53% of docs >4096 tokens. Filtered to ≤4096 to keep complete reasoning chains.
