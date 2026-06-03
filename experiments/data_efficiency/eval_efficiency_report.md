# Eval Efficiency Report — Time Budget + Cluster Arrangement

Written 2026-06-03 after fixing the HF + multi-GPU code_eval issues that turned last night's eval into a 9-hour ordeal. With the fixes in place, the same suite is ~1.5h wall-time per model — and parallelizable across nodes.

## TL;DR

**Per-model eval suite: measured 80m 57s on 8 × A100-40GB** (B4 final on gpu-dy-5, started 13:29:17 PDT, finished 14:50:14 PDT, 2026-06-03). My earlier estimate was 85 min; actual landed 5% under. Compose any subset to fit a deadline.

**6-model sweep (our 4 + phi-1 + phi-1.5):**
- 1 node per model in parallel: **~1.5h wall time, 12 GPU-hours**
- 2 models per node sequential: ~3h wall time, 12 GPU-hours
- 1 node sequential: 9h wall time, 12 GPU-hours

The 8 p4d-24xlarge nodes (`gpu-st-1..4` + `gpu-dy-1..4`) we have idle = capacity for the full sweep with one extra slot to spare. **Default to 1 node per model.**

## Per-task time on 8 × A100-40GB (multi-GPU)

Times measured 2026-06-03 with the v2 fixes (HF offline + shared cache, per-rank metrics cache). Token counts are for B4 1ep final (1.4B). Larger models scale roughly linearly with model FLOPs.

| Task | Time | Notes |
|---|---:|---|
| arc_easy + arc_challenge 25-shot | 3 min | dual task |
| hellaswag 10-shot | 6 min | large (10k items) |
| winogrande + gsm8k 5-shot | 3 min | dual task |
| mmlu 5-shot full | 3 min | (~14k items, 57 subtasks; verified clean run 2:45) |
| piqa+boolq+sciq+openbookqa+csqa+social_iqa+logiqa 0-shot | 5 min | 7-task batch |
| openbookqa_fact 0-shot | 2 min | |
| gsm8k_cot 8-shot generation | 4 min | generation, slower |
| mbpp 3-shot (lm-eval, code_eval) | 3 min | **previously 14 min single-process** |
| humaneval 0-shot (lm-eval, code_eval) | 3 min | **previously 19 min single-process** |
| minerva_math 4-shot | 9 min | generation, 7 subtasks |
| lambada+copa+wsc+agieval 0-shot | 2 min | 4-task batch |
| gpqa 0-shot | 8 min | small but slow logprob |
| bbh 3-shot (limit=0.1) | 4 min | generation; ~650 items |
| mmlu_pro 5-shot (limit=0.1) | 17 min | **the bottleneck** — generation + extraction, ~1.2k items |
| bigcode HumanEval | 6 min | generation + Python execution |
| **Total** | **~85 min** | |

**The 17-min mmlu_pro is the bottleneck.** Worth investigating: reduce to `--limit 0.05` (~600 items, ~9 min) if the SE remains acceptable.

## Total time per model under different layouts

Suite = 85 min on 8 GPUs.

| Layout | Wall time | GPU-hours |
|---|---|---|
| 1 model on 8 GPUs | 85 min (1.4h) | 11 |
| 6 models on 6 nodes (parallel) | 85 min (1.4h) | 68 |
| 6 models on 3 nodes (2-pass) | 170 min (2.8h) | 68 |
| 6 models on 2 nodes (3-pass) | 255 min (4.3h) | 68 |
| 6 models on 1 node | 510 min (8.5h) | 68 |

(GPU-hours is invariant, only wall time differs.)

For comparison, last night's actual: ~5h wall time + ~5h waiting on HF retries = 10h total elapsed — driven by transient HF outage + my bad cache hygiene, not the work itself.

## Cluster arrangement — what we actually have

8 × p4d-24xlarge (64 GPUs total): `gpu-st-p4d24xlarge-1..4` + `gpu-dy-p4d24xlarge-1..5`.

When we did the A5/B4 1-epoch training, both runs occupied 4 nodes each, leaving 0 idle. After training: all 8 free.

**Recommended layouts**

- **Full 6-model eval after a training run:** assign one node to each model. 1.5h wall time. Use the 9th node as a hot spare or for a 7th model (e.g., a phi-1.5-style cosmopedia leg if we add one).
- **Single-model spot eval:** 1 node, 85 min.
- **Partial-suite quick check (just NL benchmarks + paloma, no code_eval / new evals):** ~25 min on 1 node.

**Avoid:** running the eval on a training node while training is still active — you'll pre-empt the training. (We accidentally did this with A5 mmlu retries on gpu-st-4 last night while it was still on the main suite — caused the GPU OOM that wasted another 2 minutes.)

## What the fixes are

All committed under `/fsx/users/dongweij/marin/outputs/`:

1. **`hf_cache/datasets/`** — 560 MB shared cache containing all eval datasets we use (`cais___mmlu`, `Rowan___hellaswag`, `Idavidrein___gpqa`, `EleutherAI___{hendrycks_math,lambada_openai,logiqa}`, `SaylorTwift___bbh`, `TIGER-Lab___mmlu-pro`, `allenai___{ai2_arc,openbookqa,winogrande}`, `baber___piqa`, `google-research-datasets___mbpp`, `hails___agieval-lsat-ar`, `openai___{gsm8k,openai_humaneval}`, `super_glue`, `tau___commonsense_qa`, `sciq`, `social_i_qa`, `openbookqa`, `winogrande`, `parquet`, `downloads`). Lives on `/fsx` so all nodes see it.

2. **`HF_DATASETS_OFFLINE=1` + `HF_HUB_OFFLINE=1`** in the eval env. Skips every HF metadata roundtrip — no more 504s. Trade-off: if a new dataset shows up that's not in the cache, eval fails immediately; we add the dataset to `/fsx/.../hf_cache/datasets/` and re-run.

3. **`lm_eval_wrapper.py`** — sets `HF_METRICS_CACHE=/tmp/hf_metrics_rank_<LOCAL_RANK>` before importing lm_eval. Each multi-GPU rank uses its own `code_eval` cache file → no race. `mbpp` 3 min instead of 14, `humaneval` 3 min instead of 19. Multi-GPU is restored as the default.

4. **`run_eval_v2.sh`** — drop-in replacement for `run_eval_final.sh` that wires up all of the above. Tested end-to-end this morning on B4 final (mmlu, mbpp, humaneval, bigcode HumanEval); all pass, all match the night-time numbers.

## Counterfactual probe redesign — Wu et al + GSM-Symbolic style

You're right that the night's arithmetic decomposition probe isn't a true counterfactual probe and was unfair to phi-1.5. The Percy / Wu et al "[Reasoning or Reciting?](https://arxiv.org/abs/2307.02477)" and Mirzadeh et al "[GSM-Symbolic](https://arxiv.org/abs/2410.05229)" papers establish the right design pattern:

**Keep the task structure identical. Swap only surface tokens (names, numbers, cities, entity types). Measure the delta.**
- If accuracy(original) ≈ accuracy(perturbed) → the model has the underlying capability.
- If accuracy(original) >> accuracy(perturbed) → the score reflects surface memorization, not understanding.

Redesigned [`counterfactual_probes.md`](counterfactual_probes.md) families (will write next):

### Family CF-1: Format-invariant arithmetic (for our 4 1.4B models)

Same arithmetic problem in 4 surface formats, single fixed problem set:

| Format | Example for `5 + 7 = 12` |
|---|---|
| Bare equation (Llama-style) | `5 + 7 = ` |
| Q&A textbook | `Q: What is 5 plus 7? A: ` |
| Python REPL | `>>> 5 + 7\n` |
| Word problem (phi-1.5-style) | `Alice has 5 apples. Bob gives her 7 more. How many apples does she have?` |

Score: model gets the problem right iff it produces the right answer in **at least one** format.
- "Format-agnostic capability" = correct in ≥3 of 4 formats.
- "Format-bound capability" = correct in 1 format only.
- "No capability" = correct in 0 formats.

Predicted result:
- **A5** (DCLM-only) — should be bare-equation only (35% A1 → format-bound).
- **B4** (DCLM + code) — should be bare-equation + Python REPL (textbook code patterns; format-bound but in TWO formats). If B4 also gets the word problem right, that's a strong "code teaches transferable arithmetic" result.
- **phi-1.5** — should be word-problem only (textbook-distribution-bound; would explain the 0.305 GSM8K + 0% on bare-equation).
- **phi-1** — should be Python REPL only.

This *answers* the H1 question directly: does data X teach a real capability that survives surface change, or just teaches a format?

### Family CF-2: GSM-Symbolic for phi-1.5 (where it has signal)

Phi-1.5 scores 0.305 on GSM8K. Take its successes and perturb:
- Rename entities: `Janet → Beatrice`, `ducks → muffins`, `eggs → tickets`.
- Change numbers within the same magnitude: `16 → 22`, `2 → 3`, `4 → 5`.
- Change template wording without changing structure.

Measure: `phi-1.5(perturbed) / phi-1.5(original)`. Mirzadeh et al found:
- GPT-4: ~10 pp drop on perturbed
- Llama-3-8B: ~25 pp drop
- Smaller models: >50% drop, often near floor

If phi-1.5's 0.305 → ~0.10 on perturbed GSM8K, it's mostly surface memorization. If 0.305 → 0.28, it's mostly real reasoning. We'd write this up as "phi-1.5 reasoning vs recitation."

Not the most directly useful for our matched-token comparison (since A5/B4 floor on GSM8K either way), but it's a nice independent finding.

### Family CF-3: Counterfactual MMLU (Wu et al style)

Take 3-4 MMLU subjects with substitutable surface tokens:
- `high_school_world_history`: rename countries (`France → Atlantis`, `Napoleon → Avander`) consistently across questions. Same dates, same events, different names.
- `abstract_algebra`: rename groups (`Z_5 → R_5`), operators (`+ → ⊞`). Same algebra.
- `formal_logic`: rename predicates (`P, Q → A, B`). Same inferences.

For each of (original, counterfactual) pair, measure accuracy. Phi-1.5 should drop more than base — synthetic training memorizes surface patterns harder.

### Implementation order

1. **CF-1 first** — answers our matched-token H1 question directly and uses existing prompt-engineering, no external dataset needed. ~1 day to build, ~10 min to run on each model.
2. **CF-2 second** — only needs perturbing existing GSM8K. Can use Mirzadeh's perturbation taxonomy. ~1-2 days to build.
3. **CF-3 third** — surface substitution needs careful manual review per subject; slower but the most academically interesting. ~3-5 days.

## What I'll change before the next run

- Use `run_eval_v2.sh` (8 GPU multi-process, shared cache, offline mode) as the default.
- Avoid running eval on training nodes.
- For partial reruns, use `--limit 0.05` on the expensive `mmlu_pro` (saves 8 min/model).
- Skip the v1 arithmetic probe in its current form for phi-models; use CF-1 instead.
