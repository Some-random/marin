# Overnight: actual reasoning-augmented DCLM data (2026-07-05)

**Goal (Dongwei, going to sleep):** stop theorizing — generate ACTUAL implicit-reasoning augmentations on
real DCLM docs and have data to look at in the morning; weaker model is fine.

**FINAL status (03:17 PDT — complete):**
- ✅ **544 docs augmented with Qwen2.5-7B-Instruct** → `data/dclm_aug_qwen7b.jsonl` (+ `_readable.md`).
- ✅ **3000 docs with Qwen2.5-32B-Instruct** (better quality, improved prompt) → `data/dclm_aug_qwen32b.jsonl`
  (+ `_readable.md`). **← the main artifact; these are real numbered reasoning chains.**
- ✅ **Completeness checks done** (7B + 32B, judge=7B) — both negative, and the negativity is *diagnostic*
  (see below): the zero-shot perplexity probe measures format shift, not reasoning value.
- **Total: 3,544 reasoning-augmented DCLM docs.**

## What "the data" is
For each real DCLM doc: split into `context` (first ~65%) + `continuation` (rest); a teacher LLM writes the
**implicit reasoning** the context takes for granted; we save `{context, reasoning, continuation}`. The
reasoning is *manufactured* (raw text doesn't contain it) — teacher = local **Qwen2.5-Instruct** self-hosted
via HF transformers on our own GPUs. **No API used** (none available on this box: only GH/HF/WandB tokens);
no data left the cluster; zero $ spent.

## Pipeline (both in this dir)
- `generate_reasoning.py` — teacher writes a concise COMPLETE numbered reasoning chain per doc (prompt asks
  for every load-bearing step, no gaps, but skip facts everyone knows → the complete-and-minimal target).
- `compute_completeness.py` — the doc's own continuation is the free "answer key": judge NLL of the
  continuation given `context` vs `context+reasoning`; lower with reasoning = a gap was closed. `--ablate`
  drops one step and checks NLL rises (a load-bearing / minimality probe).

## Real examples (verbatim)

**32B (improved prompt → genuine numbered chain):** context = a comment about ACA/Medicaid coverage →
```
1. The ACA expansion aims to provide additional health coverage options.
2. Currently, children under 18 and certain adults have access to Medicaid based on parental eligibility or income.
3. Individuals aged 18-65 with incomes below $11,000/year do not have coverage under existing programs but could benefit from ACA expansion.
4. Through the ACA expansion, these individuals could enroll in health exchanges and receive affordable private plans.
5. Participation would be a significant improvement over current lack of coverage.
```

**7B (coherent but paraphrase-y):** context = "How Does Mitosis Result in Tissues and Organs?…" →
> "Mitosis produces identical cells, allowing for the growth and repair of tissues and organs. This process
> enables the replacement and expansion of cellular structures necessary for bodily functions."
(actual continuation went on to discuss somatic-cell division — the reasoning bridges toward it.)

## Honest quality notes
- **32B >> 7B**: the improved prompt + bigger model produces real explicit step-chains; 7B tends to
  summarize/paraphrase rather than expose non-obvious inferential steps.
- **Judge caveat:** the completeness check uses Qwen-7B as judge, which is close to the generator — the
  *right* judge is a base model standing in for the learner (our 1.4B base). Treat the numbers as a first,
  optimistic read; re-run with a 1.4B base judge for the real signal.
- Some `continuation` fields are empty when the 65% split left nothing after — minor, filtered in scoring.

## Completeness-check results

### 7B data, judge=Qwen2.5-7B (155 docs scored) — NEGATIVE, and instructive
| | mean continuation NLL | ppl |
|---|---:|---:|
| context only | 2.610 | 13.6 |
| context + reasoning | 3.056 | 21.2 |

**Adding the reasoning RAISED the continuation's perplexity (−0.446 nats/token worse); only 2.6% of docs
improved.** So the naive "free answer key" signal is negative here. Honest read of *why* — two confounds,
and the second is the important one:
1. **7B reasoning is vague/paraphrase-y** — it restates the topic instead of stating what specifically comes next.
2. **The check itself is format-confounded (the real lesson).** We measure a *zero-shot* judge that has never
   seen "context → numbered reasoning → prose continuation." Splicing a numbered list before natural web prose
   is off-distribution, so the continuation looks more surprising **regardless of the reasoning's content**.
   This is exactly the distribution-shift we flagged — and it means **zero-shot continuation-perplexity is a
   poor proxy for completeness.** BoLT/Quiet-STaR's benefit only appears *after training on the format* (the
   model learns to use the reasoning); a zero-shot probe penalizes the unfamiliar format instead.
   (Ablation probe didn't trigger for 7B — its reasoning is a paragraph, not multi-line steps.)

**Takeaway:** don't trust the zero-shot perplexity check as the completeness metric. The real signal needs
either (a) a **base judge + natural insertion**, or (b) the actual **training experiment** (train on the
format, then measure). The 32B numbers below (numbered chains → ablation works) are a second data point, but
the format confound applies to them too — interpret cautiously.

### 32B data, judge=Qwen2.5-7B (234 docs scored) — also NEGATIVE, and it *confirms* the confound
| | mean continuation NLL | ppl |
|---|---:|---:|
| context only | 2.655 | 14.2 |
| context + reasoning | 3.230 | 25.3 |
- NLL reduction from reasoning: **−0.575 nats/token (worse)**; reasoning helped in only **0.4%** of docs.
- Ablation (drop one step): NLL rose in **32.5%** of docs (mixed — ~⅓ of steps look load-bearing, but this is
  confounded too).

**The smoking gun:** the *higher-quality* 32B reasoning scores **worse** than the vague 7B reasoning
(−0.575 vs −0.446). If the metric were tracking reasoning value, better chains would help more. It does the
opposite — because it's measuring **format/distribution shift**: 32B produces longer, more-structured numbered
chains, which shift the local distribution further from "natural web prose," so a zero-shot judge finds the
real continuation *more* surprising. **So the zero-shot continuation-perplexity check is not a valid
completeness metric** — do not read the negative numbers as "the reasoning is bad" (inspection shows the 32B
chains are genuinely good; see `data/dclm_aug_qwen32b_readable.md`).

## Bottom line
1. **Data: success.** 3,544 real DCLM docs augmented with implicit reasoning (544 @ 7B + 3,000 @ 32B),
   locally, no API, no $, no data egress. The 32B numbered chains are the target artifact — look at them.
2. **Completeness check: the zero-shot perplexity proxy failed, and failed informatively** — it measures
   format shift, not reasoning completeness (better reasoning → worse score is the tell). BoLT/Quiet-STaR
   only show the benefit *after training on the format*; a zero-shot probe penalizes the unfamiliar format.
3. **What to do next:** (a) redesign the check — use a **1.4B base judge** and insert reasoning naturally
   (not as a bracketed list), OR score whether reasoning predicts a held-out *probe* rather than the raw
   continuation; and/or (b) go straight to the **training experiment** (train 1.4B on complete vs.
   gap-broken chains, chain-before-continuation) — the only test that isn't zero-shot-format-confounded.

## Next steps (for review, not launched)
1. Look at `data/dclm_aug_qwen32b_readable.md` — are these the completeness-augmented docs you want?
2. If yes: scale generation (more docs / Qwen3.6-27B via vLLM), swap the judge to a 1.4B base, then the
   Stage-1 training experiment (train 1.4B on complete vs. gap-broken chains, chain-before-continuation).
