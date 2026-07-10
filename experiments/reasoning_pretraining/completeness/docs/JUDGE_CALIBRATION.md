# Judge calibration — completeness test across models

Same test on every judge: for real DCLM docs whose continuation needs a ≥3-step reasoning chain, score the
NLL of the real **target** given `context` (base) vs `context+complete rationale` vs `context+incomplete`
(a load-bearing middle step deleted) vs `context+placebo` (an unrelated doc's complete rationale). All n=41.

**Columns:** `Δ` = mean NLL change vs base (nats/tok); negative = lowers the real target's perplexity.

| judge | type | era | complete−base | incomplete−base | placebo−base | **complete−placebo** | **complete−incomplete** |
|---|---|---|---:|---:|---:|---:|---:|
| DCLM-1.4B | base | ours, DCLM 2023 | +0.048 (51%↓) | +0.044 (54%↓) | +0.745 | **-0.698** | **+0.004** (17/41) |
| Llama-3.1-8B | base | Meta 2024 | +0.074 (49%↓) | +0.058 (59%↓) | +0.767 | **-0.693** | **+0.016** (17/41) |
| Qwen3.5-35B-A3B | base | Qwen newest | +0.163 (46%↓) | +0.109 (39%↓) | +0.932 | **-0.770** | **+0.053** (12/41) |
| Qwen3.5-35B-A3B | instruct | Qwen newest | +0.135 (41%↓) | +0.066 (41%↓) | +0.908 | **-0.773** | **+0.069** (11/41) |
| Qwen2.5-72B | base | Qwen 2024 | +0.336 (37%↓) | +0.283 (34%↓) | +0.747 | **-0.411** | **+0.052** (13/41) |
| Qwen2.5-72B | instruct | Qwen 2024 | +0.306 (29%↓) | +0.245 (34%↓) | +0.758 | **-0.452** | **+0.060** (18/41) |
| GLM-4.5-Air-Base | base | Zhipu 2025 (110B) | +0.216 (29%↓) | +0.177 (32%↓) | +0.859 | **-0.642** | **+0.039** (19/41) |

## What every judge agrees on

1. **Adding a rationale does NOT beat raw text** — `complete − base` is **positive on every judge**
   (+0.05 to +0.34), drop-rate ≤51% (coin-flip or worse). Same for incomplete. On zero-shot perplexity,
   inserting a rationale is neutral-to-*worse* than leaving the text alone. This is the fair "does
   augmentation help" number, and it's a no.
2. **The reasoning CONTENT is real** — `complete − placebo` is strongly negative everywhere (−0.41 to
   −0.77): the *right* reasoning beats a same-format *irrelevant* rationale, which always *hurts*
   (`placebo − base` ≈ +0.75). The gap between #1 and #2 is the **format/insertion cost** — both arms in #2
   pay it, so the placebo isolates content.
3. **Completeness does NOT matter** — `complete − incomplete` ≈ 0 (+0.00 to +0.07), ~coin-flip on every
   judge. A gap-broken chain predicts the target as well as a complete one; the model fills the deleted step.

## Two things this sweep specifically established

- **Newer/better base ≠ weaker effect.** The *newest* base (Qwen3.5-35B) shows the *strongest* content
  signal (−0.770). Qwen2.5-72B (−0.411) is the outlier, not a trend — so its weaker number was a quirk of
  that older model, not "strong models need reasoning less."
- **Instruct-tuning barely moves this measurement.** Base vs. instruct of the *same* model are nearly
  identical: Qwen2.5-72B −0.411/−0.452, Qwen3.5-35B −0.770/−0.773; completeness null on both. So the
  base-vs-instruct judge choice doesn't change any conclusion here.

**Bottom line:** across 1.4B→110B, three families (DCLM/Llama/Qwen/GLM), two eras, and base+instruct —
relevant reasoning content is real, but on zero-shot perplexity *adding* a rationale doesn't beat raw text,
and *completeness* is not the active ingredient. The only unconfounded test of completeness is training.

*Raw per-doc NLLs are the gitignored `data/complete_results*.jsonl` (on /fsx disk); this table is the record.*
