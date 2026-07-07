# Perplexity-drop hunt — can adding a rationale LOWER continuation perplexity?

**Goal (Dongwei, 2026-07-07):** find a setup where inserting the rationale makes the held-out continuation
*more* predictable (`+rationale − base < 0`), zero-shot, no training. "Try everything."

## Where we started (controlled check v2, Qwen2.5-32B-Instruct judge, 5 warranting docs)
| insertion | format penalty (placebo−base) | content (claude−placebo) | claude − base |
|---|---:|---:|---:|
| bracketed (`Reasoning:\n1.2.3`) | +0.296 | −0.118 | **+0.178** |
| natural (prose) | +0.133 | −0.021 | **+0.112** |

→ format penalty was the obstacle; natural insertion halved it but shrank content. Still no raw drop.

## Lever 1 — a BASE learner judge (our DCLM 1.4B), 500 docs, Qwen rationales
Instruct models over-penalize off-distribution inserts; a base LM (the actual learner) shouldn't. Sweep
insertion × target (`perplexity_hunt.py`, judge = `1ep_dclm_step14672_hf`, n=397 scored):

| style | target | mean_delta | %drop | drop_mean |
|---|---|---:|---:|---:|
| bracketed | full | **+0.141** | **12.3%** | −0.069 |
| bracketed | first | +0.266 | 11.3% | −0.157 |
| natural | full | +0.195 | 7.3% | −0.059 |
| nosep | full | +0.157 | 11.6% | −0.061 |

- **Mean is still positive** — over ALL docs, adding a rationale raises perplexity.
- **But ~12% of docs drop** — and that ≈ the **11.6% warrant rate** from `RATIONALE_WARRANT_AUDIT.md`.
- No memorization (`noctx` 3.98). Base judge likes `bracketed` (the "natural is better" effect was a Qwen-*instruct* artifact).

**Hypothesis this points to:** rationales lower perplexity *on the reasoning-dependent docs* and raise it on
the ~7/8 that have no latent reasoning (a rationale on a product listing is noise). The mean-over-all is the
wrong statistic; the right one is **mean over docs that warrant a rationale**.

## Lever 2 — independent R/N split + strong (Claude) rationales  *(in progress)*
Selecting on the Qwen-drop outcome is circular (the drop-subset examples — a MiFi review, a devotional —
didn't look reasoning-dependent). So: independently label 100 docs **R** (genuine latent reasoning) vs **N**
(narrative/opinion/factlist), write **context-only Claude rationales** for all, and measure `claude − base`
split by label. Prediction: **mean delta on R < 0** (drop), **on N > 0**.

### Result — the hypothesis is REFUTED
100 docs independently labeled (**19 R / 81 N**, ~19% — reproduces the warrant rate), context-only Claude
rationales, `bracketed+full`, DCLM-1.4B base judge (`perplexity_rn.py`):

| label | teacher | n | mean_delta | %drop |
|---|---|---:|---:|---:|
| **R** | claude | 19 | **+0.090** | 10.5% |
| R | qwen | 19 | +0.102 | 15.8% |
| N | claude | 81 | +0.089 | 14.8% |
| N | qwen | 81 | +0.154 | 9.9% |

**R ≈ N.** Reasoning-dependent docs show NO drop and are indistinguishable from non-reasoning docs. The
earlier 12%≈11.6% match was **coincidence**. (Claude beats Qwen on N — less harmful — but no win.)

### Diagnosis — continuation-perplexity is the wrong metric, and here's why
`NLL(continuation | context + rationale)` can only drop if the rationale carries information about the
**continuation**. But the "R" label is about the **context** containing reasoning — it does NOT imply the
**continuation** is that reasoning's *output*. In raw web docs the next text is almost always **new facts /
a topic shift** (the Apple doc's continuation is more earnings detail, not the conclusion of the
CFO-guidance inference). So the reasoning has nothing to predict, no matter the judge, insertion, or doc
selection. **Verified across 4 configs (Qwen-instruct bracketed/natural, base-1.4B sweep, R/N split) —
continuation-perplexity never drops.** This is exactly why BoLT/Quiet-STaR only show benefit *after*
training on the format: zero-shot, the reasoning doesn't predict the raw next tokens; trained, the model
learns to *use* it.

### The one avenue left that can honestly show a drop: change the TARGET
Score a target the reasoning actually **determines** — a held-out probe (a question whose answer follows
from the context by reasoning), not the raw continuation. If making the reasoning explicit lowers the
probe-answer's NLL (with a leakage guard: answer not verbatim in the rationale), that's a genuine
"+rationale lowers perplexity" — on the thing reasoning is supposed to make predictable. Trying this next.

### RESULT — a raw probe drop, but the placebo control REFUTES it as reasoning-specific ⚠️
19 probes (one per R doc; `perplexity_probe.py`), leakage-guarded (2 removed where the answer appeared
verbatim in the rationale/context), DCLM-1.4B base judge. NLL of the reasoning-derived answer, **with** the
rationale vs **without**:

- **mean delta −0.358 nats/token**, median −0.290, **12/17 (70.6%) drop**; individual drops up to −1.66.
- Clean examples: id 190 arithmetic ("5 bits→32, so 6 bits→**64**") 3.56→3.18; id 444 ("spoilers never
  deployed") 8.24→**6.59** (−1.66); id 89 chained inference ("the boy's father") 3.54→2.93. The 5 that rose
  are vague/subjective answers ("worse", "it defends Twain's character") where reasoning doesn't pin wording.

**The contrast IS the finding:**
| target | mean delta | drop? |
|---|---:|:---:|
| raw continuation — 4 configs (Qwen-instruct ×2, base-1.4B sweep, R/N split) | +0.09 … +0.18 | ❌ no |
| reasoning-determined probe | **−0.358** | ✅ **yes, 70%** |

Adding a rationale **does** lower perplexity — of the content the reasoning **determines**, not arbitrary
next-web-text. The rationale's value is real but only visible on a reasoning-dependent target. (This is also
why BoLT/Quiet-STaR only pay off *after* training on the format: zero-shot, the reasoning predicts the
reasoning-derived answer, not the raw continuation.)

### PLACEBO CONTROL — the probe drop is a FORMAT artifact, not reasoning ❌
Re-ran the probes with a third arm: `context + ANOTHER doc's (irrelevant) rationale + Q` (`perplexity_probe_placebo.py`).

| arm | mean delta | %drop |
|---|---:|---:|
| real rationale | −0.358 | 71% |
| **placebo** (irrelevant rationale) | **−0.333** | 71% |
| **real − placebo** (the reasoning-specific effect) | **−0.024** | — |

In the MEAN, an irrelevant rationale drops the answer's perplexity essentially as much as the real one
(−0.333 vs −0.358) — so most of the raw −0.358 is a **format/priming effect**, and the raw "win" is retracted.

**But the per-probe split (`data/probe_placebo_perprobe.jsonl`) is the real story — the mean −0.024 washes out
a genuine effect:**
- **8/17 probes show a real reasoning-specific drop** (`real ≪ placebo`), strongest on **specific, non-generic
  answers**: id 444 "spoilers never deployed" (real −1.66, placebo **−0.00**, r−p −1.65), id 458 "its low-cost
  advantage" (−1.21 vs −0.41), id 89 "the boy's father" (−0.61 vs −0.22). The irrelevant rationale does nothing;
  the *right* reasoning does the work.
- **Vague/common-word answers** ("worse", "it can't escape") show the opposite — a random rationale primes the
  common word as well or better (id 345: placebo −1.30 vs real +0.46). These outliers drag the mean to ~0.

So it is **not** "all artifact": there is a real reasoning-specific signal **on well-posed (specific-answer)
probes**, drowned in the mean by priming on generic answers + n=17 noise. (Post-hoc subset — flagged as such.)

### Honest bottom line for the night
- **Continuation target:** no drop (4 configs). Raw next-web-text isn't the reasoning's output.
- **Probe target (mean):** raw drop is mostly format/priming, not reasoning (real−placebo ≈ 0).
- **Probe target (per-probe):** a **real reasoning-specific drop on ~half the probes**, large on specific
  answers — suggestive, not established (n=17, post-hoc).
- **What would settle it:** a **pre-registered specific-answer probe set** (drop vague/directional answers,
  n≥100, per-probe placebo) — the clean version of tonight's most promising signal. And ultimately the
  **training experiment** (needs Dongwei's sign-off — not launched).

