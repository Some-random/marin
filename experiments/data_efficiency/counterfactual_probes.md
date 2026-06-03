# Counterfactual Probe Design

> **Revision note (2026-06-03 afternoon).** What ran the previous night under the name "counterfactual probe Phase 1" was the *arithmetic decomposition probe* below (Family A) — a **skill-decomposition probe, not a counterfactual probe in the Wu et al sense**. It usefully decomposed "GSM8K = 0" for our 4 small models (B4 has single-digit arithmetic, A5 doesn't), but it was format-bound (`a + b = ` only) and gave phi-1.5 a misleading 0 because phi-1.5 was trained on word-problem textbooks, not bare equations. Real counterfactual probes — defined below in CF-1..CF-3 — keep the task structure identical and only swap surface tokens (Wu, Geiger, Goodman, Manning, [Reasoning or Reciting?](https://arxiv.org/abs/2307.02477) 2024; Mirzadeh et al, [GSM-Symbolic](https://arxiv.org/abs/2410.05229) 2024). The doc is reorganized accordingly: counterfactual designs come first, the skill-decomposition probe is preserved at the bottom as "Family A (v1, skill-decomposition; ran 2026-06-02)" for reference.

Companion to [`next_steps_strategy.md`](next_steps_strategy.md). Detailed design for **Path B**: cheap evaluations that decompose "model can do X" from "model knows the surface pattern of X" on our existing 6 checkpoints, no new training required.

---

## Counterfactual probes (CF-1..CF-3) — the actual designs

**Design pattern (Wu et al + GSM-Symbolic):** keep task structure identical, swap only surface tokens (entities, numbers, formats). Measure delta accuracy(original) − accuracy(perturbed). A model with genuine capability shows small delta. A model leaning on surface memorization shows large delta.

### CF-1 — Format-invariant arithmetic (for our 4 1.4B models)

Each arithmetic problem presented in 4 surface formats, single fixed problem set:

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
- **A5** (DCLM-only) — bare-equation only (35% A1 → format-bound).
- **B4** (DCLM + code) — bare-equation + Python REPL (textbook code patterns; format-bound in TWO formats). If B4 also gets the word problem right, that's strong "code teaches transferable arithmetic" evidence.
- **phi-1.5** — word-problem only (textbook-distribution-bound; explains the 0.305 GSM8K + 0% on bare-equation).
- **phi-1** — Python REPL only.

This *answers* the H1 question directly: does data X teach a real capability that survives surface change, or just teach a format?

**Implementation cost:** ~1 day to write, ~10 min/model to run.

### CF-2 — GSM-Symbolic for phi-1.5

Phi-1.5 scores 0.305 on GSM8K. Take its successes and perturb per Mirzadeh et al (2024):
- Rename entities consistently: `Janet → Beatrice`, `ducks → muffins`, `eggs → tickets`.
- Change numbers within same magnitude: `16 → 22`, `2 → 3`, `4 → 5`.
- Change template wording without changing structure.

Measure `phi-1.5(perturbed) / phi-1.5(original)`. Mirzadeh found:
- GPT-4: ~10 pp drop on perturbed
- Llama-3-8B: ~25 pp drop
- Smaller open models: >50% drop

If phi-1.5: 0.305 → ~0.10 on perturbed = surface memorization. → ~0.28 = real reasoning. Either is informative.

Our 4 small models floor on GSM8K either way, so CF-2 is mostly a phi-1.5 result. Not central to the matched-token comparison, but academically interesting as a standalone finding.

**Implementation cost:** ~2 days (perturb taxonomy + generation + scoring).

### CF-3 — Counterfactual MMLU (Wu et al style)

Take 3-4 MMLU subjects with substitutable surface tokens:
- `high_school_world_history`: rename countries (`France → Atlantis`, `Napoleon → Avander`) consistently across questions. Same dates, same events, different names.
- `abstract_algebra`: rename groups (`Z_5 → R_5`), operators (`+ → ⊞`). Same algebra.
- `formal_logic`: rename predicates (`P, Q → A, B`). Same inferences.

For each (original, counterfactual) pair, measure accuracy. Phi-1.5 should drop more than base — synthetic training memorizes surface patterns harder.

**Implementation cost:** ~3-5 days (per-subject surface-substitution rules + sanity-check pass).

---

## Implementation order

1. **CF-1 first.** Answers the matched-token H1 directly, no external dataset, ~1 day.
2. **CF-2 second.** Only needs perturbation of existing GSM8K, ~2 days.
3. **CF-3 third.** Most academically interesting but most work; ~3-5 days.

---

The motivation: our 4 small-scale models score at floor on math (GSM8K 0.0-0.014) and code (bigcode HumanEval 0.000). The benchmark numbers alone don't tell us *why*. A model at 0.0 on GSM8K could have no arithmetic at all, or could have arithmetic but no word-problem parsing — those have very different implications for which data to bet on next.

## Research questions the probes answer

For each capability our models score near zero on:

1. **Q1 — Does the model have the underlying capability circuit at all?** E.g., for GSM8K = 0: does the model know `1 + 1 = 2`?
2. **Q2 — Does the model fail at composition / multi-step?** E.g., can it do `7 × 3`, but not `7 × 13`?
3. **Q3 — Are existing scores driven by surface-pattern memorization rather than the named capability?** E.g., does the same MMLU question score differently if we rename "France" to "Atlantis"?

Q1 and Q2 mostly target our 4 small models (where the bottleneck is "scale + data"). Q3 targets phi-1.5 specifically (where the bottleneck might be "synthetic data overfits to the surface distribution of its source").

---

## Family A (v1, skill-decomposition; ran 2026-06-02) — arithmetic decomposition

**This is the probe that ran 2026-06-02, before the redesign above. It is a *skill-decomposition* probe, not a counterfactual probe.** Kept here for reference and because its result (B4 has single-digit arithmetic, A5 doesn't) is real, just measured under a single fixed surface format which biases against phi-1.5. Replaced by CF-1 going forward.

**Targets Q1 + Q2 on the math axis.**

**Construction:** synthetically generate prompts at five difficulty levels. No external dataset; all generated in code with a fixed `numpy.random` seed so the probe is fully reproducible.

| Level | Format | Range | Count | Purpose |
|---|---|---|---|---|
| A1 | `a + b = ` | a, b ∈ [0, 9] | 100 | Most basic arithmetic |
| A2 | `a + b = ` | a, b ∈ [10, 99], no carry | 100 | Two-digit, no decomposition needed |
| A3 | `a + b = ` | a, b ∈ [10, 99], with carry | 100 | Two-digit with decomposition |
| A4 | `a × b = ` | a ∈ [2, 9], b ∈ [2, 9] | 100 | Single-digit multiplication |
| A5 | `a - b = ` | a, b ∈ [0, 99], a ≥ b | 100 | Subtraction |

**Scoring:**
- Greedy generate up to `max_new_tokens=4`. Strip whitespace.
- Match: numeric prefix == ground truth integer.
- Report per-level accuracy.

**Prompt format:** 0-shot, single line, model-vocab-friendly. Match phi-1.5's number format (no commas, no leading zero):

```
1 + 2 = 
```

Token-budget: ~6 tokens prompt, ~4 tokens completion. 500 problems × 6 models = 3000 single-batch generations = ~5 GPU-minutes total.

**Expected discriminating outcome:**
- DCLM baseline / A5 / B4: probably 30-70% on A1, dropping toward 0 on A3 and A5.
- phi-1.5: high on A1-A5 (it does GSM8K = 0.305, so likely 80%+ on basic arithmetic).
- phi-1: ??? code-only training. Unclear — interesting either way.

If our 4 small models hit ~0% even on A1, the GSM8K floor is "no arithmetic capability" — pushes us toward scale rather than data-mix. If they hit 60% on A1, then GSM8K floor is "can't parse word problems" — pushes us toward NL textbook style data (e.g., cosmopedia/phi-1.5).

**Implementation:** new file `probes_arithmetic.py`. Pure-Python probe generation + eval. ~150 LOC. No external dataset dependency.

---

## Probe family B — CRUXEval-style code execution

**Targets Q1 on the code axis.**

**Why:** our 4 small models can't generate Python (bigcode HumanEval = 0.000). But can they *read* Python? CRUXEval-output (Gu et al., 2024) gives a Python function + input and asks for the output. No generation required — pure code understanding.

**Dataset:** [`cruxeval-org/cruxeval`](https://huggingface.co/datasets/cruxeval-org/cruxeval). 800 problems, single-paragraph deterministic Python functions. Output is a short string. Already supported by lm-eval-harness as the `cruxeval` task family (`cruxeval_output_predict`, `cruxeval_input_predict`).

**Setup:**
- Task: `cruxeval_output_predict` (input known, predict output)
- Scoring: lm-eval generation + Python-eval comparison (the harness wraps this)
- Few-shot: 0-shot first; 1-shot if everyone is at floor.
- Runs via existing `.venv/bin/accelerate launch -m lm_eval` infra.

**Expected outcome:**
- phi-1 (code-only): should do well — it was trained on the right distribution.
- B4 (25% code mix): higher than A5 (DCLM-only)?
- phi-1.5: middling (some code in mix).
- Our 4 small models: probably all floor.

**Discriminating value:** if B4 beats A5 here even by a few pp, that's "25% code mix DOES teach code understanding, just not generation" — refines the negative-NL-result story.

**Cost:** standard lm-eval run. ~10 min/model on 8 GPUs.

---

## Probe family C — Counterfactual MMLU (Wu et al. style)

**Targets Q3 on the knowledge axis.**

**Why:** phi-1.5 scores MMLU 0.422, but it's trained on GPT-3.5-generated textbooks. We don't know if 0.422 reflects acquired knowledge or surface-pattern memorization of the synthetic distribution. Wu, Geiger, Goodman & Manning ([Reasoning or Reciting?, 2024](https://arxiv.org/abs/2307.02477)) show counterfactual rewrites can decompose this.

**Approach (lightweight version, no novel methodology):**

Pick 3 MMLU subjects where a simple surface substitution preserves the reasoning structure:

| Subject | Original | Counterfactual transform |
|---|---|---|
| `high_school_world_history` | Real country/leader names | Substitute consistently (France → Atlantis, Napoleon → Avander, etc.) using a fixed mapping table |
| `abstract_algebra` | "Find the degree of Q(√2, √3, √18) over Q." | Rename Q → R_5, √2 → ω_2, etc. Same algebra, different surface tokens. |
| `formal_logic` | Propositional symbols | Substitute symbol names (P → A, Q → B) and operator wording (∧ → "and" → "yotta"). |

Construction: scripted text substitution with a deterministic mapping. Verify by eyeballing 5 examples per subject.

**Scoring:**
- Standard MMLU MC: log-probability over the four answer letters.
- Compare model's MMLU score on (original, counterfactual) pairs.
- If `accuracy_original` >> `accuracy_counterfactual`, the model is leaning on surface memorization, not the underlying knowledge structure.

**Expected outcome:**
- phi-1.5: original 0.422; counterfactual ???
- Our 4 small models: original ~0.25 (random); counterfactual likely also ~0.25 (no signal in either direction — they may not have either knowledge or memorization at this scale)

**Discriminating value:**
- If phi-1.5 keeps ~0.4 on counterfactual, it has real subject knowledge (still impressive given its training distribution).
- If phi-1.5 drops to ~0.25 on counterfactual, MMLU 0.422 reflects surface memorization more than knowledge.
- For our small models, the probe will likely be inconclusive (they're at random floor either way), but the negative result is itself informative — "at this scale, neither memorization nor knowledge has formed."

**Implementation cost:** the substitution-mapping construction is the slow part. 3 subjects × 100 questions = 300 manual or scripted substitutions. ~1 day of work to do right.

---

## Implementation plan

**Phase 1 (this week, no GPU bottleneck):**
1. Write `probes_arithmetic.py` — fully self-contained. Run on all 6 models. ~1 day.
2. Run `cruxeval_output_predict` via lm-eval. ~30 min after Phase 1's gpu-st-4 frees up.
3. Add a §4 to EVALUATION.md with the probe results. Update headlines if probes change the story.

**Phase 2 (next week, only if Phase 1 is informative):**
4. Build counterfactual MMLU substitution dictionaries for 3 subjects. ~2 days.
5. Run paired MMLU eval (original + counterfactual) on all 6 models. Compare deltas.
6. Add §5 to EVALUATION.md.

**What success looks like:** after Phase 1, EVALUATION.md has new rows that change the *story* (not just add numbers) for at least one of:
- Whether our 4 small models have any arithmetic capability.
- Whether B4's code mix teaches code understanding even though it doesn't teach code generation.

**What failure looks like:** every model floor on every probe → conclusion is "1.4B / 30B tokens is too small for any of these capabilities" → strong evidence that the next experiment should be scale, not data.

---

## Open questions before starting

1. **Few-shot format for arithmetic probes:** phi-1.5 was trained on textbook examples — does it prefer "1 + 2 = 3" 0-shot or a 4-shot CoT prompt? Worth pilot-testing both on 10 examples first.
2. **MMLU subject choice for counterfactual:** are there subjects where surface substitution actually preserves the reasoning (vs subjects like US-history where the question IS the surface)? The 3 I listed should work but may need to be refined.
3. **Should we also probe natural-language tasks (e.g., paraphrased HellaSwag)?** Adds value if HellaSwag is the main NL benchmark we care about; not adding it for now to keep scope tight.
