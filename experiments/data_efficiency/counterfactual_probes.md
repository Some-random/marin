# Counterfactual Probe Design

Companion to [`next_steps_strategy.md`](next_steps_strategy.md). Detailed design for **Path B**: cheap evaluations that decompose "model can do X" from "model knows the surface pattern of X" on our existing 6 checkpoints, no new training required.

The motivation: our 4 small-scale models score at floor on math (GSM8K 0.0-0.014) and code (bigcode HumanEval 0.000). The benchmark numbers alone don't tell us *why*. A model at 0.0 on GSM8K could have no arithmetic at all, or could have arithmetic but no word-problem parsing — those have very different implications for which data to bet on next.

## Research questions the probes answer

For each capability our models score near zero on:

1. **Q1 — Does the model have the underlying capability circuit at all?** E.g., for GSM8K = 0: does the model know `1 + 1 = 2`?
2. **Q2 — Does the model fail at composition / multi-step?** E.g., can it do `7 × 3`, but not `7 × 13`?
3. **Q3 — Are existing scores driven by surface-pattern memorization rather than the named capability?** E.g., does the same MMLU question score differently if we rename "France" to "Atlantis"?

Q1 and Q2 mostly target our 4 small models (where the bottleneck is "scale + data"). Q3 targets phi-1.5 specifically (where the bottleneck might be "synthetic data overfits to the surface distribution of its source").

---

## Probe family A — Arithmetic decomposition

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
