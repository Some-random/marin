# Counterfactual Probe Design

> **Revision note (2026-06-03 afternoon).** What ran the previous night under the name "counterfactual probe Phase 1" was the *arithmetic decomposition probe* below (Family A) — a **skill-decomposition probe, not a counterfactual probe in the Wu et al sense**. It usefully decomposed "GSM8K = 0" for our 4 small models (B4 has single-digit arithmetic, A5 doesn't), but it was format-bound (`a + b = ` only) and gave phi-1.5 a misleading 0 because phi-1.5 was trained on word-problem textbooks, not bare equations. Real counterfactual probes — defined below in CF-1..CF-3 — keep the task structure identical and only swap surface tokens (Wu, Geiger, Goodman, Manning, [Reasoning or Reciting?](https://arxiv.org/abs/2307.02477) 2024; Mirzadeh et al, [GSM-Symbolic](https://arxiv.org/abs/2410.05229) 2024). The doc is reorganized accordingly: counterfactual designs come first, the skill-decomposition probe is preserved at the bottom as "Family A (v1, skill-decomposition; ran 2026-06-02)" for reference.

Companion to [`next_steps_strategy.md`](next_steps_strategy.md). Detailed design for **Path B**: cheap evaluations that decompose "model can do X" from "model knows the surface pattern of X" on our existing 6 checkpoints, no new training required.

---

## Counterfactual probes — replicating Wu et al + GSM-Symbolic, NOT inventing variants

> **2026-06-03 revision (PM):** The CF-1..CF-3 originally listed below were *hand-picked variants on hunch* (4 self-chosen formats, MMLU rewrites for subjects I picked). That violates the rule of following published methodology exactly when the user has named a paper. Replaced here with strict replications of:
> - **Wu, Geiger, Goodman, Manning (2024), "Reasoning or Reciting?"** — counterfactual world-model perturbations (e.g., arithmetic in base-9 instead of base-10).
> - **Mirzadeh et al (2024, Apple), "GSM-Symbolic"** — symbolic GSM8K templates with name/number perturbations.

### Honest scope: who can we actually run these on?

| Model | GSM8K | Smallest GSM-Symbolic-tested = Phi-3.5-mini (3.8B). Smallest Wu et al-tested = closed-source instruction-tuned. | Replicable on this model? |
|---|---:|---|---|
| 1.4B base / code25v2 / A5 / B4 | 0.000–0.014 | Below validated range; floor on default → counterfactual drop is 0 → 0, no signal | **Out of scope.** Run anyway only with explicit "below paper's validated range" caveat. |
| phi-1 | 0.012 | Floor on default; bigcode HumanEval 0.543 | Replicable on Wu's code task; not on math. |
| phi-1.5 | 0.305 | Below paper's 3.8B floor but enough signal | Replicable on GSM-Symbolic with size-out-of-scope caveat. |

**Implication for our matched-token H1 (what data teaches reasoning at our scale):** the published counterfactual methodologies do not work directly on our 4 1.4B base models — they floor on the default tasks the papers test. We have two honest options:
- (a) Stop calling our 4-model comparison "counterfactual" and use the v1 skill-decomposition probe (renamed accordingly) for them.
- (b) Run published probes on phi-1 + phi-1.5 only and report those as standalone results, separate from the matched-token A5-vs-B4 story.

This doc favours (a)+(b) in combination: skill-decomposition for our 4 (already done as Family A below), and CF-A / CF-B / CF-C below for phi models against the published methodologies.

### CF-A — Replicate GSM-Symbolic on phi-1.5 (and phi-1 as comparison)

**Dataset:** [`apple/GSM-Symbolic`](https://huggingface.co/datasets/apple/GSM-Symbolic) — Apple's released benchmark, downloadable from HF. Three splits: `main`, `p1` (one extra step), `p2` (two extra steps). 5000 examples per split = 100 templates × 50 instances.

**Setup (per Mirzadeh et al §3.2):**
- 8-shot CoT prompting (their preliminary experiments showed shot count not significant; we use 8 for direct comparability with the paper's tables)
- Greedy decoding
- Exact-match scoring against the question's gold answer
- Report the accuracy *distribution* across the 50 datasets of 100 examples each (per Mirzadeh Fig 2), not just a single mean

**Models:** phi-1.5 (in-scope-adjacent, has signal on GSM8K = 0.305) + phi-1 (out-of-scope, floor). Skip our 4 × 1.4B models.

**Comparison numbers:**
- phi-1.5 GSM8K (already have): 0.305
- phi-1.5 GSM-Symbolic main: TBD
- phi-1.5 GSM-Symbolic p1: TBD
- phi-1.5 GSM-Symbolic p2: TBD

The paper's Figure 2 shows GSM8K accuracy on the right tail of the GSM-Symbolic distribution for 21 of 25 tested models, with drops ranging 0.3–9.2 pp. If phi-1.5 shows a similar pattern, that's a clean reproduction of the published finding extended to a smaller model.

**Implementation cost:** ~1 day. Dataset already on HF, just need a wrapper that uses our lm-eval pipeline. (Custom task YAML for `apple/GSM-Symbolic` + run.)

### CF-B — Replicate Wu et al arithmetic counterfactual (§3.1)

**Setup (per Wu et al §3.1):**
- Default: two-digit base-10 addition (same as Brown et al 2020)
- Counterfactual bases: 8, 9, 11, 16 (chosen because 8 and 16 are programmer-familiar, 9 and 11 are uncommon)
- 0-shot and 0-shot-CoT prompts (paper reports both)
- **CCC (Counterfactual Comprehension Check)** required: ask model the successor relation under each base. If a model fails the CCC, it doesn't understand the base specification and the counterfactual score is uninformative.

**Models:** Wu et al §4 footnote 6 explicitly excluded open-source models due to "unsatisfactory instruction-following ability". Our 1.4B base models will almost certainly fail the CCC. Phi-1.5 has the strongest chance; report it with caveat. Our 4 1.4B base models we report CCC-fail-rate only, not the counterfactual score.

**Code reference:** `https://github.com/ZhaofengWu/counterfactual-evaluation` (paper's released code + data).

**Implementation cost:** ~2 days (clone repo, adapt to our lm-eval pipeline, CCC then conditional probe).

### CF-C — Replicate Wu et al programming counterfactual (§3.2)

**Setup (per Wu et al §3.2):**
- Default: Python (0-based indexing) on HumanEval
- Counterfactual: ThonPy (fictional language = Python with 1-based indexing) on HumanEval
- CCC: simple list indexing on much simpler inputs

**Models:** **phi-1 only.** Our 4 1.4B models score 0.000 on default bigcode HumanEval — can't generate Python at all, ThonPy result will also be 0.000 (no signal). Phi-1.5 = 0.342 lm-eval / similar bigcode — also runnable. Phi-1 = 0.543 bigcode — primary target.

**Headline question:** does phi-1's code-only training give it generalizable code-execution understanding (small drop on ThonPy) or 0-based-indexing memorization (large drop)?

**Implementation cost:** ~1 day. Wu et al's released code includes ThonPy prompts.

### Family A (v1, renamed: skill-decomposition probe, not counterfactual)

Still useful for the matched-token A5 vs B4 finding (B4 has single-digit arithmetic, A5 doesn't). Kept in the doc below under its honest name. Replaces the previous CF-1 framing.

---

## Implementation order (revised, honest)

1. **CF-A first** (~1 day): GSM-Symbolic on phi-1.5 + phi-1. Strict paper replication, dataset already on HF.
2. **CF-C second** (~1 day): Wu et al ThonPy on phi-1. Clean code-counterfactual signal.
3. **CF-B third** (~2 days): Wu et al arithmetic counterfactual on phi-1.5. Lower priority because most models will fail the CCC.
4. **Skill-decomposition (former Family A)** stays as the only published-style result for our 4 × 1.4B models, with the v1 caveat.

**What I am NOT doing this time:**
- Inventing 4 hand-picked surface formats and calling it counterfactual.
- Picking MMLU subjects to "rewrite" without the published methodology being on those subjects.
- Running our 4 × 1.4B models on published counterfactual protocols without the scope caveat.

---

## Original v1 designs (preserved for transparency — superseded above)

### ~~CF-1 — Format-invariant arithmetic (for our 4 1.4B models)~~ (RETRACTED — hand-picked variant, not paper-replicated)

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
