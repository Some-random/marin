# Evaluation Reference: Tasks, Taxonomy, and Model Results

Canonical reference for the evaluation suite used in the data-efficiency project. Lives outside `EXPERIMENT_LOG.md` (which is a chronological narrative) so the eval setup and current-best numbers can be read independently of any particular day's work.

---

## 1. Why this doc exists

Names like "NL reasoning benchmark" hide important mechanistic differences. Before reasoning about why a benchmark moved (or didn't), **read 2–3 actual examples** and classify the task by the cognitive mechanism it tests. The May 26 code-mix probe is the standing example: sciq and piqa are both routinely labeled "NL reasoning", but inspection shows they operate through completely different mechanisms (sciq is passage-grounded extraction; piqa is parametric commonsense). Confusing them produces false positives for "reasoning gains".

This doc:
- Classifies every eval we use by mechanism, with a complete inline example for each (§2).
- States which evals give signal at our 1.4B / 3.3B-token scale, and which don't (§3).
- Lists the models tracked (§4) and the canonical cross-model results table (§5).
- Has the per-subset Paloma comparison (§6).

---

## 2. Taxonomy of evaluations by mechanism (with examples)

### A. Continuous PPL (no task structure, just next-token loss)

**Paloma macro** — 16 held-out web/forum/code text subsets; primary continuous signal at our scale. (See `experiments/data_efficiency/run_1_4b_25code_alg.py` for the subset list.)

> Example (`c4_en` subset, `val-00000000.jsonl.gz`):
>
> ```
> Media friends to attend & cover event to affirm nation's support to jawans.
> Organisor thru their newspapers/media.
> Finally a performing arts College in Noida!
> Learning music, dance or any other art form not only soothes your inner being, but it has also been proven that people who systematically learn music have better strategizing and planning skills than other people. The world today is very competitive and every individual has to be active not only in academics or any one field but in all possible areas. Ishaan Music College has been established to promote and create just the right environment for anyone to learn art, music or dance.
> ```
>
> Eval = mean next-token cross-entropy over this text.

**dclm_200m_val** — held-out NL within the DCLM training distribution. Sensitive to overfitting on the 209M-token training slice.

> Example (DCLM raw, `dclm_1500m.jsonl`):
>
> ```
> Take the 2-minute tour ×
>
> Here what happened with me today. TimeMachine asked me whether I want to set a backup disk, I've answered yes, but then, when I've realized that in order to...
> ```

**opc_algorithmic (training-data loss)** — final eval loss on the code training slice when used as part of a mix. Signals code memorization, not generalization.

> Example (`opc_algorithmic.jsonl.gz`):
>
> ````
> Write a python function to find the kth smallest element in a Binary Search Tree (BST).
>
> ```python
> class TreeNode:
>     def __init__(self, x):
>         self.val = x
>         self.left = None
>         self.right = None
>
> class Solution:
>     def getlt(self,pRoot,lt,k):
>         if pRoot == None:
>             return
>         self.getlt(pRoot.left,lt,k)
>         if len(lt)<k+1:
>             lt.append(pRoot)
>             self.getlt(pRoot.right,lt,k)
>
>     def KthNode(self, pRoot, k):
>         if pRoot == None or k < 1:
>             return None
>         lt = []
>         self.getlt(pRoot,lt,k)
>         if len(lt) < k:
>             return None
>         return lt[k-1]
> ```
> ````

### B. Passage-grounded reading comprehension (answer is in the prompt; model attends and extracts)

**sciq** — every question comes with a `support` paragraph that literally states the answer. 4-way MC.

> Example (`samples_sciq_*.jsonl`):
> - **question**: "Compounds that are capable of accepting electrons, such as o 2 or f2, are called what?"
> - **support**: "Oxidants and Reductants Compounds that are capable of accepting electrons, such as O 2 or F2, are called oxidants (or oxidizing agents) because they can oxidize other compounds. In the process of accepting electrons, an oxidant is reduced. Compounds that are capable of donating electrons, such as sodium metal or cyclohexane (C6H12), are called reductants (or reducing agents) because they can cause the reduction of another compound."
> - **choices**: oxidants / antioxidants / Oxygen / residues
> - **correct_answer**: oxidants

**boolq** — yes/no question + Wikipedia-style passage that contains the answer.

> Example (`samples_boolq_*.jsonl`):
> - **question**: "does ethanol take more energy make that produces"
> - **passage**: "Ethanol fuel -- All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. … one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. …"
> - **label**: False (i.e., ethanol does not take more energy to produce than it produces)

**openbookqa** — 4-way MC; in the standard `open_book` variant a relevant fact is supplied alongside the question (we run the no-fact variant in lm-eval).

> Example (`samples_openbookqa_*.jsonl`):
> - **question_stem**: "A person wants to start saving money so that they can afford a nice vacation at the end of the year. After looking over their budget and expenses, they decide the best way to save money is to"
> - **choices**: make more phone calls / quit eating lunch out / buy less with monopoly money / have lunch with friends
> - **answerKey**: B

### C. Parametric world knowledge (no passage; model must recall facts from weights)

**arc_easy** — 4-way MC grade-school science.

> Example (`samples_arc_easy_*.jsonl`):
> - **question**: "Which statement best explains why photosynthesis is the foundation of most food webs?"
> - **choices**: A) Sunlight is the source of energy for nearly all ecosystems. / B) Most ecosystems are found on land instead of in water. / C) Carbon dioxide is more available than other gases. / D) The producers in all ecosystems are plants.
> - **answerKey**: A

**arc_challenge** — harder ARC subset; mostly at-random at 1.4B/3.3B.

> Example (`samples_arc_challenge_*.jsonl`):
> - **question**: "An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?"
> - **choices**: A) Planetary density will decrease. / B) Planetary years will become longer. / C) Planetary days will become shorter. / D) Planetary gravity will become stronger.
> - **answerKey**: C

**mmlu** — 4-way MC across 57 subject subtasks (abstract_algebra, business_ethics, ..., world_religions). 5-shot in our pipeline.

> Example (`samples_mmlu_abstract_algebra_*.jsonl`):
> - **subject**: abstract_algebra
> - **question**: "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
> - **choices**: 0 / 4 / 2 / 6
> - **answer**: index 1 → "4"

### D. Physical / social commonsense (no passage; intuition from weights)

**piqa** — 2-way MC; pick the physically plausible solution to an everyday task.

> Example (`samples_piqa_*.jsonl`):
> - **goal**: "How do I ready a guinea pig cage for it's new occupants?"
> - **sol1**: "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish."
> - **sol2**: "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
> - **label**: 0 (sol1)

**social_iqa** — 3-way MC; social-situation reasoning.

> Example (`samples_social_iqa_*.jsonl`):
> - **context**: "Tracy didn't go home that evening and resisted Riley's attacks."
> - **question**: "What does Tracy need to do before this?"
> - **answers**: A) make a new plan / B) Go home and see Riley / C) Find somewhere to go
> - **label**: 3 (C)

**hellaswag** — sentence-completion plausibility from ActivityNet/WikiHow contexts.

> Example (`samples_hellaswag_*.jsonl`):
> - **activity_label**: "Roof shingle removal"
> - **ctx**: "A man is sitting on a roof. he"
> - **endings**: 0) is using wrap to wrap a pair of skis. / 1) is ripping level tiles off. / 2) is holding a rubik's cube. / 3) starts pulling up roofing on a roof.
> - **label**: 3

### E. Coreference / logical / commonsense MC

**winogrande** — pronoun-resolution pairs (Winograd schema). 2-way.

> Example (`samples_winogrande_*.jsonl`):
> - **sentence**: "Sarah was a much better surgeon than Maria so _ always got the easier cases."
> - **option1**: Sarah
> - **option2**: Maria
> - **answer**: 2 (Maria)

**logiqa** — formal logical reasoning MC.

> Example (`samples_logiqa_*.jsonl`):
> - **context**: "In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park. These four communities are designated as cultural area, leisure area, commercial area and administrative service area. It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area."
> - **question**: "Based on the above statement, which of the following can be derived?"
> - **options**: A) Civic Park is north of the administrative service area / B) The leisure area is southwest of the cultural area / C) The cultural district is in the northeast of the business district / D) The business district is southeast of the leisure area
> - **label**: a

**commonsense_qa** — 5-way MC, ConceptNet-derived commonsense reasoning.

> Example (`samples_commonsense_qa_*.jsonl`):
> - **question**: "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"
> - **question_concept**: revolving door
> - **choices**: A) bank / B) library / C) department store / D) mall / E) new york
> - **answerKey**: A

### F. Math (multi-step generation)

**gsm8k** (logprob variant, 5-shot) and **gsm8k_cot** (free-generation, 8-shot CoT) — same problems, different scoring.

> Example (`samples_gsm8k_cot_*.jsonl`):
> - **question**: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
> - **answer (CoT)**: "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market. #### 18"
> - **gold**: 18

**minerva_math** — competition math, free generation, evaluated by `math_verify`.

> Example (`samples_minerva_math_algebra_*.jsonl`):
> - **type**: Algebra (Level 3)
> - **problem**: "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?"
> - **solution**: "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$. Therefore, the graph has $\\boxed{2}$ vertical asymptotes."
> - **answer**: 2

### G. Code generation

**HumanEval** — function generation from docstring; pass@1 by running unit tests.

> Example (`samples_humaneval_*.jsonl`, `HumanEval/0`):
>
> ```python
> from typing import List
>
> def has_close_elements(numbers: List[float], threshold: float) -> bool:
>     """ Check if in given list of numbers, are any two numbers closer to each other than
>     given threshold.
>     >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
>     False
>     >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
>     True
>     """
> ```
>
> Hidden test (run on the generated continuation):
>
> ```python
> def check(candidate):
>     assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
>     assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
>     assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
>     # ...
> ```

**MBPP** — short Python programming problems with explicit test asserts.

> Example (`samples_mbpp_*.jsonl`, `task_id 11`):
> - **text**: "Write a python function to remove first and last occurrence of a given character from the string."
> - **test_list** (used for scoring):
>   ```python
>   assert remove_Occ("hello","l") == "heo"
>   assert remove_Occ("abcda","a") == "bcd"
>   assert remove_Occ("PHP","P") == "H"
>   ```
> - **canonical solution** (reference; model is asked to generate from `text`):
>   ```python
>   def remove_Occ(s,ch):
>       for i in range(len(s)):
>           if (s[i] == ch):
>               s = s[0 : i] + s[i + 1:]
>               break
>       for i in range(len(s) - 1,-1,-1):
>           if (s[i] == ch):
>               s = s[0 : i] + s[i + 1:]
>               break
>       return s
>   ```

---

## 3. What's usable at our 1.4B / 3.3B-token scale

| Eval | Status at our scale |
|---|---|
| Paloma macro PPL | primary continuous signal |
| dclm_200m_val PPL | continuous; sensitive to overfit |
| arc_easy | above-random |
| sciq | above-random (passage-grounded) |
| piqa | above-random |
| boolq | barely above-random |
| arc_challenge | at-random |
| mmlu | at-random |
| hellaswag | ~ at-random |
| winogrande | at-random |
| openbookqa | ~ at-random |
| commonsense_qa | at-random |
| social_iqa | ~ at-random |
| logiqa | at-random |
| gsm8k (logprob) | floor |
| gsm8k_cot | floor |
| minerva_math | floor |
| HumanEval | floor for our recipes (phi-1: 0.49) |
| MBPP | floor (our pipeline scoring) |

The only signal-producing benchmarks at our scale are **Paloma + dclm_200m_val + arc_easy + sciq + piqa + boolq**. Everything else is logged for completeness but should be treated as noise around the model. The four discrete benchmarks split across mechanisms — sciq+boolq are passage-grounded (B), arc_easy+piqa are parametric knowledge/commonsense (C/D). Always classify deltas by mechanism, not by name.

---

## 4. Models tracked

| Label | HF repo (or local path) | Params | Train tokens | Notes |
|---|---|---|---|---|
| **1.4B baseline** | `1_4b_wd1_6_x16_nocrossblock_hf` (`peach-thunder-100` / `6xx0hu3l`) | 1.4 B | 3.3 B (DCLM-200M, x16 epochs) | wd=1.6, LR=1e-3 cosine, block_cross_doc=False |
| **1.4B code-mix 25%** | `1_4b_25code_alg_hf` (`eager-grass-104` / `p2n84bo3`) | 1.4 B | 3.3 B (75% DCLM + 25% opc_algorithmic) | Same recipe as baseline + code component |
| phi-1 | `microsoft/phi-1` | 1.3 B | ~7 B (filtered Stack + ~1B GPT-3.5 synth Python) | Code-only |
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

**Caveat on code-gen numbers.** `humaneval` and `mbpp` in our pipeline use `lm-eval-harness`'s scoring path (`pass_at_1,none` for MBPP, `pass@1,none` for HumanEval) with `--confirm_run_unsafe_code` + `HF_ALLOW_CODE_EVAL=1`. The original phi-1 paper reported MBPP 55.5% with the BigCode evaluation framework; our pipeline reports phi-1 MBPP 1.0%. The methodology difference (extraction patterns, n-shot, runner) is substantial. **Treat our code-gen pipeline numbers as conservative lower bounds; do not directly compare to published phi paper numbers.**

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

## Updating this doc

When a new model is trained or a new eval is added, update §4 (models) and §5 (results) with the new row/column. Add a brief follow-up entry in `EXPERIMENT_LOG.md` pointing here. Chronological narrative stays in `EXPERIMENT_LOG.md`; canonical reference stays here.
