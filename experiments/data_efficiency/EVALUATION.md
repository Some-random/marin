# Evaluation Reference: Tasks, Taxonomy, and Model Results

## 1. Models tracked

| Label | HF repo (or local path) | Params | Train tokens | Notes |
|---|---|---|---|---|
| **1.4B baseline** | `1_4b_wd1_6_x16_nocrossblock_hf` (`peach-thunder-100` / `6xx0hu3l`) | 1.4 B | 3.3 B (DCLM-200M, x16 epochs) | wd=1.6, LR=1e-3 cosine, block_cross_doc=False |
| **1.4B code-mix 25%** | `1_4b_25code_alg_hf` (`eager-grass-104` / `p2n84bo3`) | 1.4 B | 3.3 B (75% DCLM + 25% opc_algorithmic) | Same recipe as baseline + code component |
| **1.4B 1-ep DCLM @ step-14672** | `1ep_dclm_step14672_hf` (run `1ep-dclm-A5`, in-flight) | 1.4 B | ~15.4 B trained so far (target 30.77 B, x1 epoch over 7 DCLM shards) | wd=0.1, LR=3e-4 cosine, 4-node DP. Mid-training snapshot. |
| **1.4B 1-ep code-mix @ step-14672** | `1ep_code25_step14672_hf` (run `1ep-code25-B4`, in-flight) | 1.4 B | ~15.4 B trained so far (target 30.77 B, x1 epoch each) | Same as above + 25% code mix (aryabumi_synth 17.5% + aryabumi_web 4.4% + opc 3.1%) |
| phi-1 | `microsoft/phi-1` | 1.3 B | ~7 B (filtered Stack + ~1B GPT-3.5 synth Python) | Code-only |
| phi-1.5 | `microsoft/phi-1_5` | 1.3 B | ~30 B (phi-1 mix + ~20B synthetic NL textbooks) | 5 epochs × 30B unique |

---

## 2. Taxonomy by mechanism (with examples)

Two-way split for QA: **open-book** (the answer is in the prompt; model attends and extracts) vs. **closed-book** (no passage; model uses weights). Plus three task families that don't fit the QA frame: math, code generation, and continuous PPL.

### A. Open-book QA

**sciq** — every question comes with a `support` paragraph that literally states the answer. 4-way MC.

> - **question**: "Compounds that are capable of accepting electrons, such as o 2 or f2, are called what?"
> - **support**: "Oxidants and Reductants Compounds that are capable of accepting electrons, such as O 2 or F2, are called oxidants (or oxidizing agents) because they can oxidize other compounds. In the process of accepting electrons, an oxidant is reduced. Compounds that are capable of donating electrons, such as sodium metal or cyclohexane (C6H12), are called reductants (or reducing agents) because they can cause the reduction of another compound."
> - **choices**: oxidants / antioxidants / Oxygen / residues
> - **correct_answer**: oxidants

**boolq** — yes/no question + Wikipedia-style passage that contains the answer.

> - **question**: "does ethanol take more energy make that produces"
> - **passage**: "Ethanol fuel -- All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. … one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. …"
> - **label**: False

**openbookqa_fact** — custom variant we added in `experiments/data_efficiency/openbookqa_fact.yaml` that uses the `additional` config of `allenai/openbookqa` and prepends the dataset's `fact1` field to the question stem. This is the open-book MC eval. Re-run pending; replaces the closed-book openbookqa default that lm-eval ships with.

### B. Closed-book QA / commonsense

No passage in the prompt. The model has to recall facts, apply commonsense, or do logical deduction from its weights. Multi-way MC.

**arc_easy** — grade-school science MC.

> - **question**: "Which statement best explains why photosynthesis is the foundation of most food webs?"
> - **choices**: A) Sunlight is the source of energy for nearly all ecosystems. / B) Most ecosystems are found on land instead of in water. / C) Carbon dioxide is more available than other gases. / D) The producers in all ecosystems are plants.
> - **answerKey**: A

**arc_challenge** — harder ARC subset.

> - **question**: "An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?"
> - **choices**: A) Planetary density will decrease. / B) Planetary years will become longer. / C) Planetary days will become shorter. / D) Planetary gravity will become stronger.
> - **answerKey**: C

**mmlu** — 4-way MC across 57 subject subtasks. 5-shot in our pipeline.

> - **subject**: abstract_algebra
> - **question**: "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
> - **choices**: 0 / 4 / 2 / 6
> - **answer**: 1 → "4"

**piqa** — 2-way MC; physical-intuition continuations.

> - **goal**: "How do I ready a guinea pig cage for it's new occupants?"
> - **sol1**: "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish."
> - **sol2**: "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
> - **label**: 0 (sol1)

**social_iqa** — 3-way MC; social-situation reasoning.

> - **context**: "Tracy didn't go home that evening and resisted Riley's attacks."
> - **question**: "What does Tracy need to do before this?"
> - **answers**: A) make a new plan / B) Go home and see Riley / C) Find somewhere to go
> - **label**: 3 (C)

**hellaswag** — sentence-completion plausibility from ActivityNet/WikiHow contexts.

> - **activity_label**: "Roof shingle removal"
> - **ctx**: "A man is sitting on a roof. he"
> - **endings**: 0) is using wrap to wrap a pair of skis. / 1) is ripping level tiles off. / 2) is holding a rubik's cube. / 3) starts pulling up roofing on a roof.
> - **label**: 3

**winogrande** — pronoun-resolution pairs (Winograd schema). 2-way.

> - **sentence**: "Sarah was a much better surgeon than Maria so _ always got the easier cases."
> - **option1**: Sarah
> - **option2**: Maria
> - **answer**: 2 (Maria)

**commonsense_qa** — 5-way MC, ConceptNet-derived commonsense.

> - **question**: "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"
> - **question_concept**: revolving door
> - **choices**: A) bank / B) library / C) department store / D) mall / E) new york
> - **answerKey**: A

**logiqa** — formal logical reasoning. (The `context` here is the puzzle premise, not a knowledge passage — the model has to deduce from it, which is why this is closed-book rather than open-book.)

> - **context**: "In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park. These four communities are designated as cultural area, leisure area, commercial area and administrative service area. It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area."
> - **question**: "Based on the above statement, which of the following can be derived?"
> - **options**: A) Civic Park is north of the administrative service area / B) The leisure area is southwest of the cultural area / C) The cultural district is in the northeast of the business district / D) The business district is southeast of the leisure area
> - **label**: a

### C. Math (multi-step generation)

**gsm8k** (logprob, 5-shot) and **gsm8k_cot** (free generation, 8-shot CoT) — same problems, different scoring.

> - **question**: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
> - **answer (CoT reference)**: "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market. #### 18"
> - **gold**: 18

**minerva_math** — competition math, free generation, scored by `math_verify`.

> - **type**: Algebra (Level 3)
> - **problem**: "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?"
> - **solution**: "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$. Therefore, the graph has $\\boxed{2}$ vertical asymptotes."
> - **answer**: 2

### D. Code generation

**HumanEval** — function generation from docstring; pass@1 by running unit tests.

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
>     # ...
> ```

**MBPP** — short Python programming problems with test asserts.

> - **text**: "Write a python function to remove first and last occurrence of a given character from the string."
> - **test_list** (used for scoring):
>   ```python
>   assert remove_Occ("hello","l") == "heo"
>   assert remove_Occ("abcda","a") == "bcd"
>   assert remove_Occ("PHP","P") == "H"
>   ```

### E. Continuous PPL

**Paloma** — 16 held-out web/forum/code text subsets. Primary continuous signal at our scale. Eval = mean next-token cross-entropy.

> Example (`c4_en` subset):
>
> ```
> Media friends to attend & cover event to affirm nation's support to jawans.
> Organisor thru their newspapers/media.
> Finally a performing arts College in Noida!
> Learning music, dance or any other art form not only soothes your inner being, but it has also been proven that people who systematically learn music have better strategizing and planning skills than other people. The world today is very competitive and every individual has to be active not only in academics or any one field but in all possible areas.
> ```

**dclm_200m_val** — held-out NL within the DCLM training distribution. Sensitive to overfitting on the 209M-token slice.

> ```
> Take the 2-minute tour ×
>
> Here what happened with me today. TimeMachine asked me whether I want to set a backup disk, I've answered yes, but then, when I've realized that in order to...
> ```

**opc_algorithmic (training-data loss)** — final eval loss on the code training slice when used as part of a mix. Signals memorization on training data, not generalization.

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
>     def getlt(self, pRoot, lt, k):
>         if pRoot == None:
>             return
>         self.getlt(pRoot.left, lt, k)
>         if len(lt) < k + 1:
>             lt.append(pRoot)
>             self.getlt(pRoot.right, lt, k)
>
>     def KthNode(self, pRoot, k):
>         if pRoot == None or k < 1:
>             return None
>         lt = []
>         self.getlt(pRoot, lt, k)
>         if len(lt) < k:
>             return None
>         return lt[k - 1]
> ```
> ````

---

## 3. Canonical results — all models

All numbers from our `lm-eval-harness` pipeline (lm_eval 0.4.11) at the n-shot settings shown. `acc_norm` used where reported; `acc` otherwise. Paloma / `dclm_200m_val` loss is mean next-token cross-entropy (lower is better). Bolded cells = best for that row. Mechanism column refers to the §2 taxonomy.

**A5 1ep / B4 1ep columns are the in-flight 1-epoch experiment at the step-14672 checkpoint (~50% trained). Final-step numbers will replace these when training completes (~2026-06-02 22:00 PDT).**

| Mech | Task | Metric | n-shot | base (x16) | code25 (x16) | A5 1ep s14672 | B4 1ep s14672 | phi-1 | phi-1.5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Open | sciq | acc | 0 | 0.652 | 0.709 | **0.816** | 0.799 | 0.707 | **0.933** |
| Open | boolq | acc | 0 | 0.502 | 0.579 | 0.577 | 0.569 | 0.451 | **0.746** |
| Open | piqa | acc | 0 | 0.634 | 0.619 | 0.691 | 0.688 | 0.562 | **0.766** |
| Open | openbookqa_fact (with `fact1`) | acc_norm | 0 | 0.336 | 0.370 | 0.418 | 0.410 | 0.316 | **0.530** |
| Closed | arc_easy | acc_norm | 25 | 0.401 | 0.416 | 0.590 | 0.564 | 0.378 | **0.805** |
| Closed | arc_challenge | acc_norm | 25 | 0.242 | 0.236 | 0.297 | 0.282 | 0.232 | **0.532** |
| Closed | hellaswag | acc_norm | 10 | 0.348 | 0.341 | 0.458 | 0.427 | 0.301 | **0.635** |
| Closed | winogrande | acc | 5 | 0.504 | 0.500 | 0.530 | 0.508 | 0.498 | **0.710** |
| Closed | mmlu | acc | 5 | 0.252 | 0.249 | pending | pending | 0.248 | **0.422** |
| Closed | commonsense_qa | acc | 0 | 0.192 | 0.200 | 0.212 | 0.200 | 0.175 | **0.507** |
| Closed | social_iqa | acc | 0 | 0.366 | 0.362 | 0.408 | 0.394 | 0.364 | **0.523** |
| Closed | logiqa | acc | 0 | 0.218 | 0.234 | 0.235 | 0.212 | 0.214 | **0.240** |
| Math | gsm8k | exact_match | 5 | 0.000 | 0.000 | 0.000 | 0.014 | 0.012 | **0.305** |
| Math | gsm8k_cot | exact_match | 8 | 0.022 | 0.020 | 0.014 | 0.014 | 0.021 | **0.299** |
| Math | minerva_math | exact_match | 4 | 0.0002 | 0.0014 | 0.001 | 0.006 | 0.012 | **0.029** |
| Code | humaneval | pass@1 | 0 | 0.000 | 0.006 | 0.006 | 0.043 | **0.494** | 0.342 |
| Code | mbpp | pass@1 (or pass_at_1) | 3 | 0.000 | 0.000 | 0.000 | 0.068 | **0.416** | 0.342 |
| PPL | `dclm_200m_val` (in-domain) | nats | — | — | — | **2.996** | 3.058 | — | — |
| PPL | **paloma_macro (16 subsets)** | bits_per_byte | — | 1.631 | 1.483 | pending | pending | 1.738 | **1.174** |
| PPL | paloma 4chan_meta_sep | bits_per_byte | — | 1.561 | 1.402 | pending | pending | 1.665 | **1.199** |
| PPL | paloma c4_100_domains | bits_per_byte | — | 1.322 | 1.212 | pending | pending | 1.615 | **0.948** |
| PPL | paloma c4_en | bits_per_byte | — | 1.387 | 1.269 | pending | pending | 1.621 | **0.985** |
| PPL | paloma dolma-v1_5 | bits_per_byte | — | 1.433 | 1.293 | pending | pending | 1.479 | **0.957** |
| PPL | paloma dolma_100_programing_languages | bits_per_byte | — | 1.681 | 1.412 | pending | pending | 0.834 | **0.679** |
| PPL | paloma dolma_100_subreddits | bits_per_byte | — | 1.492 | 1.368 | pending | pending | 1.732 | **1.159** |
| PPL | paloma falcon-refinedweb | bits_per_byte | — | 1.448 | 1.327 | pending | pending | 1.679 | **1.025** |
| PPL | paloma gab | bits_per_byte | — | 2.475 | 2.277 | pending | pending | 2.645 | **1.918** |
| PPL | paloma m2d2_s2orc_unsplit | bits_per_byte | — | 1.389 | 1.274 | pending | pending | 1.273 | **0.946** |
| PPL | paloma m2d2_wikipedia_unsplit | bits_per_byte | — | 1.319 | 1.211 | pending | pending | 1.629 | **0.976** |
| PPL | paloma manosphere_meta_sep | bits_per_byte | — | 1.566 | 1.434 | pending | pending | 1.758 | **1.180** |
| PPL | paloma mc4 | bits_per_byte | — | 1.546 | 1.400 | pending | pending | 1.604 | **1.031** |
| PPL | paloma ptb | bits_per_byte | — | 1.565 | 1.451 | pending | pending | 1.644 | **1.057** |
| PPL | paloma redpajama | bits_per_byte | — | 1.491 | 1.335 | pending | pending | 1.374 | **0.928** |
| PPL | paloma twitterAAE_HELM_fixed | bits_per_byte | — | 3.077 | **2.834** | pending | pending | 3.634 | 2.826 |
| PPL | paloma wikitext_103 | bits_per_byte | — | 1.336 | 1.224 | pending | pending | 1.620 | **0.970** |

### Headlines

**Phi-1.5** wins on every NL subset (synthetic-textbook-heavy training pays off at 1.3B).

**Phi-1 is WORSE than our 1.4B text-only baseline on Paloma macro** (1.738 vs 1.631) because phi-1 was trained almost exclusively on code. Phi-1 only beats our 1.4B baseline on `programming_languages` and `m2d2_s2orc` (the code/academic subsets).

**Code-mix (1.4B code25, x16-epoch May 26)** beats our text-only baseline on every Paloma subset, but our June 1 in-domain analysis showed this was driven by the unique-tokens confound (v1 had 5× more unique tokens), not the code mix per se.

**1-epoch experiment at step-14672 (15 tasks):** Of 15 tasks both A5 and B4 report, **A5 (DCLM-only) wins 10 NL benchmarks by 1-3pp each, B4 (code-mix) wins 2 code benchmarks decisively (humaneval +3.7 pp, mbpp +6.8 pp), 3 tied at floor**. Code-mix is trading NL ability for code-gen ability under matched compute. In-domain val: A5 wins by 0.06 nats. This replicates the May 31 / June 1 v2 finding (matched-token code mix doesn't help in-domain NL at 1.4B / our compute) and adds the new signal that the cost shows up *across NL benchmarks consistently*. Full-checkpoint comparison (incl. paloma 16-subset bpb + full mmlu) pending training completion. See `1ep_experiment_plan.md` for methodology.

**Caveat on code-gen numbers.** Our `humaneval` / `mbpp` uses `lm-eval-harness`'s scoring path with `--confirm_run_unsafe_code` + `HF_ALLOW_CODE_EVAL=1`. The original phi-1 paper reported MBPP 55.5% with the BigCode evaluation framework; our pipeline reports phi-1 MBPP 1.0%. The methodology difference (extraction patterns, n-shot, runner) is substantial. **Treat our code-gen pipeline numbers as conservative lower bounds; do not directly compare to published phi paper numbers.**

---

## Updating this doc

When a new model is trained or a new eval is added, update §1 (models) and §3 (results) with the new row/column. Add a brief follow-up entry in `EXPERIMENT_LOG.md` pointing here. Chronological narrative stays in `EXPERIMENT_LOG.md`; canonical reference stays here.
