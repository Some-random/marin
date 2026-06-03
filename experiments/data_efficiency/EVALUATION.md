# Evaluation Reference: Tasks, Taxonomy, and Model Results

## 1. Models tracked

| Label | HF repo (or local path) | Params | Total trained tokens | Unique tokens (token mix proportions × epoch counts) | Notes |
|---|---|---|---|---|---|
| **1.4B base (x16)** | `1_4b_wd1_6_x16_nocrossblock_hf` (`peach-thunder-100` / `6xx0hu3l`) | 1.4 B | 3.36 B | 209 M DCLM × 16 epochs (single source) | wd=1.6, LR=1e-3 cosine, block_cross_doc=False, batch=64 × seq=4096 × 12,800 steps |
| **1.4B code25 v2 (matched)** | `1_4b_25code_alg_v2_hf` (`sage-wildflower-106` / `joqfahkl`) | 1.4 B | 3.36 B | 150 M DCLM × 16 epochs **+** 50 M opc_algorithmic × 16 epochs  →  **200 M unique total (matched to baseline 209 M)** | Same hyperparams as baseline. **This IS the matched-compute comparison**: same total trained tokens, same total unique tokens, only the data mix differs (75% NL vs 100% NL). Previous v1 (`eager-grass-104` / `p2n84bo3`) used the full 943 M opc slice at ~1 epoch which made unique-token counts unequal — see June 1 retraction in EXPERIMENT_LOG. |
| **A5 1ep DCLM @ s14672** | `1ep_dclm_step14672_hf` (run `1ep-dclm-A5`, in-flight) | 1.4 B | 15.4 B (target 30.77 B) | 7 × DCLM shards (~34.85 B unique) sampled uniformly, ~0.44 epoch per shard at this checkpoint | wd=0.1, LR=3e-4 cosine, 4-node DP, batch=256 × seq=4096 × 14,672 / 29,344 steps. Mid-training snapshot. |
| **B4 1ep code25 @ s14672** | `1ep_code25_step14672_hf` (run `1ep-code25-B4`, in-flight) | 1.4 B | 15.4 B (target 30.77 B) | 75% DCLM (23.08 B at target) + 25% code: 5.4 B aryabumi_synth + 1.35 B aryabumi_web + 0.94 B opc, each at ~1 epoch of its source at final step | Same hyperparams as A5. **Matched-compute** vs A5: same total trained tokens, same hyperparams, only data differs. |
| phi-1 | `microsoft/phi-1` | 1.3 B | ~50 B | ~7 B unique (6 B filtered Stack + ~1 B GPT-3.5 synth Python) × ~8 epochs | Code-only |
| phi-1.5 | `microsoft/phi-1_5` | 1.3 B | ~150 B | ~30 B unique (phi-1 mix + ~20 B synthetic NL textbooks) × ~5 epochs | Larger synth-textbook training |

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

**openbookqa_fact** — custom variant we added in `experiments/data_efficiency/openbookqa_fact.yaml` that uses the `additional` config of `allenai/openbookqa` and prepends the dataset's `fact1` field to the question stem. This is the open-book MC eval, and replaces the closed-book `openbookqa` default that lm-eval ships with.

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

All numbers from our `lm-eval-harness` pipeline (lm_eval 0.4.11). Rows = tasks (header format `task[nshot]`). Columns = models. Accuracy metrics use `acc_norm` where reported in §2; `acc` otherwise. PPL is `bits_per_byte` (paloma) or nats (`dclm_200m_val`), lower=better. Bolded = best in row. `—` = not run.

**The 1.4B code25 column uses the matched-token v2 run (`joqfahkl`), NOT the earlier v1 (`p2n84bo3`) where the full 943M opc slice was included and made unique-token counts unequal — see §1 + EXPERIMENT_LOG June 1 retraction.**

**A5 1ep / B4 1ep are the in-flight 1-epoch experiment at step-14672 (~50% trained). Final-step numbers will replace these when training completes.**

| Task | base (x16) | code25 v2 (x16) | A5 1ep s14672 | B4 1ep s14672 | phi-1 | phi-1.5 |
|---|---:|---:|---:|---:|---:|---:|
| **Open-book** | | | | | | |
| sciq[0] | 0.652 | 0.590 | **0.816** | 0.799 | 0.707 | **0.933** |
| boolq[0] | 0.502 | 0.567 | 0.577 | 0.569 | 0.451 | **0.746** |
| piqa[0] | 0.634 | 0.606 | 0.691 | 0.688 | 0.562 | **0.766** |
| openbookqa_fact[0] | 0.336 | 0.312 | 0.418 | 0.410 | 0.316 | **0.530** |
| **Closed-book NL** | | | | | | |
| arc_easy[25] | 0.401 | 0.388 | 0.590 | 0.564 | 0.378 | **0.805** |
| arc_challenge[25] | 0.242 | 0.241 | 0.297 | 0.282 | 0.232 | **0.532** |
| hellaswag[10] | 0.348 | 0.321 | 0.458 | 0.427 | 0.301 | **0.635** |
| winogrande[5] | 0.504 | 0.500 | 0.530 | 0.508 | 0.498 | **0.710** |
| mmlu[5] | 0.252 | 0.256 | 0.243 † | 0.252 † | 0.248 | **0.422** |
| commonsense_qa[0] | 0.192 | 0.212 | 0.212 | 0.200 | 0.175 | **0.507** |
| social_iqa[0] | 0.366 | 0.362 | 0.408 | 0.394 | 0.364 | **0.523** |
| logiqa[0] | 0.218 | 0.210 | 0.235 | 0.212 | 0.214 | **0.240** |
| lambada_openai[0] | 0.238 | 0.197 | pending § | 0.448 | 0.106 | **0.527** |
| copa[0] | 0.620 | 0.620 | pending § | 0.700 | 0.530 | **0.800** |
| wsc[0] | 0.365 | 0.365 | pending § | 0.452 | 0.442 | **0.606** |
| agieval_lsat_ar[0] | 0.226 | 0.252 | pending § | 0.217 | 0.213 | 0.183 |
| gpqa_diamond[0] | 0.268 | **0.328** | 0.258 | 0.232 | 0.197 | 0.232 |
| bbh[3] (limit=0.1) §§ | pending §§ | pending §§ | 0.127 | pending §§ | pending §§ | **0.288** |
| mmlu_pro[5] (limit=0.1) §§ | 0.050 | 0.047 | 0.098 | 0.055 | pending §§ | pending §§ |
| **Math** | | | | | | |
| gsm8k[5] | 0.000 | 0.000 | 0.000 | 0.014 | 0.012 | **0.305** |
| gsm8k_cot[8] | 0.022 | 0.005 | 0.014 | 0.014 | 0.021 | **0.299** |
| minerva_math[4] | 0.0002 | 0.000 | 0.001 | 0.006 | 0.012 | **0.029** |
| **Code** | | | | | | |
| humaneval[0] (lm-eval) | 0.000 | 0.012 | 0.006 | 0.043 | 0.494 | 0.342 |
| humaneval[0] (bigcode) ‡‡ | 0.000 | 0.000 | 0.000 | 0.000 | **0.543** | 0.342 |
| mbpp[3] | 0.000 | 0.000 | 0.000 | 0.068 | **0.416** | 0.342 |
| **Perplexity (lower=better)** | | | | | | |
| dclm_200m_val (nats) | 4.070 | 4.596 | **2.996** | 3.058 | — ‡ | — ‡ |
| paloma_macro (bpb) | 1.631 | 1.824 | 1.122 ¶ | **1.097 ¶** | 1.738 | 1.174 |

**†** = mmlu A5/B4 from `--limit 0.1` single-process run (~1.4k questions; SE ~1.2 pp). Sufficient for A vs B comparison; both within noise of random floor at this scale.

**‡** = dclm_200m_val is logged by training (Levanter in-training eval) on our runs only. phi-1/phi-1.5 are external models we never re-ran in-training eval against; their values could be computed post-hoc via bits-per-byte on raw text (tokenizer-independent) but we haven't.

**‡‡** = bigcode-evaluation-harness (the canonical code-gen runner used by the phi paper). Confirms phi-1 ≈ 54% (paper 50.6%); our 4 small models score 0 (lm-eval-harness gave partial credit that bigcode correctly rejects — see updated caveat below). MBPP via bigcode is broken upstream (`'MBPP' object has no attribute 'dataset'`), so MBPP numbers stay on lm-eval.

**§** = A5s14672 0-shot suite hit the multi-task `torch.distributed.gather_object` issue. Single-process re-run was attempted but is too slow to fill in this gap before final-step evals; will be picked up at the final-checkpoint sweep.

**§§** = bbh / mmlu_pro hit the same multi-task gather issue for several models (succeeded for some by luck). Single-process re-runs were impractically slow (~1h/task, 12 tasks remaining). Will be retried multi-GPU on subtask-by-subtask basis when the final checkpoint sweep runs.

**¶** = A5/B4 paloma_macro from Levanter in-training eval (see Table B in the per-subset details below), NOT from lm-eval-harness like the other columns (see Table A). Methodologies disagree by ~+0.05 nats on average and ~+0.55 nats on twitterAAE, so direct numerical comparison to other columns has that calibration noise. The cross-table interpretation in the per-subset section gives a rough Table-A-equivalent of 1.05–1.08 — both still lower than phi-1.5's 1.174.

<details>
<summary><b>Paloma per-subset bpb (16 subsets) — expand</b></summary>

**Table A — lm-eval-harness paloma_* values** (base/code25v2/phi-1/phi-1.5 only; A5/B4 lm-eval paloma blocked on gated `EleutherAI/paloma` dataset).

| Subset | base (x16) | code25 v2 (x16) | phi-1 | phi-1.5 |
|---|---:|---:|---:|---:|
| 4chan_meta_sep | 1.561 | 1.711 | 1.665 | **1.199** |
| c4_100_domains | 1.322 | 1.488 | 1.615 | **0.948** |
| c4_en | 1.387 | 1.558 | 1.621 | **0.985** |
| dolma-v1_5 | 1.433 | 1.599 | 1.479 | **0.957** |
| dolma_100_programing_languages | 1.681 | 1.828 | 0.834 | **0.679** |
| dolma_100_subreddits | 1.492 | 1.669 | 1.732 | **1.159** |
| falcon-refinedweb | 1.448 | 1.626 | 1.679 | **1.025** |
| gab | 2.475 | 2.754 | 2.645 | **1.918** |
| m2d2_s2orc_unsplit | 1.389 | 1.593 | 1.273 | **0.946** |
| m2d2_wikipedia_unsplit | 1.319 | 1.482 | 1.629 | **0.976** |
| manosphere_meta_sep | 1.566 | 1.749 | 1.758 | **1.180** |
| mc4 | 1.546 | 1.734 | 1.604 | **1.031** |
| ptb | 1.565 | 1.781 | 1.644 | **1.057** |
| redpajama | 1.491 | 1.658 | 1.374 | **0.928** |
| twitterAAE_HELM_fixed | 3.077 | 3.443 | 3.634 | **2.826** |
| wikitext_103 | 1.336 | 1.512 | 1.620 | **0.970** |
| **macro** | **1.631** | **1.824** | **1.738** | **1.174** |

**Table B — Levanter in-training-eval paloma_* values** (base/code25v2/A5/B4 only; phi-1/phi-1.5 are external HF checkpoints and were never put through Levanter's in-training eval pipeline).

| Subset | base (x16) | code25 v2 (x16) | A5 1ep final | B4 1ep final |
|---|---:|---:|---:|---:|
| 4chan | 1.5622 | 1.7079 | 1.0612 | **1.0715** |
| c4_100_domains | 1.3229 | 1.4912 | **0.8939** | 0.9095 |
| c4_en | 1.3966 | 1.5635 | **0.9479** | 0.9603 |
| dolma-v1_5 | 1.4595 | 1.6240 | 0.9458 | **0.9311** |
| dolma_100_programing_languages | 1.7378 | 1.8932 | 0.8816 | **0.7087** |
| dolma_100_subreddits | 1.4977 | 1.6712 | **1.0773** | 1.0841 |
| falcon-refinedweb | 1.4855 | 1.6549 | **1.0059** | 1.0193 |
| gab | 2.6040 | 2.8929 | 1.7705 | **1.7103** |
| m2d2_s2orc_unsplit | 1.3835 | 1.5874 | **0.9175** | 0.9225 |
| m2d2_wikipedia_unsplit | 1.3153 | 1.4756 | **0.8747** | 0.8931 |
| manosphere_meta_sep | 1.5570 | 1.7404 | **1.1023** | 1.1122 |
| mc4 | 1.5140 | 1.6976 | **0.9607** | 0.9673 |
| ptb | 1.6080 | 1.8303 | **1.0140** | 1.0490 |
| redpajama | 1.5471 | 1.7158 | 0.9461 | **0.9052** |
| twitterAAE_HELM_fixed | 3.6221 | 4.0445 | 2.6658 | **2.4174** |
| wikitext_103 | 1.3338 | 1.5076 | **0.8785** | 0.8852 |
| **macro** | **1.6842** | **1.8465** | **1.1215** | **1.0967** |

**Caveat on Table A vs B comparability.** The two pipelines compute bpb differently (document boundaries, eos handling, twitterAAE in particular differs ~0.5 nats). For base at step-12799, Table A says 1.631 while Table B says 1.684 — same model, same checkpoint, different methodology, ~+0.05 nat offset. Per-subset offsets vary (twitterAAE is the worst: +0.55 nats). So **don't compare a Table A value to a Table B value at face**; only compare WITHIN a table.

**Cross-table reading (rough only):** A5/B4 final macro values (Table B) of 1.12 / 1.10 are ~0.55 nats LOWER than base Table B (1.68), and ~0.55 lower than code25v2 (1.85). Translating the ~0.55-nat gap into Table A-equivalent terms (base Table A = 1.63), A5/B4 final macros would be ~1.05–1.08 — i.e., still significantly LOWER than phi-1.5's lm-eval value of 1.174. This translation is rough and only works for macro, not per-subset.

</details>

### Headlines

**Phi-1.5** wins on every NL subset (synthetic-textbook-heavy training pays off at 1.3B).

**Phi-1 is WORSE than our 1.4B text-only baseline on Paloma macro** (1.738 vs 1.631) because phi-1 was trained almost exclusively on code. Phi-1 only beats our 1.4B baseline on `programming_languages` and `m2d2_s2orc` (the code/academic subsets).

**Matched-token code-mix (code25 v2) HURTS NL at our scale:**
- paloma_macro: 1.824 vs baseline 1.631 → **+0.19 nats WORSE**
- dclm_200m_val (in-domain): 4.596 vs baseline 4.070 → **+0.53 nats WORSE**
- arc_easy, sciq, hellaswag, openbookqa_fact, piqa, social_iqa, logiqa: code25 v2 loses by 1-7pp on each
- Only boolq wins for code25 v2 (+6pp) and a couple of tied tasks

This is the **controlled** comparison (same total trained tokens, same unique tokens, only data mix differs). It confirms: at 1.4B / 3.3B-token / 16-epoch repetition, 25% code mix doesn't help NL; it actively hurts. The earlier "v1 wins" interpretation from May 26 was retracted June 1 — v1 had 5× more unique tokens, not a fair comparison.

**1-epoch experiment at step-14672 (~50% trained, A5 vs B4):** **A5 (DCLM-only) wins 10 NL benchmarks by 1-3pp each, B4 (code-mix) wins 2 code benchmarks decisively (humaneval +3.7 pp, mbpp +6.8 pp), 3 tied at floor**. In-domain val: A5 wins by 0.06 nats. Same pattern as code25 v2 above: code-mix trades NL ability for code-gen ability under matched compute, even at 1-epoch (no repetition). Full-checkpoint paloma + final downstream pending training completion. See `1ep_experiment_plan.md` for methodology.

**Caveat on code-gen numbers.** We now run HumanEval via both `lm-eval-harness` and `bigcode-evaluation-harness`. Comparison:

| Model | HE (lm-eval) | HE (bigcode) | HE (paper) |
|---|---:|---:|---:|
| phi-1 | 0.494 | **0.543** | 0.506 |
| phi-1.5 | 0.342 | 0.342 | 0.414 |
| our 4 × 1.4B | 0.000–0.043 | **0.000** | n/a |

bigcode matches phi-1 paper closely (54.3% vs 50.6%); lm-eval undercounts slightly. For our small 1.4B models, lm-eval's loose extraction gives them spurious 0-4% credit on partial generations that bigcode (which actually runs the test suite) correctly rejects as failures. **Bottom line: at 1.4B / our compute, our models really score zero on HumanEval — they can't generate executable Python.** MBPP via bigcode is broken (upstream bug); MBPP numbers stay on lm-eval-harness with its known ~14 pp gap to published numbers — internally consistent but not absolute-comparable.

---

## 4. Counterfactual probes — arithmetic decomposition (Phase 1)

Probe design lives in [`counterfactual_probes.md`](counterfactual_probes.md); implementation in [`probes_arithmetic.py`](probes_arithmetic.py). 500 problems across 5 levels (100 per level), 0-shot greedy generation with `max_new_tokens=4`, scored by parsing the first integer in the generation.

| Level | Format | Random-guess baseline | Description |
|---|---|---:|---|
| A1 | `a + b = ` | ~5% | single-digit addition (a, b ∈ [0, 9]) |
| A2 | `a + b = ` | ~1% | two-digit addition, no carry (a + b ≤ 99) |
| A3 | `a + b = ` | ~1% | two-digit addition with carry |
| A4 | `a * b = ` | ~3% | single-digit multiplication (a, b ∈ [2, 9]) |
| A5 | `a - b = ` | ~1% | two-digit subtraction |

### Results

| Model | A1 | A2 | A3 | A4 | A5 |
|---|---:|---:|---:|---:|---:|
| **1.4B base (x16)** | 0.13 | 0.01 | 0.00 | 0.02 | 0.01 |
| **1.4B code25 v2 (x16)** | 0.09 | 0.01 | 0.00 | 0.01 | 0.01 |
| **A5 1ep final (DCLM-only)** | 0.35 | 0.01 | 0.01 | 0.13 | 0.00 |
| **B4 1ep final (DCLM 75% + code 25%)** | **0.83** | **0.07** | 0.01 | **0.84** | **0.07** |
| phi-1 † | 0.14 | 0.00 | 0.00 | 0.11 | 0.03 |
| phi-1.5 † | 0.01 | 0.07 | 0.01 | 0.07 | 0.02 |

**†** = NOT directly comparable on this probe format. phi-1 and phi-1.5 receive the bare `a + b = ` prompt as Python-indent (phi-1) or word-problem (phi-1.5) context and start writing a chain-of-thought response ("Simplifying the equation...", "Answer:") rather than the bare integer. `max_new_tokens=4` cuts before any answer is produced. Re-ran with v2 (`probes_arithmetic_v2.py`: max_new_tokens=64, last-int, truncate at first `\n\n`) and phi-1.5 still scored 0 across the board because:
1. Phi-1.5's generations begin with `\n\nSimplifying...`, so the first-`\n\n` truncation grabs everything BEFORE the model writes anything — empty string, no integer.
2. Even with a more lenient parse, inspection of phi-1.5 outputs shows it generates word-problem-shaped responses (`"the width of the garden is 10 meters..."`) where the answer appears but is preceded by domain words ("garden"). Phi-1.5 was trained on synthetic word problems, not bare `1 + 2 = ` notation, so the format mismatch dominates the result.

Concretely: for `0 + 5 = ` phi-1.5 writes `\n\nSimplifying the equation, we get:\n\nx = 10\n\nTherefore...` — predicts 10 regardless of the input numbers. The model has a strong prior toward "x = 10" because that's the canonical "garden width" answer in its training distribution.

**This is itself a finding about phi-1.5:** the synthetic-NL-textbook training teaches a very specific surface format (word problems with garden/area framing) and the model's arithmetic capability is only accessible through that format. The bare `a + b = ` probe is biased toward models that learned explicit arithmetic notation — which our 4 1.4B columns are directly testing.

**Only the 4 1.4B columns are directly comparable. Phi-1/phi-1.5 results reflect format mismatch, not arithmetic capability.** For phi-1.5 specifically, GSM8K-style word-problem evaluation is the right format (that's what `gsm8k=0.305` measures).

### Headlines

**B4 (1-epoch, 25% code mix) has MASSIVELY more arithmetic than A5 (1-epoch, DCLM-only).** Single-digit addition jumps from A5's 35% to B4's 83%; single-digit multiplication from 13% to 84%. Two-digit no-carry addition jumps from 1% to 7% — small but 7× the random baseline. **Code data teaches arithmetic at our scale.**

**Both A5 and B4 floor on GSM8K (0.000 / 0.014) despite the gap above.** The probe decomposes the GSM8K floor: A5 lacks single-digit arithmetic, B4 has single-digit but lacks two-digit composition AND word-problem parsing. They share the GSM8K-relevant deficit (multi-digit + word problems) even though they differ on basic capability.

**The 1-epoch DCLM model (A5) does pick up SOME single-digit addition.** 35% vs the heavily-overfit baseline's 13% (and code25 v2's 9%, which is at floor). 30B tokens of pure web text gives the model weak single-digit addition; 7.5B tokens of mixed-in code (aryabumi synth + opc algorithmic) gives it strong single-digit add+mult.

**Phi-1's code training does NOT teach arithmetic the way B4's does.** Phi-1's 14% A1 / 11% A4 is closer to our 1.4B baseline than to B4. Phi-1 was trained on filtered Stack code + GPT-3.5 Python textbooks — the Python is mostly syntactically arithmetic-free (operators yes, "= integer" rarely). aryabumi_synth + opc_algorithmic are textbook/algorithmic-Python style where explicit arithmetic IS prevalent. **The specific code distribution matters, not just "code".** (Caveat: phi-1's prompt-format mismatch may inflate this gap; need Phase 2 reformat to confirm.)

### What this clarifies about H1

The matched-token study said "code mix HURTS NL by 0.2 nats paloma" → the question was *why* the mix didn't help any downstream metric. The probe answers: **code DOES teach a foundational capability** (single-digit arithmetic, jumps 35% → 83%), but this capability **isn't enough to surface on GSM8K** at 1.4B / 30B tokens — multi-digit composition and word-problem parsing remain at floor regardless of data. So the H1 question splits into "what teaches the foundational capability" (code-textbook-style data teaches arithmetic) and "what teaches the composition" (still unanswered at our scale).

---

## Updating this doc

When a new model is trained or a new eval is added, update §1 (models) and §3 (results) with the new row/column. Add a brief follow-up entry in `EXPERIMENT_LOG.md` pointing here. Chronological narrative stays in `EXPERIMENT_LOG.md`; canonical reference stays here.
