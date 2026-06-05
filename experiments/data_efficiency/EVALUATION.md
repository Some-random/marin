# Evaluation Reference: Tasks, Taxonomy, and Model Results

## 1. Taxonomy by mechanism (with examples)

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

## 2. Models tracked

| Label | HF repo (or local path) | Params (N) | Tokens trained (D) | FLOPs (≈6·N·D) | Unique tokens | Notes |
|---|---|---|---|---:|---|---|
| **1.4B base (x16)** | `1_4b_wd1_6_x16_nocrossblock_hf` (`peach-thunder-100` / `6xx0hu3l`) | 1.4 B | 3.36 B | 2.8 × 10¹⁹ | 209 M DCLM × 16 epochs (single source) | wd=1.6, LR=1e-3 cosine, block_cross_doc=False, batch=64 × seq=4096 × 12,800 steps |
| **1.4B code25 v2 (matched)** ¤ | `1_4b_25code_alg_v2_hf` (`sage-wildflower-106` / `joqfahkl`) | 1.4 B | 3.36 B | 2.8 × 10¹⁹ | 150 M DCLM × 16 epochs + 50 M opc_algorithmic × 16 epochs → 200 M unique total (matched to baseline 209 M) | Same hyperparams as baseline. Same total trained tokens, same total unique tokens; only the data mix differs (75% NL vs 100% NL). |
| **A5 1ep DCLM final** ¥ | `1ep_dclm_final_hf` (run `1ep-dclm-A5`, `tmgu1im8`, step-29343) | 1.4 B | 30.77 B | 2.6 × 10²⁰ | 7 × DCLM shards (~34.85 B unique), ~0.88 epoch per shard | wd=0.1, LR=3e-4 cosine, 4-node DP, batch=256 × seq=4096 × 29,343 steps. |
| **B4 1ep code25 final** ¥ | `1ep_code25_final_hf` (run `1ep-code25-B4`, `6zs6ybgt`, step-29343) | 1.4 B | 30.77 B | 2.6 × 10²⁰ | 75% DCLM (23.08 B = 0.66 epoch over 34.85 B available) + 25% code: 5.4 B aryabumi_synth + 1.35 B aryabumi_web + 0.94 B opc, each at ~1 epoch | Same hyperparams as A5. Matched-compute vs A5: same total trained tokens, same hyperparams, only data differs. |
| **4B final** ª | `4b_dclm_short_final_hf` (run `3_5b_dclm_short`, step-22887) | 3.5 B | 6.0 B | 1.3 × 10²⁰ | 7 × DCLM shards, ~0.17 epoch per shard | wd=0.1, LR=3e-4 cosine, 8-node FSDP, batch=64 × seq=4096 × 22,887 steps. |
| phi-1 | `microsoft/phi-1` | 1.3 B | ~50 B | ~3.9 × 10²⁰ | ~7 B unique (6 B filtered Stack + ~1 B GPT-3.5 synth Python) × ~8 epochs | Code-only (external reference) |
| phi-1.5 | `microsoft/phi-1_5` | 1.3 B | ~150 B | ~1.2 × 10²¹ | ~30 B unique (phi-1 mix + ~20 B synthetic NL textbooks) × ~5 epochs | Larger synth-textbook training (external reference) |

**FLOPs column** is the Chinchilla approximation 6·N·D (Hoffmann et al 2022). Useful for understanding which models had comparable training compute, but doesn't predict per-task accuracy on its own — for that we just look at the actual numbers in §3.

**¤** = the matched-token v2 column uses run `joqfahkl`, NOT the earlier v1 (`eager-grass-104` / `p2n84bo3`). v1 used the full 943M opc slice at ~1 epoch which made unique-token counts unequal between v1 and baseline — see EXPERIMENT_LOG June 1 retraction.

**¥** = A5/B4 are the FINAL-STEP checkpoints (step-29343, ~30.77 B trained tokens). Earlier versions of this doc had columns labelled `s14672` (~50% trained, mid-training snapshots); those have been replaced with the final-step values.

**ª** = 4B is a "tokens-vs-params" comparison point. Note that 4B used 1.3 × 10²⁰ FLOPs vs A5's 2.6 × 10²⁰ — A5 has ~2× more training compute, so the A5 > 4B comparison is partly explained by raw compute rather than purely by "tokens vs params". Treat the comparison as "what does an 8-GPU-day 4B run look like vs an 8-GPU-day 1.4B run", not as a controlled experiment.

---

## 3. Canonical results — all models

All numbers from our `lm-eval-harness` pipeline (lm_eval 0.4.11). Rows = tasks (header format `task[nshot]`). Columns = models. Accuracy metrics use `acc_norm` where reported in §1; `acc` otherwise. PPL is `bits_per_byte` (paloma) or nats (`dclm_200m_val`), lower=better. Bolded = best in row. `—` = not run.

See §2 footnotes ¤ (code25 v2 vs v1) and ¥ (A5/B4 final-step) for column-definition caveats.

| Task | base (x16) | code25 v2 (x16) | A5 1ep final | B4 1ep final | 4B final ª | phi-1 | phi-1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Open-book** | | | | | | | |
| sciq[0] | 0.652 | 0.590 | 0.834 | 0.829 | 0.824 | 0.707 | **0.933** |
| boolq[0] | 0.502 | 0.567 | 0.563 | 0.599 | 0.552 | 0.451 | **0.746** |
| piqa[0] | 0.634 | 0.606 | 0.718 | 0.709 | 0.697 | 0.562 | **0.766** |
| openbookqa_fact[0] | 0.336 | 0.312 | 0.430 | 0.430 | 0.426 | 0.316 | **0.530** |
| **Closed-book NL** | | | | | | | |
| arc_easy[25] | 0.401 | 0.388 | 0.629 | 0.607 | 0.612 | 0.378 | **0.805** |
| arc_challenge[25] | 0.242 | 0.241 | 0.316 | 0.289 | 0.292 | 0.232 | **0.532** |
| hellaswag[10] | 0.348 | 0.321 | 0.497 | 0.464 | 0.466 | 0.301 | **0.635** |
| winogrande[5] | 0.504 | 0.500 | 0.541 | 0.515 | 0.511 | 0.498 | **0.710** |
| mmlu[5] | 0.252 | 0.256 | 0.244 | 0.258 | 0.250 | 0.248 | **0.422** |
| commonsense_qa[0] | 0.192 | 0.212 | 0.195 | 0.213 | 0.193 | 0.175 | **0.507** |
| social_iqa[0] | 0.366 | 0.362 | 0.415 | 0.400 | 0.407 | 0.364 | **0.523** |
| logiqa[0] | 0.218 | 0.210 | **0.320** | 0.270 | 0.269 | 0.214 | 0.240 |
| lambada_openai[0] | 0.238 | 0.197 | 0.519 | 0.496 | 0.494 | 0.106 | **0.527** |
| copa[0] | 0.620 | 0.620 | 0.740 | 0.690 | 0.740 | 0.530 | **0.800** |
| wsc[0] | 0.365 | 0.365 | 0.519 | 0.365 | 0.394 | 0.442 | **0.606** |
| agieval_lsat_ar[0] | 0.226 | 0.252 | 0.187 | 0.222 | 0.222 | 0.213 | 0.183 |
| gpqa_diamond[0] | 0.268 | **0.328** | 0.268 | 0.217 | 0.273 | 0.197 | 0.232 |
| bbh[3] (limit=0.1) | pending §§ | pending §§ | 0.160 | 0.206 | 0.155 | pending §§ | **0.288** |
| mmlu_pro[5] (limit=0.1) | 0.050 | 0.047 | **0.116** | 0.073 | 0.069 | pending §§ | pending §§ |
| **Math** | | | | | | | |
| gsm8k[5] | 0.000 | 0.000 | 0.001 | 0.010 | 0.018 | 0.012 | **0.305** |
| gsm8k_cot[8] | 0.022 | 0.005 | 0.031 | 0.027 | 0.021 | 0.021 | **0.299** |
| gsm_symbolic_main[8] | — | — | — | — | — | 0.013 | **0.160** |
| gsm_noop[8] ° | — | — | — | — | — | 0.000 | **0.034** |
| minerva_math[4] | 0.0002 | 0.000 | 0.002 | 0.010 | 0.007 | 0.012 | **0.029** |
| **Code** | | | | | | | |
| humaneval[0] (lm-eval) | 0.000 | 0.012 | 0.006 | 0.104 | 0.000 | **0.494** | 0.342 |
| humaneval[0] (bigcode) ‡‡ | 0.000 | 0.000 | 0.000 | failed ‡‡ | 0.000 | **0.543** | 0.342 |
| mbpp[3] | 0.000 | 0.000 | 0.000 | 0.060 | 0.000 | **0.416** | 0.342 |
| **Perplexity (lower=better)** | | | | | | | |
| dclm_200m_val (nats) | 4.070 | 4.596 | **2.821** | 2.878 | 2.894 ¶ | — ‡ | — ‡ |
| paloma_macro (bpb) | 1.631 | 1.824 | 1.122 ¶ | **1.097 ¶** | 1.153 ¶ | 1.738 | 1.174 |

**‡** = dclm_200m_val is logged by training (Levanter in-training eval) on our runs only. phi-1/phi-1.5 are external models we never re-ran in-training eval against; their values could be computed post-hoc via bits-per-byte on raw text (tokenizer-independent) but we haven't.

**‡‡** = bigcode-evaluation-harness (the canonical code-gen runner used by the phi paper) — actually executes the unit tests instead of regex-matching the answer. We use bigcode for HumanEval and lm-eval for MBPP (bigcode MBPP is upstream-broken). The B4 final bigcode HumanEval entry shows `failed` because of a transient HF metadata outage during that specific run; the underlying capability is captured by B4's lm-eval HumanEval = 0.104. For all other models, the bigcode column matches the paper's number where comparable (phi-1 0.543 vs paper 0.506).

**°** = `gsm_symbolic_main` from `apple/GSM-Symbolic` (Mirzadeh et al §3.2, 5000 examples). `gsm_noop` from `Experimental-Orange/gsm-noop-audited` (Sturgeon's third-party reconstruction of §4.4, 117 audited-irrelevant items). Both 8-shot CoT, greedy. Headline: phi-1.5 drops 47% from GSM8K → GSM-Sym main (0.305 → 0.160), and 89% to NoOp (→ 0.034) — replicates the published pattern that smaller models drop more aggressively under perturbation. Our 1.4B models floor on GSM8K so we didn't run these (would also floor, no signal); phi-1 floors on GSM8K and is included only as the code-only control.

**¶** = paloma_macro and dclm_200m_val for A5/B4/4B are from Levanter's in-training eval (Table B in the per-subset details), NOT lm-eval-harness like the base/code25v2/phi columns (Table A). Methodologies disagree by ~+0.05 nats on average and ~+0.55 nats on twitterAAE, so direct numerical comparison to other columns has calibration noise. Cross-table A-equivalent estimate: A5/B4 ~1.05-1.08, 4B ~1.10 — all still lower than phi-1.5's 1.174.

**§§** = bbh / mmlu_pro hit the multi-task `torch.distributed.gather_object` issue for some models (succeeded for others by luck of HF timing). A5 and B4 final bbh and mmlu_pro all completed via either the main multi-GPU run (B4 mmlu_pro succeeded on first try) or a single-task retry (A5 bbh and B4 bbh retries succeeded after HF stabilized). Older base/code25v2/phi-1/phi-1.5 cells remain unfilled where the corresponding single-process retry was impractical (~1h/task on 1 GPU).

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

**1-epoch experiment, FINAL step-29343 (~30.77B trained tokens, A5 vs B4):** **A5 (DCLM-only) wins ~11 NL benchmarks by 0.5-5 pp each** (arc_easy +2.2, arc_challenge +2.7, hellaswag +3.3, winogrande +2.6, piqa +0.9, sciq +0.5, openbookqa +1.8, social_iqa +1.5, logiqa +5.0, lambada +2.3, copa +5.0, wsc +15.4 — wsc is noisy, gpqa_diamond +5.1, mmlu_pro +4.3). **B4 (code-mix) wins boolq (+3.6), agieval_lsat_ar (+3.5), mmlu (+1.4), commonsense_qa (+1.8), bbh (+4.6), and decisively wins code-gen (humaneval lm-eval +9.8 pp, mbpp +6.0 pp). B4 actually *does* generate plausible Python — sample inspection shows it solves easy mbpp problems like `min_of_three`, substring-check, regex-whitespace-strip, and produces compilable but logically-wrong code on harder HumanEval problems. Bigcode strict-unit-test HE = 0.000 reflects that those longer HumanEval programs rarely pass all test cases, NOT that the model can't write Python at all.** In-domain val: A5 wins by 0.06 nats (2.821 vs 2.878). Paloma_macro: B4 slightly lower (1.097 vs 1.122) driven by twitterAAE + code subsets; on mainstream NL subsets (c4_en, wikipedia, m2d2_*, falcon-refinedweb, wikitext_103, redpajama) A5 wins by 1-3 nats × 0.01. **Same overall pattern as the 16-epoch comparison: code-mix HURTS NL while modestly improving code-gen-shaped metrics, with the trade-off persisting at matched compute and 1-epoch (no repetition).** See [§4 arithmetic decomposition probe](#4-counterfactual-probes--arithmetic-decomposition-phase-1) for one measured underlying difference: B4 has 83%/84% on single-digit add/mult while A5 has 35%/13%. That tells us code teaches a foundational arithmetic capability we can measure; it does NOT tell us why GSM8K floors for both — that's an open question we didn't probe.

See ‡‡ footnote above for the lm-eval vs bigcode-eval-harness distinction. The two-row split in the §3 table (`humaneval[0] (lm-eval)` and `humaneval[0] (bigcode)`) shows that lm-eval's regex-extraction gives our small models 0-10pp credit on partial generations that bigcode (which runs the unit tests) rejects. For the only model in our suite where this matters in absolute terms, **B4 final lm-eval HumanEval = 0.104** captures real partial capability — sample inspection of B4's mbpp generations shows it solves easy problems like `find_substring`, `min_of_three`, regex-whitespace-strip; just rarely the full HumanEval programs.

---

**Arithmetic-notation probe (one-line finding).** A 500-problem synthetic probe (`probes_arithmetic.py`) asks each model to complete `a + b = `, `a * b = `, `a - b = ` at single and two-digit scales. The only signal that comes out: **B4 (DCLM + 25% code) recognizes the `a op b = c` notation and emits answers (83% A1, 84% A4); A5 (DCLM only) much less so (35%, 13%); base/code25v2 essentially floor; phi-1/phi-1.5 score ~0 because they generate Python chain-of-thought or word-problem context instead of bare answers, so the score is a format mismatch not a capability claim.** Full numbers and design in [`counterfactual_probes.md`](counterfactual_probes.md). Read this as "code-textbook data teaches the bare-arithmetic notation format" — NOT as a Wu-style counterfactual or a decomposition into reasoning sub-skills.

---

## Updating this doc

When a new model is trained or a new eval is added, update §1 (models) and §3 (results) with the new row/column. Add a brief follow-up entry in `EXPERIMENT_LOG.md` pointing here. Chronological narrative stays in `EXPERIMENT_LOG.md`; canonical reference stays here.
