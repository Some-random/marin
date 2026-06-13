# Evaluation Reference: Tasks, Taxonomy, and Model Results

<details>
<summary><h2>1. Taxonomy by mechanism (with examples)</h2></summary>


Two-way split for QA: **open-book** (the answer is in the prompt; model attends and extracts) vs. **closed-book** (no passage; model uses weights). Plus four task families that don't fit the QA frame: math (with perturbation-robust variants), code generation, multi-domain aggregates, and continuous PPL.

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

**copa** — Choice of Plausible Alternatives (super_glue/copa). 2-way MC: given a premise, pick the more plausible **cause** or **effect** of it. The `question` field is the literal string "cause" or "effect" and tells the model which direction to reason.

> - **premise**: "The man turned on the faucet."
> - **question**: effect
> - **choice1**: "The toilet filled with water."
> - **choice2**: "Water flowed from the spout."
> - **label**: 1 (choice2)

**wsc** — Winograd Schema Challenge (super_glue config `wsc.fixed`). Binary coreference: given a sentence with two flagged spans, decide whether the second span (a pronoun) co-refers with the first span (a noun phrase). Adversarially constructed so that pure pattern-matching fails — the model has to use world knowledge.

> - **text**: "Bernard, who had not told the government official that he was less than 21 when he filed for a homestead claim, did not consider that he had done anything dishonest. Still, anyone who knew that he was 19 years old could take his claim away."
> - **span1_text**: "anyone"  • **span2_text**: "him"
> - **label**: 0 (the pronoun "him" does NOT refer to "anyone")

**storycloze_2018_local** — Story Cloze Test (Mostafazadeh et al 2016, 2018 split). 2-way MC: given the first 4 sentences of a 5-sentence story, pick the more plausible ending. Tests narrative coherence and commonsense expectations. We use a local custom YAML (`storycloze_2018_local.yaml`) because the canonical lm-eval `storycloze_2018` task requires manual dataset acceptance terms.

> - **input_sentence_1**: "Karen was assigned a roommate her first year of college."
> - **input_sentence_2**: "Her roommate asked her to go to a nearby city for a concert."
> - **input_sentence_3**: "Karen agreed happily."
> - **input_sentence_4**: "The show was absolutely exhilarating."
> - **ending_1**: "Karen became good friends with her roommate."
> - **ending_2**: "Karen hated her roommate."
> - **answer_right_ending**: 1

**cb** — CommitmentBank (super_glue config `cb`). 3-way NLI: given a premise (typically a complex sentence with an embedded clause) and a hypothesis (the embedded clause taken at face value), classify entailment / contradiction / neutral. The premise often contains a presupposition trigger that may or may not project; the task tests whether the model can identify when an embedded proposition is being asserted vs hedged. Small dataset (250 train, 56 dev).

> - **premise**: "He thought that one prick was enough. Now he wasn't so sure."
> - **hypothesis**: "One prick was enough."
> - **label**: 2 (neutral)

**quac_first_turn** — Question Answering in Context (Choi et al 2018, `allenai/quac`), first-turn-only adaptation: 1000 single-shot examples (Q0 of each dialogue). Format: background + section title + context passage, then "Q: ... A:". Model generates a free-form span. Scored by token-level F1 (and EM) against ≤ 4 annotators' gold answers via transformers' `squad_metrics`. We report F1 as the canonical "Acc."-equivalent (Aryabumi et al 2024 list QUAC under their NL Reasoning suite with Acc.; F1 is the standard QUAC metric). Custom YAML at `quac_first_turn.yaml` + utility functions at `quac_utils.py`.

> - **context** (first 200 chars): "In May 1983, she married Nikos Karvelas, a composer, with whom she collaborated in 1975 and in November she gave birth to her daughter Sofia. After their marriage, she started a close collaboration wi…"
> - **question** (Q0): "what happened in 1983?"
> - **gold answers**: ["In May 1983, she married Nikos Karvelas,", "In May 1983, she married Nikos Karvelas, a composer,"]

**lambada_openai** — last-word prediction from a narrative passage. The model sees the full passage minus its final word and is scored by whether the most-likely next token matches the gold word. Tests long-range narrative coherence and discourse continuation, not QA.

> Passage end (final word elided):
>
> > "…'Figured if you're going to be out at night getting hit by cars, you might as well have some backup.' I look at him, feeling stunned. Like this is some sort of sign. But as I stare at Harlin, his mouth curved in a confident grin, I don't care about ___"
>
> Gold word: `signs`.

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

**gsm_symbolic_main** — GSM-Symbolic main split (Mirzadeh et al 2024 §3.1, `apple/GSM-Symbolic`, 5000 examples). Re-instantiations of GSM8k templates where every named entity and integer is re-sampled while keeping the underlying arithmetic structure. Compared against gsm8k, isolates "is the model solving the problem, or pattern-matching the surface form?" 8-shot CoT, greedy decoding. Headline result from the paper: phi-1.5 drops 47% from GSM8K (0.305) → GSM-Sym main (0.160) — replicates the published pattern that smaller models drop more aggressively under perturbation. Our 1.4B models floor on regular GSM8K so they floor on this row too — included for completeness.

> - **original_question** (GSM8k #473): "Benny saw a 10-foot shark with 2 6-inch remoras attached to it. What percentage of the shark's body length is the combined length of the remoras?" — gold 10
> - **question** (symbolic instance): "Rania saw a 210-foot whale with 7 72-inch remoras attached to it. What percentage of the whale's body length is the combined length of the remoras?" — gold 20
> - Same template, name and numbers re-drawn, arithmetic structure (inches → feet conversion, then percentage) preserved.

**gsm_noop** — GSM-NoOp (Mirzadeh et al 2024 §4.4). GSM8k problems with one extra clause inserted that is grammatically plausible but mathematically irrelevant. Tests whether the model can ignore irrelevant context or gets distracted into incorporating it. Our eval uses `Experimental-Orange/gsm-noop-audited`, a 117-item third-party reconstruction (the Apple-original NoOp split was not released). 8-shot CoT, greedy. Paper headline: phi-1.5 drops 89% from GSM8K to NoOp (→ 0.034). Our 1.4B models floor (consistent with their GSM8K floor).

> - **original_question** (GSM8k #1223): "To make a call from a phone booth, you must pay ₣0.6 for each minute of your call. After 30 minutes, that price drops to ₣0.5 per minute. How much would a 78-minute call cost?" — gold 42
> - **question** (with NoOp clause): "To make a call from a phone booth, you must pay ₣0.6 for each minute of your call. After 30 minutes, that price drops to ₣0.5 per minute. **If you had placed the same call on a weekend, the initial per-minute rate would have been 15% cheaper.** How much would a 78-minute call cost on a weekday?" — gold still 42.
> - The weekend-discount clause is a hypothetical that does not apply (the question specifies a weekday call). A model that applies the 15% discount has been distracted by the no-op.

### D. Code generation

**HumanEval** — function generation from docstring; pass@1 by running unit tests. 0-shot: the function signature + docstring **is** the prompt (no demonstration examples). Reported in §3 with two rows — `humaneval[0] (lm-eval)` uses lm-eval-harness's `pass@1,create_test` (regex-match on the generated function body), `humaneval[0] (bigcode)` uses bigcode-evaluation-harness which actually **executes** the unit tests against the generated code. The bigcode number is the trustworthy one (phi-1: 0.543 bigcode vs paper's 0.506); the lm-eval column is included for comparison since it was used historically and is faster to run. We use bigcode for HumanEval but lm-eval for MBPP (bigcode-MBPP is upstream-broken).

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

### E. Aggregate / multi-domain reasoning

Tasks that aggregate many subtasks across heterogeneous domains. Reported as the unweighted mean of per-subtask scores. Useful as a single "frontier reasoning" number, but a poor diagnostic since subtask composition is fixed and individual signals are averaged out.

**bbh** — Big-Bench Hard. 27 subtasks (`bbh_boolean_expressions`, `bbh_causal_judgement`, `bbh_logical_deduction_five_objects`, …) that the original BIG-bench paper identified as the hardest splits. Each subtask is free-generation, scored by an answer-extraction regex (`exact_match,get-answer`). 3-shot in our pipeline.

> Subtask example (`bbh_boolean_expressions`):
> - **input**: "not ( True ) and ( True ) is"
> - **target**: "False"

**mmlu_pro** — harder MMLU successor. 10-way MC (vs. MMLU's 4-way) drawn from STEM exams and textbooks with longer, more reasoning-heavy questions; subtasks are organized by `category` (math, physics, chemistry, …). Scored by `exact_match,custom-extract` after letter extraction. 5-shot, 2048-context (phi-1/phi-1.5 cannot run this — `n/a (ctx)`).

> - **category**: math
> - **question**: "The symmetric group $S_n$ has $n!$ elements. … Find the characteristic of the ring 2Z."
> - **options** (10): "0" / "30" / "3" / "10" / "12" / "50" / "2" / "100" / "20" / "5"
> - **answer**: A (= "0")

**agieval_lsat_ar** — LSAT Analytical Reasoning subset of AGIEval (Zhong et al 2023). Verbal logic puzzles (scheduling, grouping, ordering) with 5 candidate solutions per question; each candidate is a fully specified assignment that must satisfy the puzzle constraints.

> - **query**: "Of the eight students—George, Helen, Irving, Kyle, Lenore, Nina, Olivia, and Robert—in a seminar, exactly six will give individual oral reports during three consecutive days—Monday, Tuesday, and Wednesday. Exactly two reports will be given each day … [further constraints]"
> - **choices** (5, each a complete schedule): "(A) Mon. morning: Helen; Mon. afternoon: Robert; Tues. morning: Olivia; …" / "(B) Mon. morning: Irving; …" / …
> - **gold**: option C

**gpqa_diamond** — graduate-level physics, chemistry, biology MC questions written by domain PhDs (Rein et al 2023). "Diamond" is the highest-quality validated subset (~198 questions). 4-way MC; expert validators score >65%, non-expert validators with web access score <35%.

> - **subdomain**: Physics (general)
> - **question**: "Two quantum states with energies E1 and E2 have lifetimes of 10⁻⁹ s and 10⁻⁸ s respectively. We want to clearly distinguish these two energy levels. Which of the following could be their energy difference?"
> - **correct answer**: 10⁻⁴ eV
> - **distractors**: 10⁻¹¹ eV / 10⁻⁸ eV / 10⁻⁹ eV

### F. Continuous PPL

**Paloma** — 16 held-out web/forum/code text subsets. Primary continuous signal at our scale. Eval = mean next-token cross-entropy.

> Example (`c4_en` subset):
>
> ```
> Media friends to attend & cover event to affirm nation's support to jawans.
> Organisor thru their newspapers/media.
> Finally a performing arts College in Noida!
> Learning music, dance or any other art form not only soothes your inner being, but it has also been proven that people who systematically learn music have better strategizing and planning skills than other people. The world today is very competitive and every individual has to be active not only in academics or any one field but in all possible areas.
> ```

**dclm_200m_val** — held-out NL within the DCLM training distribution. Sensitive to overfitting on the 209M-token slice. Reported in `bits_per_byte` (tokenizer-independent) via lm-eval's `loglikelihood_rolling` on a fixed 5000-document DCLM slice (`outputs/raw/dclm_5000docs.jsonl`); the in-training Levanter `eval/dclm_200m_val/loss` values (logged in nats per Llama-3.1 token during training) were converted to bpb via the factor measured on the same dclm text (Llama-3.1 ≈ 4.408 bytes/token → bpb = nats × 0.3273). Sanity check: A5 via direct lm-eval bpb = 0.906 vs A5 via converted nats = 0.923 — agreement within 0.017 bpb.

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


</details>

---

<details>
<summary><h2>2. Models tracked</h2></summary>


| Label | HF repo (or local path) | Params (N) | Tokens trained (D) | FLOPs (≈6·N·D) | Unique tokens | Notes |
|---|---|---|---|---:|---|---|
| **1.4B base (x16)** | `1_4b_wd1_6_x16_nocrossblock_hf` (`peach-thunder-100` / `6xx0hu3l`) | 1.4 B | 3.36 B | 2.8 × 10¹⁹ | 209 M DCLM × 16 epochs (single source) | wd=1.6, LR=1e-3 cosine, block_cross_doc=False, batch=64 × seq=4096 × 12,800 steps |
| **1.4B code25 v2 (matched)** ¤ | `1_4b_25code_alg_v2_hf` (`sage-wildflower-106` / `joqfahkl`) | 1.4 B | 3.36 B | 2.8 × 10¹⁹ | 150 M DCLM × 16 epochs + 50 M opc_algorithmic × 16 epochs → 200 M unique total (matched to baseline 209 M) | Same hyperparams as baseline. Same total trained tokens, same total unique tokens; only the data mix differs (75% NL vs 100% NL). |
| **C5-v2 small stage-1 (code-only)** ‖§ | `c5v2_small_stage1_step6400_hf` (run `stoic-hill-135` / `5hb7vl3u`, step-6400) | 1.4 B | 1.68 B | 1.4 × 10¹⁹ | 100% clean code+markup, same sources as C5-v2 stage-1 (Stack-Edu Python @ score>3.0, Nemotron Code-Concepts, Nemotron Unconditional-Algorithmic, Stack-Edu Markdown @ score>3.0) | wd=0.1, LR=3e-4 cosine, 1-node DP, batch=64 × seq=4096 × 6,400 steps. Mid-cosine snapshot. |
| **C5-v2 small final (matched-budget)** ‖§ | `c5v2_small_step12799_hf` (run `stoic-hill-135` / `5hb7vl3u`, step-12799) | 1.4 B | 3.36 B | 2.8 × 10¹⁹ | Stage 1 (1.68 B, as above) + stage 2 (1.68 B): 90% DCLM + 10% (80% clean code + 20% Stack-Edu Markdown) | Same hparams as base/code25 v2. Same total trained tokens, matched-budget probe — does clean code recovery hold at 1/9.2 the C5-v2 full budget? Direct comparison to base x16 + code25 v2 columns. |
| **A5 1ep DCLM final** ¥ | `1ep_dclm_final_hf` (run `1ep-dclm-A5`, `tmgu1im8`, step-29343) | 1.4 B | 30.77 B | 2.6 × 10²⁰ | 7 × DCLM shards (~34.85 B unique), ~0.88 epoch per shard | wd=0.1, LR=3e-4 cosine, 4-node DP, batch=256 × seq=4096 × 29,343 steps. |
| **B4 1ep code25 final** ¥ | `1ep_code25_final_hf` (run `1ep-code25-B4`, `6zs6ybgt`, step-29343) | 1.4 B | 30.77 B | 2.6 × 10²⁰ | 75% DCLM (23.08 B = 0.66 epoch over 34.85 B available) + 25% code: 5.4 B aryabumi_synth + 1.35 B aryabumi_web + 0.94 B opc, each at ~1 epoch | Same hyperparams as A5. Matched-compute vs A5: same total trained tokens, same hyperparams, only data differs. |
| **C5 stage-1 (code-only)** † | `c5_stage1_step14672_hf` (run `vocal-microwave-132` / `7mnu0nch`, step-14672) | 1.4 B | 15.39 B | 1.3 × 10²⁰ | 100% code+markup: 80% Aryabumi multi-lang Stack (12.31 B ≈ 1 epoch at Aryabumi Table 3 ratios) + 20% markup (3.08 B ≈ 1 epoch at Aryabumi Table 4 ratios) | wd=0.1, LR=3e-4 cosine, 8-node DP, batch=256 × seq=4096 × 14,672 steps. End-of-stage-1 snapshot of the code→text recipe — LR continues into stage 2, so this is mid-cosine, not a fully-cooled checkpoint. |
| **C5 1ep code→text final** † | `c5_final_step29343_hf` (run `rural-forest-133` / `vj95091k`, step-29343 — resumed from `7mnu0nch`/step-20914 after a dy-9 power-cycle crash at step 21,201) | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Stage 1 (15.39 B, as above) then stage 2 (15.39 B): 90% DCLM (13.85 B ≈ 0.40 epoch over 34.85 B available) + 10% (80% Stack + 20% markup) | Same hyperparams as A5/B4. Matched-compute vs A5/B4: same total trained tokens, same hyperparams, only data ordering and ratios differ. Single continuous cosine LR across both stages. |
| **C5-v2 stage-1 (clean code-only)** ‖ | `c5v2_stage1_step14672_hf` (run `glorious-sun-134` / `u23atfbm`, step-14672) | 1.4 B | 15.39 B | 1.3 × 10²⁰ | 100% clean code+markup: 80% code = Stack-Edu Python @ score>3.0 (~54%) + Nemotron Code-Concepts (~45%) + Nemotron Unconditional-Algorithmic (~1%); 20% markup = Stack-Edu Markdown @ score>3.0 | Same hparams as C5. End-of-stage-1 snapshot (mid-cosine). |
| **C5-v2 1ep clean code→text final** ‖ | `c5v2_final_step29343_hf` (run `glorious-sun-134` / `u23atfbm`, step-29343) | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Stage 1 (15.39 B, as above) + stage 2 (15.39 B): 90% DCLM + 10% (80% clean code + 20% Stack-Edu Markdown), same code-mix ratios as stage 1 | Same hparams as A5/B4/C5. Matched-compute vs C5: same total trained tokens, same hparams, same 80/20 + 90/10 recipe — **only the code+markup data quality differs** (classifier-filtered Stack-Edu + Nemotron synthetic textbook code, vs C5's raw multi-language StarCoderData). Single continuous cosine LR across both stages. |
| **C5-v3 phase 1 (code-LM, separate cosine)** ◊ | `c5v3_p1_a6_step14671_hf` (run `8dtdcear`, step-14671) | 1.4 B | 15.39 B | 1.3 × 10²⁰ | 100% clean code+markup at 80/20 (same caches as C5-v2 stage-1: Stack-Edu Python @ score>3.0, Nemotron Code-Concepts, Nemotron Unconditional-Algorithmic, Stack-Edu Markdown @ score>3.0) | Same hparams as A5/B4/C5/C5-v2 (wd=0.1, LR=3e-4 cosine, batch=256 × seq=4096 × 14,672 steps). **Difference from C5/C5-v2:** this phase 1 ends with a fully-cooled cosine (LR → 0) instead of mid-cosine, because phase 2 starts a fresh cosine via `initialize_from_checkpoint_path`. |
| **C5-v3 final (hero, Aryabumi-faithful)** ◊ | `c5v3_p2_a6_step14671_hf` (run `85ip8s5o`, step-14671) | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Phase 1 (15.39 B, as above) + phase 2 (15.39 B): 90% DCLM + 10% (80% clean code + 20% markup), same code-mix ratios as phase 1 | Same per-phase hparams. Phase 2 launched as a separate process with `initialize_from_checkpoint_path` pointing at phase 1's step-14671 — fresh cosine 3e-4 → 0 over phase 2's own budget, fresh optimizer state, step counter restarts at 0. **The "fix" relative to C5/C5-v2 is the separate-cosine-per-phase recipe**, faithful to Aryabumi et al §3.1. |
| **C5-v3-small phase 1** ◊§ | `c5v3_small_phase1_step6399_hf` (run `ex8j1nax`, step-6399) | 1.4 B | 1.68 B | 1.4 × 10¹⁹ | 100% clean code+markup, same caches as C5-v3 phase 1 | wd=0.1, LR=3e-4 cosine, 1-node DP, batch=64 × seq=4096 × 6,400 steps. Matched-budget probe — does the separate-cosine-per-phase recipe still help at 1/9.2 the budget? Direct comparison to C5-v2-small. |
| **C5-v3-small final** ◊§ | (phase 2 currently training on dy-5; final HF path TBD) | 1.4 B | 3.36 B | 2.8 × 10¹⁹ | Phase 1 (1.68 B, as above) + phase 2 (1.68 B): 90% DCLM + 10% (80% code + 20% markup) | Same hparams as C5-v3-small phase 1. Phase 2 inits from phase 1 step-6399 with FRESH cosine. **Direct apples-to-apples comparison vs C5-v2-small final** — same total budget, same data, only LR-schedule recipe differs. |
| **4B final** ª | `4b_dclm_short_final_hf` (run `3_5b_dclm_short`, step-22887) | 3.5 B | 6.0 B | 1.3 × 10²⁰ | 7 × DCLM shards, ~0.17 epoch per shard | wd=0.1, LR=3e-4 cosine, 8-node FSDP, batch=64 × seq=4096 × 22,887 steps. |
| **C5-v4 final** ⚛ | `c5v4_p2_step14671_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Phase 1 (15.39 B, same as C5-v3 phase 1) + phase 2 (15.39 B): **90% SlimPajama-NL** (CC + C4 + Books + ArXiv + Wikipedia, English-only Wiki) + 10% (80% code + 20% markup); code+markup ratios same as C5-v3 | Same per-phase hparams as C5-v3 (wd=0.1, LR=3e-4 fresh cosine, batch=256 × seq=4096 × 14,672 steps). **Difference from C5-v3:** phase 2's 90% NL slot uses SlimPajama-NL (English-only Wiki filtered) instead of DCLM. Tests whether reasoning-dense text (Books/ArXiv/Wiki) elicits reasoning circuits learned in phase 1. |
| **A5-SP** ⚛⚛ | `a5_sp_step29343_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | 100% SlimPajama-NL (English-only Wiki filtered): CC + C4 + Books + ArXiv + Wikipedia. Same as C5-v4's text source. | A5 recipe (single-phase, fresh init from scratch, single continuous cosine 3e-4 → 0) but with **SlimPajama-NL** replacing DCLM. Pure data-axis control vs A5. Tests: is the SP-NL lift seen in C5-v4 just a 'better text data' main effect, or does it require the code phase 1 to be elicited? |
| **c5v6_phase2_step14671** ★ | `c5v6_phase2_step14671_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | C5-v3 phase 1 (15.39 B) + 70% DCLM + 30% (80% code + 20% markup) | C5-v6: 30% code+markup replay in phase 2 (vs 10% in C5-v3); separate cosine init |
| phi-1 | `microsoft/phi-1` | 1.3 B | ~50 B | ~3.9 × 10²⁰ | ~7 B unique (6 B filtered Stack + ~1 B GPT-3.5 synth Python) × ~8 epochs | Code-only (external reference). **NOT a base model**: `microsoft/phi-1` is the *fine-tuned* model — phi-1-base (pretrained on CodeTextbook = filtered Stack-Python + synthetic textbooks) was then post-trained on 180M tokens of synthetic CodeExercises (synthetic HumanEval-shaped problems → solutions). Per the phi-1 paper §3.3: phi-1-base = 29.0% HumanEval, phi-1 (post-fine-tune) = 50.6% — the +22 pp lift comes from format-matched supervised data. phi-1-base is not publicly released. **When comparing our base models (A5/B4/C5/C5-v2/C5-v3/C5-v4) on HumanEval, phi-1.5 is the apples-to-apples reference, NOT phi-1.** Phi-1 is included as a "best-case post-trained code-only model at this scale" reference. |
| phi-1.5 | `microsoft/phi-1_5` | 1.3 B | ~150 B | ~1.2 × 10²¹ | ~30 B unique (phi-1 mix + ~20 B synthetic NL textbooks) × ~5 epochs | Larger synth-textbook training (external reference). **Base model — no instruction or format fine-tune** (per phi-1.5 paper). The right apples-to-apples reference for our base models on code-gen tasks. |

**FLOPs column** is the standard 6·N·D approximation (Hoffmann et al 2022 / Kaplan 2020) — a rough compute marker, not a capability number.

**¤** = the matched-token v2 column uses run `joqfahkl`, NOT the earlier v1 (`eager-grass-104` / `p2n84bo3`). v1 used the full 943M opc slice at ~1 epoch which made unique-token counts unequal between v1 and baseline — see EXPERIMENT_LOG June 1 retraction.

**¥** = A5/B4 are the FINAL-STEP checkpoints (step-29343, ~30.77 B trained tokens). Earlier versions of this doc had columns labelled `s14672` (~50% trained, mid-training snapshots); those have been replaced with the final-step values.

**ª** = 4B is a "tokens-vs-params" comparison point. Note that 4B used 1.3 × 10²⁰ FLOPs vs A5's 2.6 × 10²⁰ — A5 has ~2× more training compute, so the A5 > 4B comparison is partly explained by raw compute rather than purely by "tokens vs params". Treat the comparison as "what does an 8-GPU-day 4B run look like vs an 8-GPU-day 1.4B run", not as a controlled experiment.

**§** = "small" denotes a matched-budget scale-down of the C5-v2 recipe to 3.36 B trained tokens (1/9.2 of the full C5-v2 budget). Same data mix and same two-stage 80/20 code+markup → 90/10 DCLM+code recipe as C5-v2, but at batch=64 × 12,800 steps (matching base x16 / code25 v2 hparams). Single node × 8 GPUs. Purpose: test whether the clean-code recovery seen at full budget also holds when compute budget matches base x16 / code25 v2.

**‖** = C5-v2 is the matched-recipe re-run of C5 using **clean code data** instead of raw StarCoderData. Same exact 80% code + 20% markup stage-1 mix and 90% DCLM + 10% (80% code + 20% markup) stage-2 mix, same hparams (wd=0.1, LR=3e-4 cosine, batch=256 × seq=4096 × 29,343 steps). Only difference: the "code" component is **Stack-Edu Python @ score > 3.0 + Nemotron Code-Concepts + Nemotron Unconditional-Algorithmic** (token-proportional within the 80% slot), and "markup" is **Stack-Edu Markdown @ score > 3.0**. Code-mix proportions within the 80% code slot: SE-Py ~54%, Nemotron-CC ~45%, Nemotron-UA ~1% (token-proportional based on available clean tokens). See `experiments/data_efficiency/code_data_source_samples.md` for raw samples + size rationale. This isolates "does data quality alone fix the code-first NL damage?" — directly comparable to C5 (raw code) and A5 (DCLM-only).

**†** = C5 implements the Aryabumi et al "To Code, or Not To Code?" (2408.10914) two-stage code→text recipe at our scale. Stage 1 is code-only (Aryabumi Tables 3+4 ratios: 80% multi-language StarCoderData + 20% markup); stage 2 reverts to mostly NL (90% DCLM + 10% mixed). C5-stage1 and C5-final share the same wandb run logically but are split across two run-ids (`7mnu0nch` and `vj95091k`) because the original run crashed at step 21,201 when AWS pcluster's auto-scaler (SuspendTime=600s) power-cycled compute node `dy-9` mid-training. The resume reloaded from step-20914 with `WANDB_RUN_ID=7mnu0nch + WANDB_RESUME=allow`, but a new run-id was generated; the optimizer state, LR schedule position, and data position were all restored from the checkpoint — only the wandb log is cosmetically split.

**◊** = C5-v3 implements the **Aryabumi-faithful separate-cosine-per-phase** version of the C5 recipe. Where C5 and C5-v2 used a single continuous cosine across both stages (so stage 2 inherited a half-decayed LR), C5-v3 runs phase 2 as a **separate process** initialized via Levanter's `initialize_from_checkpoint_path` — model weights only, fresh optimizer state, fresh cosine LR (3e-4 → 0) over phase 2's own budget. This matches our reading of Aryabumi et al §3.1 footnote 5; we have an open email to the authors to confirm. Phase 1 (code+markup) and phase 2 (90% DCLM + 10% code+markup) match C5-v2's data mixes exactly — only the LR schedule across the stage boundary differs.


</details>

---

## 3. Canonical results — all models

All numbers from our `lm-eval-harness` pipeline (lm_eval 0.4.11). Rows = tasks (header format `task[nshot]`). Columns = models. Accuracy metrics use `acc_norm` where reported in §1; `acc` otherwise. PPL is `bits_per_byte` (lower=better) for both paloma_macro and dclm_200m_val. Bolded = best in row. `—` = not run.

See §2 footnotes ¤ (code25 v2 vs v1), ¥ (A5/B4 final-step), † (C5 code→text stages + resume forensics), and ‖ (C5-v2 clean-code recipe) for column-definition caveats.

| Task | base (x16) | code25 v2 (x16) | **C5-v2 small stage-1 ‖§** | **C5-v2 small ‖§** | A5 1ep final | B4 1ep final | C5 stage-1 † | C5-v2 stage-1 ‖ | C5 final † | C5-v2 final ‖ | **C5-v3 phase 1** ◊ | **C5-v3 final** ◊ | **C5-v3-small phase 1** ◊§ | **C5-v3-small final** ◊§ | **C5-v4 final** ⚛ | **A5-SP** ⚛⚛ | **c5v6_phase2_step14671** ★ | 4B final ª | phi-1 | phi-1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| **Open-book** | | | | | | | | | | | | | | | | | | | | |
| sciq[0] | 0.652 | 0.590 | 0.545 | 0.601 | 0.834 | 0.829 | 0.707 | 0.727 | 0.754 | 0.715 | 0.720 | 0.728 | 0.541 | 0.712 | 0.799 | 0.781 | 0.806 | 0.824 | 0.707 | **0.933** |
| boolq[0] | 0.502 | 0.567 | 0.617 | 0.614 | 0.563 | 0.599 | 0.619 | 0.593 | 0.623 | 0.580 | 0.595 | 0.443 | 0.618 | 0.614 | 0.607 | 0.561 | 0.546 | 0.552 | 0.451 | **0.746** |
| piqa[0] | 0.634 | 0.606 | 0.566 | 0.577 | 0.718 | 0.709 | 0.583 | 0.584 | 0.591 | 0.600 | 0.581 | 0.649 | 0.554 | 0.647 | 0.680 | 0.683 | 0.688 | 0.697 | 0.562 | **0.766** |
| openbookqa_fact[0] | 0.336 | 0.312 | 0.304 | 0.294 | 0.430 | 0.430 | 0.306 | 0.312 | 0.316 | 0.326 | 0.236 | 0.296 | 0.198 | 0.252 | 0.294 | 0.300 | 0.306 | 0.426 | 0.316 | **0.530** |
| **Mean Open-book** | *0.531* | *0.519* | *0.508* | *0.521* | *0.636* | *0.642* | *0.554* | *0.554* | *0.571* | *0.555* | *0.533* | *0.529* | *0.478* | *0.556* | *0.595* | *0.581* | *0.587* | *0.625* | *0.509* | *0.744* |
| **Closed-book NL** | | | | | | | | | | | | | | | | | | | | |
| arc_easy[25] | 0.401 | 0.388 | 0.312 | 0.335 | 0.629 | 0.607 | 0.362 | 0.395 | 0.385 | 0.418 | 0.397 | 0.536 | 0.322 | 0.485 | 0.552 | 0.563 | 0.590 | 0.612 | 0.378 | **0.805** |
| arc_challenge[25] | 0.242 | 0.241 | 0.223 | 0.218 | 0.316 | 0.289 | 0.209 | 0.220 | 0.208 | 0.215 | 0.183 | 0.214 | 0.181 | 0.206 | 0.226 | 0.251 | 0.250 | 0.292 | 0.232 | **0.532** |
| hellaswag[10] | 0.348 | 0.321 | 0.275 | 0.280 | 0.497 | 0.464 | 0.292 | 0.304 | 0.298 | 0.311 | 0.275 | 0.323 | 0.268 | 0.291 | 0.338 | 0.350 | 0.350 | 0.466 | 0.301 | **0.635** |
| winogrande[5] | 0.504 | 0.500 | 0.505 | 0.507 | 0.541 | 0.515 | 0.513 | 0.507 | 0.517 | 0.484 | 0.514 | 0.504 | 0.502 | 0.513 | 0.512 | 0.515 | 0.521 | 0.511 | 0.498 | **0.710** |
| mmlu[5] | 0.252 | 0.256 | 0.262 | 0.261 | 0.244 | 0.258 | 0.265 | 0.253 | 0.269 | 0.245 | 0.259 | 0.238 | 0.267 | 0.260 | 0.259 | 0.264 | 0.244 | 0.250 | 0.248 | **0.422** |
| commonsense_qa[0] | 0.192 | 0.212 | 0.196 | 0.193 | 0.195 | 0.213 | 0.196 | 0.198 | 0.196 | 0.194 | 0.203 | 0.193 | 0.196 | 0.215 | 0.202 | 0.192 | 0.199 | 0.193 | 0.175 | **0.507** |
| social_iqa[0] | 0.366 | 0.362 | 0.349 | 0.342 | 0.415 | 0.400 | 0.346 | 0.360 | 0.354 | 0.359 | 0.358 | 0.383 | 0.346 | 0.382 | 0.390 | 0.387 | 0.396 | 0.407 | 0.364 | **0.523** |
| logiqa[0] | 0.218 | 0.210 | 0.280 | 0.278 | **0.320** | 0.270 | 0.295 | 0.286 | 0.287 | 0.270 | 0.214 | 0.220 | 0.266 | 0.212 | 0.221 | 0.218 | 0.203 | 0.269 | 0.214 | 0.240 |
| lambada_openai[0] | 0.238 | 0.197 | 0.089 | 0.124 | 0.519 | 0.496 | 0.144 | 0.213 | 0.185 | 0.250 | 0.187 | 0.357 | 0.077 | 0.349 | 0.439 | 0.379 | 0.469 | 0.494 | 0.106 | **0.527** |
| copa[0] | 0.620 | 0.620 | 0.540 | 0.540 | 0.740 | 0.690 | 0.550 | 0.560 | 0.540 | 0.550 | 0.550 | 0.680 | 0.570 | 0.660 | 0.690 | 0.690 | 0.700 | 0.740 | 0.530 | **0.800** |
| wsc[0] | 0.365 | 0.365 | 0.356 | 0.346 | 0.519 | 0.365 | 0.365 | 0.596 | 0.365 | 0.558 | 0.404 | 0.654 | 0.365 | 0.365 | 0.365 | 0.587 | 0.375 | 0.394 | 0.442 | **0.606** |
| storycloze_2018_local[0] | 0.591 | 0.568 | 0.534 | 0.528 | 0.663 | 0.654 | 0.516 | 0.535 | 0.529 | 0.545 | 0.541 | 0.603 | 0.515 | 0.586 | 0.640 | 0.637 | 0.642 | 0.658 | 0.531 | 0.531 |
| cb[0] | 0.250 | 0.429 | 0.411 | 0.411 | 0.321 | 0.214 | 0.411 | 0.393 | 0.393 | 0.375 | 0.375 | 0.411 | 0.411 | 0.429 | 0.411 | 0.304 | 0.393 | 0.393 | 0.464 | 0.464 |
| quac_first_turn[0] | 0.142 | 0.136 | 0.077 | 0.092 | 0.176 | 0.179 | 0.126 | 0.123 | 0.146 | 0.136 | 0.122 | 0.110 | 0.079 | 0.122 | 0.185 | 0.201 | 0.161 | 0.168 | 0.169 | 0.169 |
| **Mean Closed-book NL** | *0.338* | *0.343* | *0.315* | *0.318* | *0.435* | *0.401* | *0.328* | *0.353* | *0.334* | *0.351* | *0.327* | *0.388* | *0.312* | *0.363* | *0.388* | *0.396* | *0.392* | *0.418* | *0.332* | *0.534* |
| **Aggregate / multi-domain reasoning** | | | | | | | | | | | | | | | | | | | | |
| agieval_lsat_ar[0] | 0.226 | 0.252 | 0.230 | 0.204 | 0.187 | 0.222 | 0.248 | 0.209 | 0.235 | 0.230 | 0.235 | 0.226 | 0.222 | 0.226 | 0.230 | 0.217 | 0.252 | 0.222 | 0.213 | 0.183 |
| gpqa_diamond[0] | 0.268 | **0.328** | 0.263 | 0.263 | 0.268 | 0.217 | 0.263 | 0.258 | 0.263 | 0.283 | 0.258 | 0.258 | 0.263 | 0.268 | 0.268 | 0.263 | 0.232 | 0.273 | 0.197 | 0.232 |
| bbh[3] (limit=0.1) | 0.025 | 0.026 | 0.127 | 0.178 | 0.160 | 0.206 | 0.199 | 0.218 | 0.235 | 0.215 | 0.204 | 0.124 | 0.100 | 0.094 | 0.222 | 0.175 | 0.235 | 0.155 | 0.238 | **0.288** |
| mmlu_pro[5] (limit=0.1) | 0.050 | 0.047 | 0.063 | 0.064 | **0.116** | 0.073 | 0.051 | 0.071 | 0.065 | 0.080 | 0.089 | 0.064 | 0.075 | 0.097 | 0.085 | 0.075 | 0.089 | 0.069 | n/a (ctx) ™ | n/a (ctx) ™ |
| **Mean Aggregate** | *0.142* | *0.163* | *0.171* | *0.177* | *0.183* | *0.179* | *0.190* | *0.189* | *0.200* | *0.202* | *0.196* | *0.168* | *0.165* | *0.171* | *0.201* | *0.182* | *0.202* | *0.180* | *0.216* | *0.234* |
| **Math (standard)** | | | | | | | | | | | | | | | | | | | | |
| gsm8k[5] | 0.000 | 0.000 | 0.015 | 0.009 | 0.001 | 0.010 | 0.003 | 0.017 | 0.010 | 0.014 | 0.008 | 0.002 | 0.009 | 0.008 | 0.016 | 0.002 | 0.015 | 0.018 | 0.012 | **0.305** |
| gsm8k_cot[8] | 0.022 | 0.005 | 0.017 | 0.021 | 0.031 | 0.027 | 0.016 | 0.021 | 0.024 | 0.033 | 0.014 | 0.004 | 0.007 | 0.011 | 0.022 | 0.014 | 0.017 | 0.021 | 0.021 | **0.299** |
| minerva_math[4] | 0.0002 | 0.000 | 0.006 | 0.005 | 0.002 | 0.010 | 0.002 | 0.006 | 0.007 | 0.009 | 0.003 | 0.001 | 0.001 | 0.002 | 0.010 | 0.013 | 0.008 | 0.007 | 0.012 | **0.029** |
| **Mean Math (standard)** | *0.007* | *0.002* | *0.013* | *0.012* | *0.011* | *0.016* | *0.007* | *0.015* | *0.014* | *0.019* | *0.008* | *0.002* | *0.006* | *0.007* | *0.016* | *0.010* | *0.013* | *0.015* | *0.015* | *0.211* |
| **Math (perturbation-robust)** | | | | | | | | | | | | | | | | | | | | |
| gsm_symbolic_main[8] | 0.009 | 0.004 | 0.001 | 0.002 | 0.007 | 0.005 | 0.005 | 0.004 | 0.006 | 0.009 | 0.003 | 0.003 | 0.003 | 0.006 | 0.008 | 0.010 | 0.007 | 0.006 | 0.013 | **0.160** |
| gsm_noop[8] | 0.009 | 0.017 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.017 | 0.009 | 0.000 | 0.009 | 0.000 | 0.000 | **0.034** |
| **Mean Math (perturbation-robust)** | *0.009* | *0.011* | *0.005* | *0.001* | *0.004* | *0.003* | *0.003* | *0.006* | *0.003* | *0.004* | *0.002* | *0.002* | *0.002* | *0.011* | *0.009* | *0.005* | *0.008* | *0.003* | *0.006* | *0.097* |
| **Code** | | | | | | | | | | | | | | | | | | | | |
| humaneval[0] (lm-eval) | 0.000 | 0.012 | 0.183 | 0.159 | 0.006 | 0.104 | 0.037 | 0.262 | 0.061 | 0.280 | 0.256 | 0.165 | 0.116 | 0.116 | 0.171 | 0.000 | 0.213 | 0.000 | **0.494** | 0.342 |
| humaneval[0] (bigcode) | 0.000 | 0.000 | 0.055 | 0.098 | 0.000 | 0.000 | 0.012 | 0.073 | 0.037 | 0.055 | 0.122 | 0.024 | 0.055 | 0.030 | 0.079 | — | 0.012 | 0.000 | **0.543** | 0.342 |
| mbpp[3] | 0.000 | 0.000 | 0.098 | 0.136 | 0.000 | 0.052 | 0.052 | 0.210 | 0.098 | 0.290 | 0.208 | 0.048 | 0.088 | 0.050 | 0.126 | 0.000 | 0.204 | 0.000 | **0.416** | 0.342 |
| **Mean Code** | *0.000* | *0.006* | *0.141* | *0.148* | *0.003* | *0.078* | *0.044* | *0.236* | *0.080* | *0.285* | *0.232* | *0.107* | *0.102* | *0.083* | *0.149* | *0.000* | *0.208* | *0.000* | *0.455* | *0.342* |
| **Perplexity (lower=better)** | | | | | | | | | | | | | | | | | | | | |
| dclm_200m_val (bpb) | 1.332 | 1.504 | 1.514 | 1.449 | **0.923** | 0.942 | 1.313 | 1.308 | 1.286 | 1.260 | 1.308 | 1.110 | 1.525 | 1.077 | 1.019 | 1.054 | 0.955 | 0.947 | 1.636 | 1.041 |
| paloma_macro (bpb) | 1.631 | 1.824 | 1.639 | 1.566 | 1.077 | 1.074 | 1.374 | 1.370 | 1.326 | 1.326 | 1.377 | 1.315 | 1.582 | 1.216 | 1.093 | 1.142 | 1.087 | 1.114 | 1.738 | 1.174 |

**™** = mmlu_pro 5-shot prompts run 2000-2400 tokens; phi-1 and phi-1.5 both have `max_position_embeddings = 2048`, so the eval cannot run without truncating the prompt past the few-shot examples. Not a missing run — a model context limitation.

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

## Updating this doc

When a new model is trained or a new eval is added, update §1 (models) and §3 (results) with the new row/column. Add a brief follow-up entry in `EXPERIMENT_LOG.md` pointing here. Chronological narrative stays in `EXPERIMENT_LOG.md`; canonical reference stays here.
