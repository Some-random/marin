# Evaluation Reference: Tasks, Taxonomy, and Model Results

<details>
<summary><h2>1. Taxonomy by mechanism (with examples)</h2></summary>

Two-way split for QA: **open-book** (the answer is in the prompt; model attends and extracts) vs. **closed-book** (no passage; model uses weights). Plus two task families that don't fit the QA frame: code generation and continuous PPL. (Math and the multi-domain aggregates have been **dropped** from the kept suite — see the **Collapse** subsection at the end of this taxonomy for why.)

### A. Open-book QA

**sciq** — every question comes with a `support` paragraph that literally states the answer. 4-way MC.

> - **question**: "Compounds that are capable of accepting electrons, such as o 2 or f2, are called what?"
> - **support**: "Oxidants and Reductants Compounds that are capable of accepting electrons, such as O 2 or F2, are called oxidants (or oxidizing agents) because they can oxidize other compounds. In the process of accepting electrons, an oxidant is reduced. Compounds that are capable of donating electrons, such as sodium metal or cyclohexane (C6H12), are called reductants (or reducing agents) because they can cause the reduction of another compound."
> - **choices**: oxidants / antioxidants / Oxygen / residues
> - **correct_answer**: oxidants

**boolq** — yes/no question + Wikipedia-style passage that contains the answer. **Open-book**: the passage **is** in the prompt (`{{passage}}\nQuestion: …?\nAnswer:`), scored on ` no`/` yes` — identical to Marin's lm-eval default setup; 0-shot in our pipeline (Marin runs it 10-shot, which our small models don't benefit from). Our 1.4B models do read the passage (+6pp vs a no-passage ablation) but sit below the 62% yes-majority — a real reading-comprehension task that scales in (Marin-8B Base 85.9%).

> - **question**: "does ethanol take more energy make that produces"
> - **passage**: "Ethanol fuel -- All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. … one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. …"
> - **label**: False

**openbookqa_fact** — custom variant we added in `experiments/reasoning_pretraining/code_ladder/eval/openbookqa_fact.yaml` that uses the `additional` config of `allenai/openbookqa` and prepends the dataset's `fact1` field to the question stem. This is the open-book MC eval, and replaces the closed-book `openbookqa` default that lm-eval ships with.

### B. Closed-book QA / commonsense

No passage in the prompt. The model has to recall facts, apply commonsense, or do logical deduction from its weights. Multi-way MC.

**arc_easy** — grade-school science MC.

> - **question**: "Which statement best explains why photosynthesis is the foundation of most food webs?"
> - **choices**: A) Sunlight is the source of energy for nearly all ecosystems. / B) Most ecosystems are found on land instead of in water. / C) Carbon dioxide is more available than other gases. / D) The producers in all ecosystems are plants.
> - **answerKey**: A

**mmlu** — 4-way MC across 57 subject subtasks. **Text-scored** (`mmlu_text.yaml`, `cais/mmlu` `all` config): the candidates are the answer *texts*, not the bare letter ` A`/` B`/` C`/` D`. The letter-scored lm-eval default collapses weak base models onto a letter-frequency prior; text-scoring reads the actual answer. 0-shot (mmlu_text is flat across 0/5/10-shot at our scale). Kept because our models clear chance (~28–30% vs 25%) — a genuine, if small, knowledge signal, unlike the letter-scored version which floored at ~24–25%.

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

**commonsense_qa** — 5-way MC, ConceptNet-derived commonsense. **Text-scored, 5-shot** (`commonsense_qa_text.yaml`): candidates are the answer *texts*, not bare letters. The letter-scored default pins weak models to ~chance (20%) via a letter-frequency prior (A5 fires ` A` ~98% of the time); text-scoring recovers real signal (c5v6 20→35%, A5 20→41% at 0-shot; 5-shot adds a few points). See [`docs/COMMONSENSE_QA_SCORING_DIFF.md`](docs/COMMONSENSE_QA_SCORING_DIFF.md).

> - **question**: "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"
> - **question_concept**: revolving door
> - **choices**: A) bank / B) library / C) department store / D) mall / E) new york
> - **answerKey**: A

**copa** — Choice of Plausible Alternatives (super_glue/copa). 2-way MC: given a premise, pick the more plausible **cause** or **effect** of it. The `question` field is the literal string "cause" or "effect" and tells the model which direction to reason.

> - **premise**: "The man turned on the faucet."
> - **question**: effect
> - **choice1**: "The toilet filled with water."
> - **choice2**: "Water flowed from the spout."
> - **label**: 1 (choice2)

**wsc273** — Winograd Schema Challenge, 273-example **referent-choice** version (`winograd_wsc`, config `wsc273`; needs `HF_DATASETS_TRUST_REMOTE_CODE=1`). Given a sentence with an ambiguous pronoun and two candidate referents, substitute each referent for the pronoun and score the two resulting sentences; pick the higher-likelihood one. This is the version **Marin** evaluates — we switched from the SuperGLUE binary `wsc.fixed` (a yes/no "does span2 refer to span1" that our models couldn't beat its majority baseline on; moved to Collapse). See [`docs/WSC273_PREDICTIONS.md`](docs/WSC273_PREDICTIONS.md). 2-way.

> - **text**: "The city councilmen refused the demonstrators a permit because they feared violence."
> - **pronoun**: "they"  • **options**: the city councilmen / the demonstrators
> - **label**: the city councilmen

**storycloze_2018_local** — Story Cloze Test (Mostafazadeh et al 2016, 2018 split). 2-way MC: given the first 4 sentences of a 5-sentence story, pick the more plausible ending. Tests narrative coherence and commonsense expectations. We use a local custom YAML (`storycloze_2018_local.yaml`) because the canonical lm-eval `storycloze_2018` task requires manual dataset acceptance terms.

> - **input_sentence_1**: "Karen was assigned a roommate her first year of college."
> - **input_sentence_2**: "Her roommate asked her to go to a nearby city for a concert."
> - **input_sentence_3**: "Karen agreed happily."
> - **input_sentence_4**: "The show was absolutely exhilarating."
> - **ending_1**: "Karen became good friends with her roommate."
> - **ending_2**: "Karen hated her roommate."
> - **answer_right_ending**: 1

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

### D. Code generation

**HumanEval** — function generation from docstring; pass@1 by **executing** the real unit tests. 0-shot: the function signature + docstring **is** the prompt (no demonstration examples). Reported in §3 with two rows, `humaneval[0] (lm-eval)` and `humaneval[0] (bigcode)`. **Both actually execute byte-identical unit tests** — bigcode's `custom_metrics/execute.py` is a vendored copy of HF `code_eval`, which lm-eval loads via `evaluate.load('code_eval')`; both build `candidate + "\n" + test_case` and run the same `check()`. So the two rows differ ONLY in the prompt/generation config, **not in what counts as passing** (the earlier "lm-eval = regex-match" / "bigcode is the trustworthy one" descriptions were both wrong). The gap between them (e.g. c5v6: lm-eval 0.213 vs bigcode 0.012) is a **prompt-format artifact**: bigcode calls `doc['prompt'].strip()`, removing the prompt's trailing newline, which flips weak/undertrained base models from continuing the indented body to emitting an empty stub (**49% of bigcode HumanEval generations are empty for c5v6**) — the same checkpoint writes a full working body under lm-eval. So **for our weak base models lm-eval HE is the more faithful number, not bigcode**; the two converge only at strong (phi) scale, where the model writes an indented body regardless of the trailing newline (phi-1: 0.543 bigcode ≈ paper's 0.506). We use lm-eval for MBPP (bigcode-MBPP is upstream-broken). See [`docs/EVAL_TRUSTWORTHINESS_AUDIT.md`](docs/EVAL_TRUSTWORTHINESS_AUDIT.md).

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

### G. Collapse (removed from the kept suite)

<details>
<summary>Dropped tasks + rationale — click to expand.</summary>

Tasks pulled from the suite because, at our scale (≤1.4B), they don't produce a trustworthy signal — the model floors at chance, collapses onto a scoring prior, or the metric is too noisy to read. Full per-task rationale + 0/5/10-shot numbers + answer-choice distributions in [`docs/REMOVED_TASKS.md`](docs/REMOVED_TASKS.md).

**From closed-book QA (B):**

- **arc_challenge** (25-shot) — hard grade-school science MC. c5v6 sits at chance (~27.6% acc_norm); only A5 clears it weakly (~31.6%). Real task, but too hard to discriminate our models.
- **logiqa** — formal logical deduction. All our models ~26–29% (4-way chance = 25%); few-shot doesn't help, and even phi barely moves. No signal at our scale.
- **wsc** (binary `wsc.fixed`) — yes/no coreference. **Replaced by wsc273** (referent-choice, Marin-aligned, kept in B): our models couldn't beat the binary majority baseline.
- **cb** (CommitmentBank, 3-way NLI) — N=56 dev set (one example = 1.8%, inherently noisy); our models collapse to 2 classes and never predict "Neither", stuck ~40–48% near the ~50% majority. Not in Marin's suite (came from Aryabumi's NL set). No scoring lever (labels are the answer).

**Math (former section C)** — gsm8k, gsm8k_cot, minerva_math, gsm_symbolic_main, gsm_noop. All free-generation exact-match on a final number. Our 1.4B models floor (≈0) on plain GSM8K, so every perturbation variant floors too — nothing to measure. (The GSM-Symbolic / NoOp robustness story needs a model that can do the base task first.)

**Aggregate / multi-domain (former section E)** — bbh, mmlu_pro, agieval_lsat_ar, gpqa_diamond. Heterogeneous subtask means; each is at/near chance for our models and averages out any remaining signal. None are used by the scale-matched references (phi-1.5, Aryabumi, Suhas); bbh + mmlu_pro are Marin-8B-lineage tasks meant for 8B+.

The kept **mmlu** and **commonsense_qa** were *rescued* (not collapsed) by switching from letter-scoring to **text-scoring** — see their entries in B.

</details>

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
| **C5-v3 phase 1 (code-LM, separate cosine)** ◊ | `c5v3_phase1_step14671_hf` (run `8dtdcear`, step-14671) | 1.4 B | 15.39 B | 1.3 × 10²⁰ | 100% clean code+markup at 80/20 (same caches as C5-v2 stage-1: Stack-Edu Python @ score>3.0, Nemotron Code-Concepts, Nemotron Unconditional-Algorithmic, Stack-Edu Markdown @ score>3.0) | Same hparams as A5/B4/C5/C5-v2 (wd=0.1, LR=3e-4 cosine, batch=256 × seq=4096 × 14,672 steps). **Difference from C5/C5-v2:** this phase 1 ends with a fully-cooled cosine (LR → 0) instead of mid-cosine, because phase 2 starts a fresh cosine via `initialize_from_checkpoint_path`. |
| **C5-v3 final (hero, Aryabumi-faithful)** ◊ | `c5v3_p2_a6_step14671_hf` (run `85ip8s5o`, step-14671) | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Phase 1 (15.39 B, as above) + phase 2 (15.39 B): 90% DCLM + 10% (80% clean code + 20% markup), same code-mix ratios as phase 1 | Same per-phase hparams. Phase 2 launched as a separate process with `initialize_from_checkpoint_path` pointing at phase 1's step-14671 — fresh cosine 3e-4 → 0 over phase 2's own budget, fresh optimizer state, step counter restarts at 0. **The "fix" relative to C5/C5-v2 is the separate-cosine-per-phase recipe**, faithful to Aryabumi et al §3.1. |
| **C5-v3-small phase 1** ◊§ | `c5v3_small_phase1_step6399_hf` (run `ex8j1nax`, step-6399) | 1.4 B | 1.68 B | 1.4 × 10¹⁹ | 100% clean code+markup, same caches as C5-v3 phase 1 | wd=0.1, LR=3e-4 cosine, 1-node DP, batch=64 × seq=4096 × 6,400 steps. Matched-budget probe — does the separate-cosine-per-phase recipe still help at 1/9.2 the budget? Direct comparison to C5-v2-small. |
| **C5-v3-small final** ◊§ | (phase 2 currently training on dy-5; final HF path TBD) | 1.4 B | 3.36 B | 2.8 × 10¹⁹ | Phase 1 (1.68 B, as above) + phase 2 (1.68 B): 90% DCLM + 10% (80% code + 20% markup) | Same hparams as C5-v3-small phase 1. Phase 2 inits from phase 1 step-6399 with FRESH cosine. **Direct apples-to-apples comparison vs C5-v2-small final** — same total budget, same data, only LR-schedule recipe differs. |
| **4B final** ª | `4b_dclm_short_final_hf` (run `3_5b_dclm_short`, step-22887) | 3.5 B | 6.0 B | 1.3 × 10²⁰ | 7 × DCLM shards, ~0.17 epoch per shard | wd=0.1, LR=3e-4 cosine, 8-node FSDP, batch=64 × seq=4096 × 22,887 steps. |
| **C5-v4 final** ⚠ | `c5v4_p2_step14671_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Phase 1 (15.39 B, same as C5-v3 phase 1) + phase 2 (15.39 B): **90% SlimPajama-NL** (CC + C4 + Books + ArXiv + Wikipedia, English-only Wiki) + 10% (80% code + 20% markup); code+markup ratios same as C5-v3. **⚠ Audit (2026-06-15):** SP-NL slot is weighted uniformly per shard directory, NOT per token. SP-NL has 128 chunk1 parts (12.83 B tokens) + 100 chunk2 parts (51.94 B tokens). Actual sampling ≈ 56% chunk1 / 44% chunk2, not the intended token-proportional 19.8% / 80.2%. Model saw a chunk1-biased SP-NL distribution, not the full 64.77 B token-proportional SP-NL. | Same per-phase hparams as C5-v3 (wd=0.1, LR=3e-4 fresh cosine, batch=256 × seq=4096 × 14,672 steps). **Difference from C5-v3:** phase 2's 90% NL slot uses (part-uniform-biased) SlimPajama-NL instead of DCLM. |
| **A5-SP** ⚠ | `a5_sp_step29343_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | 100% SlimPajama-NL (English-only Wiki filtered): CC + C4 + Books + ArXiv + Wikipedia. Same as C5-v4's text source. **⚠ Same part-uniform vs token-proportional bug as C5-v4 ⚠.** | A5 recipe (single-phase, fresh init from scratch, single continuous cosine 3e-4 → 0) but with (part-uniform-biased) SlimPajama-NL replacing DCLM. |
| **C5-v6 final** ★ | `c5v6_phase2_step14671_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | C5-v3 phase 1 (15.39 B) + 70% DCLM + 30% (80% code + 20% markup). **Phase 2's code+markup is STRICT REPLAY of phase 1's first ~30% of code+markup data, NOT new code** — see ★. | C5-v6: 30% code+markup REPLAY in phase 2 (vs 10% in C5-v3); separate cosine init. **Important: phase 2 sees a strict prefix-subset of phase 1's code+markup tokens, NOT new code** (see ★). |
| **C5-v5 final** ⚠ | `c5v5_step29343_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | 100% SlimPajama-NL (English-only Wiki) as text source; code+markup at C5-v2/v3/v4 ratios. Phase 1: 100% code+markup. Phase 2: 90% SP-NL + 10% (80% code + 20% markup). Total 30.77 B tokens, 29,343 steps. **⚠ Same SP-NL part-uniform vs token-proportional bug as C5-v4/A5-SP ⚠.** | C5-v5: C5-v2 recipe (single continuous cosine across both stages) but with (part-uniform-biased) SlimPajama-NL replacing DCLM in the 90% text slot. |
| **C5-v6-NEW final** ⚠ | `c5v6new_final_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Same data mix as C5-v6 (70% DCLM + 30% (80% code + 20% markup)) but the 30% code+markup is **partially fresh**, NOT fully disjoint as originally described. SE-Python is genuinely new (fresh score-[2.8, 3.0) cache, c5v6new_stack_edu_python_low, 3.27 B tokens). Nemotron-CC, Nemotron-UA, Markdown reuse existing caches with `DatasetComponent.offset = phase-1-consumed-count`. **⚠ Audit (2026-06-15):** `offset` slices the underlying cache in RAW INDEX space, but training's per-component reads go through a Feistel shuffle. So phase 2's reads of Nemotron-CC and Markdown OVERLAP with phase 1's reads at ~394K sequences ≈ 1.62 B tokens ≈ 10.4% of phase 2's total token budget ≈ 34.7% of phase 2's code+markup slice. C5-v6-NEW is a **"partially fresh"** comparison, NOT a clean replay-vs-new contrast. | C5-v6-NEW: was intended to test REPLAY vs NEW at matched 30%. Actual experiment is "partial-new" (only SE-Python disjoint). Same per-phase hparams as C5-v6. |
| **C5-V7 final** | `c5v7_final_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Same recipe as C5-v6 (separate-cosine init from C5-v3 phase 1 step-14671, strict-replay code+markup) but phase 2's code+markup share is **50%** (vs C5-v6's 30%). Phase 2: 50% DCLM + 50% (80% code + 20% markup). Forms the scaling-axis 10% → 30% → 50% with C5-v3, C5-v6, C5-v7. | C5-v7: replay-axis scaling study. Phase 2 code+markup are strict prefix replay of phase 1's data (same mechanism as C5-v6 — same caches, same data_seed). Same per-phase hparams as C5-v6: wd=0.1, LR=3e-4 fresh cosine, batch=256 × seq=4096 × 14,672 steps. |
| **c5v8r_step14671** ◊⚠ | `c5v8r_step14671_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Random-code phase 1 (init from C5 step-14672, the StarCoderData-only continuous-cosine endpoint) → SP-NL phase 2 with 10% curated code+markup, separate cosine. Matches C5-v4 phase 2 exactly; only phase 1 init changes from C5-v3 curated to C5 random. | Matching-data follow-up for the C5 vs C5-v2 null-transfer finding. Compared to C5-v4 (curated code init), C5-v8r isolates whether curated code is contributing latent signal that needed SP-NL to surface. ⚠ Caveat: C5's step-14672 was the midpoint of a continuous-cosine run (LR mid-decay ~1.5e-4), not a fully-cooled phase-1 endpoint like C5-v3 phase 1's step-14671. Comparison isn't perfectly clean. |
| **code25b_step23746** ⚙ | `code25b_step23746_hf` | 1.4 B | 24.9 B | 2.1 × 10²⁰ | 1.4B single-phase code-only base: 24.9 B Llama tokens of curated code (Stack-Edu Python score≥3.0/2.8-3.0/2.7-2.8/2.5-2.7 + Nemotron Code-Concepts + Nemotron Unconditional-Algorithmic), row-proportional sampling. A5-style recipe: LR 3e-4 cosine to 0, wd 0.1, batch 256 × seq 4096, 23,747 steps. | Designed to be the 'code-as-reasoning-prior' base for follow-up continued-pretraining experiments. Expect terrible NL/perplexity (model has never seen text) but strong code. The 1.4 B/24.9 B compute (~2.1e20 FLOPs) is slightly under A5 (2.6e20) due to lower token count. num_train_steps was originally set from a 683 tok/row heuristic giving 26,300 steps; fixed mid-run to 23,747 from measured .stats.json, restart from step 0 so cosine LR ends exactly at end-of-corpus. |
| **c5v8r_p1_step14671** ◊r | `c5v8r_p1_step14671_hf` | 1.4 B | 15.39 B | 1.3 × 10²⁰ | Clean random-code phase 1: same data as C5/C5-v3 stage 1 (raw Stack 10 langs + raw markup 5 langs at Aryabumi Table 3/4 ratios), but SEPARATE cosine 3e-4→0 over 14671 steps (fully cooled), matching C5-v3 phase 1's recipe structure. | Built as the clean random-code init for C5-v8r phase 2 (replaces the confounded C5 step-14672 mid-cosine endpoint used in the original c5v8r). Direct apples-to-apples comparison with C5-v3 phase 1 (curated code, same recipe) on the code-data axis. |
| **code25b_clean_step23511** ⚙c | `code25b_clean_step23511_hf` | 1.4 B | 24.65 B | 2.1 × 10²⁰ | 1.4B code-only base, CLEAN data composition: 80% curated code at threshold ≥2.7 (SE-Py clean+low+mid + Nemotron-CC + Nemotron-UA = 19.73B, ~1.0 epoch) + 20% Stack-Edu Markdown (4.93B target, 0.50 epoch). Row-proportional weighting within slots. A5-style cosine 3e-4→0, batch 256×4096, 23,512 steps, no Levanter overrides. | Replaces code25b v2 (which dropped markup and added lower SE-Py bands without user confirmation). Matches C5-v6 Stage 1's 80/20 code+markup ratio, just at 1.6× the compute. Tests whether 'more curated code + markup' helps Code beyond C5-v6 Stage 1's 0.195 baseline. |
| **code25b_clean_p2_step4767** ⊗ | `code25b_clean_p2_step4767_hf` | 1.4 B | 29.65 B | 2.5 × 10²⁰ | Phase 2 over the code25b_clean base (⚙c, 24.65 B curated code ≥2.7): 4,768 steps = 5.0 B tokens, 70% DCLM + 30% (80% code + 20% markup) replay, separate fresh cosine 3e-4→0, wd=0.1, DCLM text source. Total trained = 24.65 B (phase 1) + 5.0 B (phase 2). | High-code / low-text **diagonal** point of the code↔text scaling grid: vs C5-v6 (15.39 B code → 15.39 B text) it moves code UP (+9.3 B) and text DOWN (−10.4 B) at once. Buys Code (+30.9% Mean Code: lm-eval HE +8.8%, mbpp +4.9%), pays in NL (−7.5% Mean Closed-book, text-scored) + perplexity (dclm 1.020 vs 0.955, paloma 1.130 vs 1.087). Both axes move → confounded; (a) code25b_clean_p2_15bt (24.65 B → 15.39 B text) isolates the code axis (vs C5-v6) and the text axis (vs this run). |
| **c5v3_half_p2_step14671** ½ | `c5v3_half_p2_step14671_hf` | 1.4 B | 23.09 B | 1.9 × 10²⁰ | Phase 1 = HALF of C5-v6's ≥3.0 code base (all 4 sources scaled 0.5×, 7.70 B, 7,336 steps, every source <1 epoch, no repeat) → phase 2 = 15.39 B text (70% DCLM + 30% [80% code + 20% markup] replay), separate fresh cosine 3e-4→0, wd=0.1. Total trained = 7.70 B + 15.39 B. | The **0.5× rung** of the code-budget ladder (fixed 15.4 B text, fixed ≥3.0 quality). vs C5-v6 isolates code budget at matched quality: halving code costs −16.3% Mean Code (0.120 vs 0.143); NL ~flat, marginally worse (−2.1% Closed-book, text-scored — NL peaks at the 1× rung). Clean low end of the ladder {½ 0.120, C5-v6 0.143, ⊕ 0.146}. |
| **code25b_clean_p2_15bt_step14671** ⊕ | `code25b_clean_p2_15bt_step14671_hf` | 1.4 B | 40.04 B | 3.4 × 10²⁰ | Phase 2 over the code25b_clean base (⚙c, 24.65 B curated code ≥2.7): 14,672 steps = 15.39 B tokens, 70% DCLM + 30% (80% code + 20% markup) replay, separate fresh cosine 3e-4→0, wd=0.1, DCLM text source. Total trained = 24.65 B + 15.39 B. | The **1.6× rung** of the code-budget ladder (fixed 15.4 B text). Same phase-2 recipe as C5-v6, bigger code base. vs C5-v6 isolates code budget (+2.1% Mean Code, 0.146 vs 0.143 — saturated); vs ⊗ isolates text budget (restoring text → +6.8% NL, +5.5% dclm bpb, −21.9% Code). Caveat: code base is ≥2.7 (vs ≥3.0 for ½/C5-v6), so the 1×→1.6× flatness is saturation OR quality dilution — not separable. |
| **c5v8r_p2_step14671** ◊r | `c5v8r_p2_step14671_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Clean phase 2 follow-up to c5v8r_p1: same SP-NL phase-2 recipe as C5-v4 (90% SP-NL + 10% [80% code + 20% markup], separate cosine 14,672 steps), but initialized from the fully-cooled clean c5v8r_p1_step14671 base. Replaces the confounded mid-cosine C5 step-14672 base used by the original c5v8r. | Apples-to-apples with C5-v4 (curated-code init) on the code-data axis — only the phase-1 init changes from curated to clean random. Phase 1 endpoint is c5v8r_p1_step14671 (also in §3). |
| **c5v6_strict_step14671** ◊* | `c5v6_strict_step14671_hf` | 1.4 B | 30.77 B | 2.6 × 10²⁰ | Strict-prefix-replay variant of C5-v6: same 70% DCLM + 30% (80% code + 20% markup) recipe and same init (C5-v3 phase 1 step-14671), but the phase-2 components dict puts code+markup BEFORE dclm so Levanter's per-component Feistel shuffle keys match phase 1's. Phase 2 reads = strict prefix of phase 1's shuffled code+markup stream (what the original c5v6_phase2 docstring promised but didn't deliver due to component-order bug). | Isolates strict prefix replay (specific docs rehearsed) vs same-cache different-shuffle (same distribution, fresh draws). Compare directly to C5-v6 final (same data, same ratios, same seed — only the components dict order differs). |
| phi-1 | `microsoft/phi-1` | 1.3 B | ~50 B | ~3.9 × 10²⁰ | ~7 B unique (6 B filtered Stack + ~1 B GPT-3.5 synth Python) × ~8 epochs | Code-only (external reference). **NOT a base model**: `microsoft/phi-1` is the *fine-tuned* model — phi-1-base (pretrained on CodeTextbook = filtered Stack-Python + synthetic textbooks) was then post-trained on 180M tokens of synthetic CodeExercises (synthetic HumanEval-shaped problems → solutions). Per the phi-1 paper §3.3: phi-1-base = 29.0% HumanEval, phi-1 (post-fine-tune) = 50.6% — the +22 pp lift comes from format-matched supervised data. phi-1-base is not publicly released. **When comparing our base models (A5/B4/C5/C5-v2/C5-v3/C5-v4) on HumanEval, phi-1.5 is the apples-to-apples reference, NOT phi-1.** Phi-1 is included as a "best-case post-trained code-only model at this scale" reference. |
| phi-1.5 | `microsoft/phi-1_5` | 1.3 B | ~150 B | ~1.2 × 10²¹ | ~30 B unique (phi-1 mix + ~20 B synthetic NL textbooks) × ~5 epochs | Larger synth-textbook training (external reference). **Base model — no instruction or format fine-tune** (per phi-1.5 paper). The right apples-to-apples reference for our base models on code-gen tasks. |

**FLOPs column** is the standard 6·N·D approximation (Hoffmann et al 2022 / Kaplan 2020) — a rough compute marker, not a capability number.

**¤** = the matched-token v2 column uses run `joqfahkl`, NOT the earlier v1 (`eager-grass-104` / `p2n84bo3`). v1 used the full 943M opc slice at ~1 epoch which made unique-token counts unequal between v1 and baseline — see EXPERIMENT_LOG June 1 retraction.

**¥** = A5/B4 are the FINAL-STEP checkpoints (step-29343, ~30.77 B trained tokens). Earlier versions of this doc had columns labelled `s14672` (~50% trained, mid-training snapshots); those have been replaced with the final-step values.

**ª** = 4B is a "tokens-vs-params" comparison point. Note that 4B used 1.3 × 10²⁰ FLOPs vs A5's 2.6 × 10²⁰ — A5 has ~2× more training compute, so the A5 > 4B comparison is partly explained by raw compute rather than purely by "tokens vs params". Treat the comparison as "what does an 8-GPU-day 4B run look like vs an 8-GPU-day 1.4B run", not as a controlled experiment.

**§** = "small" denotes a matched-budget scale-down of the C5-v2 recipe to 3.36 B trained tokens (1/9.2 of the full C5-v2 budget). Same data mix and same two-stage 80/20 code+markup → 90/10 DCLM+code recipe as C5-v2, but at batch=64 × 12,800 steps (matching base x16 / code25 v2 hparams). Single node × 8 GPUs. Purpose: test whether the clean-code recovery seen at full budget also holds when compute budget matches base x16 / code25 v2.

**‖** = C5-v2 is the matched-recipe re-run of C5 using **clean code data** instead of raw StarCoderData. Same exact 80% code + 20% markup stage-1 mix and 90% DCLM + 10% (80% code + 20% markup) stage-2 mix, same hparams (wd=0.1, LR=3e-4 cosine, batch=256 × seq=4096 × 29,343 steps). Only difference: the "code" component is **Stack-Edu Python @ score > 3.0 + Nemotron Code-Concepts + Nemotron Unconditional-Algorithmic** (token-proportional within the 80% slot), and "markup" is **Stack-Edu Markdown @ score > 3.0**. Code-mix proportions within the 80% code slot: SE-Py ~54%, Nemotron-CC ~45%, Nemotron-UA ~1% (token-proportional based on available clean tokens). See `experiments/reasoning_pretraining/code_ladder/data/code_data_source_samples.md` for raw samples + size rationale. This isolates "does data quality alone fix the code-first NL damage?" — directly comparable to C5 (raw code) and A5 (DCLM-only).

**†** = C5 implements the Aryabumi et al "To Code, or Not To Code?" (2408.10914) two-stage code→text recipe at our scale. Stage 1 is code-only (Aryabumi Tables 3+4 ratios: 80% multi-language StarCoderData + 20% markup); stage 2 reverts to mostly NL (90% DCLM + 10% mixed). C5-stage1 and C5-final share the same wandb run logically but are split across two run-ids (`7mnu0nch` and `vj95091k`) because the original run crashed at step 21,201 when AWS pcluster's auto-scaler (SuspendTime=600s) power-cycled compute node `dy-9` mid-training. The resume reloaded from step-20914 with `WANDB_RUN_ID=7mnu0nch + WANDB_RESUME=allow`, but a new run-id was generated; the optimizer state, LR schedule position, and data position were all restored from the checkpoint — only the wandb log is cosmetically split.

**◊** = C5-v3 implements the **Aryabumi-faithful separate-cosine-per-phase** version of the C5 recipe. Where C5 and C5-v2 used a single continuous cosine across both stages (so stage 2 inherited a half-decayed LR), C5-v3 runs phase 2 as a **separate process** initialized via Levanter's `initialize_from_checkpoint_path` — model weights only, fresh optimizer state, fresh cosine LR (3e-4 → 0) over phase 2's own budget. This matches our reading of Aryabumi et al §3.1 footnote 5; we have an open email to the authors to confirm. Phase 1 (code+markup) and phase 2 (90% DCLM + 10% code+markup) match C5-v2's data mixes exactly — only the LR schedule across the stage boundary differs.

**★** = C5-v6's phase 2 reuses the same code+markup caches as phase 1 (same C5-v3 phase 1 checkpoint reused — separate cosine init). Because phase 2 uses `data_seed=0` (same as phase 1), fresh `initialize_from_checkpoint_path` (loads model weights only, not data-loader state), and the same component caches, Levanter's `MixtureDataset` re-indexes each component starting at sequence-index 0. Per `lib/levanter/src/levanter/data/mixture.py:221-232` with `mixture_block_size=2048` (default in `LmDataConfig`), the sequence-index for SE-Python at block T is `block_id * counts_per_block[se_python]`, where `counts_per_block ≈ component_weight × block_size`. Phase 1's SE-Python `counts_per_block ≈ 886` (weight 0.432 × 2048); phase 2's SE-Python `counts_per_block ≈ 265` (weight 0.130 × 2048). Over 14,672 steps × batch 256 = 3.76 M sequences = 1834 mixture blocks. Phase 2's SE-Python sequence range `[0 .. 1834×265]` ≈ `[0 .. 486 K]` is a strict prefix of phase 1's `[0 .. 1834×886]` ≈ `[0 .. 1.62 M]`. Same applies to Nemotron-CC, Nemotron-UA, Stack-Edu-Markdown. **All 4.6 B code+markup tokens phase 2 sees are tokens phase 1 already saw**, in the same shuffled order — this is approximately strict replay of the first ~30% of phase 1's code+markup data. The original C5-v6 design intent was "30% new code"; the actual implementation is "30% replay". A planned contrast run (C5-v6-NEW) would use an explicit per-component sequence-offset to start phase 2 where phase 1 ended, but the current code+markup caches are mostly consumed by phase 1 already (SE-Python cache total ≈ 1.66 M seqs, phase 1 consumed ≈ 1.62 M; Nemotron-CC ≈ 1.71 M vs 1.34 M consumed). A clean "new-code" run requires tokenizing additional code shards first; with the existing caches, only Stack-Edu-Markdown has enough headroom for genuinely-new phase 2 data.

**⊗** = code25b_clean phase-2 (5 B-text **diagonal** continuation). The 5.0 B-token (4,768-step) phase-2 continuation of the code25b_clean base (⚙c). Identical phase-2 recipe to C5-v6 (★) — 70% DCLM + 30% (80% code + 20% markup) replay, separate fresh cosine 3e-4→0, DCLM text source — but only 5.0 B phase-2 tokens vs C5-v6's 15.39 B, on a 24.65 B code base vs C5-v6's 15.39 B. It is the high-code / low-text corner of the (code, text) scaling grid {(b') 7.7→15.4, C5-v6 15.4→15.4, (a) 24.65→15.4, ⊗ 24.65→5.0}. Compared to C5-v6, **both** budget axes differ (code ↑, text ↓), so the diff is a diagonal — run (a) `code25b_clean_p2_15bt` to decompose it into single-axis edges (code-axis vs C5-v6, text-axis vs ⊗).

**½** = c5v3_half_p2: the **0.5× rung** of the code-budget ladder. Phase 1 is HALF of C5-v6's phase-1 code base — all 4 ≥3.0 sources (SE-Py ≥3.0, Nemotron-CC, Nemotron-UA, SE-Markdown) scaled 0.5× to 7.70 B (7,336 steps, every source <1 epoch, no repeat) — then the same 15.39 B phase 2 as C5-v6 (70% DCLM + 30% replay, separate cosine). Isolates code budget at FIXED ≥3.0 quality vs C5-v6 (1×). Ladder Mean Code: ½ 0.120 → C5-v6 0.143 → ⊕ 0.146 (rises with diminishing returns; NL ~flat).

**⊕** = code25b_clean_p2_15bt: the **1.6× rung** of the code-budget ladder. The code25b_clean base (⚙c, 24.65 B ≥2.7 code) continued on the FULL 15.39 B text (same phase-2 recipe as C5-v6 and ⊗). Removes ⊗'s text-budget confound: same 24.65 B code base as ⊗ but 15.39 B text instead of 5.0 B. With C5-v6 it isolates the code axis (both 15.4 B text); with ⊗ it isolates the text axis (both 24.65 B code). Caveat: ≥2.7 code (vs ≥3.0 for ½/C5-v6) — the near-flat 1×→1.6× Code (0.143→0.146) is saturation OR lower-quality dilution, not separable from this data.

**⬥ Smaller-scale models (300M / 600M) — cross-size scaling battery.** To test whether the 1.4B code→text findings hold as a *scaling law*, the core recipes were replicated at 300M and 600M at Chinchilla-optimal budget (≈20 × params: 300M → 6.0 B total tokens, 600M → 12.0 B). LLaMA family (300M = 12 layers / 768 hidden, seq 4096), same AdamW recipe as the 1.4B runs (LR 3e-4 cosine→0, wd 0.1, β=0.9/0.95, warmup 1%, batch 64 × seq 4096, data_seed 0), 1 node × 8 GPUs. Two-phase models init from a code-only phase-1 base with a **fresh** cosine (except `c5v2cont` = one continuous cosine across both phases). Scripts: `code_ladder/scripts/run_smallscale_{single_phase,code_p1,phase2,phase2_strict,c5v2_continuous}.py`; results in §3c. **The smaller-scale SP-NL runs use row-proportional (token-weighted) SlimPajama-NL, fixing the part-uniform ⚠ bug of the 1.4B A5-SP/C5-v4/C5-v5** — so these SP-NL numbers are on the intended distribution.

| Label | Sizes | Tokens D (300M / 600M) | Recipe |
|---|---|---|---|
| **a5** | 300M, 600M | 6.0 / 12.0 B | Single-phase, 100% DCLM — text-only baseline. |
| **a5sp** | 300M, 600M | 6.0 / 12.0 B | Single-phase, 100% SlimPajama-NL (row-proportional). |
| **code_p1_half** | 300M, 600M | 3.0 / 6.0 B | ½-budget **code-only** phase-1 base (80% code + 20% markup; C5-v3 phase-1 caches). Evaluated directly — no text phase. |
| **c5v3** | 300M, 600M | 6.0 / 12.0 B | code base → 90% DCLM + 10% (80% code + 20% markup), separate cosine (C5-v3 / 10%-replay). |
| **c5v4** | 300M | 6.0 B | c5v3 recipe but phase-2 text = SlimPajama-NL (C5-v4). |
| **c5v2cont** | 300M | 6.0 B | code→text as one **continuous** cosine (C5-v2 / continuous-cosine). |
| **c5v6** | 300M, 600M | 6.0 / 12.0 B | 70% DCLM + **30%** replay, separate cosine (C5-v6). |
| **c5v6_strict** | 300M, 600M | 6.0 / 12.0 B | C5-v6 with strict-prefix component ordering. |
| **c5v7** | 300M, 600M | 6.0 / 12.0 B | 50% DCLM + **50%** replay, separate cosine (C5-v7). |

</details>

---

## 3. Canonical results — all models

> **Structure:** four tables — **§3a** original 1.4B/30B recipe ablations, **§3b** the code↔text budget-scaling grid, **§3c** smaller-scale (300M/600M), and **§3d** (collapsible) misc / off-ramp probes. C5-v6 appears in both §3a (as a replay ablation) and §3b (as the 1× ladder anchor).
>
> **Symbols** (the only two kept): **★** = phase-2 code is *replay* of phase-1, not new; **⚠** = has a data-setup caveat — see §2. Everything else is spelled out in §2 or implied by which table a model sits in.

All numbers from our `lm-eval-harness` pipeline (lm_eval 0.4.11). Rows = tasks (header format `task[nshot]`). Columns = models. Accuracy metrics use `acc_norm` where reported in §1; `acc` otherwise. PPL is `bits_per_byte` (lower=better) for both paloma_macro and dclm_200m_val. Bolded = best in row. `—` = not run.

See §2 for full per-model recipe and caveat descriptions.

### 3a. Original ablations — 1.4B / 30B (old conclusions)

The founding recipe ablations at fixed ~30.8 B-token compute (text-only vs mixed vs staged code; code quality; separate vs continuous cosine; text source; replay fraction), plus the phi-1/1.5 external refs. The matched-budget "small" / ×16 probes and the 4B tokens-vs-params point now live in **§3d**. This is the table `eval_section3.py` manages.

| Task | A5 | B4 | C5 s1 | C5-v2 s1 | C5 | C5-v2 | C5-v3 p1 | C5-v8r p1 | C5-v3 | C5-v4 ⚠ | A5-SP ⚠ | C5-v6 ★ | C5-v5 ⚠ | C5-v6-NEW ⚠ | C5-v6 strict ★ | C5-v7 ★ | C5-v8r ⚠ | C5-v8r p2 | phi-1 | phi-1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Open-book** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| sciq[0] | 0.834 | 0.829 | 0.707 | 0.727 | 0.754 | 0.715 | 0.720 | 0.683 | 0.728 | 0.782 | 0.800 | 0.806 | 0.762 | 0.814 | 0.792 | 0.797 | 0.788 | 0.798 | 0.707 | 0.933 |
| boolq[0] | 0.563 | 0.598 | 0.619 | 0.593 | 0.623 | 0.580 | 0.595 | 0.622 | 0.443 | 0.598 | 0.565 | 0.546 | 0.604 | 0.582 | 0.583 | 0.589 | 0.605 | 0.601 | 0.450 | 0.746 |
| piqa[0] | 0.718 | 0.709 | 0.583 | 0.584 | 0.591 | 0.600 | 0.581 | 0.587 | 0.649 | 0.688 | 0.699 | 0.688 | 0.589 | 0.684 | 0.683 | 0.678 | 0.665 | 0.687 | 0.562 | 0.766 |
| openbookqa_fact[0] | 0.430 | 0.430 | 0.306 | 0.312 | 0.316 | 0.326 | 0.320 | 0.300 | 0.378 | 0.388 | 0.400 | 0.386 | 0.308 | 0.396 | 0.390 | 0.394 | 0.378 | 0.374 | 0.316 | 0.530 |
| **Mean Open-book** | *0.636* | *0.642* | *0.554* | *0.554* | *0.571* | *0.555* | *0.554* | *0.548* | *0.549* | *0.614* | *0.616* | *0.607* | *0.566* | *0.619* | *0.612* | *0.615* | *0.609* | *0.615* | *0.509* | *0.744* |
| **Closed-book NL** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| arc_easy[25] | 0.630 | 0.610 | 0.373 | 0.406 | 0.389 | 0.420 | 0.397 | 0.374 | 0.536 | 0.552 | 0.576 | 0.590 | 0.415 | 0.584 | 0.574 | 0.572 | 0.529 | 0.552 | 0.386 | 0.802 |
| hellaswag[10] | 0.497 | 0.464 | 0.292 | 0.304 | 0.298 | 0.311 | 0.297 | 0.288 | 0.377 | 0.415 | 0.454 | 0.427 | 0.313 | 0.428 | 0.404 | 0.400 | 0.391 | 0.419 | 0.301 | 0.635 |
| winogrande[5] | 0.541 | 0.515 | 0.513 | 0.507 | 0.517 | 0.484 | 0.514 | 0.503 | 0.504 | 0.502 | 0.522 | 0.521 | 0.488 | 0.509 | 0.517 | 0.517 | 0.530 | 0.516 | 0.496 | 0.711 |
| mmlu_text[0] | 0.290 | 0.286 | 0.244 | 0.251 | 0.247 | 0.250 | 0.247 | 0.241 | 0.263 | 0.273 | 0.271 | 0.277 | 0.253 | 0.280 | 0.271 | 0.276 | 0.269 | 0.276 | 0.243 | 0.337 |
| commonsense_qa_text[5] | 0.523 | 0.508 | 0.290 | 0.326 | 0.305 | 0.340 | 0.309 | 0.279 | 0.432 | 0.459 | 0.450 | 0.484 | 0.342 | 0.467 | 0.446 | 0.456 | 0.431 | 0.444 | 0.271 | 0.609 |
| social_iqa[0] | 0.415 | 0.400 | 0.346 | 0.360 | 0.354 | 0.359 | 0.358 | 0.346 | 0.383 | 0.387 | 0.394 | 0.396 | 0.364 | 0.396 | 0.384 | 0.389 | 0.386 | 0.386 | 0.364 | 0.523 |
| lambada_openai[0] | 0.519 | 0.496 | 0.144 | 0.212 | 0.185 | 0.250 | 0.187 | 0.138 | 0.357 | 0.435 | 0.409 | 0.469 | 0.249 | 0.469 | 0.441 | 0.446 | 0.412 | 0.444 | 0.106 | 0.527 |
| copa[0] | 0.740 | 0.690 | 0.550 | 0.560 | 0.540 | 0.550 | 0.550 | 0.540 | 0.680 | 0.680 | 0.690 | 0.700 | 0.510 | 0.760 | 0.670 | 0.710 | 0.750 | 0.690 | 0.530 | 0.800 |
| wsc273[0] | 0.586 | 0.575 | 0.535 | 0.524 | 0.516 | 0.502 | 0.520 | 0.495 | 0.505 | 0.586 | 0.553 | 0.601 | 0.516 | 0.593 | 0.557 | 0.564 | 0.538 | 0.564 | 0.502 | 0.769 |
| storycloze_2018_local[0] | 0.663 | 0.654 | 0.516 | 0.535 | 0.529 | 0.545 | 0.541 | 0.502 | 0.603 | 0.636 | 0.637 | 0.642 | 0.549 | 0.646 | 0.628 | 0.637 | 0.618 | — | 0.531 | 0.531 |
| quac_first_turn[0] | 0.176 | 0.179 | 0.126 | 0.123 | 0.146 | 0.136 | 0.122 | 0.130 | 0.110 | 0.197 | 0.201 | 0.161 | 0.135 | 0.179 | 0.142 | 0.150 | 0.149 | — | 0.169 | 0.169 |
| **Mean Closed-book NL** | *0.507* | *0.489* | *0.357* | *0.373* | *0.366* | *0.377* | *0.367* | *0.349* | *0.432* | *0.466* | *0.469* | *0.479* | *0.376* | *0.483* | *0.458* | *0.465* | *0.455* | *0.477* | *0.354* | *0.583* |
| **Code** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| humaneval[0] (lm-eval) | 0.006 | 0.104 | 0.037 | 0.268 | 0.061 | 0.280 | 0.256 | 0.049 | 0.165 | 0.177 | 0.000 | 0.213 | 0.305 | 0.195 | 0.207 | 0.250 | 0.134 | 0.140 | 0.494 | 0.335 |
| humaneval[0] (bigcode) | 0.000 | 0.000 | 0.012 | 0.073 | 0.037 | 0.055 | 0.122 | 0.006 | 0.024 | 0.067 | 0.000 | 0.012 | 0.061 | 0.006 | 0.043 | 0.085 | 0.073 | 0.116 | 0.543 | 0.341 |
| mbpp[3] | 0.000 | 0.060 | 0.050 | 0.212 | 0.104 | 0.298 | 0.208 | 0.046 | 0.048 | 0.124 | 0.000 | 0.204 | 0.306 | 0.138 | 0.146 | 0.230 | 0.030 | 0.116 | 0.416 | 0.342 |
| **Mean Code** | *0.002* | *0.055* | *0.033* | *0.184* | *0.067* | *0.211* | *0.195* | *0.034* | *0.079* | *0.123* | *0.000* | *0.143* | *0.224* | *0.113* | *0.132* | *0.188* | *0.079* | *0.124* | *0.484* | *0.339* |
| **Perplexity (lower=better)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dclm_200m_val (bpb) | 0.923 | 0.942 | 1.313 | 1.308 | 1.286 | 1.260 | 1.308 | 1.318 | 1.110 | 1.019 | 1.054 | 0.955 | 1.245 | 0.954 | 0.997 | 0.973 | 1.046 | — | 1.636 | 1.041 |
| paloma_macro (bpb) | 1.077 | 1.074 | 1.374 | 1.370 | 1.326 | 1.326 | 1.377 | 1.376 | 1.315 | 1.098 | 1.142 | 1.087 | 1.331 | 1.082 | 1.121 | 1.099 | 1.150 | — | 1.738 | 1.174 |

### 3b. Scaling — code ↔ text budget (1.4B / 30B+)

The code/text budget-scaling grid: the code-only bases (`code25b`, `code25b-clean`) and the code-budget ladder at fixed 15.4 B text — `code 0.5×`, `C5-v6 1×` (= its §3a column), `code 1.6×` — plus the `code diag` point (24.65 B code → 5 B text). See §2 for the per-model recipes.

| Task | code25b | code25b-clean | code 0.5× | C5-v6 1× | code 1.6× | code diag |
|---|---:|---:|---:|---:|---:|---:|
| **Open-book** | | | | | | |
| sciq[0] | 0.686 | 0.737 | 0.788 | 0.806 | 0.806 | 0.783 |
| boolq[0] | 0.620 | 0.547 | 0.530 | 0.546 | 0.502 | 0.560 |
| piqa[0] | 0.567 | 0.600 | 0.692 | 0.688 | 0.679 | 0.659 |
| openbookqa_fact[0] | 0.306 | 0.308 | 0.408 | 0.386 | 0.392 | 0.386 |
| **Mean Open-book** | *0.545* | *0.548* | *0.604* | *0.607* | *0.595* | *0.597* |
| **Closed-book NL** | | | | | | |
| arc_easy[25] | 0.361 | 0.403 | 0.584 | 0.590 | 0.588 | 0.531 |
| hellaswag[10] | 0.291 | 0.305 | 0.423 | 0.427 | 0.434 | 0.369 |
| winogrande[5] | 0.500 | 0.501 | 0.519 | 0.521 | 0.508 | 0.506 |
| mmlu_text[0] | 0.243 | 0.248 | 0.277 | 0.277 | 0.280 | 0.267 |
| commonsense_qa_text[5] | 0.284 | 0.310 | 0.454 | 0.484 | 0.486 | 0.437 |
| social_iqa[0] | 0.347 | 0.358 | 0.393 | 0.396 | 0.385 | 0.383 |
| lambada_openai[0] | 0.140 | 0.232 | 0.469 | 0.469 | 0.468 | 0.411 |
| copa[0] | 0.560 | 0.530 | 0.660 | 0.700 | 0.670 | 0.660 |
| wsc273[0] | 0.549 | 0.531 | 0.575 | 0.601 | 0.557 | 0.546 |
| storycloze_2018_local[0] | 0.509 | 0.548 | 0.638 | 0.642 | 0.644 | 0.621 |
| quac_first_turn[0] | 0.135 | 0.140 | 0.170 | 0.161 | 0.178 | 0.147 |
| **Mean Closed-book NL** | *0.356* | *0.373* | *0.469* | *0.479* | *0.473* | *0.443* |
| **Code** | | | | | | |
| humaneval[0] (lm-eval) | 0.232 | 0.226 | 0.189 | 0.213 | 0.226 | 0.232 |
| humaneval[0] (bigcode) | 0.061 | 0.128 | 0.012 | 0.012 | 0.018 | 0.116 |
| mbpp[3] | 0.234 | 0.194 | 0.158 | 0.204 | 0.194 | 0.214 |
| **Mean Code** | *0.176* | *0.183* | *0.120* | *0.143* | *0.146* | *0.187* |
| **Perplexity (lower=better)** | | | | | | |
| dclm_200m_val (bpb) | 1.402 | 1.256 | 0.970 | 0.955 | 0.964 | 1.020 |
| paloma_macro (bpb) | 1.457 | 1.337 | 1.091 | 1.087 | 1.079 | 1.130 |

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

### 3c. Smaller-scale results (300M / 600M)

Cross-size replication of the code→text battery (see §2 for recipes). Same lm-eval v2 pipeline + aux runners as §3, extracted from `outputs/eval_results/{v2,paloma,gsm,aryabumi_nl,quac}_{300m,600m}_*` with the same metric-fallback logic as the 1.4B table (via `code_ladder/eval/eval_section3.py`). `—` = not run: **mmlu** crashed on the pre-fix NCCL gather-OOM bug and was only recovered for the four 600M u-shape models; **dclm_200m_val bpb** was not evaluated at these sizes. Per-axis writeups live in `outputs/eval_results/COMPARISON_{300m,600m}_*.md`.

Column order: text-only baselines (a5, a5sp), the ½-budget code base (code_p1_half), then the code→text variants (c5v3=10%-replay/DCLM, c5v4=SP-NL, c5v2cont=continuous cosine, c5v6=30%, c5v6_strict, c5v7=50%). c5v4 and c5v2cont are 300M-only.

| Task | 300M a5 | 300M a5sp | 300M codeP1 | 300M c5v3 | 300M c5v4 | 300M c5v2cont | 300M c5v6 | 300M c5v6-str | 300M c5v7 | 600M a5 | 600M a5sp | 600M codeP1 | 600M c5v3 | 600M c5v6 | 600M c5v6-str | 600M c5v7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Open-book** | | | | | | | | | | | | | | | | |
| sciq[0] | 0.676 | 0.645 | 0.485 | 0.656 | 0.654 | 0.546 | 0.660 | 0.656 | 0.652 | 0.765 | 0.734 | 0.577 | 0.770 | 0.741 | 0.737 | 0.736 |
| boolq[0] | 0.589 | 0.535 | 0.619 | 0.571 | 0.597 | 0.623 | 0.596 | 0.599 | 0.603 | 0.581 | 0.535 | 0.621 | 0.592 | 0.596 | 0.603 | 0.581 |
| piqa[0] | 0.624 | 0.603 | 0.543 | 0.605 | 0.610 | 0.548 | 0.604 | 0.596 | 0.594 | 0.670 | 0.662 | 0.563 | 0.664 | 0.653 | 0.649 | 0.646 |
| openbookqa_fact[0] | 0.316 | 0.338 | 0.294 | 0.328 | 0.316 | 0.292 | 0.304 | 0.328 | 0.304 | 0.386 | 0.352 | 0.286 | 0.380 | 0.376 | 0.386 | 0.360 |
| **Mean Open-book** | *0.551* | *0.530* | *0.485* | *0.540* | *0.544* | *0.502* | *0.541* | *0.545* | *0.538* | *0.601* | *0.571* | *0.512* | *0.602* | *0.592* | *0.594* | *0.581* |
| **Closed-book NL** | | | | | | | | | | | | | | | | |
| arc_easy[25] | 0.454 | 0.416 | 0.298 | 0.441 | 0.403 | 0.319 | 0.428 | 0.434 | 0.422 | 0.563 | 0.519 | 0.334 | 0.535 | 0.511 | 0.518 | 0.495 |
| hellaswag[10] | 0.299 | 0.295 | 0.267 | 0.285 | 0.284 | 0.272 | 0.284 | 0.282 | 0.278 | 0.390 | 0.365 | 0.275 | 0.358 | 0.343 | 0.346 | 0.329 |
| winogrande[5] | 0.507 | 0.519 | 0.501 | 0.519 | 0.523 | 0.481 | 0.513 | 0.511 | 0.519 | 0.510 | 0.523 | 0.518 | 0.509 | 0.503 | 0.504 | 0.503 |
| mmlu_text[0] | 0.255 | 0.252 | 0.234 | 0.252 | 0.251 | 0.240 | 0.253 | 0.251 | 0.250 | 0.277 | 0.263 | 0.244 | 0.263 | 0.262 | 0.261 | 0.262 |
| commonsense_qa_text[5] | 0.350 | 0.308 | 0.229 | 0.334 | 0.315 | 0.251 | 0.318 | 0.320 | 0.301 | 0.443 | 0.389 | 0.267 | 0.423 | 0.409 | 0.429 | 0.392 |
| social_iqa[0] | 0.371 | 0.371 | 0.333 | 0.361 | 0.365 | 0.343 | 0.355 | 0.357 | 0.355 | 0.400 | 0.394 | 0.351 | 0.387 | 0.387 | 0.393 | 0.386 |
| lambada_openai[0] | 0.297 | 0.219 | 0.049 | 0.276 | 0.243 | 0.077 | 0.256 | 0.256 | 0.234 | 0.432 | 0.327 | 0.099 | 0.394 | 0.376 | 0.374 | 0.353 |
| copa[0] | 0.670 | 0.660 | 0.550 | 0.630 | 0.610 | 0.590 | 0.600 | 0.610 | 0.640 | 0.710 | 0.700 | 0.530 | 0.720 | 0.700 | 0.710 | 0.710 |
| wsc273[0] | 0.542 | 0.535 | 0.502 | 0.527 | 0.513 | 0.516 | 0.498 | 0.505 | 0.495 | 0.542 | 0.546 | 0.513 | 0.568 | 0.542 | 0.542 | 0.531 |
| storycloze_2018_local[0] | 0.579 | 0.569 | 0.516 | 0.561 | 0.557 | 0.526 | 0.558 | 0.550 | 0.553 | 0.633 | 0.612 | 0.526 | 0.612 | 0.615 | 0.613 | 0.602 |
| quac_first_turn[0] | 0.100 | 0.124 | 0.068 | 0.101 | 0.113 | 0.071 | 0.100 | 0.094 | 0.091 | 0.136 | 0.152 | 0.090 | 0.133 | 0.135 | 0.139 | 0.131 |
| **Mean Closed-book NL** | *0.402* | *0.388* | *0.322* | *0.390* | *0.380* | *0.335* | *0.378* | *0.379* | *0.376* | *0.458* | *0.435* | *0.341* | *0.446* | *0.435* | *0.439* | *0.427* |
| **Aggregate** | | | | | | | | | | | | | | | | |
| **Code** | | | | | | | | | | | | | | | | |
| humaneval[0] (lm-eval) | 0.000 | 0.000 | 0.104 | 0.049 | 0.049 | 0.134 | 0.049 | 0.085 | 0.098 | 0.000 | 0.000 | 0.140 | 0.110 | 0.146 | 0.152 | 0.159 |
| humaneval[0] (bigcode) | 0.000 | 0.000 | 0.018 | 0.000 | 0.000 | 0.079 | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 0.134 | 0.061 | 0.085 | 0.085 | 0.079 |
| mbpp[3] | 0.000 | 0.000 | 0.016 | 0.004 | 0.006 | 0.078 | 0.024 | 0.016 | 0.020 | 0.000 | 0.000 | 0.110 | 0.046 | 0.078 | 0.066 | 0.116 |
| **Mean Code** | *0.000* | *0.000* | *0.046* | *0.018* | *0.018* | *0.097* | *0.024* | *0.034* | *0.041* | *0.000* | *0.000* | *0.128* | *0.072* | *0.103* | *0.101* | *0.118* |
| **Perplexity (lower=better)** | | | | | | | | | | | | | | | | |
| dclm_200m_val (bpb) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| paloma_macro (bpb) | 1.320 | 1.382 | 1.677 | 1.309 | 1.325 | 1.600 | 1.318 | 1.318 | 1.332 | 1.279 | 1.355 | 1.720 | 1.174 | 1.181 | 1.183 | 1.197 |

**Cross-scale headline (see §3c vs §3):** the two *positive* 1.4B code→text findings do **not** replicate downward — (1) the "30% replay sweet spot" is a pure monotonic trade-off at 600M (Code rises 10→30→50%, NL falls; no peak); (2) "SP-NL > DCLM over a code prior" flips at 300M (c5v3 DCLM ≥ c5v4 SP-NL). What *does* replicate: DCLM > SP-NL single-phase (a5 > a5sp), and continuous-cosine-wins-Code / separate-cosine-wins-NL (c5v2cont Code 0.097 vs c5v6 0.024; c5v6 NL higher).

### 3d. Misc / off-ramp probes

<details>
<summary>Matched-budget "small" scale-downs, the ×16 short-budget baselines, and the 4B tokens-vs-params point — not part of the ~30B recipe ablations. Click to expand.</summary>

| Task | base | code25 v2 | C5-v2-sm s1 | C5-v2-sm | C5-v3-sm p1 | C5-v3-sm | 4B |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Open-book** |  |  |  |  |  |  |  |
| sciq[0] | 0.652 | 0.590 | 0.601 | 0.601 | 0.541 | 0.712 | 0.824 |
| boolq[0] | 0.502 | 0.567 | 0.614 | 0.614 | 0.618 | 0.614 | 0.552 |
| piqa[0] | 0.634 | 0.606 | 0.577 | 0.577 | 0.554 | 0.647 | 0.697 |
| openbookqa_fact[0] | 0.336 | 0.312 | 0.294 | 0.294 | 0.296 | 0.356 | 0.426 |
| **Mean Open-book** | *0.531* | *0.519* | *0.521* | *0.521* | *0.502* | *0.582* | *0.625* |
| **Closed-book NL** |  |  |  |  |  |  |  |
| arc_easy[25] | 0.401 | 0.388 | 0.350 | 0.350 | 0.322 | 0.485 | 0.612 |
| hellaswag[10] | 0.348 | 0.321 | 0.280 | 0.280 | 0.276 | 0.322 | 0.466 |
| winogrande[5] | 0.504 | 0.500 | 0.507 | 0.507 | 0.502 | 0.513 | 0.511 |
| mmlu_text[0] | 0.250 | 0.248 | 0.238 | 0.244 | 0.241 | 0.261 | 0.284 |
| commonsense_qa_text[5] | 0.339 | 0.296 | 0.254 | 0.277 | 0.247 | 0.387 | 0.524 |
| social_iqa[0] | 0.366 | 0.362 | 0.342 | 0.342 | 0.346 | 0.382 | 0.407 |
| lambada_openai[0] | 0.238 | 0.197 | 0.124 | 0.124 | 0.077 | 0.349 | 0.494 |
| copa[0] | 0.620 | 0.620 | 0.540 | 0.540 | 0.570 | 0.660 | 0.740 |
| wsc273[0] | 0.516 | 0.527 | 0.484 | 0.498 | 0.487 | 0.524 | 0.634 |
| storycloze_2018_local[0] | 0.591 | 0.568 | 0.534 | 0.528 | 0.515 | 0.586 | 0.658 |
| quac_first_turn[0] | 0.142 | 0.136 | 0.077 | 0.092 | 0.079 | 0.122 | 0.168 |
| **Mean Closed-book NL** | *0.392* | *0.378* | *0.339* | *0.344* | *0.333* | *0.417* | *0.500* |
| **Code** |  |  |  |  |  |  |  |
| humaneval[0] (lm-eval) | 0.000 | 0.012 | 0.159 | 0.159 | 0.116 | 0.116 | 0.000 |
| humaneval[0] (bigcode) | 0.000 | 0.000 | 0.098 | 0.098 | 0.055 | 0.030 | 0.000 |
| mbpp[3] | 0.000 | 0.000 | 0.130 | 0.130 | 0.088 | 0.050 | 0.000 |
| **Mean Code** | *0.000* | *0.004* | *0.129* | *0.129* | *0.086* | *0.065* | *0.000* |
| **Perplexity (lower=better)** |  |  |  |  |  |  |  |
| dclm_200m_val (bpb) | 1.332 | 1.504 | 1.514 | 1.449 | 1.525 | 1.077 | 0.947 |
| paloma_macro (bpb) | 1.631 | 1.824 | 1.639 | 1.566 | 1.582 | 1.216 | 1.114 |

</details>

## Updating this doc

When a new model is trained or a new eval is added, update §1 (models) and §3 (results) with the new row/column. Add a brief follow-up entry in `EXPERIMENT_LOG.md` pointing here. Chronological narrative stays in `EXPERIMENT_LOG.md`; canonical reference stays here.
