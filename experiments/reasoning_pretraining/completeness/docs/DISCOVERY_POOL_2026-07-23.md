# Discovery pool — 2026-07-23 wide zero-seed sweep (rounds 1+2)

Candidate pool for the reasoning-in-pretraining literature review (`../REASONING_CONTENT_LIT.md`). Nothing is
dropped — every candidate the searchers returned is listed here with its triage verdict, so the pool stays
queryable. Triage is CHEAP (title + one-line relevance only; no paper was fetched) — a `skip` here is a weak
signal, a `must_read` is a candidate for the next adversarial full-read batch, not a settled judgment.

**Provenance.** Round 1 (`wf_438a8a3c-3b1`): 12 concept angles × 4–6 hand-seeded query variations, 12 zero-seed
searchers → 300 raw → 246 deduped, triaged in-workflow. Round 2 recall pass (`wf_66465130-feb`): per-angle
concept-expansion agents rewrote each angle into 12–15 new queries (cross-community synonyms, sub-concepts,
method-name families, benchmark vocabulary, adjacent-field phrasings), 2 searchers per angle → 490 raw → 287
new titles; increment triaged by `wf_6e18afd1-6ab`. Zero-seed rule throughout: no paper names in any query.

**Counts.** 533 unique titles: **126 must-read**, 248 maybe, 26 known (already covered by the lit review), 121 skip, 12 untriaged (triage agents failed to return these — re-triage before dismissing).

Must-read by bucket: H1.1 25, H1.2 12, H1.3 26, H2.4 14, H2.5 25, H2.6 12, H2.7 12, other 0


## Must-read shortlist


### H1.1 — shortcuts instead of reasoning (25)

- **Shortcut Learning in Deep Neural Networks** (Robert Geirhos, 2020, Nature Machine Intelligence) — <https://arxiv.org/abs/2004.07780>
  - Canonical conceptual framing for H1's shortcut-wins story *(r1: shortcut-learning)*
- **The Pitfalls of Simplicity Bias in Neural Networks** (Harshay Shah, 2020, NeurIPS) — <https://arxiv.org/abs/2006.07710>
  - Mechanism for why simple shortcuts beat complex features (WON'T vs CAN'T) *(r1: shortcut-learning)*
- **Teaching Pre-Trained Models to Systematically Reason Over Implicit Knowledge** (Alon Talmor, 2020, NeurIPS) — <https://proceedings.neurips.cc/paper/2020/file/e992111e4ab9985366e806733383bd8c-Paper.pdf>
  - Combining explicit statements with implicit weights knowledge — bears on can't-vs-won't *(r2: enthymemes-completeness)*
- **Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little** (Koustuv Sinha, 2021, EMNLP 2021) — <https://aclanthology.org/2021.emnlp-main.230/>
  - Direct evidence pretraining objective is satisfied by distributional shortcuts over structure. *(r2: shortcut-learning)*
- **Gradient Starvation: A Learning Proclivity in Neural Networks** (Mohammad Pezeshki, 2021, NeurIPS 2021) — <https://arxiv.org/pdf/2011.09468>
  - Optimization-level mechanism for why cross-entropy prefers shortcuts — core WON'T account. *(r2: shortcut-learning)*
- **Transformers Learn Shortcuts to Automata** (Bingbin Liu, 2022, ICLR 2023 / arXiv) — <https://arxiv.org/pdf/2210.10749>
  - Core evidence transformers prefer shallow shortcuts over full computation *(r2: mechanistic-formation)*
- **Faith and Fate: Limits of Transformers on Compositionality** (Nouha Dziri, 2023, NeurIPS) — <https://arxiv.org/abs/2305.18654>
  - Direct evidence of subgraph pattern-matching instead of full inference *(r1: shortcut-learning)*
- **Physics of Language Models: Part 3.2, Knowledge Manipulation** (Zeyuan Allen-Zhu, 2023, ICLR 2025) — <https://arxiv.org/abs/2309.14402>
  - Controlled CAN'T-vs-WON'T separation; CoT-in-training-data requirement bears on H2.6 *(r1: latent-multihop)*
- **The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"** (Lukas Berglund, 2023, ICLR 2024) — <https://arxiv.org/abs/2309.12288>
  - Simplest case of text implying an inference next-token training never learns *(r1: latent-multihop)*
- **Towards a Mechanistic Interpretation of Multi-Step Reasoning Capabilities of Language Models** (Yifan Hou, 2023, arXiv/EMNLP) — <https://arxiv.org/abs/2310.14491>
  - Evidence whether next-token training yields real multi-step inference vs shortcuts *(r2: format-internalization)*
- **The Pitfalls of Next-Token Prediction** (Gregor Bachmann, 2024, ICML) — <https://arxiv.org/abs/2403.06963>
  - Objective-level argument that NTP itself learns Clever Hans shortcuts *(r1: shortcut-learning)*
- **A Peek into Token Bias: Large Language Models Are Not Yet Genuine Reasoners** (Bowen Jiang, 2024, EMNLP) — <https://arxiv.org/abs/2406.11050>
  - Direct CAN'T-vs-WON'T evidence: performance hinges on token bias, not inference. *(r2: shortcut-learning)*
- **Premise Order Matters in Reasoning with Large Language Models** (Xinyun Chen, 2024, ICML) — <https://arxiv.org/html/2402.08939>
  - Surface-order dependence over logical structure — clean H1 shortcut signal. *(r2: shortcut-learning)*
- **Investigating Multi-Hop Factual Shortcuts in Knowledge Editing of Large Language Models** (Tianjie Ju, 2024, ACL 2024 / arXiv) — <https://arxiv.org/abs/2402.11900>
  - Direct evidence that pretraining co-occurrence shortcuts replace latent hopping *(r2: latent-multihop, cot-faithfulness)*
- **Do LLMs Overcome Shortcut Learning? An Evaluation of Shortcut Challenges in Large Language Models** (Yu Yuan, 2024, arXiv) — <https://arxiv.org/abs/2410.13343>
  - Directly benchmarks shortcut reliance vs genuine reasoning — core H1 shortcut evidence. *(r2: cot-faithfulness, fresh-2026)*
- **Reasoning Bias of Next Token Prediction Training** (2025, preprint) — <https://arxiv.org/abs/2502.02007>
  - Directly analyzes how the NTP objective biases learned reasoning *(r1: shortcut-learning, hard-tokens)*
- **Implicit Reasoning in Transformers is Reasoning through Shortcuts** (Tianhe Lin, 2025, ACL Findings 2025) — <https://arxiv.org/abs/2503.07604>
  - Direct evidence shortcut pattern-matching wins over full inference *(r1: latent-multihop, mechanistic-formation, fresh-2026, format-internalization)*
- **The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction** (Yihuai Hong, 2025, preprint (arXiv)) — <https://arxiv.org/html/2503.23084>
  - Mechanistic handle on retrieval-vs-reasoning mode choice — the shortcut decision itself *(r1: mechanistic-formation)*
- **Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens** (Chengshuai Zhao, 2025, preprint) — <https://arxiv.org/abs/2508.01191>
  - CoT as pattern-matching bounded by training data distribution — bridges shortcut behavior to pretraining data (H2) *(r1: cot-faithfulness, fresh-2026)*
- **Rethinking the Chain-of-Thought: The Roles of In-Context Learning and Pretrained Priors** (unknown, 2025, preprint) — <https://arxiv.org/abs/2509.01236>
  - CoT behavior governed by pretrained priors — directly on what pretraining instills vs prompting elicits (H1). *(r1: format-internalization)*
- **How Reinforcement Learning After Next-Token Prediction Facilitates Learning** (2025, preprint) — <https://arxiv.org/abs/2510.11495>
  - Theory on tasks where NTP alone fails but RL after NTP generalizes — directly on shortcut-vs-full-inference under NTP. *(r1: hard-tokens)*
- **RLP: Reinforcement as a Pretraining Objective** (2025, preprint) — <https://arxiv.org/abs/2510.01265>
  - Alternative pretraining objective targeting NTP shortcut failure — direct H1 remedy candidate *(r1: fresh-2026)*
- **Reinforcement Learning on Pre-Training Data** (2025, preprint) — <https://arxiv.org/abs/2509.19249>
  - RL directly on pretraining text as alternative to NTP shortcuts, with scaling laws — core H1 *(r1: fresh-2026)*
- **Composition Collapse: Stable Factual Knowledge Does Not Imply Compositional Reasoning** ((see arXiv), 2026, preprint (arXiv)) — <https://arxiv.org/pdf/2605.26789>
  - Directly separates CAN'T vs WON'T: facts stored but not composed *(r1: mechanistic-formation, latent-multihop)*
- **How Does Unfaithful Reasoning Emerge from Autoregressive Training? A Study of Synthetic Experiments** (Fuxin Wang, 2026, arXiv) — <https://arxiv.org/abs/2602.01017>
  - Directly asks how next-token pretraining produces shortcut reasoning in controlled settings *(r2: cot-faithfulness)*

### H1.2 — latent multi-hop / one-pass composition (12)

- **Probing for Bridging Inference in Transformer Language Models** (Onkar Pandit, 2021, NAACL) — <https://arxiv.org/abs/2104.09400>
  - probes whether pretraining internalizes implicit bridging inferences — direct H1 evidence *(r1: enthymemes-completeness)*
- **Measuring and Narrowing the Compositionality Gap in Language Models** (Ofir Press, 2022, EMNLP Findings 2023) — <https://arxiv.org/abs/2210.03350>
  - Canonical compositionality-gap framing; gap not shrinking with scale is core H1 *(r1: latent-multihop)*
- **The Two-Hop Curse / Lessons from Studying Two-Hop Latent Reasoning** (Mikita Balesni, 2024, preprint (Apollo Research)) — <https://arxiv.org/abs/2411.16353>
  - Sharp CAN'T vs data-arrangement result; co-occurrence lever for H2.5 too *(r1: latent-multihop)*
- **Lessons from Studying Two-Hop Latent Reasoning** (Mikita Balesni, 2024, preprint (arXiv)) — <https://arxiv.org/pdf/2411.16353>
  - Latent two-hop needs training co-occurrence — direct H1 shortcut/exposure evidence *(r1: mechanistic-formation, cot-faithfulness)*
- **Let's Think Dot by Dot: Hidden Computation in Transformer Language Models** (Jacob Pfau, 2024, preprint) — <https://arxiv.org/abs/2404.15758>
  - Key evidence on when computation is genuinely latent vs token-carried, and that dense supervision is needed — bears on H1 training-signal question. *(r1: format-internalization, hard-tokens)*
- **Think-to-Talk or Talk-to-Think? When LLMs Come Up with an Answer in Multi-Step Reasoning** (Keito Kudo, 2024, arXiv) — <https://arxiv.org/abs/2412.01113>
  - Tests whether answers are computed latently before CoT — directly on latent multi-hop vs shortcut *(r2: format-internalization)*
- **Do Larger Language Models Generalize Better? A Scaling Law for Implicit Reasoning at Pretraining Time** ((see arXiv page), 2025, preprint) — <https://arxiv.org/abs/2504.03635>
  - U-shaped implicit-reasoning-vs-size at pretraining; memorization crowds out inference — direct H1 shortcut tension *(r1: enthymemes-completeness)*
- **Unveiling the Mechanisms of Multi-Hop Reasoning in Transformers via Identity Bridge** (Pengxiao Lin, 2025, arXiv) — <https://arxiv.org/abs/2509.24653>
  - Circuit-level cause of two-hop failure plus a data fix (identity-bridge) for OOD composition. *(r2: latent-multihop)*
- **Examining Two Hop Reasoning Through Information Content Scaling** (David Johnston, 2025, arXiv) — <https://arxiv.org/abs/2502.03490>
  - Capacity account of when latent two-hop is learnable — CAN'T side of H1. *(r2: latent-multihop)*
- **SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models** (Ken Gu, 2025, arXiv) — <https://arxiv.org/abs/2510.24427>
  - Clean CAN'T-vs-WON'T separation design via parallel corpora *(r2: latent-multihop)*
- **How does Transformer Learn Implicit Reasoning?** ((unknown), 2025, arXiv) — <https://www.themoonlight.io/en/review/how-does-transformer-learn-implicit-reasoning>
  - When latent composition emerges in training; close to Grokked Transformers but distinct study *(r2: latent-multihop)*
- **The Scaling Properties of Implicit Deductive Reasoning in Transformers** (Enrico Vompa, 2026, arXiv) — <https://arxiv.org/abs/2605.04330>
  - How no-scratchpad deduction scales with model/data — directly on H1 can't-vs-won't boundary *(r2: format-internalization)*

### H1.3 — persistence through post-training (26)

- **The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL)** (Bill Yuchen Lin, 2023, arXiv) — <https://arxiv.org/abs/2312.01552>
  - Alignment is elicitation not new capability — direct persistence evidence. *(r2: persistence-posttraining)*
- **The False Promise of Imitating Proprietary LLMs** (Arnav Gudibande, 2023, ICLR 2024) — <https://arxiv.org/abs/2305.15717>
  - Post-training can't add capability unsupported in pretraining. *(r2: persistence-posttraining)*
- **Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking** (Nikhil Prakash, 2024, ICLR 2024 (arXiv:2402.14811)) — <https://arxiv.org/abs/2402.14811>
  - Mechanistic evidence fine-tuning reuses pretrained circuits — persistence at circuit level. *(r1: persistence-posttraining)*
- **Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining** (Rosie Zhao, 2025, preprint (arXiv:2504.07912)) — <https://arxiv.org/abs/2504.07912>
  - Directly ties RL persistence to pretraining mixture — core H1.3 evidence *(r1: persistence-posttraining)*
- **Base Models Know How to Reason, Thinking Models Learn When** (Constantin Venhoff, 2025, preprint (arXiv:2510.07364)) — <https://arxiv.org/abs/2510.07364>
  - Elicitation-not-creation at mechanism level — persistence question head-on *(r1: persistence-posttraining)*
- **Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs** (Xumeng Wen, 2025, preprint (arXiv:2506.14245)) — <https://arxiv.org/abs/2506.14245>
  - Direct rebuttal to Yue boundedness (already covered) — needed for balance *(r1: persistence-posttraining)*
- **Why Distillation can Outperform Zero-RL: The Role of Flexible Reasoning** (Xiao Hu, 2025, preprint (arXiv:2505.21067)) — <https://arxiv.org/pdf/2505.21067>
  - Directly on RL bounded by base vs distillation injecting new reasoning — elicitation/creation boundary. *(r1: persistence-posttraining)*
- **Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination** (Mingqi Wu, 2025, preprint (arXiv:2507.10532)) — <https://arxiv.org/abs/2507.10532>
  - RL gains as elicitation of memorized pretraining content — CAN'T/WON'T and persistence question. *(r1: persistence-posttraining)*
- **Demystifying Long Chain-of-Thought Reasoning in LLMs** (Edward Yeo, 2025, preprint (arXiv:2502.03373)) — <https://arxiv.org/abs/2502.03373>
  - Traces long-CoT to latent pretraining data (forum dialogue) unlocked by RL — pretraining-data provenance of reasoning. *(r1: persistence-posttraining)*
- **Assessing Robustness to Spurious Correlations in Post-Training Language Models** ((see paper), 2025, preprint (arXiv:2505.05704)) — <https://arxiv.org/abs/2505.05704>
  - Directly tests whether shortcut reliance (H1 WON'T) persists through SFT/DPO/KTO. *(r1: persistence-posttraining)*
- **Reasoning Models Don't Always Say What They Think** (Yanda Chen, 2025, preprint (Anthropic)) — <https://arxiv.org/abs/2505.05410>
  - Unfaithful shortcut use persists after RL — directly H1's persistence-through-post-training question *(r1: cot-faithfulness)*
- **On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models** (2025, ICML 2026 Spotlight) — <https://arxiv.org/abs/2512.07783>
  - Controlled study of mid-training reasoning data vs RL under fixed compute — directly on persistence and H2.5 *(r1: fresh-2026)*
- **Spurious Rewards: Rethinking Training Signals in RLVR** (Rulin Shao, 2025, arXiv) — <https://arxiv.org/abs/2506.10947>
  - RLVR gains from random rewards show RL elicits latent pretrained behavior — directly on persistence/elicitation vs new capability. *(r2: persistence-posttraining)*
- **Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs** (Kanishk Gandhi, 2025, arXiv) — <https://arxiv.org/abs/2503.01307>
  - RL self-improvement bounded by base-model behaviors (priming unlocks gains) — squarely CAN'T-vs-WON'T and persistence through post-training. *(r2: persistence-posttraining)*
- **Understanding R1-Zero-Like Training: A Critical Perspective** (Zichen Liu, 2025, arXiv) — <https://arxiv.org/abs/2503.20783>
  - Base models already possess reasoning; RL elicits not creates — core WON'T evidence. *(r2: persistence-posttraining)*
- **Reinforcement Learning for Reasoning in Large Language Models with One Training Example** (Yiping Wang, 2025, arXiv) — <https://arxiv.org/abs/2504.20571>
  - 1-shot RLVR surfaces pretrained reasoning — persistence/elicitation. *(r2: persistence-posttraining)*
- **LIMO: Less is More for Reasoning** (Yixin Ye, 2025, arXiv) — <https://arxiv.org/abs/2502.03387>
  - ~1% data elicits reasoning already present from pretraining. *(r2: persistence-posttraining)*
- **s1: Simple Test-Time Scaling** (Niklas Muennighoff, 2025, arXiv) — <https://arxiv.org/abs/2501.19393>
  - 1k-example SFT elicits latent base reasoning. *(r2: persistence-posttraining)*
- **Small Models Struggle to Learn from Strong Reasoners** (Yuetai Li, 2025, arXiv) — <https://arxiv.org/abs/2502.12143>
  - CAN'T side: reasoning won't transfer if base capacity absent. *(r2: persistence-posttraining)*
- **Are DeepSeek R1 and Other Reasoning Models More Faithful?** (James Chua, 2025, arXiv) — <https://arxiv.org/abs/2501.08156>
  - Whether RL post-training fixes unverbalized shortcut cues — direct H1.3 evidence *(r2: cot-faithfulness)*
- **Understanding Reasoning from Pretraining to Post-Training** ((see paper), 2026, preprint (arXiv:2607.16097)) — <https://arxiv.org/abs/2607.16097>
  - Quantifies how pretraining bounds post-training (loss predicts post-RL) — core H1.3. *(r1: persistence-posttraining, data-selection-reasoning, fresh-2026)*
- **Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model** ((see paper), 2026, preprint (arXiv:2606.22317)) — <https://arxiv.org/html/2606.22317v1>
  - Direct counter-evidence to boundedness thesis (Yue) — must weigh both sides. *(r1: persistence-posttraining)*
- **Training on Documents About Monitoring Leads to CoT Obfuscation** ((see arXiv), 2026, preprint) — <https://arxiv.org/pdf/2605.15257>
  - Direct pretraining-corpus-content → reasoning-behavior link; data changes downstream CoT faithfulness *(r1: cot-faithfulness)*
- **Mid-Training with Self-Generated Data Improves Reinforcement Learning in Language Models** (2026, preprint) — <https://arxiv.org/abs/2605.08472>
  - Self-generated reasoning data in mid-training shaping RL gains — pretraining-state-to-RL persistence *(r1: fresh-2026)*
- **Distributional Clarity: The Hidden Driver of RL-Friendliness in Large Language Models** (2026, preprint) — <https://arxiv.org/abs/2601.06911>
  - Base-model properties predicting RL reasoning gains — pretraining→post-training persistence link *(r1: fresh-2026)*
- **Operationalising the Superficial Alignment Hypothesis via Task Complexity** (Tomas Vergara-Browne, 2026, arXiv) — <https://arxiv.org/abs/2602.15829>
  - Formal lens on elicitation vs creation in post-training. *(r2: persistence-posttraining)*

### H2.4 — identify reasoning-rich text (14)

- **Textbooks Are All You Need** (Suriya Gunasekar, 2023, arXiv) — <https://arxiv.org/abs/2306.11644>
  - Origin of textbook-quality classifier selection driving reasoning gains; foundational for H2.4/H2.5. *(r2: data-selection-reasoning)*
- **Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models** (Laura Ruis, 2024, ICLR 2025 / arXiv) — <https://arxiv.org/abs/2411.12580>
  - Influence-function evidence for which documents drive reasoning — core H2.4 *(r1: mechanistic-formation, data-selection-reasoning, fresh-2026)*
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (Zhihong Shao, 2024, arXiv) — <https://arxiv.org/abs/2402.03300>
  - Iterative fastText classifier mines reasoning corpus from CC — canonical H2.4. *(r2: data-selection-reasoning)*
- **MAmmoTH2: Scaling Instructions from the Web** (Xiang Yue, 2024, NeurIPS 2024 / arXiv) — <https://arxiv.org/abs/2405.03548>
  - Shows raw web contains extractable naturally-occurring reasoning QA — bears directly on identifying reasoning-rich text. *(r2: data-selection-reasoning)*
- **Exploring the Mystery of Influential Data for Mathematical Reasoning** (Xinzhe Ni, 2024, arXiv 2404.01067) — <https://arxiv.org/abs/2404.01067>
  - Directly asks which training data drives math reasoning — core H2.4 identification question *(r2: fresh-2026)*
- **Which Data Attributes Stimulate Math and Code Reasoning? An Investigation via Influence Functions** (2025, preprint) — <https://arxiv.org/abs/2505.19949>
  - Influence functions identifying reasoning-driving data attributes — directly H2.4. *(r1: data-selection-reasoning)*
- **Essential-Web v1.0: 24T tokens of organized web data** (2025, preprint) — <https://arxiv.org/abs/2506.14111>
  - Per-doc reasoning-depth taxonomy labels = direct infrastructure for identifying reasoning-rich documents *(r1: data-selection-reasoning)*
- **Data Recipes for Reasoning Models** (2025, ICLR 2026) — <https://arxiv.org/abs/2506.04178>
  - Systematic curation of reasoning training data — squarely H2.4/H2.5 *(r1: fresh-2026)*
- **The Data-Quality Illusion: Rethinking Classifier-Based Quality Filtering for LLM Pretraining** (2025, preprint) — <https://arxiv.org/abs/2510.00866>
  - Cautionary evidence against classifier filtering — load-bearing for H2.4 method choice *(r1: fresh-2026)*
- **Reinforcement Pre-Training on General-Domain Corpora: Scaling Next-Token Reasoning Beyond Mathematical Text** (Abdulloh Abdulloh, 2025, TechRxiv) — <https://doi.org/10.36227/techrxiv.176315661.10663467/v1>
  - NTP-as-reasoning objective on general web corpora — directly H1 objective and H2 reasoning-rich text. *(r2: fresh-2026)*
- **What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code** (2026, preprint) — <https://arxiv.org/html/2605.19762>
  - Dissects what in code carries reasoning signal — refines 'reasoning-rich' definition (H2.4/6) *(r1: data-selection-reasoning, fresh-2026)*
- **Unlocking Latent Value: Taxonomy-Guided Recovery of High-Performing Data from Low-Tier Web Corpora** (2026, preprint) — <https://arxiv.org/html/2606.07778>
  - Shows composite quality scores miss reasoning dimensions — directly on the H2.4 open question *(r1: data-selection-reasoning, fresh-2026)*
- **Reasoning Quality Emerges Early: Data Curation for Reasoning Models** (Hongyi Henry Jin, 2026, arXiv) — <https://arxiv.org/abs/2606.26797>
  - Detect/curate reasoning content cheaply from initial tokens — H2(4)/(6). *(r2: fresh-2026)*
- **Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text** (Ximing Lu, 2026, arXiv 2601.22975) — <https://arxiv.org/abs/2601.22975>
  - Mines latent reasoning/verifiable tasks from raw web text — direct H2.4 corpus-mining approach *(r2: fresh-2026)*

### H2.5 — augment text with reasoning (25)

- **Textbooks Are All You Need (phi-1)** (Suriya Gunasekar, 2023, preprint (arXiv)) — <https://arxiv.org/abs/2306.11644>
  - Canonical evidence that explanation-dense generated text is reasoning-rich pretraining data. *(r1: augmentation-synthetic)*
- **Synthetic Continued Pretraining** (Zitong Yang, 2024, ICLR 2025) — <https://arxiv.org/abs/2409.07431>
  - Rewriting corpora into synthetic text so knowledge internalizes — directly on augmenting pretraining text (EntiGraph). *(r1: format-internalization)*
- **Programming Every Example: Lifting Pre-training Data Quality Like Experts at Scale (ProX)** (Fan Zhou, 2024, arXiv) — <https://arxiv.org/abs/2409.17115>
  - Program-based doc refinement/upgrade improving reasoning benchmarks. *(r2: data-selection-reasoning)*
- **MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code** (Zimu Lu, 2024, arXiv) — <https://arxiv.org/abs/2410.08196>
  - Augments corpus with interleaved reasoning steps + code for continued pretraining — direct H2.5 instance. *(r2: data-selection-reasoning, augmentation-synthetic)*
- **ToW: Thoughts of Words Improve Reasoning in Large Language Models** (Zhikun Xu, 2024, arXiv) — <https://arxiv.org/pdf/2410.16235>
  - Injects inter-word thoughts into pretraining text — directly H2.5/H2.6 *(r2: enthymemes-completeness)*
- **Grokking in the Wild: Data Augmentation for Real-World Multi-Hop Reasoning with Transformers** (2025, preprint) — <https://arxiv.org/abs/2504.20752>
  - Data augmentation to push multi-hop past memorization — direct H2.5 intervention *(r1: shortcut-learning, latent-multihop, mechanistic-formation, format-internalization)*
- **Effect of Document Packing on the Latent Multi-Hop Reasoning Capabilities of Large Language Models** ((see paper), 2025, preprint) — <https://arxiv.org/abs/2512.14427>
  - Pretraining data-arrangement lever directly actionable for H2.5 *(r1: latent-multihop)*
- **The Kinetics of Reasoning: How Chain-of-Thought Shapes Learning in Transformers** ((see arXiv), 2025, preprint (arXiv)) — <https://arxiv.org/pdf/2510.25791>
  - CoT-in-training changes learning kinetics — links reasoning-augmented text to formation *(r1: mechanistic-formation, format-internalization)*
- **Rewriting Pre-Training Data Boosts LLM Performance in Math and Code** (2025, preprint) — <https://arxiv.org/abs/2505.02881>
  - Direct H2.5 evidence: LLM-rewriting pretraining corpora improves reasoning (dup of #16) *(r1: data-selection-reasoning, fresh-2026)*
- **Rewriting Pre-Training Data Boosts LLM Performance in Math and Code (SwallowCode/SwallowMath)** (Kazuki Fujii, 2025, preprint (arXiv)) — <https://arxiv.org/abs/2505.02881>
  - Same paper as #1's rewrite item; step-by-step rewrites with large GSM8K/HumanEval gains — read once *(r1: augmentation-synthetic)*
- **MIND: Math Informed syNthetic Dialogues for Pretraining LLMs** (Syeda Nahida Akter, 2025, ICLR 2025) — <https://arxiv.org/abs/2410.12881>
  - Converts math docs to decomposed step-by-step dialogues — squarely H2.5 with H2.6 flavor *(r1: augmentation-synthetic)*
- **Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls** ((see arXiv 2510.01631), 2025, preprint (arXiv)) — <https://arxiv.org/abs/2510.01631>
  - Quantifies when reasoning-style/rephrased augmentation helps vs hurts at scale — core to H2.5. *(r1: augmentation-synthetic, enthymemes-completeness, fresh-2026)*
- **Mining Hidden Thoughts from Texts: Evaluating Continual Pretraining with Synthetic Data for LLM Reasoning** ((see paper), 2025, preprint (arXiv)) — <https://arxiv.org/pdf/2505.10182>
  - generates latent reasoning from STEM/law text for continual pretraining — core H2.5 *(r1: perplexity-signals, format-internalization, hard-tokens, data-selection-reasoning)*
- **Recycling the Web: A Method to Enhance Pre-training Data by Guided Rewriting** (Thao Nguyen, 2025, preprint) — <https://arxiv.org/abs/2506.04689>
  - LLM-guided rewriting of web docs into better pretraining text — direct H2.5 vehicle *(r1: enthymemes-completeness)*
- **CCI4.0: A Bilingual Pretraining Dataset for Enhancing Reasoning in Large Language Models** ((BAAI), 2025, arXiv) — <https://arxiv.org/pdf/2506.07463>
  - 400B tokens of CoT reconstructed from web docs — large-scale H2.5 augmentation evidence. *(r2: data-selection-reasoning)*
- **Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity** (unknown, 2025, arXiv) — <https://arxiv.org/abs/2506.04689>
  - Rewriting low-quality web docs to boost pretraining value — directly on H2.5 augmentation. *(r2: augmentation-synthetic)*
- **LLM Pretraining with Continuous Concepts (CoCoMix)** (Jihoon Tack, 2025, arXiv (Meta)) — <https://arxiv.org/abs/2502.08524>
  - Pretraining objective change injecting latent concepts — directly on augmenting pretraining with reasoning-like signal *(r2: format-internalization)*
- **Pretraining Language Models to Ponder in Continuous Space** (Boyi Zeng, 2025, arXiv) — <https://arxiv.org/abs/2505.20674>
  - Pretraining-time latent computation induction; tests whether reasoning depth can be built in at pretraining *(r2: format-internalization)*
- **RePro: Training Language Models to Faithfully Recycle the Web for Pretraining** (Zichun Yu, 2025, arXiv 2510.10681) — <https://arxiv.org/abs/2510.10681>
  - Faithful rephrasing/recycling web into higher-value tokens — core H2(5) augmentation. *(r2: fresh-2026)*
- **WRAP++: Web discoveRy Amplified Pretraining** ((see arXiv 2604.06829), 2026, preprint (arXiv)) — <https://arxiv.org/abs/2604.06829>
  - Synthesizes cross-doc multi-hop QA — augmentation adding reasoning absent from single docs. *(r1: augmentation-synthetic, enthymemes-completeness, fresh-2026)*
- **Transformers Provably Learn to Internalize Chain-of-Thought** (2026, arXiv) — <https://arxiv.org/pdf/2605.28600>
  - Theory of how explicit CoT in training data becomes latent computation — core to whether augmenting text with reasoning changes what pretraining learns. *(r2: mechanistic-formation, cot-faithfulness, format-internalization)*
- **Data-efficient pre-training by scaling synthetic megadocs** (Konwoo Kim, 2026, arXiv) — <https://arxiv.org/abs/2603.18534>
  - Synthetic rephrase megadocs improve loss scaling in data-constrained regime — core H2.5. *(r2: augmentation-synthetic)*
- **Pretraining with Token-Level Adaptive Latent Chain-of-Thought** (Boyi Zeng, 2026, arXiv) — <https://arxiv.org/abs/2602.08220>
  - Per-token latent CoT during pretraining — pretraining-format augmentation for H2 *(r2: format-internalization)*
- **FOL-Traces: Verified First-Order Logic Reasoning Traces at Scale** (Isabelle Lee, 2026, Findings of EACL 2026) — <https://doi.org/10.18653/v1/2026.findings-eacl.115>
  - Large-scale verified explicit reasoning traces — resource for H2(5)/(6) complete-reasoning augmentation. *(r2: fresh-2026)*
- **Procedural Knowledge at Scale Improves Reasoning** (Di Wu, 2026, arXiv 2604.01348) — <https://arxiv.org/abs/2604.01348>
  - Scales procedural-knowledge pretraining data and measures reasoning gains — build-on for H2(5). *(r2: fresh-2026)*

### H2.6 — completeness of reasoning chains (12)

- **RATIONALYST: Mining Implicit Rationales for Process Supervision of Reasoning** (Dongwei Jiang, 2024, preprint (arXiv)) — <https://arxiv.org/abs/2410.01044>
  - Mines implicit rationales in web text — directly on H2.4/6; also the user's own paper *(r1: augmentation-synthetic)*
- **Can Language Models Learn to Skip Steps?** (Tengxiao Liu, 2024, NeurIPS) — <https://arxiv.org/abs/2411.01855>
  - controlled train on complete vs step-skipped traces — closest study of completeness *(r1: enthymemes-completeness, format-internalization)*
- **Synthetic Continued Pretraining (EntiGraph)** (Zitong Yang, 2024, ICLR 2025) — <https://arxiv.org/abs/2409.07431>
  - Makes implicit entity connections explicit for continued pretraining — closest analog to enthymeme-explicitation (H2.6). *(r2: augmentation-synthetic)*
- **Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus** (Terufumi Morishita, 2024, NeurIPS 2024) — <https://arxiv.org/abs/2411.12498>
  - Synthetic deduction corpus with fully explicit inference steps — directly probes completeness of reasoning chains. *(r2: augmentation-synthetic)*
- **On the Bias of Next-Token Predictors Toward Systematically Inefficient Reasoning: A Shortest-Path Case Study** (2025, preprint) — <https://arxiv.org/abs/2507.05362>
  - Redundant/longer traces generalize better — direct evidence on chain completeness/verbosity *(r1: shortcut-learning, fresh-2026)*
- **Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation** (unknown, 2025, arXiv) — <https://arxiv.org/html/2509.05226>
  - Required reasoning granularity vs difficulty — direct evidence on how complete chains must be, via training *(r2: enthymemes-completeness)*
- **Rethinking the Role of Text Complexity in Language Model Pretraining** (Dan John Velasco, 2025, BabyLM Workshop 2025) — <https://doi.org/10.18653/v1/2025.babylm-main.1>
  - How inferential load/complexity of pretraining text affects learning — directly H2(6) completeness. *(r2: fresh-2026)*
- **Chain of Execution Supervision Promotes General Reasoning in Large Language Models** (Nuo Chen, 2025, arXiv 2510.23629) — <https://arxiv.org/abs/2510.23629>
  - Execution-trace supervision makes latent steps explicit — completeness augmentation H2(5/6). *(r2: fresh-2026)*
- **The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning** (2026, preprint) — <https://arxiv.org/html/2603.29025>
  - Surface cue beating unstated premise — bridges H1 shortcuts and H2.6 implicit-premise completeness *(r1: shortcut-learning, format-internalization)*
- **Making Implicit Premises Explicit in Logical Understanding of Enthymemes** ((see arXiv 2603.06114), 2026, preprint (arXiv)) — <https://arxiv.org/abs/2603.06114>
  - Reconstructs missing enthymeme premises — the exact completeness operation of H2.6 (distinct from covered Feng&Hunter). *(r1: augmentation-synthetic, enthymemes-completeness, fresh-2026, cot-faithfulness)*
- **Zipping the Thought: When and How Compressed Reasoning Data Works in LLM Post-Training** (Kohsei Matsutani, 2026, arXiv) — <https://arxiv.org/abs/2605.28008>
  - Compressed vs full reasoning traces in training data — directly on how complete reasoning must be *(r2: format-internalization)*
- **From Implicit to Explicit: Token-Efficient Logical Supervision for Mathematical Reasoning in LLMs** (Shaojie Wang, 2026, Findings of ACL 2026) — <https://doi.org/10.18653/v1/2026.findings-acl.1420>
  - Makes implicit logical steps explicit as supervision — directly H2(6) enthymemes made explicit. *(r2: fresh-2026)*

### H2.7 — perplexity / model-gap signals (12)

- **Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt (RHO-LOSS)** (Sören Mindermann, 2022, ICML 2022) — <https://arxiv.org/abs/2206.07137>
  - Excess loss = train minus reference-model loss — the foundational model-gap signal for H2.7. *(r2: perplexity-signals, hard-tokens)*
- **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining** (Sang Michael Xie, 2023, NeurIPS 2023) — <https://arxiv.org/abs/2305.10429>
  - Foundational excess-loss (proxy vs reference) valuation — the reference-model gap signal. *(r1: perplexity-signals, data-selection-reasoning)*
- **Generalization v.s. Memorization: Tracing Language Models' Capabilities Back to Pretraining Data** (Xinyi Wang, 2024, ICLR 2025) — <https://arxiv.org/abs/2407.14985>
  - Corpus-frequency signal separating recall from inference; touches H1 and H2.7 *(r1: shortcut-learning, mechanistic-formation)*
- **Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models** (Zachary Ankner, 2024, preprint (arXiv)) — <https://arxiv.org/abs/2405.20541>
  - Canonical small-reference-model perplexity pruning signal for H2.7. *(r1: perplexity-signals, data-selection-reasoning)*
- **ScalingFilter: Assessing Data Quality through Inverse Utilization of Scaling Laws** (Ruihang Li, 2024, preprint (arXiv)) — <https://arxiv.org/abs/2408.08310>
  - Scores docs by perplexity gap between two model sizes — exactly the weak/strong model-gap signal. *(r1: perplexity-signals)*
- **Rethinking KenLM: Good and Bad Model Ensembles for Efficient Text Quality Filtering in Large Web Corpora** (Yungi Kim, 2024, preprint (arXiv)) — <https://arxiv.org/abs/2409.09613>
  - good-vs-bad LM perplexity-gap ensemble for filtering — direct H2.7 two-model signal *(r1: perplexity-signals)*
- **Compression Represents Intelligence Linearly** (Yuzhen Huang, 2024, COLM 2024) — <https://arxiv.org/abs/2404.09937>
  - Model bits-per-char/loss correlates linearly with reasoning scores — direct perplexity-gap signal. *(r2: perplexity-signals)*
- **Sequence Reducible Holdout Loss for Language Model Pretraining** (Simin Fan, 2024, LREC-COLING 2024) — <https://aclanthology.org/2024.lrec-main.1281/>
  - Token/sequence-level reducible-holdout-loss selection signal, directly on the perplexity/model-gap axis. *(r2: hard-tokens)*
- **Predicting LLM Reasoning Performance with Small Proxy Model (rBridge)** (2025, preprint) — <https://arxiv.org/abs/2509.21013>
  - Cheap model-signal ranking data for reasoning value — directly H2.7. *(r1: data-selection-reasoning)*
- **The Signal is in the Steps: Local Scoring for Reasoning Data Selection** ((see paper), 2025, preprint (arXiv)) — <https://arxiv.org/pdf/2510.03988>
  - Frames CE as missing long-range dependencies (reasoning gap), motivating step-level scoring — bridges H1 shortcuts and loss signal. *(r1: perplexity-signals)*
- **Systematic Generalization in Language Models Scales with Information Entropy** (Sondre Wold, 2025, arXiv (ACL Findings 2025)) — <https://arxiv.org/abs/2505.13089>
  - Corpus-level measurable signal predicting compositional generalization *(r2: latent-multihop)*
- **Reasoning Stabilization Point: A Training-Time Signal for Stable Evidence and Shortcut Reliance** (Sahil Rajesh Dhayalkar, 2026, arXiv 2601.11625) — <https://arxiv.org/abs/2601.11625>
  - Training-time signal separating evidence-based reasoning from shortcuts — bears on both H1.1 and H2.7 detection *(r2: fresh-2026)*

## Maybe


### H1.1 — shortcuts instead of reasoning (69)

- **Annotation Artifacts in Natural Language Inference Data** (Suchin Gururangan, 2018, NAACL 2018) — <https://arxiv.org/pdf/1803.02324>
  - Canonical shortcut evidence but finetuning-era NLI, not pretraining-level. *(r2: shortcut-learning)*
- **Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference** (R. Thomas McCoy, 2019, ACL) — <https://aclanthology.org/P19-1334/>
  - Classic shortcut diagnosis but pre-LLM fine-tuned NLI, not pretraining-level *(r1: shortcut-learning)*
- **Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training, and Model Development for Multi-Hop QA** (Yichen Jiang, 2019, ACL 2019 / arXiv) — <https://arxiv.org/abs/1906.07132>
  - Classic shortcut-vs-reasoning result, but pre-LLM QA-model era *(r2: latent-multihop)*
- **Evading the Simplicity Bias: Training a Diverse Set of Models Discovers Solutions with Superior OOD Generalization** (Damien Teney, 2021, CVPR 2022) — <https://arxiv.org/abs/2105.05612>
  - Vision-domain evidence shortcuts are a preference (WON'T), not capacity limit *(r1: shortcut-learning)*
- **Shortcut Learning of Large Language Models in Natural Language Understanding** (Mengnan Du, 2022, Communications of the ACM) — <https://arxiv.org/abs/2208.11857>
  - Survey of LLM shortcut reliance; useful map but not new evidence *(r1: shortcut-learning, cot-faithfulness, hard-tokens, mechanistic-formation)*
- **Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting** (Miles Turpin, 2023, NeurIPS 2023) — <https://arxiv.org/abs/2305.04388>
  - Foundational post-hoc CoT evidence but prompting-time, no pretraining/data lever *(r1: cot-faithfulness)*
- **Measuring Faithfulness in Chain-of-Thought Reasoning** (Tamera Lanham, 2023, preprint (Anthropic)) — <https://arxiv.org/abs/2307.13702>
  - Canonical causal-use intervention tests; analysis of trained models, not data-level *(r1: cot-faithfulness, shortcut-learning)*
- **Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations** (Yanda Chen, 2023, ICML 2024) — <https://arxiv.org/abs/2307.08678>
  - Behavioral def of explanation-vs-computation match; adjacent, no training angle *(r1: cot-faithfulness)*
- **Overcoming Simplicity Bias in Deep Networks using a Feature Sieve** (Rishabh Tiwari, 2023, arXiv (ICML 2023)) — <https://arxiv.org/abs/2301.13293>
  - Simplicity-bias mitigation, vision-centric; principle relevant, method not. *(r2: shortcut-learning)*
- **Counterfactual reasoning: Testing language models' understanding of hypothetical scenarios** (Jiaxuan Li, 2023, ACL) — <https://arxiv.org/pdf/2305.16572>
  - Clean lexical-cue vs reasoning separation method; probing not training-level. *(r2: shortcut-learning)*
- **Analyzing the Effectiveness of the Underlying Reasoning Tasks in Multi-hop Question Answering** (Xanh Ho, 2023, Findings of EACL 2023 / arXiv) — <https://arxiv.org/abs/2302.05963>
  - Sub-task supervision vs shortcuts; benchmark-centric, adjacent *(r2: latent-multihop)*
- **A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis** (2023, arXiv) — <https://arxiv.org/abs/2305.15054>
  - Arithmetic circuit evidence adjacent to Bag-of-Heuristics (already covered); marginal new signal. *(r2: mechanistic-formation)*
- **Reasoning in Transformers — Mitigating Spurious Correlations and Reasoning Shortcuts** (2024, preprint) — <https://arxiv.org/abs/2403.11314>
  - Controlled logic-training shortcut test; relevance depends on setup scale *(r1: shortcut-learning)*
- **Neural Networks Learn Statistics of Increasing Complexity** (Nora Belrose, 2024, ICML) — <https://arxiv.org/abs/2402.04362>
  - Distributional-simplicity trajectory; background for shortcut-first learning *(r1: shortcut-learning)*
- **Understanding Transformers via N-gram Statistics** (Timothy Nguyen, 2024, NeurIPS) — <https://arxiv.org/abs/2407.12034>
  - Quantifies surface-statistics share of NTP behavior; background evidence *(r1: shortcut-learning)*
- **How Do Large Language Models Acquire Factual Knowledge During Pretraining?** (Hoyeon Chang, 2024, NeurIPS 2024) — <https://proceedings.neurips.cc/paper_files/paper/2024/file/6fdf57c71bc1f1ee29014b8dc52e723f-Paper-Conference.pdf>
  - Memorization-dynamics baseline; useful contrast but not reasoning per se *(r1: mechanistic-formation)*
- **Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning** (Debjit Paul, 2024, EMNLP Findings 2024) — <https://arxiv.org/abs/2402.13950>
  - Causal mediation + FRODO training to make answers depend on rationale; post-training not pretraining *(r1: cot-faithfulness)*
- **Deceptive Semantic Shortcuts on Reasoning Chains: How Far Can Models Go without Hallucination?** (Ming Shen, 2024, NAACL 2024) — <https://arxiv.org/abs/2311.09702>
  - EUREQA shows semantic-association shortcuts over stated multi-hop steps; benchmark-ish *(r1: cot-faithfulness)*
- **Learning Shortcuts: On the Misleading Promise of NLU in Language Models** (Geetanjali Bihani, 2024, preprint) — <https://arxiv.org/abs/2401.09615>
  - General shortcut-learning position; H1 background but not CoT/pretraining specific *(r1: cot-faithfulness, shortcut-learning)*
- **Think before you speak: Training Language Models With Pause Tokens** (Sachin Goyal, 2024, ICLR 2024) — <https://arxiv.org/abs/2310.02226>
  - Pause tokens add compute per prediction — bears on CAN'T (capacity) side of H1; training-time intervention. *(r1: hard-tokens, format-internalization)*
- **Grokking as the Transition from Lazy to Rich Training Dynamics** (unknown, 2024, ICLR 2024) — <https://proceedings.iclr.cc/paper_files/paper/2024/file/63ed15a46a143ff57484b38cd6b85d91-Paper-Conference.pdf>
  - Mechanism for why shortcuts win early; adjacent theory, not LM-data-level. *(r2: shortcut-learning)*
- **The Pitfalls of Memorization: When Memorization Hurts Generalization** (Reza Bayat, 2024, arXiv) — <https://arxiv.org/pdf/2412.07684>
  - Mechanistic account of shortcut+memorize-exceptions beating full rules. *(r2: shortcut-learning)*
- **Unveiling Factual Recall Behaviors of Large Language Models through Knowledge Neurons** ((unknown), 2024, arXiv) — <https://arxiv.org/abs/2408.03247>
  - Recall-vs-surface-cue probing; adjacent knowledge-use evidence *(r2: latent-multihop)*
- **The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains** (Benjamin L. Edelman, 2024, NeurIPS 2024) — <https://proceedings.neurips.cc/paper_files/paper/2024/file/75b0edb869e2cd509d64d0e8ff446bc1-Paper-Conference.pdf>
  - Staged capability formation (n-gram before induction) — shortcut-first dynamics background *(r2: mechanistic-formation)*
- **Learning Syntax Without Planting Trees: Understanding Hierarchical Generalization in Transformers** (Kabir Ahuja, 2024, TACL / arXiv) — <https://arxiv.org/abs/2404.16367>
  - Surface-heuristic vs hierarchical generalization dynamics; adjacent *(r2: mechanistic-formation)*
- **Navigating the Shortcut Maze: A Comprehensive Analysis of Shortcut Learning in Text Classification by Language Models** (Unknown, 2024, arXiv (EMNLP Findings)) — <https://arxiv.org/html/2409.17455v1>
  - Shortcut taxonomy, but text-classification setting distant from pretraining *(r2: mechanistic-formation)*
- **When can transformers compositionally generalize in-context?** (2024, arXiv (ICML 2024 workshop)) — <https://arxiv.org/abs/2407.12275>
  - Bottleneck condition for compositional generalization bears on shortcut-vs-inference, but in-context/toy setting. *(r2: mechanistic-formation)*
- **Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference** (2024, arXiv) — <https://arxiv.org/html/2408.08590>
  - Circuit for multi-step syllogistic inference could show whether models do full inference or pattern-match. *(r2: mechanistic-formation)*
- **A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning** (2024, OpenReview) — <https://openreview.net/pdf?id=M0U8wUow8c>
  - Isolates rule-application vs pattern-matching components — directly the shortcut distinction, but circuit-level. *(r2: mechanistic-formation)*
- **Dissociating Language and Thought in Large Language Models** (Kyle Mahowald, 2024, Trends in Cognitive Sciences) — <https://arxiv.org/abs/2301.06627>
  - Conceptual competence-vs-performance frame; not pretraining/data-level. *(r2: persistence-posttraining)*
- **Minds versus Machines: Rethinking Entailment Verification with Language Models** (Soumya Sanyal, 2024, arXiv) — <https://arxiv.org/html/2402.03686v1>
  - Multi-premise/implicit-knowledge entailment; relevant to whether models fill missing steps, but eval-time *(r2: enthymemes-completeness)*
- **Dissociation of Faithful and Unfaithful Reasoning in LLMs** (Evelyn Yee, 2024, arXiv) — <https://arxiv.org/abs/2405.15092>
  - Silent error correction dissociates stated reasoning from computation; shortcut-adjacent, inference-time *(r2: cot-faithfulness)*
- **Break the Chain: Large Language Models Can be Shortcut Reasoners** (Mengru Ding, 2024, arXiv) — <https://arxiv.org/abs/2406.06580>
  - Shortcut-over-CoT behavior relevant to WON'T, but framed as efficiency method. *(r2: cot-faithfulness)*
- **Preemptive Answer "Attacks" on Chain-of-Thought Reasoning** (Rongwu Xu, 2024, ACL Findings) — <https://doi.org/10.18653/v1/2024.findings-acl.876>
  - Shows chain rationalizes a pre-set answer; prompting-time but H1-relevant evidence. *(r2: cot-faithfulness)*
- **Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers** ((see arXiv), 2025, preprint (arXiv)) — <https://arxiv.org/abs/2506.09099>
  - Capacity axis of CAN'T vs WON'T; adjacent unless it addresses reasoning tasks *(r1: mechanistic-formation)*
- **Reason to Rote: Rethinking Memorization in Reasoning** ((see arXiv), 2025, preprint (arXiv)) — <https://arxiv.org/pdf/2507.04782>
  - Memorization riding on reasoning mechanisms — nuances shortcut framing *(r1: mechanistic-formation)*
- **Chain-of-Thought Reasoning In The Wild Is Not Always Faithful** (Iván Arcuschin, 2025, preprint / OpenReview) — <https://arxiv.org/abs/2503.08679>
  - Natural-use post-hoc rationalization; adjacent shortcut evidence, no training angle *(r1: cot-faithfulness, shortcut-learning)*
- **Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps** (Martin Tutek, 2025, preprint) — <https://arxiv.org/abs/2502.14829>
  - Parameter-level test of causal load-bearing steps; measurement, not data method *(r1: cot-faithfulness)*
- **Post-Hoc Reasoning in Chain-of-Thought: Evidence from Pre-CoT Probes and Activation Steering** ((OpenReview submission), 2025, OpenReview) — <https://openreview.net/forum?id=UMUYpeXtJQ>
  - Answer decodable before CoT tokens — strong decoupling evidence but analysis-only *(r1: cot-faithfulness)*
- **Robust Answers, Fragile Logic: Probing the Decoupling Hypothesis in LLM Reasoning** ((see arXiv), 2025, preprint) — <https://arxiv.org/abs/2505.17406>
  - Quantifies reasoning/answer decoupling H1 posits; behavioral probing, no data lever *(r1: cot-faithfulness)*
- **Is it Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort** ((see arXiv), 2025, preprint / OpenReview) — <https://arxiv.org/abs/2510.01367>
  - Operational shortcut-vs-reasoning detector via effort; RL-eval, no data angle *(r1: cot-faithfulness)*
- **Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization** ((see arXiv), 2025, preprint) — <https://arxiv.org/pdf/2512.23032>
  - Counterpoint balancing hint-verbalization tests; keeps treatment balanced *(r1: cot-faithfulness)*
- **Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers** (Charles London, 2025, preprint) — <https://arxiv.org/abs/2505.21024>
  - Theory on why extra tokens expand compute; grounds shortcut/capacity (CAN'T vs WON'T) distinction. *(r1: format-internalization, hard-tokens)*
- **Next-Token Prediction Should be Ambiguity-Sensitive** (2025, preprint) — <https://arxiv.org/abs/2506.16288>
  - Conceptual argument that hard tokens get too little compute; supports H1 framing but likely no data-level experiments *(r1: hard-tokens)*
- **Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR** (2025, preprint) — <https://arxiv.org/abs/2507.15778>
  - Operationalizes knowledge-vs-reasoning token split (CAN'T vs WON'T) but at RL stage, not pretraining data *(r1: hard-tokens)*
- **Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries** (2025, preprint) — <https://arxiv.org/abs/2510.14751>
  - Objective-level account of NTP long-horizon weakness; H1-relevant but objective engineering, not data *(r1: fresh-2026)*
- **Supervised and Unsupervised Probing of Shortcut Learning** (Unknown, 2025, ACL Findings 2025) — <https://aclanthology.org/2025.findings-acl.499.pdf>
  - Probing for internal shortcut representations is adjacent to H1.1 but method-level, not pretraining-data-level. *(r2: mechanistic-formation)*
- **Tracing Multilingual Factual Knowledge Acquisition in Pretraining** (Unknown, 2025, EMNLP Findings 2025) — <https://aclanthology.org/2025.findings-emnlp.113/>
  - Checkpoint-level frequency-driven knowledge acquisition informs CAN'T-vs-WON'T exposure effects, but factual recall not reasoning. *(r2: mechanistic-formation)*
- **Analyzing the Inner Workings of Transformers in Compositional Generalization** (2025, arXiv) — <https://arxiv.org/html/2502.15277>
  - Mechanistic memorization-vs-compositional-structure evidence is adjacent to shortcut-vs-inference. *(r2: mechanistic-formation)*
- **Complexity Control Facilitates Reasoning-Based Compositional Generalization in Transformers** (2025, arXiv) — <https://arxiv.org/html/2501.08537v1>
  - Directly links training regime to reasoning-based vs memorized solutions, but toy-scale and not corpus-level. *(r2: mechanistic-formation)*
- **Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation** (unknown, 2025, arXiv) — <https://arxiv.org/abs/2505.13792>
  - Whether distilled reasoning traces actually drive answers vs shortcut — adjacent to H1. *(r2: augmentation-synthetic)*
- **When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors** (Scott Emmons, 2025, arXiv) — <https://arxiv.org/abs/2507.05246>
  - Delineates when CoT is load-bearing computation vs decorative; capability-boundary relevant but safety-framed *(r2: cot-faithfulness)*
- **Can Aha Moments Be Fake? Towards Quantifying Decorative and True Thinking in Chain-of-Thought** (Jiachen Zhao, 2025, arXiv) — <https://arxiv.org/abs/2510.24941>
  - Quantifies decorative vs used reasoning steps; diagnostic, no data-level intervention *(r2: cot-faithfulness)*
- **Unveiling Confirmation Bias in Chain-of-Thought Reasoning** (Yue Wan, 2025, arXiv) — <https://arxiv.org/abs/2506.12301>
  - Answer-before-rationale evidence; inference-time but supports shortcut framing. *(r2: cot-faithfulness)*
- **Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness** (Jiachun Li, 2025, ACL Findings) — <https://doi.org/10.18653/v1/2025.findings-acl.560>
  - Separates accuracy from faithfulness — adjacent to CAN'T vs WON'T. *(r2: cot-faithfulness)*
- **On the Causal Identifiability of Chain-of-Thought in Language Models** (Jitesh Uikey, 2025, TechRxiv preprint) — <https://doi.org/10.36227/techrxiv.176153550.09433591/v1>
  - Formal framework for whether CoT drives answers; theory, not data-level. *(r2: cot-faithfulness)*
- **Entropy-Guided Token Dropout: Training Autoregressive Language Models with Limited Domain Data** (Jiapeng Wang, 2025, arXiv) — <https://arxiv.org/abs/2512.23422>
  - Masks low-entropy shortcut tokens; token difficulty as anti-shortcut training criterion. *(r2: hard-tokens)*
- **All Code, No Thought: Current Language Models Struggle to Reason in Ciphered Language** (Shiyuan Guo, 2025, arXiv 2510.09714) — <https://arxiv.org/abs/2510.09714>
  - Diagnoses format-bound vs genuine reasoning (shortcut evidence), but eval-only with no clear data lever *(r2: fresh-2026)*
- **Are Arithmetic Heuristic Neurons Form-Invariant? A Mechanistic Analysis of Symbols, Text, and Code in LLMs** (2026, preprint) — <https://arxiv.org/abs/2607.16693>
  - Follow-up to covered bag-of-heuristics; format-generality of shortcut circuits *(r1: shortcut-learning)*
- **Opening the Black Box: A Survey on the Mechanisms of Multi-Step Reasoning in Large Language Models** ((see paper), 2026, preprint (survey)) — <https://arxiv.org/pdf/2601.14270>
  - Consolidates shortcuts-vs-hops interpretability evidence; convenience read *(r1: latent-multihop, cot-faithfulness, mechanistic-formation)*
- **Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence Between Thinking Tokens and Answers in Open-Weight Reasoning Models** ((see arXiv), 2026, preprint) — <https://arxiv.org/abs/2603.26410>
  - Locates where unfaithfulness lives across models; analysis not data-level *(r1: cot-faithfulness)*
- **Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought Reasoning** ((see arXiv), 2026, preprint) — <https://arxiv.org/pdf/2602.11201>
  - Causal force of reasoning tokens concentrates early/decays — decoupling evidence *(r1: cot-faithfulness)*
- **When Benchmarks Mislead: Shortcut Learning, Length Confounds, and the Limits of Cross-Dataset Generalization** (unknown, 2026, arXiv 2607.14131) — <https://arxiv.org/html/2607.14131>
  - Measurement caution for shortcut vs capability claims; not data-level intervention. *(r2: shortcut-learning)*
- **What Does Loss Optimization Actually Teach, If Anything? Knowledge Dynamics in Continual Pre-training of LLMs** (Unknown, 2026, arXiv) — <https://arxiv.org/pdf/2601.03858>
  - Whether NTP loss reduction reflects inference vs memorization touches H1, but focus is knowledge dynamics in CPT. *(r2: mechanistic-formation)*
- **Model Capacity Determines Grokking through Competing Memorisation and Generalisation Speeds** (2026, arXiv) — <https://arxiv.org/pdf/2605.09724>
  - Competing memorization-vs-generalization speeds is a mechanistic account of when the shortcut wins (WON'T vs CAN'T). *(r2: mechanistic-formation)*
- **Measuring and curing reasoning rigidity: from decorative chain-of-thought to genuine faithfulness** (unknown, 2026, arXiv) — <https://arxiv.org/html/2603.22816>
  - Decorative vs load-bearing CoT relevant to shortcut diagnosis; unclear training-level angle *(r2: enthymemes-completeness)*
- **Adaptive Loops and Memory in Transformers: Think Harder or Know More?** (Markus Frey, 2026, arXiv) — <https://arxiv.org/abs/2603.08391>
  - Separates looping compute from memory/knowledge, mirrors CAN'T vs WON'T; architecture-level. *(r2: format-internalization)*
- **CREST: A causal framework for mitigating shortcut learning in language models through counterfactual reasoning** (Zhonghua Liu, 2026, Information Processing & Management) — <https://doi.org/10.1016/j.ipm.2025.104418>
  - Shortcut mitigation via counterfactuals — task-level, loosely on H1 shortcut framing. *(r2: fresh-2026)*
- **To Reason or to Fabricate: Reasoning Without Shortcuts via Hint-Anchored Pairwise Aggregation** (Jiuheng Lin, 2026, arXiv 2606.29481) — <https://arxiv.org/abs/2606.29481>
  - Targets shortcut reliance directly but appears to be a method/mitigation, likely post-training not pretraining-level *(r2: fresh-2026)*

### H1.2 — latent multi-hop / one-pass composition (30)

- **Implicit Chain of Thought Reasoning via Knowledge Distillation** (Yuntian Deng, 2023, preprint) — <https://arxiv.org/abs/2311.01460>
  - Internalizing explicit CoT into hidden-state computation; relates to latent multi-hop internalization *(r1: format-internalization, cot-faithfulness)*
- **Do LLMs Really Think Step-by-step In Implicit Reasoning?** (Yijiong Yu, 2024, preprint) — <https://arxiv.org/abs/2411.15862>
  - Corroborates shortcut/intuition claim but likely redundant with Lin 2025 + SOCRATES *(r1: latent-multihop, format-internalization)*
- **Training Large Language Models to Reason in a Continuous Latent Space (Coconut)** (Shibo Hao, 2024, preprint (FAIR)) — <https://arxiv.org/abs/2412.06769>
  - Latent-space reasoning via training-format change; mechanism-level, not pretraining-data-level, but bears on latent computation. *(r1: format-internalization, hard-tokens)*
- **MoreHopQA: More Than Multi-Hop Reasoning** (Julian Schnitzler, 2024, arXiv) — <https://arxiv.org/abs/2406.13397>
  - Eval instrument only; useful for measuring gap, not data-level insight *(r2: latent-multihop)*
- **Understanding and Patching Compositional Reasoning in LLMs** ((unknown), 2024, ACL Findings 2024) — <https://arxiv.org/abs/2402.14328>
  - Localizes where latent chaining fails — capacity vs failed composition *(r2: latent-multihop)*
- **Chain-of-Thought Reasoning Without Prompting** (Xuezhi Wang, 2024, NeurIPS) — <https://doi.org/10.52202/079017-2123>
  - Latent reasoning paths in decoding; bears on CAN'T vs WON'T but decoding-time. *(r2: cot-faithfulness)*
- **Distilling System 2 into System 1** (Ping Yu, 2024, arXiv (Meta)) — <https://arxiv.org/abs/2407.06023>
  - Amortizing explicit reasoning into forward pass; post-training distillation, adjacent to latent-reasoning question *(r2: format-internalization)*
- **Compressed Chain of Thought: Efficient Reasoning Through Dense Representations** (Jeffrey Cheng, 2024, arXiv) — <https://arxiv.org/abs/2412.13171>
  - Latent contemplation tokens replace explicit CoT; efficiency method, adjacent *(r2: format-internalization)*
- **A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task** (Jannik Brinkmann, 2024, arXiv/ACL) — <https://arxiv.org/abs/2402.11917>
  - Toy-task mechanistic account of learned multi-step algorithm; informative but synthetic setting *(r2: format-internalization)*
- **How Do Language Models Compose Functions?** ((see paper), 2025, preprint) — <https://arxiv.org/pdf/2510.01685>
  - Mechanistic single-pass composition probe; supporting evidence only *(r1: latent-multihop)*
- **Where to find Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test** (Ziyue Li, 2025, preprint (arXiv)) — <https://arxiv.org/abs/2506.21551>
  - Checkpoint dynamics of circuit formation in real pretraining; adjacent *(r1: mechanistic-formation)*
- **Examining Two-Hop Reasoning Through Information Content Scaling** (David Johnston, 2025, preprint (arXiv)) — <https://arxiv.org/pdf/2502.03490>
  - Capacity-theoretic CAN'T-side account of two-hop; theoretical rather than data-level *(r1: mechanistic-formation)*
- **CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation** (Zhenyi Shen, 2025, preprint) — <https://arxiv.org/abs/2502.21074>
  - Scratchpad-to-latent distillation; adjacent to latent multi-hop but a fine-tuning method, not pretraining data. *(r1: format-internalization)*
- **Implicit Reasoning in Large Language Models: A Comprehensive Survey** (Jindong Li, 2025, preprint (survey)) — <https://arxiv.org/abs/2509.02350>
  - Survey useful as background/citation index for latent reasoning; not load-bearing itself. *(r1: format-internalization)*
- **Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning** (DiJia Su, 2025, preprint) — <https://arxiv.org/abs/2502.03275>
  - Training-format manipulation for partial internalization; mechanism-adjacent, not data selection. *(r1: format-internalization)*
- **DART: Distilling Autoregressive Reasoning to Silent Thought** (Nan Jiang, 2025, arXiv) — <https://arxiv.org/abs/2506.11752>
  - Scratchpad-to-latent distillation; adjacent, post-training not pretraining data *(r2: format-internalization)*
- **Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought** (Hanlin Zhu, 2025, arXiv) — <https://arxiv.org/abs/2505.12514>
  - Theory of latent superposed reasoning states; conceptual background, not data-level *(r2: format-internalization)*
- **Reasoning with Latent Thoughts: On the Power of Looped Transformers** (Nikunj Saunshi, 2025, arXiv/ICLR) — <https://arxiv.org/abs/2502.17416>
  - Depth/looping vs latent reasoning capacity; bears on CAN'T (capacity) side but architectural *(r2: format-internalization)*
- **Unsupervised decoding of encoded reasoning using language model interpretability** (Ching Fang, 2025, arXiv) — <https://arxiv.org/abs/2512.01222>
  - Interpretability probe of internalized reasoning; adjacent to latent multi-hop. *(r2: format-internalization)*
- **Layer-Order Inversion: Rethinking Latent Multi-Hop Reasoning in Large Language Models** ((see paper), 2026, preprint) — <https://arxiv.org/abs/2601.03542>
  - Complicates hop-by-hop mechanistic story; interpretive caveat, not data-level *(r1: latent-multihop)*
- **Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers** ((see paper), 2026, preprint) — <https://arxiv.org/abs/2604.07822v1>
  - Architectural-capacity CAN'T evidence; adjacent to data-level focus *(r1: latent-multihop, format-internalization, mechanistic-formation)*
- **Is Grokking Worthwhile? Functional Analysis and Transferability of Generalization Circuits in Transformers** ((see paper), 2026, preprint) — <https://arxiv.org/pdf/2601.09049>
  - Caveat for grokking-reliant H2 augmentation recipes *(r1: latent-multihop, shortcut-learning)*
- **Reading Between the Dots: Decoding Hidden Computation across Filler Tokens** (unknown, 2026, preprint) — <https://arxiv.org/abs/2607.03502>
  - Probing what latent computation occurs without visible steps; follow-up to Dot-by-Dot. *(r1: format-internalization)*
- **Geometric Factual Recall in Transformers** (Shauli Ravfogel, 2026, arXiv) — <https://arxiv.org/abs/2605.12426>
  - Mechanistic background on whether stored facts can be latently chained *(r2: latent-multihop)*
- **The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models** (Michael Rizvi-Martel, 2026, arXiv) — <https://arxiv.org/abs/2604.06374>
  - Caution against overclaiming latent reasoning; adjacent to H1.2 *(r2: latent-multihop, fresh-2026)*
- **Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning** (Zijun Chen, 2026, AAAI) — <https://doi.org/10.1609/aaai.v40i44.41061>
  - Probes latent computation behind CoT — adjacent to latent multi-hop question. *(r2: cot-faithfulness)*
- **Internalizing LLM Reasoning via Discovery and Replay of Latent Actions** (Zhenning Shi, 2026, arXiv) — <https://arxiv.org/abs/2602.04925>
  - Latent-action internalization of CoT traces; adjacent scratchpad-to-latent method *(r2: format-internalization)*
- **Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers** (Ying Fan, 2026, arXiv) — <https://arxiv.org/abs/2606.31779>
  - When latent matches explicit CoT; architectural but informs internalization gap *(r2: format-internalization)*
- **Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure** (Zirui Li, 2026, arXiv) — <https://arxiv.org/abs/2602.08783>
  - Probes whether latent CoT is genuine multi-step compute vs shortcut; adjacent to H1 but not data-level. *(r2: format-internalization)*
- **Uncovering Latent Reasoning Strategies in Language Models** (Awni Altabaa, 2026, arXiv 2607.17674) — <https://arxiv.org/abs/2607.17674>
  - Characterizes internal latent reasoning strategies; interpretability angle, unclear pretraining/data lever *(r2: fresh-2026)*

### H1.3 — persistence through post-training (27)

- **Emergent Abilities of Large Language Models** (Jason Wei, 2022, TMLR 2022) — <https://arxiv.org/abs/2206.07682>
  - Origin-of-skills backdrop; background framing. *(r2: persistence-posttraining)*
- **LIMA: Less Is More for Alignment** (Chunting Zhou, 2023, NeurIPS 2023) — <https://openreview.net/pdf?id=KBMOKmX2he>
  - Superficial Alignment Hypothesis origin; background for persistence framing *(r1: persistence-posttraining)*
- **The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets** (Samuel Marks, 2023, arXiv) — <https://arxiv.org/abs/2310.06824>
  - Probing capability present pre-finetune; adjacent, not data-level. *(r2: persistence-posttraining)*
- **Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs** (Oded Ovadia, 2023, arXiv) — <https://arxiv.org/abs/2312.05934>
  - Separates knowledge injection from alignment; adjacent to CAN'T/WON'T. *(r2: persistence-posttraining)*
- **An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning** (Yun Luo, 2023, arXiv) — <https://arxiv.org/abs/2308.08747>
  - Forgetting during SFT; peripheral to persistence. *(r2: persistence-posttraining)*
- **Are Emergent Abilities of Large Language Models a Mirage?** (Rylan Schaeffer, 2023, NeurIPS 2023) — <https://arxiv.org/abs/2304.15004>
  - Metric-artifact critique bearing on capability emergence. *(r2: persistence-posttraining)*
- **Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?** (Zorik Gekhman, 2024, EMNLP 2024 (arXiv:2405.05904)) — <https://arxiv.org/abs/2405.05904>
  - Capabilities-come-from-pretraining evidence, but knowledge not reasoning *(r1: persistence-posttraining)*
- **Revisiting the Superficial Alignment Hypothesis** (Mohit Raghavendra, 2024, preprint (arXiv:2410.03717)) — <https://arxiv.org/pdf/2410.03717>
  - Bears on how much post-training adds, but general alignment scaling, not reasoning-specific. *(r1: persistence-posttraining)*
- **From Distributional to Overton Pluralism: Investigating Large Language Model Alignment** (Thom Lake, 2024, arXiv) — <https://arxiv.org/abs/2406.17692>
  - Supports superficial-alignment reading; secondary. *(r2: persistence-posttraining)*
- **LoRA Learns Less and Forgets Less** (Dan Biderman, 2024, TMLR 2024) — <https://arxiv.org/abs/2405.09673>
  - Acquisition vs preservation of capability in FT; adjacent. *(r2: persistence-posttraining)*
- **On the Hardness of Faithful Chain-of-Thought Reasoning in Large Language Models** (Sree Harsha Tanneru, 2024, arXiv) — <https://arxiv.org/abs/2406.10625>
  - Tests whether fine-tuning can fix unfaithful CoT — adjacent to persistence of under-reasoning *(r2: cot-faithfulness)*
- **Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought** (James Chua, 2024, arXiv) — <https://arxiv.org/abs/2403.05518>
  - Training intervention to make CoT causally responsible; post-training fix angle *(r2: cot-faithfulness)*
- **MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations** (Kaixuan Huang, 2025, ICML) — <https://arxiv.org/abs/2502.06453>
  - Behavioral shortcut test incl. post-RL persistence, but GSM-Symbolic already covered similar ground *(r1: shortcut-learning)*
- **LLMs Learn New Skills in RL by Composing Old Ones** (Lifan Yuan, 2025, preprint (arXiv:2509.25123)) — <https://arxiv.org/pdf/2509.25123>
  - Middle-ground elicitation-vs-creation; RL-side but bears on capability boundary *(r1: persistence-posttraining)*
- **MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Model** (Prasanna Mayilvahanan, 2025, preprint (arXiv:2510.11653)) — <https://arxiv.org/html/2510.11653v1>
  - Benchmark for boundary-expansion; useful eval tool but not itself a data/pretraining finding. *(r1: persistence-posttraining)*
- **Activation Control for Efficiently Eliciting Long Chain-of-thought Ability of Language Models** (Zekai Zhao, 2025, preprint (arXiv:2505.17697)) — <https://arxiv.org/html/2505.17697v1>
  - Steering-based 'already in base model' evidence; supportive but inference-time mechanism. *(r1: persistence-posttraining)*
- **Thinking Sparks!: Emergent Attention Heads in Reasoning Models During Post Training** ((see paper), 2025, preprint (arXiv:2509.25758)) — <https://arxiv.org/abs/2509.25758>
  - Circuit-level nuance on elicitation vs creation; secondary to Prakash-style evidence. *(r1: persistence-posttraining)*
- **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation** (Bowen Baker, 2025, preprint (OpenAI)) — <https://arxiv.org/abs/2503.11926>
  - RL pressure decouples CoT from behavior; persistence angle but safety-flavored *(r1: cot-faithfulness)*
- **Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning** (Miles Turpin, 2025, preprint) — <https://www.researchgate.net/publication/393183382_Teaching_Models_to_Verbalize_Reward_Hacking_in_Chain-of-Thought_Reasoning>
  - Fine-tuning intervention to close faithfulness gap; post-training remedy *(r1: cot-faithfulness)*
- **Absolute Zero: Reinforced Self-play Reasoning with Zero Data** (Andrew Zhao, 2025, arXiv) — <https://arxiv.org/abs/2505.03335>
  - RL self-play capability-boundary angle; algorithm-heavy. *(r2: persistence-posttraining)*
- **Reinforcement Pre-Training** (Qingxiu Dong, 2025, arXiv) — <https://arxiv.org/abs/2506.08007>
  - Blurs pretrain/post-train boundary for reasoning formation. *(r2: persistence-posttraining)*
- **ArgInstruct: Specialized Instruction Fine-Tuning for Computational Argumentation** (unknown, 2025, arXiv) — <https://arxiv.org/pdf/2505.22076>
  - SFT on implicit-premise reconstruction; adjacent to persistence through post-training *(r2: enthymemes-completeness)*
- **Examining the Faithfulness of DeepSeek R1's Chain-of-Thought Reasoning** (Chrisanna Cornish, 2025, CHOMPS Workshop) — <https://doi.org/10.18653/v1/2025.chomps-main.2>
  - Whether reasoning-trained model's CoT is genuine touches persistence through post-training. *(r2: cot-faithfulness)*
- **Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection** (Xingwu Chen, 2025, arXiv) — <https://arxiv.org/abs/2506.04695>
  - Theory of what RL changes vs base (sparse critical tokens) — bears on persistence-through-RL. *(r2: hard-tokens)*
- **Mitigating Shortcut Reasoning in Language Models: A Gradient-Aware Training Approach** (2026, preprint) — <https://arxiv.org/abs/2603.20899>
  - Training-time mitigation; relevant to persistence but algorithm-engineering flavored *(r1: shortcut-learning)*
- **On Distinguishing Capability Elicitation from Capability Creation in Post-Training: A Free-Energy Perspective** ((see paper), 2026, preprint (arXiv:2605.08368)) — <https://arxiv.org/pdf/2605.08368>
  - Theory framework for elicitation vs creation; conceptual, unclear empirical load-bearing. *(r1: persistence-posttraining)*
- **New Skills or Sharper Primitives? A Probabilistic Perspective on the Emergence of Reasoning in RLVR** ((see paper), 2026, preprint (arXiv:2602.08281)) — <https://arxiv.org/pdf/2602.08281>
  - On elicitation/creation, but analysis-of-RLVR angle; adjacent to the Yue paper already covered. *(r1: persistence-posttraining)*

### H2.4 — identify reasoning-rich text (34)

- **Datamodels: Predicting Predictions from Training Data** (Andrew Ilyas, 2022, ICML 2022) — <https://arxiv.org/abs/2202.00622>
  - Backbone estimation method behind datamodel-based selection; methodological. *(r2: perplexity-signals)*
- **OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text** (Keiran Paster, 2023, ICLR 2024) — <https://arxiv.org/abs/2310.06786>
  - Baseline math-web curation reference; useful context but not a novel selection signal *(r1: data-selection-reasoning)*
- **Data Selection for Language Models via Importance Resampling (DSIR)** (Sang Michael Xie, 2023, NeurIPS) — <https://arxiv.org/abs/2302.03169>
  - Importance-resampling selection mechanism reusable for reasoning targeting. *(r2: data-selection-reasoning)*
- **MathPile: A Billion-Token-Scale Pretraining Corpus for Math** (Zengzhi Wang, 2023, NeurIPS 2024 D&B / arXiv) — <https://arxiv.org/abs/2312.17120>
  - Math corpus curation comparison point; corpus construction, not a selection-signal study. *(r2: data-selection-reasoning, fresh-2026)*
- **QuRating: Selecting High-Quality Data for Training Language Models** (Alexander Wettig, 2024, ICML 2024) — <https://arxiv.org/abs/2402.09739>
  - Quality/expertise rating close to but not specifically reasoning-richness. *(r1: data-selection-reasoning, perplexity-signals)*
- **InfiMM-WebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning** (Xiaotian Han, 2024, arXiv) — <https://arxiv.org/abs/2409.12568>
  - Math-corpus mining pipeline but multimodal focus; adjacent to reasoning-doc identification. *(r2: data-selection-reasoning)*
- **DataComp-LM: In search of the next generation of training sets for language models** (Jeffrey Li, 2024, NeurIPS 2024 D&B) — <https://arxiv.org/abs/2406.11794>
  - Standard testbed for selection methods; background rather than reasoning-specific. *(r2: data-selection-reasoning)*
- **DsDm: Model-Aware Dataset Selection with Datamodels** (Logan Engstrom, 2024, ICML 2024) — <https://arxiv.org/abs/2401.12926>
  - Datamodels loss-impact selection paradigm; reusable to target reasoning text. *(r2: perplexity-signals)*
- **MATES: Model-Aware Data Selection for Language Model Pretraining with Data Influence Models** (Zichun Yu, 2024, NeurIPS 2024) — <https://arxiv.org/abs/2406.06046>
  - Influence-model-driven pretraining selection at scale; applicable to reasoning targets. *(r2: perplexity-signals)*
- **Scalable Influence and Fact Tracing for Large Language Model Pretraining (TrackStar)** (Tyler Chang, 2024, arXiv) — <https://arxiv.org/abs/2410.17413>
  - Influential examples often don't express the fact — bears on shortcut vs full inference. *(r2: perplexity-signals)*
- **Data Shapley in One Training Run** (Jiachen T. Wang, 2024, ICLR 2025) — <https://arxiv.org/abs/2406.11011>
  - In-run gradient Shapley valuation; could detect valuable reasoning text. *(r2: perplexity-signals)*
- **LESS: Selecting Influential Data for Targeted Instruction Tuning** (Mengzhou Xia, 2024, ICML 2024) — <https://arxiv.org/abs/2402.04333>
  - Gradient-similarity targeted selection; template for scoring docs toward a reasoning task. *(r2: perplexity-signals)*
- **Exploring the Role of Discourse Structure and Tropes in Detecting Enthymemes in Social Media Posts** (unknown, 2024, OpenReview) — <https://openreview.net/forum?id=m7xVR2V8nQ>
  - Discourse features flagging argument gaps could inform reasoning-rich doc detection *(r2: enthymemes-completeness)*
- **MASS: Mathematical Data Selection via Skill Graphs for Pretraining Large Language Models** (2025, preprint) — <https://arxiv.org/abs/2503.14917>
  - Math-targeted selection; relevant but skill-graph mechanism is domain-narrow. *(r1: data-selection-reasoning, fresh-2026)*
- **Language Models Improve When Pretraining Data Matches Target Tasks** (2025, preprint) — <https://arxiv.org/abs/2507.12466>
  - Benchmark-targeted selection with reasoning trade-offs; adjacent evidence for H2.4. *(r1: data-selection-reasoning)*
- **Influence Functions for Efficient Data Selection in Reasoning** (2025, preprint) — <https://arxiv.org/html/2510.06108v1>
  - Selection-for-reasoning method but likely fine-tuning-level, not pretraining *(r1: data-selection-reasoning)*
- **Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models** (2025, preprint) — <https://arxiv.org/abs/2504.14194>
  - Multi-axis rater includes reasoning dimension; useful only if reasoning axis is isolable *(r1: data-selection-reasoning)*
- **MegaMath: Pushing the Limits of Open Math Corpora** (2025, preprint) — <https://arxiv.org/abs/2504.02807>
  - Recent math-corpus curation practice; incremental over OpenWebMath for our questions *(r1: data-selection-reasoning, fresh-2026)*
- **Enhancing Multilingual LLM Pretraining with Model-Based Data Selection** (Bettina Messmer, 2025, preprint (arXiv)) — <https://arxiv.org/abs/2502.10361>
  - perplexity- vs classifier-based selection; breadth check on loss signals *(r1: perplexity-signals)*
- **Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset** (2025, preprint) — <https://arxiv.org/abs/2508.15096>
  - Math corpus extraction at scale; useful background but dataset-engineering rather than a reasoning-richness signal *(r1: fresh-2026, data-selection-reasoning)*
- **RegMix: Data Mixture as Regression for Language Model Pre-training** (Qian Liu, 2025, ICLR) — <https://arxiv.org/abs/2407.01492>
  - Mixture optimization; applicable but not reasoning-specific. *(r2: data-selection-reasoning)*
- **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance** (Jiasheng Ye, 2025, ICLR) — <https://arxiv.org/abs/2403.16952>
  - Predictive mixture scaling; general selection method. *(r2: data-selection-reasoning)*
- **Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training** (Shizhe Diao, 2025, NeurIPS) — <https://arxiv.org/abs/2504.13161>
  - Skill-targeted mixture search incl. reasoning; adjacent. *(r2: data-selection-reasoning)*
- **SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model** (Loubna Ben Allal, 2025, arXiv) — <https://arxiv.org/abs/2502.02737>
  - FineMath classifier + mid-training mixture relevant, but broad model paper, not selection-mechanism study. *(r2: data-selection-reasoning)*
- **Data Efficacy for Language Model Training** (Yalun Dai, 2025, arXiv) — <https://arxiv.org/abs/2506.21545>
  - Learnability-Quality scoring; recent model-based valuation signal. *(r2: perplexity-signals)*
- **BLISS: A Lightweight Bilevel Influence Scoring Method for Data Selection in Language Model Pretraining** (Jie Hao, 2025, arXiv) — <https://arxiv.org/pdf/2510.06048>
  - Generic influence-based pretraining data selection; not reasoning-specific but could support doc identification *(r2: perplexity-signals)*
- **Influence-driven Curriculum Learning for Pre-training on Limited Data** (unknown, 2025, arXiv) — <https://arxiv.org/pdf/2508.15475>
  - Influence-signal ordering of pretraining data; adjacent to selecting reasoning-rich text *(r2: perplexity-signals)*
- **FIRE: Flexible Integration of Data Quality Ratings for Effective Pretraining** (Liangyu Xu, 2025, EMNLP 2025) — <https://doi.org/10.18653/v1/2025.emnlp-main.735>
  - Combining quality raters to select data — mechanism for prioritizing reasoning-rich text. *(r2: fresh-2026)*
- **Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training** (Kailai Yang, 2025, arXiv 2507.15640) — <https://arxiv.org/abs/2507.15640>
  - Domain reweighting mechanism could upweight reasoning-rich data, but not reasoning-specific *(r2: fresh-2026)*
- **Shaping capabilities with token-level data filtering** (2026, preprint) — <https://arxiv.org/html/2601.21571>
  - Token-level selection mechanism, capability-general; adjacent unless reasoning-specific results *(r1: data-selection-reasoning)*
- **A Resource for Enthymeme Detection in Controversial Political Discourse** ((see arXiv page), 2026, preprint) — <https://arxiv.org/abs/2606.12186>
  - enthymeme detection resource — possible signal for reasoning-incomplete docs *(r1: enthymemes-completeness, augmentation-synthetic)*
- **Target-Oriented Pretraining Data Selection via Neuron-Activated Graph** (unknown, 2026, arXiv) — <https://arxiv.org/abs/2604.15706>
  - Target-capability selection signal could be pointed at reasoning; relevance unverified. *(r2: data-selection-reasoning)*
- **Hubs or Fringes: Pretraining Data Selection via Web Graph Centrality** (Vedant Badoni, 2026, arXiv) — <https://arxiv.org/abs/2606.11499>
  - Novel non-content selection signal with claimed reasoning gains; worth a look, not core. *(r2: data-selection-reasoning, fresh-2026)*
- **DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models** (Hao Liang, 2026, arXiv) — <https://arxiv.org/abs/2603.26164>
  - Unified dynamic model-driven selection framework; generic but on-angle. *(r2: perplexity-signals)*

### H2.5 — augment text with reasoning (26)

- **Counterfactually-Augmented SNLI Training Data Does Not Yield Better Generalization Than Unaugmented Data** (William Huang, 2020, arXiv 2010.04762) — <https://arxiv.org/pdf/2010.04762>
  - Negative augmentation result relevant to H2.5, but SNLI finetuning scale. *(r2: shortcut-learning)*
- **Syntactic Data Augmentation Increases Robustness to Inference Heuristics** (Junghyun Min, 2020, ACL) — <https://arxiv.org/pdf/2004.11999>
  - Data-side fix for lexical-overlap heuristics; finetuning-scale but H2.5-shaped. *(r2: shortcut-learning)*
- **CO-NNECT: A Framework for Revealing Commonsense Knowledge Paths as Explicitations of Implicit Knowledge in Texts** (Maria Becker, 2021, IWCS) — <https://arxiv.org/abs/2105.03157>
  - explicitates implicit knowledge into paths — mechanism for augmentation *(r1: enthymemes-completeness, augmentation-synthetic)*
- **Learning by Distilling Context** (Charlie Snell, 2022, preprint) — <https://arxiv.org/abs/2209.15189>
  - Internalizing scratchpad reasoning into weights — relevant precedent for reasoning-augmented training, but SFT-time. *(r1: format-internalization)*
- **Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset** (2024, ACL 2025) — <https://arxiv.org/abs/2412.02595>
  - Production curation+rephrase pipeline; context, not a reasoning-specific result *(r1: data-selection-reasoning, augmentation-synthetic)*
- **Instruction Pre-Training: Language Models are Supervised Multitask Learners** (Daixuan Cheng, 2024, EMNLP 2024) — <https://arxiv.org/abs/2406.14491>
  - Instruction augmentation at pretraining scale; task-general rather than reasoning-completeness-focused *(r1: augmentation-synthetic)*
- **Rephrasing natural text data with different languages and quality levels for LLM pre-training** ((see arXiv 2410.20796), 2024, preprint (arXiv)) — <https://arxiv.org/abs/2410.20796>
  - Boundary conditions for WRAP-style rephrasing; useful caveat, not core *(r1: augmentation-synthetic)*
- **Cosmopedia: synthetic textbooks, blogposts, and stories generated from web-derived prompts** (Loubna Ben Allal, 2024, HuggingFace dataset/report) — <https://huggingface.co/datasets/HuggingFaceTB/cosmopedia>
  - Canonical synthetic textbook corpus; dataset report, relevant reference for augmentation. *(r2: augmentation-synthetic)*
- **Identity Bridge: Enabling Implicit Reasoning via Shared Latent Memory** ((see paper), 2025, preprint) — <https://arxiv.org/abs/2509.24653>
  - Training-data-side trick unlocking two-hop; niche but relevant to H2.5 *(r1: latent-multihop)*
- **Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning** (Xinghao Chen, 2025, preprint (survey)) — <https://arxiv.org/html/2505.16782v2>
  - Survey mapping to reasoning-augmented pretraining; useful index, not load-bearing *(r1: latent-multihop, format-internalization, mechanistic-formation)*
- **BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining** ((DatologyAI), 2025, preprint (arXiv)) — <https://arxiv.org/abs/2508.10975>
  - Which source docs are worth rewriting touches H2.4/5, but lessons are quality-general *(r1: augmentation-synthetic, fresh-2026)*
- **Fast Quiet-STaR: Thinking Without Thought Tokens** ((see arXiv 2505.17746), 2025, preprint (arXiv)) — <https://arxiv.org/abs/2505.17746>
  - Compressing explicit rationales back to implicit is interesting but derivative of covered Quiet-STaR *(r1: augmentation-synthetic, hard-tokens)*
- **Regularization Through Reasoning: Systematic Improvements in LM Classification via Explanation-Enhanced Fine-Tuning** ((see arXiv 2511.02044), 2025, preprint (arXiv)) — <https://arxiv.org/abs/2511.02044>
  - Small-scale evidence appended rationales change what's learned; fine-tuning not pretraining. *(r1: augmentation-synthetic)*
- **BRiTE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning** (2025, preprint) — <https://arxiv.org/abs/2501.18858>
  - Latent-thought bootstrapping like BoLT/Quiet-STaR (covered); read only if it adds beyond those *(r1: hard-tokens)*
- **Beyond Answers: Transferring Reasoning Capabilities to Smaller LLMs Using Multi-Step Rationales** (unknown, 2025, WSDM 2025) — <https://dl.acm.org/doi/10.1145/3701551.3703577>
  - Rationale-injection but framed as distillation to smaller models, not pretraining-corpus augmentation. *(r2: augmentation-synthetic)*
- **Safety Pretraining: Toward the Next Generation of Safe AI** (Pratyush Maini, 2025, arXiv 2504.16980) — <https://arxiv.org/abs/2504.16980>
  - Scale rephrasing/annotation of pretraining data for a target property — methodological parallel to reasoning augmentation, safety objective *(r2: fresh-2026)*
- **Self-Improving Pretraining: using post-trained models to pretrain better models** ((see arXiv 2601.21343), 2026, preprint (arXiv)) — <https://arxiv.org/abs/2601.21343>
  - Closed-loop reasoning augmentation; relevant but not clearly load-bearing. *(r1: augmentation-synthetic)*
- **Understanding by Reconstruction: Reversing the Software Development Process for LLM Pretraining** ((see arXiv 2603.11103), 2026, preprint (arXiv)) — <https://arxiv.org/abs/2603.11103>
  - Reconstructs latent derivation behind code as pretraining text; completeness-adjacent for code. *(r1: augmentation-synthetic)*
- **Thinking Mid-training: Reinforcement Learning over augmented thinking data (Meta RAM)** ((Meta RAM team), 2026, blog (points to Meta RAM report)) — <https://facebookresearch.github.io/RAM/blogs/thinking_midtraining/>
  - Data thinking-augmentation making implicit logic explicit, but mid-training/RL blog — adjacent. *(r1: augmentation-synthetic)*
- **How Can We Synthesize High-Quality Pretraining Data?** ((see arXiv page), 2026, preprint) — <https://arxiv.org/abs/2604.13977>
  - systematic synthesis study; context for which rewriting styles help *(r1: enthymemes-completeness)*
- **Thinking into the Future: Latent Lookahead Training for Transformers** (unknown, 2026, preprint) — <https://arxiv.org/abs/2603.20219>
  - Placement/scaling of latent thinking tokens in training sequences; training-format manipulation, unclear load-bearing. *(r1: format-internalization)*
- **Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training** (2026, preprint) — <https://arxiv.org/abs/2603.02208>
  - Procedural symbolic pretraining data; synthetic rather than found text, adjacent to H2.5 *(r1: fresh-2026)*
- **Privileged Information Distillation for Language Models** (Emiliano Penaloza, 2026, arXiv) — <https://arxiv.org/abs/2602.04942>
  - Distills train-time rationales into a model running without them; related to internalizing augmented reasoning text. *(r2: format-internalization)*
- **Domain-Aware Scaling Laws Uncover Data Synergy** (Kimia Hamidieh, 2026, arXiv) — <https://arxiv.org/abs/2607.11052>
  - Cross-domain data interactions (code→math) in pretraining mixtures — H2(5) composition. *(r2: fresh-2026)*
- **How Can Synthetic Data Improve Multilingual Language Model Pretraining? A Data Quality Perspective** (Tongyao Zhu, 2026, ACL 2026) — <https://doi.org/10.18653/v1/2026.acl-long.1002>
  - Synthetic pretraining data quality analysis — adjacent H2(5) augmentation evidence. *(r2: fresh-2026)*
- **Generating Pretraining Tokens from Organic Data for Data-Bound Scaling** (Zichun Yu, 2026, arXiv 2605.17849) — <https://arxiv.org/abs/2605.17849>
  - Synthesize new tokens from organic data in data-bounded regime — H2(5), general not reasoning-specific. *(r2: fresh-2026)*

### H2.6 — completeness of reasoning chains (21)

- **The Argument Reasoning Comprehension Task: Identification and Reconstruction of Implicit Warrants** (Ivan Habernal, 2018, NAACL) — <https://arxiv.org/abs/1708.01425>
  - canonical implicit-warrant benchmark; relevant to gap-filling but a benchmark *(r1: enthymemes-completeness)*
- **Implicit Knowledge in Argumentative Texts: An Annotated Corpus** (Maria Becker, 2020, LREC) — <https://arxiv.org/abs/1912.10161>
  - annotated omitted/implied info — evidence of how incomplete text reasoning is *(r1: enthymemes-completeness, augmentation-synthetic)*
- **Abductive Commonsense Reasoning** (Chandra Bhagavatula, 2020, ICLR) — <https://arxiv.org/abs/1908.05739>
  - ART missing-explanation benchmark; foundational but a benchmark *(r1: enthymemes-completeness)*
- **Implicit Premise Generation with Discourse-aware Commonsense Knowledge Models** (Tuhin Chakrabarty, 2021, EMNLP) — <https://aclanthology.org/2021.emnlp-main.504/>
  - generates implicit premise of enthymeme; older task-specific but on-topic for H2.6 *(r1: enthymemes-completeness, augmentation-synthetic, cot-faithfulness, fresh-2026)*
- **A Comparative Study on Collecting High-Quality Implicit Reasonings at a Large-scale** (Keshav Singh, 2021, ArgMining Workshop) — <https://arxiv.org/pdf/2104.07924>
  - Collecting warrants at scale bears on annotating reasoning-completeness in corpora *(r2: enthymemes-completeness)*
- **Mind the Gap: Automated Corpus Creation for Enthymeme Detection and Reconstruction in Learner Arguments** (Maja Stahl, 2023, Findings of EMNLP) — <https://arxiv.org/abs/2310.18098>
  - ADU-deletion corpus template for measuring/repairing incomplete reasoning *(r1: enthymemes-completeness)*
- **From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step** (Yuntian Deng, 2024, preprint) — <https://arxiv.org/abs/2405.14838>
  - Training-side converse of completeness; internalization not pretraining data per se *(r1: latent-multihop, format-internalization)*
- **An Argumentation Scheme-Based Framework for Automatic Reconstruction of Natural Language Enthymemes** ((see COMMA proceedings), 2024, COMMA (IOS Press)) — <https://ebooks.iospress.nl/DOI/10.3233/FAIA240310>
  - entailment-verified enthymeme reconstruction pipeline — structurally verifiable completion *(r1: enthymemes-completeness)*
- **SIM-CoT: Supervised Implicit Chain-of-Thought** ((see arXiv 2509.20317), 2025, preprint (arXiv)) — <https://arxiv.org/abs/2509.20317>
  - Explicit vs implicit reasoning supervision trade-off; adjacent to completeness question. *(r1: augmentation-synthetic, format-internalization)*
- **A Logic-based Framework for Decoding Enthymemes in Argument Maps Involving Implicitness in Premises and Claims** (Victor David (et al., see proceedings), 2025, IJCAI) — <https://www.ijcai.org/proceedings/2025/0495.pdf>
  - formal default-logic account of what completing an argument means *(r1: enthymemes-completeness)*
- **TokenSkip: Controllable Chain-of-Thought Compression in LLMs** (Heming Xia, 2025, preprint) — <https://arxiv.org/abs/2502.12067>
  - Shows which reasoning tokens are dispensable — weak evidence on required completeness of chains. *(r1: format-internalization)*
- **Analysing Chain of Thought Dynamics: Active Guidance or Unfaithful Post-hoc Rationalisation?** (unknown, 2025, arXiv 2508.19827) — <https://huggingface.co/papers/2508.19827>
  - Whether explicit reasoning text does inferential work; inference-time focus though. *(r2: shortcut-learning, cot-faithfulness)*
- **DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking** (Lanni Bu, 2025, arXiv) — <https://arxiv.org/abs/2510.17013>
  - Bridging inference in natural text — analogue of implicit premises *(r2: latent-multihop)*
- **Enthymemes in Large Language Models: A Survey** (unknown, 2025, ResearchGate preprint) — <https://www.researchgate.net/publication/397263941_Enthymemes_in_Large_Language_Models_A_Survey>
  - Survey mapping implicit-premise work; useful scoping, not load-bearing evidence *(r2: enthymemes-completeness)*
- **Chain of Draft: Thinking Faster by Writing Less** (Silei Xu, 2025, arXiv) — <https://arxiv.org/pdf/2502.18600>
  - Prompting-time compression of CoT; touches completeness question only indirectly *(r2: enthymemes-completeness)*
- **An Empirical Study of Reasoning Steps in Thinking Code LLMs** (unknown, 2025, arXiv) — <https://arxiv.org/pdf/2511.05874>
  - Step-granularity/reduction tolerance quantifies load-bearing steps; likely inference-time study *(r2: enthymemes-completeness)*
- **Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions** (Leonardo Ranaldi, 2025, ACL (Long)) — <https://doi.org/10.18653/v1/2025.acl-long.843>
  - Making reasoning structure explicit relates to completeness, but prompting-method framing. *(r2: cot-faithfulness)*
- **Mining implicit arguments for reasoning: A survey** (Ekaterina Sviridova, 2026, journal (Sage)) — <https://journals.sagepub.com/doi/10.1177/19462174251344764>
  - recent survey mapping implicit-argument landscape; useful orientation *(r1: enthymemes-completeness)*
- **The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies** (Gabriel Garcia, 2026, arXiv) — <https://arxiv.org/abs/2605.10799>
  - Eval-design caution for step-corruption/completeness experiments we might run. *(r2: cot-faithfulness)*
- **Structural Rationale Distillation via Reasoning Space Compression** (Jialin Yang, 2026, arXiv) — <https://arxiv.org/abs/2605.07139>
  - Rationale representation compression affects student; adjacent to completeness question *(r2: format-internalization)*
- **When Reasoning Hurts Legal Drafting: The Verbalization Bottleneck in Patent Claim Generation** (Lekang Jiang, 2026, arXiv) — <https://arxiv.org/abs/2607.10480>
  - Data point on when explicit verbalized reasoning is unnecessary/harmful; touches completeness. *(r2: format-internalization)*

### H2.7 — perplexity / model-gap signals (33)

- **Deep Learning on a Data Diet: Finding Important Examples Early in Training (EL2N / GraNd)** (Mansheej Paul, 2021, NeurIPS 2021) — <https://arxiv.org/abs/2107.07075>
  - Origin of EL2N/GraNd importance scores — conceptual root of loss-based valuation. *(r2: perplexity-signals)*
- **Revisiting the Uniform Information Density Hypothesis** (Clara Meister, 2021, EMNLP 2021) — <https://arxiv.org/abs/2109.11635>
  - UID/surprisal operationalization — background for information-density token signals. *(r2: hard-tokens)*
- **Irreducible Curriculum for Language Model Pretraining** (Simin Fan, 2023, preprint (arXiv)) — <https://arxiv.org/abs/2310.15389>
  - RHO-loss-style excess-loss curriculum for pretraining; adjacent loss-gap valuation. *(r1: perplexity-signals, hard-tokens)*
- **Self-Influence Guided Data Reweighting for Language Model Pre-training** (Megh Thakkar, 2023, EMNLP 2023) — <https://arxiv.org/abs/2311.00913>
  - Influence-style pretraining reweighting without external reference; adjacent. *(r1: perplexity-signals)*
- **Farewell to Aimless Large-scale Pretraining: Influential Subset Selection for Language Model** (Xiao Wang, 2023, ACL 2023 Findings) — <https://arxiv.org/abs/2305.12816>
  - Influence-function pretraining subset selection; adjacent, not reasoning-specific. *(r1: perplexity-signals)*
- **MiLe Loss: a New Loss for Mitigating the Bias of Learning Difficulties in Generative Language Models** (Zhenpeng Su, 2023, NAACL 2024 Findings) — <https://arxiv.org/abs/2310.19531>
  - Entropy-weighted per-token pretraining loss — token-difficulty signal, adjacent to perplexity-gap idea. *(r1: hard-tokens)*
- **When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale** (Max Marion, 2023, arXiv) — <https://arxiv.org/abs/2309.04564>
  - Perplexity-ranked pruning baseline for perplexity-as-signal; not reasoning-specific. *(r2: data-selection-reasoning, perplexity-signals)*
- **DavIR: Data Selection via Implicit Reward for Large Language Models** (Haotian Zhou, 2023, arXiv) — <https://arxiv.org/abs/2310.13008>
  - Relative loss-reduction learnability — links excess-loss selection to implicit reward. *(r2: perplexity-signals)*
- **Studying Large Language Model Generalization with Influence Functions** (Roger Grosse, 2023, arXiv (Anthropic)) — <https://arxiv.org/abs/2308.03296>
  - EK-FAC influence at LLM scale; foundation for attributing reasoning behavior to source text. *(r2: perplexity-signals)*
- **No Train No Gain: Revisiting Efficient Training Algorithms for Transformer-based Language Models** (Jean Kaddour, 2023, NeurIPS 2023 (arXiv 2307.06440)) — <https://arxiv.org/abs/2307.06440>
  - Skeptical compute-matched counterweight for token-selection-signal claims. *(r2: hard-tokens)*
- **Revisiting Entropy Rate Constancy in Text** (Vivek Verma, 2023, EMNLP Findings 2023) — <https://arxiv.org/abs/2305.12084>
  - Cautions against uniform per-token info load; weak background for surprisal-based detection. *(r2: hard-tokens)*
- **MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models** (Zichun Yu, 2024, NeurIPS 2024) — <https://arxiv.org/abs/2406.06046>
  - Influence-model data selection; loss-based valuation, adjacent to model-gap signal. *(r1: perplexity-signals, data-selection-reasoning, fresh-2026)*
- **Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning** (Ming Li, 2024, ACL 2024) — <https://arxiv.org/abs/2402.00530>
  - Validates weak-model perplexity scoring, but instruction-tuning not pretraining. *(r1: perplexity-signals, hard-tokens)*
- **SmallToLarge (S2L): Scalable Data Selection for Fine-tuning LLMs by Summarizing Training Trajectories of Small Models** (Yu Yang, 2024, NeurIPS 2024) — <https://arxiv.org/abs/2403.07384>
  - Small-model loss-trajectory valuation; fine-tuning setting, transferable idea. *(r1: perplexity-signals)*
- **Harnessing Diversity for Important Data Selection in Pretraining Large Language Models (Quad)** (Chi Zhang, 2024, preprint (arXiv)) — <https://arxiv.org/abs/2409.16986>
  - Cluster-level influence + diversity selection; adjacent data valuation. *(r1: perplexity-signals)*
- **Compute-Constrained Data Selection** (Junjie Oscar Yin, 2024, preprint (arXiv)) — <https://arxiv.org/abs/2410.16208>
  - Cost-benefit comparison of perplexity/influence/classifier selection — context for choosing a signal. *(r1: perplexity-signals)*
- **Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability** (Zicheng Lin, 2024, preprint) — <https://arxiv.org/abs/2411.19943>
  - Token-level reasoning-difficulty signal; RL-time method but signal idea relevant to H2.7. *(r1: hard-tokens)*
- **Surprise! Uniform Information Density Isn't the Whole Story: Predicting Surprisal Contours in Long-form Discourse** (Eleftheria Tsipidi, 2024, EMNLP 2024) — <https://arxiv.org/abs/2410.16062>
  - Surprisal contours as background for locating reasoning-dense spans. *(r2: hard-tokens)*
- **Data-Efficient Pretraining with Group-Level Data Influence Modeling (Group-MATES)** (Zichun Yu, 2025, preprint (arXiv)) — <https://arxiv.org/abs/2502.14709>
  - Group-level influence modeling; adjacent extension of loss-based valuation. *(r1: perplexity-signals)*
- **Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining** (Daouda Sow, 2025, ICLR 2025) — <https://arxiv.org/html/2502.06733v1>
  - instance-level loss reweighting; adjacent but not reasoning-content-specific *(r1: perplexity-signals)*
- **Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning** (Shenzhi Wang, 2025, preprint) — <https://arxiv.org/abs/2506.01939>
  - High-entropy forking tokens as reasoning locus — token-level signal transferable to H2.7, but RL-stage. *(r1: hard-tokens)*
- **Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning** (Jinlong Pang, 2025, ICML 2025) — <https://arxiv.org/abs/2502.01968>
  - Token-level quality selection machinery; SFT-stage but transferable to reasoning-token signals. *(r1: hard-tokens)*
- **Token Weighting for Long-Range Language Modeling** (Falko Helm, 2025, NAACL 2025 Findings) — <https://arxiv.org/abs/2503.09202>
  - Per-token weighting by long- vs short-context predictability — plausible signal for tokens needing multi-hop info. *(r1: hard-tokens)*
- **Token-Level Uncertainty-Aware Objective for Language Model Post-Training** (2025, preprint) — <https://arxiv.org/abs/2503.16511>
  - Hard-token selective training echoes RHO-1; possibly redundant with covered work *(r1: hard-tokens)*
- **Enhancing Large Language Model Reasoning via Selective Critical Token Fine-Tuning** (2025, preprint) — <https://arxiv.org/abs/2510.10974>
  - Critical-token selection in reasoning traces; fine-tuning-time, adjacent to token-level signals *(r1: hard-tokens)*
- **What Makes a Good Curriculum? Disentangling the Effects of Data Ordering on LLM Mathematical Reasoning** (2025, preprint) — <https://arxiv.org/abs/2510.19099>
  - Perplexity as curriculum ordering signal; adjacent to perplexity-gap detection *(r1: fresh-2026)*
- **Perplexity-Aware Data Scaling Law: Perplexity Landscapes Predict Performance for Continual Pre-training** (2025, preprint) — <https://arxiv.org/abs/2512.21515>
  - Perplexity landscapes as diagnostic for CPT — methodological neighbor of H2.7, not reasoning-specific *(r1: fresh-2026)*
- **ESLM: Risk-Averse Selective Language Modeling for Efficient Pretraining** ((see arXiv), 2025, arXiv) — <https://arxiv.org/abs/2505.19893>
  - Online reference-free token-level entropy/loss selection, on the difficulty-signal axis. *(r2: hard-tokens)*
- **Revisiting the Uniform Information Density Hypothesis in LLM Reasoning** (Minju Gwak, 2025, arXiv) — <https://arxiv.org/abs/2510.06953>
  - Entropy profiles locate where info/compute load sits in reasoning text; detection-signal relevance. *(r2: hard-tokens)*
- **Revisiting the UID Hypothesis in LLM Reasoning Traces** (Minju Gwak, 2025, arXiv) — <https://arxiv.org/abs/2510.13850>
  - Hard reasoning steps concentrate entropy; relevant to which tokens carry the reasoning. *(r2: hard-tokens)*
- **Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data** ((see paper), 2026, preprint (arXiv)) — <https://arxiv.org/abs/2601.19936>
  - per-token top-1-vs-target gap statistic repurposable as shortcut/reasoning probe *(r1: perplexity-signals)*
- **Forecasting Downstream Performance of LLMs With Proxy Metrics** (Arkil Patel, 2026, arXiv) — <https://arxiv.org/abs/2605.18607>
  - Cheap token-level proxy signals predicting reasoning outcomes — adjacent to perplexity-gap question. *(r2: data-selection-reasoning, fresh-2026)*
- **Understanding Dynamic Compute Allocation in Recurrent Transformers** (Ibraheem Muhammad Moosa, 2026, arXiv) — <https://arxiv.org/abs/2602.08864>
  - Probes whether token-level compute tracks input complexity; the which-tokens-are-hard question. *(r2: hard-tokens)*

### Other / unbucketed (8)

- **In-context Learning and Induction Heads** (Catherine Olsson, 2022, Transformer Circuits Thread / arXiv) — <https://arxiv.org/abs/2209.11895>
  - Canonical circuit-formation template but ICL-specific, not reasoning/data-level *(r1: mechanistic-formation)*
- **Loss Landscape Degeneracy and Stagewise Development in Transformers** (Jesse Hoogland, 2024, arXiv (devinterp)) — <https://arxiv.org/pdf/2402.02364>
  - Devinterp stagewise formation is background methodology, not directly on shortcut-vs-inference or reasoning-rich data. *(r2: mechanistic-formation)*
- **Chain-of-Thought Is Not Explainability** (Fazl Barez, 2025, preprint (Oxford WhiteBox)) — <https://aigi.ox.ac.uk/wp-content/uploads/2025/07/Cot_Is_Not_Explainability.pdf>
  - Position/synthesis survey of unfaithfulness mechanisms; useful framing not evidence *(r1: cot-faithfulness)*
- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach** (Jonas Geiping, 2025, arXiv) — <https://arxiv.org/abs/2502.05171>
  - Architecture for latent iteration; pretrained but architectural, not data/text-level *(r2: format-internalization)*
- **Detecting Unfaithful Chain-of-Thought via Circuit-Guided Internal-External Discrepancy** ((see arXiv), 2026, preprint) — <https://arxiv.org/pdf/2605.25603>
  - Mechanistic detector of internal/external divergence; tooling not data *(r1: cot-faithfulness)*
- **daVinci-LLM: Towards the Science of Pretraining** (Yiwei Qin, 2026, arXiv) — <https://arxiv.org/abs/2603.27164>
  - Curriculum ordering toward reasoning-intensive data; curriculum, not selection/augmentation core. *(r2: data-selection-reasoning)*
- **What do Language Models Learn and When? The Implicit Curriculum Hypothesis** (Emmy Liu, 2026, arXiv) — <https://arxiv.org/abs/2604.08510>
  - When skills emerge constrains H1/H2 background; not directly on either hypothesis. *(r2: data-selection-reasoning)*
- **PonderLM-3: Adaptive Token-Wise Pondering with Differentiable Masking** (He Li, 2026, arXiv) — <https://arxiv.org/abs/2603.02023>
  - Train-time per-token compute allocation by difficulty; architecture-side, adjacent to hard-token reasoning. *(r2: hard-tokens)*

## Known (already covered by the lit review)

- Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics
- Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization
- Do Large Language Models Latently Perform Multi-Hop Reasoning?
- Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?
- Language models can learn implicit multi-hop reasoning, but only if they have lots of training data
- Multi-Hop Knowledge Composition is Bound by Pretraining Exposure
- Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries
- Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data
- Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
- AttentionInfluence: Adopting Attention Head Influence for Weak-to-Strong Pretraining Data Selection
- Improving Pretraining Data Using Perplexity Correlations
- Predictive Data Selection: The Data That Predicts Is the Data That Teaches (PreSelect)
- Thinking Augmented Pre-training
- To Code, or Not To Code? Exploring Impact of Code in Pre-training
- The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale
- Reasoning to Learn from Latent Thoughts
- STaR: Bootstrapping Reasoning With Reasoning
- Rho-1: Not All Tokens Are What You Need (Selective Language Modeling)
- Rho-1: Not All Tokens Are What You Need
- Reasoning to Learn from Latent Thoughts (BoLT)
- Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning
- Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling
- Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts
- Grokking of Implicit Reasoning in Transformers: A Mechanistic Journey to the Edge of Generalization
- RHO-1: Not All Tokens Are What You Need for Pretraining

## Skip (out of scope on cheap triage — weak signal, kept for queryability)

- Modular Arithmetic: Language Models Solve Math Digit by Digit — Finds sparse heuristic neurons for digit-wise arithmetic patterns — more mechanistic evidence for heuristic (not algorithmic) computation under NTP.
- The Lookahead Limitation: Why Multi-Operand Addition is Hard for LLMs — Attributes arithmetic failure to reliance on pattern-matching without lookahead over carry structure — a concrete shortcut-mechanism case study.
- Training Dynamics of Contextual N-Grams in Language Models — Traces gradual formation of n-gram circuits during pretraining — micro-level evidence of surface-statistics circuits that shortcuts build on.
- In-context Learning in Presence of Spurious Correlations — Studies how ICL is derailed by spurious features — adjacent evidence that shortcut reliance persists into inference-time behavior.
- On the Shortcut Learning in Multilingual Neural Machine Translation — Shows shortcut overfitting emerges late in training and is aggravated by multilingual pretraining — timing-of-shortcut-formation evidence from an adjacent task.
- Back Attention: Understanding and Enhancing Multi-Hop Reasoning in Large Language Models — Analyzes where multi-hop fails internally and adds a back-attention mechanism that lets later-layer info feed earlier layers, boosting latent two-hop accuracy — architectural fix for the 'hopping too late' bottleneck.
- Too Late to Recall: Explaining the Two-Hop Problem in Multimodal Knowledge Retrieval — VLM version of the two-hop timing failure (entity resolved too late to reuse the factual-recall circuit) — corroborates the layer-timing mechanism from a different modality.
- A Survey on Latent Reasoning — Broad survey of reasoning performed in hidden states rather than tokens — background scaffolding for the latent-multihop angle across both hypotheses.
- Predicting the Emergence of Induction Heads in Language Model Pretraining — Attempts to forecast when induction circuits will emerge during pretraining — relevant to predicting reasoning-circuit formation from training dynamics.
- Evolution of Concepts in Language Model Pre-Training — Traces concept/feature evolution across pretraining checkpoints, distinguishing phase-transition vs gradual acquisition — when-does-what-form evidence.
- The Elicitation Game: Evaluating Capability Elicitation Techniques — Evaluates how well fine-tuning/prompting elicit hidden capabilities from password-locked models — operationalizes latent-capability elicitation.
- Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration — Analyzes how exploration depth/breadth govern whether RLVR recovers or extends base-model reasoning — supporting evidence for boundary dynamics.
- Prompting Contrastive Explanations for Commonsense Reasoning Tasks — Early work generating explanations to support commonsense reasoning — background for the generated-explanations-as-training-signal lineage.
- What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions (LoGra) — Makes influence-function data valuation tractable at LLM scale via gradient-structure exploitation — infrastructure for influence-style estimates in H2(7).
- DoGE: Domain Reweighting with Generalization Estimation — Proxy-model gradient-alignment estimates of domain generalization contribution — influence-flavored alternative to DoReMi's excess loss.
- Abductive Inference in Retrieval-Augmented Language Models: Generating and Validating Missing Premises — Detects insufficient evidence and generates+validates candidate missing premises in RAG — recent operationalization of premise-gap filling (H2-6).
- Reverse Thinking Enhances Missing Information Detection in Large Language Models — Shows forward CoT fails to systematically notice omitted information and proposes reverse reasoning to recover it — bears on missing-step recovery (H1).
- An Axiomatic Study of a Modular Evaluation of Enthymeme Decoding — Axiomatic criteria for evaluating quality of enthymeme decodings — useful if the review needs a principled metric for reconstruction completeness.
- Finding the Missing Link: An Algorithmic Approach to Reconstructing Enthymemes — Argumentation-theory algorithm for reconstructing missing premises — theory-side grounding for what counts as the missing link.
- A Step Towards Enthymeme Reconstruction in Online Reviews — Early work reconstructing implicit premises from opinions in reviews — historical anchor for the enthymeme-reconstruction task.
- Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity — Decomposes monitorability into whether influential factors are verbalized at all (faithfulness) vs drowned in verbosity — a measurement framework for the angle.
- Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation — Shows CoT-faithfulness scores swing with the judge/classifier used — methodological caution for any faithfulness metric the review adopts.
- Generative Context Distillation — Lightweight prompt-internalization via joint generative training — a newer variant of internalizing external context into parameters.
- Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains — Compresses reasoning chains into dynamic-length latent representations during training — scratchpad-to-latent compression variant.
- System-1.5 Reasoning: Traversal in Language and Latent Spaces with Dynamic Shortcuts — Learns dynamic shortcuts between language-space and latent-space reasoning — relevant to H1's shortcut-vs-full-inference framing at the mechanism level.
- Skip-Thinking: Chunk-wise Chain-of-Thought Distillation Enable Smaller Language Models to Reason Better and Faster — Chunk-wise CoT distillation with skipping of non-reasoning chunks — distillation-format effects on small-model latent reasoning.
- TLDR: Token Loss Dynamic Reweighting for Reducing Repetitive Utterance Generation — Early token-level dynamic loss reweighting (upweight hard tokens, downweight easy ones) — precursor to hard-token training signals.
- Adaptive Pre-training Data Detection for Large Language Models via Surprising Tokens — Uses 'surprising tokens' (high-loss outliers) as a per-token signal for membership detection — evidence per-token surprisal carries exploitable structure, tangential to H2-7.
- Hierarchical Simplicity Bias of Neural Networks — Peripheral theory extension; covered by stronger simplicity-bias entries.
- Augmenting NLP data to counter Annotation Artifacts for NLI Tasks — Peripheral NLI augmentation; superseded by stronger entries.
- Counterfactual reasoning: Do language models need world knowledge for causal understanding? — Earlier version of the same probing line; redundant with the 2023 paper.
- Spurious Correlations in Machine Learning: A Survey — Generic survey; taxonomy only, no load-bearing result.
- Navigating Shortcuts, Spurious Correlations, and Confounders: From Origins via Detection to Mitigation — Survey; useful framing but not load-bearing for H1/H2.
- Spurious Correlations and Beyond: Understanding and Mitigating Shortcut Learning in SDOH Extraction with LLMs — Narrow applied domain case study.
- Explore Spurious Correlations at the Concept Level in Language Models for Text Classification — Text-classification detection lens; tangential to pretraining reasoning.
- Unveiling Memorization–Generalization Coexistence: A Case Study on Arithmetic Tasks with Label Noise — Narrow label-noise framing; adjacent but not load-bearing.
- Selection Biases: Exploring Order and Token Sensitivity in Large Language Models — Corroborating bias documentation; covered by premise-order and token-bias papers.
- Right for the Wrong Reasons (ACL Anthology page) — Duplicate anthology mirror of the HANS arXiv entry.
- DiscoLoop: Looping Discrete Embeddings and Continuous Hidden States for Multi-hop Reasoning — Architecture-side looping method, not pretraining data level
- LoopRPT: Reinforcement Pre-Training for Looped Language Models — Looped-architecture pretraining objective, not text/data question
- Shattering the Shortcut: A Topology-Regularized Benchmark for Multi-hop Medical Reasoning in LLMs — Domain benchmark design; no pretraining/data implication
- Fine-Tuning vs. RAG for Multi-Hop Question Answering with Novel Knowledge — Knowledge-injection comparison, not capability-boundary or data signal
- CRiT-QA: Evaluating Multi-hop Reasoning with Counterfactual Chains and Distractor Traps — Eval-methodology benchmark only
- Deliberation in Latent Space via Differentiable Cache Augmentation — Inference-time latent compute method, no data-level angle
- Position: Pause Recycling LoRAs and Prioritize Mechanisms to Uncover Limits and Effectiveness — Adapter-composition position paper, tangential
- Selective Induction Heads: How Transformers Select Causal Structures In Context — Mechanistic ICL circuit detail, not pretraining-data question
- How Transformers Implement Induction Heads: Approximation and Optimization Analysis — Theory of IH formation, no data/capability-boundary angle
- A Review of Developmental Interpretability in Large Language Models — Survey of the devinterp program; methodological background, no direct H1/H2 bearing.
- Stagewise Development in Neural Networks (In-Context Learning stages) — ICL stage formation in small models; not about reasoning shortcuts or reasoning-rich text.
- Refined Local Learning Coefficients: Progressive Differentiation of Attention Heads — Component-level LLC methodology; too far from data-level under-reasoning questions.
- Exploring Compositional Generalization (COGS/ReCOGS) by Transformers using RASP — RASP expressivity analysis; theoretical capability characterization, not pretraining/data-level H1/H2.
- Interpreting Arithmetic Mechanism in Large Language Models through Comparative Neuron Analysis — Neuron-level arithmetic dissection largely redundant with covered Arithmetic-Without-Algorithms line.
- A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models — Methods survey; no direct bearing on H1/H2 questions.
- Information-Theoretic Progress Measures reveal Grokking is an Emergent Phase Transition — Grokking phase-transition analysis on toy tasks; covered adequately by Grokked Transformers line.
- Grokking as a Falsifiable Finite-Size Transition — Grokking phase-transition theory; not pretraining-data-level.
- A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization — Architecture/regularization mapping of grokking timing; off the data-level questions.
- What Can Grokking Teach Us About Learning Under Nonstationarity? — Grokking under nonstationarity; tangential to both hypotheses.
- The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology — Architectural fix for grokking plateau; architecture-level, not data-level.
- CRAW4LLM: Efficient Web Crawling for LLM Pretraining — Crawl prioritization by generic influence, not reasoning-specific selection.
- Demystifying Data Organization for Enhanced LLM Training — Generic data-organization study, no reasoning-specific angle stated.
- Train a Unified Multimodal Data Quality Classifier with Synthetic Data — Multimodal quality classifier; generic quality filtering, not reasoning-targeted.
- TiKMiX: Take Data Influence into Dynamic Mixture for Language Model Pre-training — Generic influence-based mixture reweighting, no reasoning-content angle.
- Beyond Repetition: Text Simplification and Curriculum Learning for Data-Constrained Pretraining — Curriculum/simplification in low-data regime, not reasoning-specific.
- Analyzing Similarity Metrics for Data Selection for Language Model Pretraining — Embedding-similarity methodology, not reasoning-targeted selection.
- A Knowledge-Injected Curriculum Pretraining Framework for Question Answering — Small-scale QA curriculum framework, tangential to both hypotheses.
- Synthetic Rewriting as a Quality Multiplier: Evidence from Portuguese Continued Pretraining — Non-English rewriting case study; redundant with stronger WRAP/Recycling evidence.
- Self-Improvement of Large Language Models: A Technical Overview and Future Outlook — Secondary survey of self-generated reasoning loops; not load-bearing.
- A.X K1 Technical Report — Peripheral model report, only speculative synthetic-data mention.
- GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts — Multilingual MoE report, tangential to reasoning augmentation.
- Efficient Online Data Mixing for Language Model Pre-Training (ODM) — Online domain reweighting by per-group loss; mixture engineering, not reasoning-specific.
- Enhancing Training Data Attribution for Large Language Models with Fitting Error Consideration (Debias & Denoise Attribution) — Tangential TDA refinement, not a selection method.
- SLAP: Stratified Loss-based Pruning for On-Policy Data-Efficient Instruction Tuning — Instruction-tuning pruning; post-pretraining stage, narrow.
- Automatic Document Selection for Efficient Encoder Pretraining (Cynical Data Selection) — Older cross-entropy domain-selection baseline; encoder pretraining, dated.
- Scalable Influence and Fact Tracing for Large Language Model Pretraining — Duplicate of TrackStar entry (item 135).
- Learning to Refine Hidden States for Reliable LLM Reasoning — Latent-state refinement method; no pretraining/data implication
- Implicit Reasoning for Large Language Model-based Generative Recommendation — Recommendation application; out of scope
- Faithfulness Tests for Natural Language Explanations — Explanation-faithfulness evaluation methodology; no pretraining/data angle
- Are self-explanations from Large Language Models faithful? — Faithfulness measurement of self-explanations; no training/data implication
- A Causal Lens for Evaluating Faithfulness Metrics — Inference-time faithfulness metric methodology; no pretraining/data implication.
- On Measuring Faithfulness or Self-consistency of Natural Language Explanations — Explanation-metric methodology, not training/data-level.
- Towards Faithful Natural Language Explanations: A Study Using Activation Patching in Large Language Models — Mechanistic faithfulness method, no data/pretraining angle.
- Disentangling the Effects of Unlearning in Measuring Parametric Faithfulness of Chain-of-Thought — Metric-scrutiny follow-up, out of scope.
- A Closer Look at Bias and Chain-of-Thought Faithfulness of Large (Vision) Language Models — Turpin-style bias eval extended to VLMs; no training/data implication.
- C2-Faith: Benchmarking LLM Judges for Causal and Coverage Faithfulness in Chain-of-Thought Reasoning — Benchmark for LLM judges; out of scope.
- NSF-CoT: Neuro-Symbolic Formal Verification of Chain-of-Thought Faithfulness in Contextual QA — Formal verification tooling, no training/data implication.
- FACET: Measuring Attribution Faithfulness in Multi-Factor LLM Reasoning — Attribution-faithfulness metric, out of scope.
- Evaluating Step-by-Step Reasoning through Symbolic Verification — Step-validity checking method, no data-level angle.
- Stepwise Verification and Remediation of Student Reasoning Errors with Large Language Models — Tutoring/process-reward application, out of scope.
- Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning — PRM engineering with no capability-boundary or data angle.
- Faithful Chain-of-Thought Reasoning Through Question-Guided Faithfulness Verification Prompting — Pure prompting-time trick.
- A Comprehensive Evaluation of Chain-of-Thought Faithfulness in Persian Classification Tasks — Language-specific faithfulness benchmark, out of scope.
- SPD-Faith Bench: Diagnosing and Improving Faithfulness in Chain-of-Thought for Multimodal LLMs — Multimodal CoT faithfulness benchmark; no pretraining/data-level bearing
- Neutralizing Bias in LLM Reasoning using Entailment Graphs — Inference-time debiasing via entailment graphs; tangential to H1/H2
- A Comprehensive Evaluation of Multilingual Chain-of-Thought Reasoning: Performance, Consistency, Faithfulness — Broad multilingual eval, not causal or data-level
- Reasoning That Leaks, Fine-Tuning That Amplifies: Exposing the Hidden Threats of Chain-of-Thought — Security/privacy angle on CoT leakage; not capability-boundary or data-level
- SKIntern: Internalizing Symbolic Knowledge for Distilling Better CoT Capabilities into Small Language Models — Small-model CoT distillation engineering; no pretraining-data angle
- KaVa: Latent Reasoning via Compressed KV-Cache Distillation — KV-cache distillation efficiency method; no data/capability-boundary angle
- CIRF: Tokenizing Chain-of-Thoughts into Reusable Functional Units for Efficient Latent Reasoning — CoT compression for efficiency; not pretraining/data-level
- Skip-Thinking: Chunk-wise Chain-of-Thought Distillation — Distillation/prompting-time compression, no pretraining/data angle.
- Latent Thoughts Tuning: Bridging Context and Reasoning with Fused Information in Latent Tokens — Training-format latent-token architecture, no corpus/data implication.
- Reinforcement Learning for Latent-Space Thinking in LLMs — RL algorithm engineering for latent reasoning, no capability-boundary/data angle.
- Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs — Latent-computation architecture, no data/pretraining implication.
- Latent Chain-of-Thought Improves Structured-Data Transformers — Narrow structured-data latent CoT, no reasoning-corpus angle.
- LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models — Architecture conversion to looped reasoner, tangential to data/format.
- Language Modeling with Learned Meta-Tokens — Pause/meta-token architecture for extra compute, no data implication.
- Seq-VCR: Preventing Collapse in Intermediate Transformer Representations for Enhanced Reasoning — Representation regularizer + pause tokens, architecture-level.
- Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production — Generation-time adaptive pausing, no pretraining/data angle.
- AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth — Adaptive-depth architecture answer to per-token difficulty, no data implication.
- CURE: Critical-Token-Guided Re-Concatenation for Entropy-Collapse Prevention — RL entropy-collapse engineering; no pretraining/capability-boundary angle.
- Probing the Difficulty Perception Mechanism of Large Language Models — Internal difficulty encoding; inference-time probing, no pretraining/data angle.
- Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification — Inference-time critical-token identification; pure prompting trick.
- EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer — Distillation token scheduling, not pretraining data or capability boundary.
- From Input Perception to Predictive Insight: Modeling Model Blind Spots Before They Become Errors — Diagnostic token-level failure prediction; not pretraining/data selection.
- Data Mixing for Large Language Models Pretraining: A Survey and Outlook — Survey of mixture methods; orientation only, not load-bearing.
- Structured Thoughts For Improved Reasoning And Context Pruning — Inference-time structured reasoning format + context pruning; no training/data implication.
- CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning — Preference-model pretraining for RM; peripheral to reasoning-data-in-pretraining.
- A Readability-Driven Curriculum Learning Method for Data-Efficient Small Language Model Pretraining — Readability-ordered curriculum; weak-signal, not load-bearing for completeness.
- Implicit Reasoning Steering via Concept Chaining — Inference-time steering of latent multi-hop; no pretraining/data implication
- Characterizing Narrative Content in Web-scale LLM Pretraining Data — Corpus content characterization but about narrative, not reasoning-richness
- An Asymptotic Theory of Chain-of-Thought in In-Context Learning — ICL theory of CoT; no pretraining-data-level implication
- To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining — RAG-oriented memorization scaling; only loosely touches shortcut framing, off-scope

## Untriaged (triage agents did not return these — re-triage before dismissing)

- **What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation** (Aaditya Singh, 2024) — <https://proceedings.mlr.press/v235/singh24c.html>
- **Crosscoding Through Time: Tracking Emergence & Consolidation of Linguistic Representations Throughout LLM Pretraining** ((see arXiv), 2025) — <https://arxiv.org/pdf/2509.05291>
- **When Do Attention Circuits Form? Developmental Trajectories of Capability and Attention-Sink Emergence Across Three 1B-Class Architectures** ((see arXiv), 2026) — <https://arxiv.org/pdf/2606.02378>
- **Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs** ((see arXiv), 2025) — <https://arxiv.org/html/2507.21914v1>
- **SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training** (Tianzhe Chu, 2025) — <https://arxiv.org/abs/2501.17161>
- **Spurious Rewards: Rethinking Training Signals in RLVR / Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs** ((see paper), 2026) — <https://arxiv.org/abs/2601.11061>
- **Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models (REWIRE)** (2025) — <https://arxiv.org/abs/2506.04689>
- **Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling (WRAP)** (Pratyush Maini, 2024) — <https://arxiv.org/abs/2401.16380>
- **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** (Eric Zelikman, 2024) — <https://arxiv.org/abs/2403.09629>
- **How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data** ((see arXiv 2604.13977), 2026) — <https://arxiv.org/abs/2604.13977>
- **The Pragmatic Persona: Discovering LLM Persona through Bridging Inference** (unknown, 2026, arXiv) — <https://arxiv.org/html/2604.24079v1>
- **LoRi: Low-Rank Distillation for Implicit Reasoning** (Ryan Solgi, 2026, arXiv) — <https://arxiv.org/abs/2606.05315>

## Read batch 1 (launched 2026-07-23, workflow `wf_4006ecb6-289`) — 40 papers

Curated from the 126 must-reads: prioritized H2.5/H2.6 (augmentation + completeness, load-bearing for open question
#2), 2024–2026 pretraining-level evidence, and anything that could change the reverse-filter plan (H2.4/H2.7); the
strongest H1 items included, pre-LLM background classics deferred. Curation notes: `2603.06114` in the must-reads is
the already-read Feng & Hunter enthymemes paper (triage error → known); RATIONALYST excluded (Dongwei's own paper);
4 within-list duplicates collapsed (Two-Hop Curse, Recycling-the-Web, Rewriting-Pre-Training-Data, EntiGraph). The
~86 must-reads NOT in this batch remain queued above — nothing is dropped.

H2.5 (10): EntiGraph 2409.07431 · ToW 2410.16235 · MIND 2410.12881 · SwallowCode/Math rewriting 2505.02881 ·
Recycling the Web 2506.04689 · Demystifying Synthetic Data 2510.01631 · Kinetics of Reasoning 2510.25791 ·
Provably Internalize CoT 2605.28600 · Grokking in the Wild 2504.20752 · Procedural Knowledge at Scale 2604.01348
H2.6 (6): Skip Steps 2411.01855 · Inefficient-Reasoning Bias 2507.05362 · Synthetic Logic Corpus 2411.12498 ·
Zipping the Thought 2605.28008 · Less is More Tokens 2509.05226 · The Model Says Walk 2603.29025
H2.4 (6): Procedural Knowledge Drives Reasoning 2411.12580 · Influence-Function Attributes 2505.19949 ·
Essential-Web 2506.14111 · Data-Quality Illusion 2510.00866 · Reasoning Quality Emerges Early 2606.26797 ·
Beyond Pure Code 2605.19762
H2.7 (5): ScalingFilter 2408.08310 · Perplexed by Perplexity 2405.20541 · rBridge 2509.21013 ·
Generalization-vs-Memorization 2407.14985 · Signal in the Steps 2510.03988
H1.1 (5): Pitfalls of NTP 2403.06963 · Faith and Fate 2305.18654 · Physics of LM 3.2 2309.14402 ·
Implicit Reasoning through Shortcuts 2503.07604 · Composition Collapse 2605.26789
H1.2 (4): Two-Hop Curse 2411.16353 · Identity Bridge 2509.24653 · SynthWorlds 2510.24427 ·
U-Shaped Implicit-Reasoning Scaling 2504.03635
H1.3 (4): Echo Chamber 2504.07912 · Base Models Know How 2510.07364 · Spurious Correlations Post-Training
2505.05704 · Pre/Mid-Training × RL Interplay 2512.07783
