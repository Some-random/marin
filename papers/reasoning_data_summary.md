# Reasoning Data in Pretraining: Paper Summaries

## Synthetic Data Composition

### Phi-1 "Textbooks Are All You Need" (Microsoft, 2023)

**Data approach**: Three datasets for code:
- **Filtered code-language** (~6B tokens): Subset of The Stack + StackOverflow, filtered by a classifier trained on GPT-4 annotations. Classifier predicts "educational value" using embeddings from a pretrained codegen model.
- **Synthetic textbook** (<1B tokens): GPT-3.5 generated Python textbooks. Diversity achieved by conditioning on random topics from a seed vocabulary.
- **Synthetic exercises** (~180M tokens): Python exercises with solutions, also GPT-3.5 generated.

**Key insight**: Quality >> quantity. 1.3B model on 7B tokens beats 16B models on 577B tokens. The filtered+synthetic "CodeTextbook" data is the key — not model architecture.

**Datasets available**: Model released, but training data NOT released (proprietary synthetic generation).

---

### Phi-3 Technical Report (Microsoft, 2024)

**Data approach**: Two-phase pretraining:
- **Phase 1**: Mostly web data filtered by "educational level" — general knowledge and language understanding
- **Phase 2**: More heavily filtered web data (subset of Phase 1) + synthetic data teaching "logical reasoning and various niche skills"

**Key insight**: "Data Optimal Regime" — for small models, filter data to match the model's capacity. Remove content that needs world knowledge (e.g., sports scores) to "leave more model capacity for reasoning." Phi-3-mini (3.8B) on 3.3T tokens matches GPT-3.5/Mixtral 8x7B.

**Datasets available**: NOT released (proprietary).

---

### Phi-4 Technical Report (Microsoft, 2024, arxiv 2412.08905)

40% synthetic data in pretraining. Synthetic data acts as "spoonfeeding" — each token predictable from context, making reasoning patterns easier to learn. Must be diverse and mixed with organic data.

**Key insight**: Structure matters more than domain content. The predictability of each token from its context is what makes synthetic data effective for teaching reasoning patterns.

**Datasets available**: NOT released (proprietary).

---

### Demystifying Synthetic Data in LLM Pre-training (Kang et al., EMNLP 2025, arxiv 2510.01631)

Large-scale study (1000+ LLMs, 100k+ GPU hours) comparing natural web data vs. synthetic types (rephrased text, generated textbooks) and their mixtures.

**Key findings**:
- Pure synthetic data is NOT superior to CommonCrawl
- Textbook-style data alone performs notably WORSE
- Mixing ~30% rephrased synthetic data with web text speeds up training 5-10x
- Explains why OpenThoughts (pure reasoning traces) hurts in our experiments

---

### EntiGraph: Synthetic Continued Pretraining (Gao et al., Stanford, ICLR 2025 Oral, arxiv 2409.07431)

**Recipe:** Extract entities from source documents with an LLM → enumerate all entity pairs and triplets → prompt gpt-4-turbo to "analyze relations among given entities" for each pair/triplet → collect outputs as synthetic corpus. No explicit knowledge graph built — the graph is implicit in the entity combinations.

**Scale:** 265 source articles (1.3M tokens) → 455M synthetic tokens (350x amplification). Student: Llama 3 8B, continued pretraining for 2 epochs with RedPajama replay.

**Results:** QuALITY QA accuracy: base 39.5% → EntiGraph CPT 56.2% (vs GPT-4 closed-book 51.3%). Provides 80% of RAG's improvement. Scaling is log-linear in synthetic token count.

**Key insight**: The combinatorial relational structure drives diversity and quality. Directly relevant to the causal bridge idea — relational structure between real entities is what makes synthetic data effective.

**Applicability to us:** At our scale (200M source tokens), 350x amplification = 70B synthetic tokens — impractical for generation cost. More importantly, EntiGraph targets domain-specific knowledge acquisition (memorizing facts from niche articles), NOT general reasoning. The causal bridge idea extends this by targeting causal/relational reasoning rather than factual recall.

---

## Pretraining Sets the Ceiling

### Echo Chamber (Zhao, Kakade, Jelassi, Malach et al., 2025, arxiv 2504.07912)

End-to-end study training models from scratch with controlled pretraining data, then RL fine-tuning.

**Key finding**: RL consistently converges toward the dominant patterns in pretraining data — it amplifies rather than creates reasoning. Pretraining data composition is the PRIMARY determinant of reasoning ability; post-training cannot compensate.

---

### Pre-Training vs Mid-Training vs RL (Zhang, Neubig, Yue, CMU, 2025, arxiv 2512.07783)

Controlled experiments with synthetic reasoning tasks isolating contributions of each training stage.

**Key findings**:
- Pretraining sets the ceiling
- Mid-training significantly enhances performance vs RL alone
- RL produces real capability gains ONLY when pretraining leaves sufficient headroom AND RL targets the model's "edge of competence"
- Contextual generalization requires minimal but sufficient pretraining exposure

---

### Front-Loading Reasoning (NVIDIA, 2025, arxiv 2510.03264)

**Recipe:** 8B hybrid transformer (Mamba 2 + attention + FFN), 1T tokens, 512 H100s. 80% base corpus + 20% reasoning data during pretraining.

**Reasoning data sources:**
- D_LDQ (Large-Scale, Diverse): Nemotron-Pretraining-SFT-v1 — 336B tokens. ~56% math, ~17% code, ~27% science/general. Heterogeneous quality.
- D_SHQ (Small-Scale, High-Quality): OpenThoughts 1.2M examples. 71% math, 21% code, 8% science. Long CoT traces.
- D_LMQ (Mixed-Quality): D_LDQ ∪ D_SHQ
- D_ALF (Answer-Length Filtered): Subset of D_LDQ where answer > 4096 tokens

**Post-training:** SFT on 4.8M examples (AdamW, LR=5e-6, context 32k), then GRPO RL (1 epoch, LR=1e-6, 128 prompts x 8 rollouts).

**Key numbers:** After full pipeline: M_base 37.9% → M_LMQ 56.7% average across MATH-500/GSM8K/AIME/GPQA/MMLU/MMLU-Pro/LiveCodeBench. The 19% gap cannot be closed by doubling SFT data.

**Critical ablations:**
- Diversity beats quality in pretraining (D_LDQ +9.1% over D_SHQ at base model stage)
- Quality beats diversity in SFT (opposite direction)
- "Latent effect": adding high-quality data to diverse pretraining shows minimal immediate benefit but unlocks +4.25% after SFT
- Catch-up fails: doubling SFT data for M_base (26.6% → 34.0%) still falls short of even the weakest reasoning-pretrained model (37.3%)

**Applicability to us:** The 20% reasoning mix ratio is testable at our scale. The asymmetric principle (diversity for pretraining, quality for SFT) is actionable. Scale gap is enormous (1T tokens vs our 200M-1B). The 336B diverse reasoning dataset is not public.

---

## Code and Reasoning

### How Does Code Pretraining Affect Task Performance? (Petty, van Steenkiste, Linzen, TMLR 2025, arxiv 2409.04556)

Controlled experiments interleaving code and natural language at varying ratios.

**Key findings**:
- Code improves compositional tasks (semantic parsing, math)
- Code HARMS tasks requiring linguistic structure (syntax, morphology) and world knowledge
- Matches our finding that code alone hurts all benchmarks

---

### To Code, or Not To Code? (Meta, 2024, arxiv 2408.10914)

Ablations at 470M-2.8B scale.

**Key findings**:
- Code yields up to 8.2% relative increase in NL reasoning, 4.2% in world knowledge, 12x boost in code performance
- Code QUALITY has outsized impact
- Tension with Petty et al.: suggests the type/quality of code matters more than simple presence/absence

---

## Abstract Reasoning Transfer

### Warm Up Before You Train (Shrestha et al., NYU Abu Dhabi, 2025, arxiv 2505.13718)

**Toy domain:** Knights & Knaves (K&K) logic puzzles — 5,000 examples with 3-7 characters. Domain requires extensive multi-step boolean logic reasoning with zero domain-specific knowledge.

**Recipe:** Generate long CoT traces from QwQ-32B teacher on K&K puzzles (no filtering, wrong answers kept). SFT on traces with very low LR (1e-6), 3 epochs, ~20 min on 6 H100s for 1.5B model. Then RLVR on target domain with only 50-100 examples.

**Key numbers:**
- Qwen2.5-3B warmup only: MATH 43.8% → 54.0% (+10.2), HumanEval+ 32.5% → 47.8% (+15.3), MMLU-Pro 29.2% → 38.2% (+9.0)
- Qwen2.5-14B warmup only: MATH 55.6% → 77.4% (+21.8), MMLU-Pro 52.7% → 62.7% (+10.0)
- Cross-domain: RLVR on HumanEval+ causes -13.8% MATH regression for base model, but only -1.4% for warmed-up model
- Warmup + 100 MATH examples RLVR matches full-data RLVR (7,500 examples)

**Critical detail:** This is a POST-training intervention (SFT on pretrained base model), NOT pretraining from scratch. Model sizes tested: 1.5B-14B. K&K dataset is tiny (5,000 examples) but traces are ~1600-3600 tokens.

**Applicability to us:** Very cheap to try as a bonus on any pretrained model. Doesn't address H1 (what data to pretrain on) — it's a post-training trick. Could test whether our 300M models have latent reasoning capacity. Model size (1.5B-14B) is larger than our 300M — unclear if 300M can absorb long reasoning traces.

---

### Scaling Laws for Implicit Reasoning at Pretraining (2025, arxiv 2504.03635)

Pretrains LMs from scratch on a synthetic implicit multihop reasoning environment replicating knowledge graph structure.

**Key finding**: Overparameterization can IMPAIR implicit reasoning due to excessive memorization. Establishes scaling laws for implicit reasoning during pretraining. Non-trivial relationship between model size and reasoning generalization.

---

## Data Selection and Curriculum

### Perplexity Correlations for Data Selection (Thrush, Potts et al., ICLR 2025, arxiv 2409.05816)

Statistical framework: LLM losses on pretraining texts correlate with downstream benchmark performance. Selecting high-correlation documents is an effective data selection method. Uses 90 LLMs from Open LLM Leaderboard.

**Key finding**: Outperforms DSIR on every benchmark and matches DataComp-LM's hand-engineered classifier. Scales to 1.4B. Practical tool for identifying which data domains actually predict downstream reasoning.

---

### Curriculum Learning for LLM Pretraining (Elgaar, Amiri et al., 2026, arxiv 2601.21698)

Trains Pythia models (14M-1B) for 300B tokens under three linguistically motivated curricula vs. random ordering.

**Key finding**: Training follows a SHARED sequence of latent phases regardless of ordering; curricula mainly change within-phase data exposure. At smaller models (up to 160M), random ordering has higher gradient noise and worse final accuracy. At 300M-1.4B, curriculum effects are modest but real.

---

### General Intelligence Requires Reward-based Pretraining (Gershman et al., Harvard, 2025, arxiv 2502.19402)

Argues that next-token prediction pretraining constrains models to a local minimum of reasoning ability. Proposes RL from scratch on a curriculum of synthetic tasks as an alternative.

**Key insight**: Token-level prediction lets models exploit spurious correlations instead of learning underlying reasoning algorithms. Explains why OpenThoughts teaches surface CoT patterns rather than reasoning procedures.

---

## Procedural Knowledge

### Procedural Knowledge in Pretraining Drives Reasoning (Ruis et al., 2024, arxiv 2411.12580)

**Key finding**: Procedural knowledge — data demonstrating how to derive something step by step — is 10x overrepresented in influential pretraining documents for reasoning. Code on StackExchange is particularly influential. The same procedural documents help across different reasoning questions of the same type.

**Relevance to our experiments**: Explains why OpenWebMath (math web pages showing procedures) helped SciQ while OpenThoughts (exploratory CoT) didn't — OWM contains procedural demonstrations, OT contains exploratory reasoning traces.

---

### Content-Free Synthetic Tasks (Percy Liang et al., 2022, arxiv 2206.10139)

Content-free synthetic pretraining tasks can close ~65% of the gap to natural pretraining.

**Key insight**: Structural patterns matter even without semantic content — the procedural structure of data, not its domain content, drives reasoning capability.

---

### Procedural Pretraining (2026, arxiv 2601.21725)

Extends the content-free approach with more structured procedural tasks.

**Relevance**: Further evidence that abstract procedural structure transfers to downstream reasoning.

---

## OLMo 2 Data Recipe (AI2, 2025)

**Pretraining data (OLMo 2 Mix 1124)** — 3.9T tokens:
- DCLM-Baseline: 3.71T tokens (web pages)
- StarCoder: 83B tokens (code)
- peS2o: 58.6B tokens (academic papers)
- arXiv: 20.8B tokens (STEM papers)
- OpenWebMath: 12.2B tokens (math web pages)
- Algebraic Stack: 11.8B tokens (math proofs code)
- Wikipedia & Wikibooks: 3.7B tokens

**Mid-training data (Dolmino Mix 1124)** — two parts:
1. **High Quality Subset** (~833B tokens): DCLM top 7%, FLAN, peS2o, Wikipedia, Stack Exchange
2. **Math Mix** (~10.7B tokens): TuluMath, Dolmino SynthMath, TinyGSM-MIND, MathCoder2, Metamath, GSM8K

**Key insight**: Two-stage training. Stage 1 (90-95% FLOPs) on mostly web data. Stage 2 "mid-training" (5-10% FLOPs) on high-quality filtered web + synthetic math data.

**Datasets available**: YES — all open via HuggingFace (`olmo-mix-1124`, `dolmino-mix-1124`).

---

## Downloadable Reasoning Datasets (ordered by relevance)

### Directly Available on HuggingFace:
1. **OpenWebMath** (`open-web-math/open-web-math`) — 12.2B tokens of math web pages
2. **Algebraic Stack** (from ProofPile II, `EleutherAI/proof-pile-2`) — 11.8B tokens of math proofs
3. **peS2o** (from Dolma, `allenai/dolma`) — 58.6B tokens of academic papers
4. **arXiv** (from RedPajama, `togethercomputer/RedPajama-Data-1T`) — 20.8B STEM papers
5. **FLAN** (`Muennighoff/flan`) — 17B tokens of instruction data
6. **TuluMath** (check `allenai/tulu-*`) — 230M synthetic math
7. **GSM8K** (`openai/gsm8k`) — 2.74M math word problems
8. **Stack Exchange** — 1.26B Q&A tokens
9. **Wikipedia** — 3.7B tokens

### Likely Available (need to verify):
10. **Nemotron-Pretraining-SFT-v1** — 336B tokens reasoning data (NVIDIA, check HF)
11. **OpenMathInstruct** / Guha et al. dataset — 1.2M high-quality CoT examples
12. **TinyGSM-MIND** — 6.48B synthetic math
13. **Dolmino SynthMath** — 28.7M synthetic math
14. **MathCoder2 Synth Books** — 3.87B synthetic math

### NOT Available (proprietary):
- Phi-1/2/3/4 synthetic textbook data
- Phi-1 code quality classifier
- Phi-3 "educational level" web filter

---

## Synthesis: What the Literature Says About H1

1. **Pure reasoning data hurts; ~30% mixed with web data is optimal.** Our OpenThoughts results are consistent with the broader finding (Kang et al.).
2. **Diversity of reasoning patterns matters more than domain specificity.** OWM's SciQ gains saturate because math domain knowledge doesn't transfer. NVIDIA's front-loading paper shows BROAD diversity (+11%) is the key.
3. **Relational/combinatorial structure drives quality.** EntiGraph's entity-relationship approach and the causal bridge idea share the same insight.
4. **Abstract reasoning from toy domains DOES transfer.** Knights & Knaves → MATH/code transfer (Warm Up paper) suggests content-free procedural reasoning is a viable path.
5. **Pretraining is the ceiling — post-training amplifies but cannot create.** Validated by Echo Chamber, Front-Loading, and Pre-Training vs RL papers.
