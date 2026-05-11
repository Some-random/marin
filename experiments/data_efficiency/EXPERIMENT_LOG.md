# Experiment Log: Data Efficiency & Reasoning Pretraining

## Pre-May 2: Paper Replication

### Hypothesis
Replicate "Pre-training Under Infinite Compute" paper results on local 8x A100-40G GPUs.

### Runs

| Run | Model | Data | Tokens | Epochs | WD | LR | Steps | Time | dclm_val | Notes |
|-----|-------|------|--------|--------|-----|-----|-------|------|----------|-------|
| 300M baseline | 300M (seq_len=4k) | dclm_200m | 200M | 8 | 0.1 | 1e-3 | 6400 | ~1.5h | **3.797** | Paper gets 3.785. Match. |
| 1.4B regularized (dclm_200m) | 1.4B (seq_len=4k) | dclm_200m | 200M | 8 | 3.2 | 1e-3 | 6400 | ~3.5h (TE) | **3.413** | Paper single-model best: 3.462. We beat it slightly — likely dclm_200m is a curated subset. |
| 1.4B (dclm_shard73) | 1.4B (seq_len=4k) | dclm shard73 | 655M | ~2.6 | 3.2 | 1e-3 | 6400 | ~4.5h (TE) | **3.309** | More unique tokens → less repetition → lower val loss. |
| 8B (dclm_200m) | 8B | dclm_200m | 200M | 1 | 0.1 | 3e-3 | 6104 | ~5h | **6.897** | 8B on 200M tokens is massively undertrained. |
| 1.4B OpenThoughts (unfiltered) | 1.4B (seq_len=4k) | openthoughts_flat | 795M | ~2.1 | 3.2 | 1e-3 | 6400 | ~4.7h (TE) | **5.647** | Pure reasoning data → bad at web text. Expected. |

### Key Findings (Pre-May 2)
- Successfully replicated paper's single-model results within 0.05 nats
- The 3.174 number we chased was an **ensemble** result, not single-model (paper's best single 1.4B = 3.462)
- `max_train_batches=800` slices a fixed 51,200 sequences — every epoch sees the same data
- Transformer Engine 2.13 works with Levanter after adapting attention code (~30% speedup)
- High weight decay (3.2 vs 0.1) is critical for multi-epoch training

---

## May 1: Reasoning Data Curriculum Experiments (300M)

### Hypothesis
Does mixing reasoning data (OpenThoughts-114k) with web data (DCLM) during pretraining improve perplexity or reasoning benchmarks?

### Data
- **DCLM 200M**: 164K web documents, ~200M tokens
- **OpenThoughts filtered**: 54K reasoning traces (math/code/science CoT), ~170M tokens. Filtered to docs ≤4096 tokens to avoid truncating reasoning chains (53% of original data was >4096 tokens and would lose conclusions).

### Runs (300M)

| Run | Description | Phase 1 | Phase 2 | Steps | dclm_val | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|---------|---------|-------|----------|-------|-----------|------------|------|
| A (baseline) | DCLM only | DCLM 6400 steps | — | 6400 | **3.797** | 0.175 | 0.274 | 0.504 | — |
| B | OT only | OT 6400 steps | — | 6400 | **6.187** | 0.226 | 0.267 | 0.500 | 0.259 |
| C | OT→DCLM | OT 3200 steps | DCLM 3200 steps | 6400 | **5.051** | 0.218 | 0.266 | 0.507 | 0.253 |
| D | DCLM→OT | DCLM 3200 steps | OT 3200 steps | 6400 | **3.906** | 0.214 | 0.272 | 0.505 | 0.269 |

### Key Findings (300M)
- Pure reasoning data pretraining (B) is bad for web text perplexity (6.187 vs 3.797)
- OT first then DCLM (C) doesn't recover — 5.051 still far from baseline
- DCLM first then OT (D) barely hurts perplexity (3.906 vs 3.797) but reasoning benchmarks near random
- All eval harness scores near random chance for 300M — model too small to show reasoning signal
- Model D learned **structure** of reasoning (markdown, numbered steps, "therefore") but not actual reasoning

### Text Generation Samples (300M)
Saved to `outputs/generations/300m_generations.json` and `outputs/generations/300m_runC_benchmark_generations.json`.
Key observation: Models produce fluent-looking but factually wrong text. Model D (DCLM→OT) produces formatted reasoning that is wrong.

---

## May 2: Reasoning Data Curriculum Experiments (600M)

**NOTE:** These 600M runs used LR=3e-3 (same as 300M), but the paper specifies LR=1e-3 for 600M. This was fixed in commit `0aa2c60a6` but the runs below have NOT been re-run with the correct LR. Results may be slightly off.

### Hypothesis
Same as 300M experiments but at 600M scale — does larger model show clearer signal from reasoning data?

### Runs (600M)

| Run | Description | Phase 1 | Phase 2 | Steps | dclm_val | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|---------|---------|-------|----------|-------|-----------|------------|------|
| A (baseline) | DCLM only | DCLM 6400 steps | — | 6400 | **3.789** | 0.170 | 0.264 | 0.487 | — |
| B | OT only | OT 6400 steps | — | 6400 | **6.151** | 0.225 | 0.275 | 0.500 | 0.263 |
| C | OT→DCLM | OT 3200 steps | DCLM 3200 steps | 6400 | **5.668** | 0.172 | 0.261 | 0.509 | 0.258 |
| D | DCLM→OT | DCLM 3200 steps | OT 3200 steps | 6400 | **4.074** | 0.177 | 0.262 | 0.493 | 0.252 |

### Key Findings (600M)
- Same pattern as 300M — reasoning data hurts DCLM perplexity, order matters (DCLM first is better)
- Eval harness still near random — 600M not enough to show reasoning gains
- 600M doesn't show improvement from reasoning data on any metric vs 300M

---

### Open Questions / Next Steps

1. **Need DCLM-only baselines with eval harness** for both 300M and 600M to compare properly
2. **Paper's benchmarks are easier** (arc_easy, piqa, sciq) than what we used (arc_challenge, hellaswag, winogrande, mmlu). Paper's 300M model gets 44% arc_easy. Should switch to their benchmarks.
3. **Paper's models are on HuggingFace** (`konwoo/300m4k-*`) — can download and replicate their exact eval numbers
4. **Scale question**: Do we need 1.4B+ to see reasoning data benefits? NVIDIA front-loading paper used 8B.
5. **Data mixing**: Haven't tried simultaneous mixing (80% DCLM + 20% OT) — only sequential curriculum.
6. **The half-baked idea**: Cross-document causal bridges — still unexplored. Requires generating bridges, not just selecting data.

---

### Infrastructure Notes

- **Transformer Engine 2.13**: Required 3 changes to Levanter attention.py (global mesh resource, AttnSoftmaxType, keyword args for fused_attn). ~30% speedup (2.5s → 1.8s/step for 1.4B).
- **Tokenization**: Full DCLM tokenization infeasible (~60 days estimated). Got 8 usable shards (~36B tokens).
- **Bug fixes**: OverflowError in iris backoff, VersionedValue tokenizer bug, GPU support in DataEfficiencyConfig.
- **OpenThoughts truncation**: 53% of docs >4096 tokens. Filtered to ≤4096 to keep complete reasoning chains.

---

## May 3: Eval Consolidation, Reference Models & 1.4B Experiments

### Paper's Benchmarks (arc_easy, piqa, sciq)
The paper evaluates on easier benchmarks than what we initially used. Results on these:

**300M models:**

| Model | ARC Easy | PIQA | SciQ |
|---|---|---|---|
| Paper 300M (16ep, WD=1.6) | **43.8%** | **62.5%** | **72.1%** |
| Our 300M A (DCLM baseline) | 39.6% | 60.3% | 63.2% |
| Our 300M C (OT→DCLM) | 32.1% | 54.5% | 50.3% |
| Our 300M D (DCLM→OT) | 37.5% | 57.6% | 58.8% |
| Random | 25% | 50% | 25% |

**600M models (our experiments):**

| Run | ARC Easy | PIQA | SciQ | dclm_val |
|---|---|---|---|---|
| A (DCLM baseline) | **37.3%** | **58.2%** | **58.1%** | 3.789 |
| C (OT→DCLM) | 30.9% | 53.4% | 47.5% | 5.668 |
| D (DCLM→OT) | 34.1% | 56.2% | 47.6% | 4.074 |

Reasoning data hurts all benchmarks at 600M — even the easier ones. DCLM baseline is best.

### Reference: OLMo 1B Models

| Model | Params | Tokens | ARC Easy | ARC-C | PIQA | SciQ |
|---|---|---|---|---|---|---|
| OLMo 1B | 1B | 3T | 63.3% | 28.5% | 75.0% | 86.7% |
| OLMo 1B 0724 | 1B | 3T | 61.1% | 30.5% | 74.7% | 92.7% |
| OLMo 2 1B | 1B | 4T | **72.4%** | **38.7%** | **75.7%** | **95.2%** |

Massive gap between our 300M-600M models (200M tokens) and properly trained 1B models (3-4T tokens).

### Key Finding: PIQA Test Split Has No Labels
PIQA test split returns label=-1 for all examples. Must use validation split for per-example eval. The lm-eval-harness handles this correctly but our manual eval script initially didn't.

### 1.4B Reasoning Experiments (completed ~4:02 AM PST May 4)

#### Runs (1.4B)

| Run | Description | dclm_val | ARC Easy | PIQA | SciQ | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|----------|----------|------|------|-------|-----------|------------|------|
| A (baseline, from earlier) | DCLM 200M, 8ep | **3.413** | 43.6% | 62.6% | 71.7% | 18.5% | 28.3% | 50.0% | 23.2% |
| B | OT only, 6400 steps | 6.211 | 31.3% | 53.6% | 51.5% | 18.8% | 26.2% | 49.9% | 23.0% |
| C | OT→DCLM (3200+3200) | 5.935 | 28.6% | 54.5% | 42.4% | 17.0% | 26.3% | 49.4% | 23.1% |
| D | DCLM→OT (3200+3200) | 4.331 | 32.1% | 57.1% | 44.9% | 17.4% | 26.0% | 50.0% | 23.4% |

#### Key Findings (1.4B)
- Same pattern as 300M/600M — reasoning data hurts both DCLM perplexity AND downstream benchmarks
- Run D (DCLM→OT) best among reasoning runs but still worse than DCLM baseline on all metrics
- 1.4B model shows same U-shape in dclm_val during OT-only training: drops, recovers, plateaus
- dclm_val trajectory for Run B: 12.3 → 7.8 → 9.5 → 6.5 → 6.2 (interesting overfitting then recovery)
- No model size from 300M to 1.4B shows benefit from OpenThoughts reasoning data on any benchmark

#### Cross-Scale Summary (all models, same experiment design)

**dclm_val loss:**
| Run | 300M | 600M | 1.4B |
|-----|------|------|------|
| A (DCLM baseline) | 3.797 | 3.789 | 3.413 |
| B (OT only) | 6.187 | 6.151 | 6.211 |
| C (OT→DCLM) | 5.051 | 5.668 | 5.935 |
| D (DCLM→OT) | 3.906 | 4.074 | 4.331 |

**ARC Easy:**
| Run | 300M | 600M | 1.4B |
|-----|------|------|------|
| A (DCLM baseline) | 39.6% | 37.3% | 43.6% |
| B (OT only) | — (not eval'd) | — (not eval'd) | 31.3% |
| C (OT→DCLM) | 32.1% | 30.9% | 28.6% |
| D (DCLM→OT) | 37.5% | 34.1% | 32.1% |

#### Conclusion (OpenThoughts)
At 200M token data budget with models 300M–1.4B, pretraining on reasoning data (OpenThoughts CoT traces) provides NO benefit over standard web text (DCLM) on any metric — perplexity, ARC, PIQA, SciQ, HellaSwag, WinoGrande, or MMLU. The reasoning data actively hurts performance. This holds regardless of curriculum order (reasoning first or web first).

---

## May 4: Procedural Knowledge Experiments (300M)

### Motivation
Based on "Procedural Knowledge in Pretraining Drives Reasoning" (Ruis et al., arxiv:2411.12580):
- Models learn reasoning from **code and math that demonstrates procedures**, not from explicit CoT traces
- Code on StackExchange is 10x overrepresented in influential documents for reasoning
- The same procedural documents help across different reasoning questions of the same type

This explains why OpenThoughts (explicit CoT) failed — it's the wrong type of reasoning data. We should test procedural knowledge sources: code and math web pages.

### Data
- **DCLM 200M**: 164K web documents, ~200M tokens (baseline)
- **Code Procedural 218M**: ~218M tokens of Python, JavaScript, C, C++ code from The Stack
- **OpenWebMath 219M**: ~219M tokens of math web pages with formulas and procedures
- **OpenThoughts filtered 170M**: ~170M tokens of CoT traces (for comparison)

### Runs (300M, all with LR=3e-3, WD=3.2, 6400 steps)

| Run | Data | ARC Easy | PIQA | SciQ | dclm_val |
|---|---|---|---|---|---|
| Baseline | DCLM 200M | 39.6% | 60.3% | 63.2% | 3.797 |
| Code only | Code 218M (Python/JS/C/C++) | 26.1% | 49.4% | 49.4% | 5.947 |
| **OpenWebMath only** | OWM 219M (math web pages) | 34.9% | 48.9% | **73.2%** | 4.304 |
| OpenThoughts only | OT 170M (CoT traces) | — (not eval'd on easy benchmarks) | — | — | 6.187 |

### Key Findings (Procedural Knowledge)
1. **OpenWebMath beats DCLM on SciQ**: 73.2% vs 63.2% — first reasoning data to beat baseline on ANY benchmark
2. **Code alone doesn't help**: Hurts all benchmarks (ARC Easy 26.1%, PIQA 49.4%, SciQ 49.4%)
3. **OpenThoughts confirmed bad**: Worst dclm_val loss (6.187), no benchmark improvements
4. **Procedural knowledge hypothesis validated**: Math web pages (which show HOW to solve problems) help more than explicit reasoning traces (which show step-by-step solutions)
5. **Sequential curriculum still fails**: When we tried OWM→DCLM sequentially, the model forgot the SciQ gains

### Open Questions
1. **Simultaneous mixing untested**: 80% DCLM + 20% OpenWebMath mixed during training — might preserve both web text quality AND SciQ gains
2. **600M with correct LR**: 600M v2 runs crashed, need restart with LR=1e-3
3. **Code + DCLM mixing**: 80% DCLM + 20% Code — code alone fails but mixed might help
4. **Causal bridges**: The cross-document bridge idea from `half-baked-idea.txt` — still unexplored

---

## May 5: Mixed DCLM+OWM Run & Research Hypotheses

### Mixed Run: 80% DCLM + 20% OpenWebMath (300M)

This is an **off-ramp exploration** from the original staged curriculum hypothesis. The original idea was that reasoning-style data (first OpenThoughts, then OpenWebMath) should be staged sequentially — reasoning first, then web data, or vice versa. Sequential curriculum failed in both directions:
- OWM→DCLM: model forgets SciQ gains
- DCLM→OWM: model forgets language/world knowledge

Simultaneous mixing is a fallback to see if we can get OWM's SciQ benefit without losing DCLM's general capabilities.

**Run config:** 300M, LR=3e-3, WD=3.2, 6400 steps, 80% DCLM + 20% OWM mixed throughout training.

| Metric | Mixed 80/20 | DCLM baseline | OWM only |
|---|---|---|---|
| dclm_val | **3.687** | 3.797 | 4.304 |
| ARC Easy | 38.2% | 39.6% | 34.9% |
| PIQA | 58.0% | 60.3% | 48.9% |
| SciQ | **64.5%** | 63.2% | **73.2%** |
| ARC-C | 17.7% | 17.5% | — |
| HellaSwag | 26.6% | 27.4% | — |
| WinoGrande | 52.1% | 50.4% | — |

**Analysis:** The mixed run slightly improves SciQ over DCLM baseline (64.5% vs 63.2%) but ARC Easy and PIQA are flat or slightly down. This supports **H3 (domain-specific knowledge)**: OWM's benefit is concentrated on science benchmarks, not a general reasoning improvement. The dclm_val improvement (3.687 vs 3.797) suggests the model benefits from data diversity for perplexity, but this doesn't translate to broad benchmark gains.

### Research Hypotheses

We now have a clear empirical pattern: OpenWebMath trains a model that excels at SciQ (73.2% vs 63.2% DCLM baseline) but hurts ARC Easy (34.9% vs 39.6%) and PIQA (48.9% vs 60.3%). Sequential curriculum in either direction loses one set of gains. Three hypotheses explain different aspects of this pattern.

#### H1: Model needs language/world knowledge first before reasoning data is useful

The idea: a model that already understands language and the world can extract more value from procedural math content than a model learning both from scratch.

- **Prediction for DCLM→OWM:** SciQ > 73.2% (language foundation makes reasoning data more useful)
- **Prediction:** ARC Easy/PIQA stay decent (world knowledge partially survives from DCLM phase)
- **How to test:** Vary DCLM phase length before switching to OWM. Run 1600/3200/4800 steps of DCLM, then OWM for the remaining steps (4800/3200/1600). If more DCLM first leads to better SciQ, that supports H1.

#### H2: Catastrophic forgetting — later data overwrites earlier

The idea: whatever the model learns last dominates. Earlier training is largely wasted because the model overwrites those representations.

- **Prediction for DCLM→OWM:** SciQ ≈ 73.2% (same as OWM-only; the DCLM phase is wasted)
- **Prediction:** ARC Easy/PIQA drop to OWM-only levels (~34.9% and ~48.9%)
- **How to test:** Run DCLM→OWM with DCLM replay during phase 2 (10% DCLM + 90% OWM in the second phase). If replay mitigates forgetting (ARC Easy/PIQA stay higher), that confirms H2 as the mechanism.
- **Note:** H1 and H2 can both be true simultaneously — the model may need prior knowledge AND suffer from forgetting.

#### H3: OWM teaches domain-specific science knowledge, not general reasoning

The idea: OWM's SciQ improvement comes from memorizing science facts and math procedures, not from learning transferable reasoning skills.

- **Prediction for mixed run:** SciQ improves but ARC Easy/PIQA stay flat (science knowledge helps science benchmarks only)
- **How to test:** Evaluate OWM-trained models on reasoning benchmarks outside math/science domains. If OWM only helps science-related tasks, it is domain knowledge transfer, not general reasoning improvement.

### Discriminating Experiments

These experiments produce different predictions under each hypothesis, allowing us to distinguish between them:

| Experiment | H1 predicts | H2 predicts | H3 predicts |
|---|---|---|---|
| DCLM→OWM (3200+3200) | SciQ > 73.2% | SciQ ≈ 73.2% | SciQ ≈ 73.2% |
| DCLM→OWM varying lengths | More DCLM → better SciQ | SciQ always ≈ 73.2% | — |
| Mixed run (80/20) | SciQ + ARC + PIQA all improve | — | SciQ up, ARC/PIQA flat |
| OWM + DCLM replay in phase 2 | — | Forgetting mitigated | — |
| OWM model on non-science reasoning | — | — | No improvement (domain-specific) |

The mixed run (80% DCLM + 20% OWM) is already complete and benchmark results will directly test H1 vs H3: if all three benchmarks improve, that favors H1 (general synergy); if only SciQ improves, that favors H3 (domain-specific knowledge).

---

## May 8: H1 Experiment — Reasoning Data in the Middle of Training

### Hypothesis
Model needs language/world knowledge first before reasoning data is useful.
If we inject reasoning data after initial pretraining, the model should perform better
on reasoning benchmarks compared to training on web data only.

### Design
- **Treatment**: Run A (3B DCLM pretrained) → 200M OT → 400M DCLM
- **Control**: Run A (3B DCLM pretrained) → 200M DCLM → 400M DCLM
- Both use `initialize_from_checkpoint_path` with fresh cosine LR schedule per phase
- Phase1: 763 steps (200M tokens), Phase2: 1526 steps (400M tokens)
- Model: 300M, batch_size=64, seq_len=4096, LR=3e-3, WD=1.6

### Fixes Applied
1. **LR schedule counter reset**: `initialize_from_checkpoint_path` now resets optimizer schedule counters (was loading stale counters from source checkpoint, giving wrong LR)
2. **Force checkpoint save**: `LambdaCallback.on_step` now passes `force` parameter (was being dropped, so final checkpoint never saved)
3. **Checkpoint wait**: Trainer now waits for async checkpoint save to complete before returning

### WandB Runs
| Phase | Run ID | Tags |
|-------|--------|------|
| Treatment Phase1 (OT) | 06va0rn2 | h1-v2, treatment, phase1, ot |
| Control Phase1 (DCLM) | ncpocjta | h1-v2, control, phase1, dclm |
| Treatment Phase2 (DCLM) | d47v5z8y | h1-v2, treatment, phase2, dclm |
| Control Phase2 (DCLM) | vothg0mz | h1-v2, control, phase2, dclm |

### Results

| Benchmark | Treatment (OT→DCLM) | Control (DCLM→DCLM) | Diff |
|-----------|---------------------|----------------------|------|
| ARC Easy | 35.0% | 35.0% | 0.0% |
| ARC Challenge | 19.0% | 18.9% | +0.2% |
| PIQA | 48.9% | 49.2% | -0.4% |
| SciQ | 70.9% | 69.0% | **+1.9%** |
| HellaSwag | 26.2% | 26.4% | -0.2% |
| Winogrande | 50.9% | 51.0% | -0.1% |
| MMLU | 25.8% | 26.7% | -1.0% |
| **Macro avg** | **27.0%** | **28.3%** | **-1.3%** |

DCLM val loss: Treatment 3.743 vs Control 3.720

### Conclusion
**H1 is not supported.** Injecting 200M tokens of reasoning data (OpenThoughts) in the
middle of training does not help reasoning benchmarks. The control (pure DCLM) slightly
outperforms on most benchmarks (macro avg -1.3%). Treatment only wins on SciQ (+1.9%),
consistent with H3 (domain-specific knowledge transfer) rather than general reasoning
improvement.

### Caveats
- Each phase gets a fresh cosine LR from max → 0. This means there's a LR jump at the
  phase boundary. Both conditions have the same jump so the comparison is fair, but a
  continuous cosine schedule would be more representative of real training.
- The 200M tokens of OT may not be enough to teach reasoning at 300M model scale.
- Fresh optimizer (Adam moments reset) at each phase means the model "forgets" gradient
  history, which may hurt the treatment more since it switches domains twice.

---

## May 10: H1 Revisited — Continuous Cosine LR, OWM+Code Treatment

### Motivation
The May 8 H1 experiment had two problems:
1. **Fresh cosine LR per phase** — LR jumps at phase boundaries, optimizer moments reset
2. **OpenThoughts as treatment** — already conclusively shown to be useless at all scales (300M–1.4B)

This run fixes both: continuous cosine LR across phases 1+2 (via `initialize_from_step`), and uses OWM+Code as treatment data since OWM showed the only positive signal (SciQ 73.2% vs 63.2% baseline).

### Technical Implementation
Added `initialize_from_step` to `TrainLmConfig` in `lib/levanter/src/levanter/main/train_lm.py`:
- Loads weights+optimizer from checkpoint via `initialize_from_checkpoint_path`
- Sets optimizer schedule counter AND `state.step` to specified value
- Enables continuous cosine LR across phases without `load_checkpoint_path` (which OOMs)
- Verified with smoke test: 40-step single run vs 20+20 split has 0.00e+00 max LR difference

### Design
```
Phase 0 (shared):     Train from scratch on 203M DCLM, 4 epochs = 3,096 steps
Phase 1 (1,667 steps / 437M tokens):
  Treatment: OWM (219M) + Code (218M) mixed 50/50
  Control:   Disjoint DCLM (~407M tokens)
Phase 2 (3,052 steps / 800M tokens):
  Both arms: Disjoint DCLM (~778M tokens)
```

LR schedule: Phases 1+2 share one continuous cosine over 4,719 total steps.
- Phase 1: `stop_step=1667`, `num_train_steps=4719`
- Phase 2: `initialize_from_step=1667`, `num_train_steps=4719`

All DCLM data is disjoint across phases (phase 0: 203M, phase 1 control: 407M, phase 2: 778M — downloaded 1.52B total from DCLM baseline).

Model: 300M, batch_size=64, seq_len=4096, LR=3e-3, WD=1.6

### WandB Runs
| Phase | Run ID | Description |
|-------|--------|-------------|
| Phase 0 (pretrain) | hvu9zzrj | 300M on DCLM 200M, 4 epochs |
| Treatment Phase 1 | ja7ty1se | OWM+Code mix, 1667 steps |
| Control Phase 1 | rd5wfmmu | Disjoint DCLM, 1667 steps |
| Treatment Phase 2 | un39dx11 | Disjoint DCLM, 3052 steps (from step 1667) |
| Control Phase 2 | m67nooef | Disjoint DCLM, 3052 steps (from step 1667) |

### Results

| Benchmark | Treatment (OWM+Code) | Control (DCLM only) | Delta |
|-----------|---------------------|---------------------|-------|
| ARC Easy | 35.5% | 36.7% | -1.1% |
| ARC Challenge | 22.3% | 22.5% | -0.3% |
| PIQA | 50.0% | 50.2% | -0.2% |
| SciQ | 74.1% | 74.1% | 0.0% |
| HellaSwag | 27.3% | 27.4% | -0.0% |
| WinoGrande | 50.4% | 51.1% | -0.6% |
| MMLU | 26.7% | 25.3% | **+1.4%** |
| **Macro avg** | **27.6%** | **26.9%** | **+0.7%** |

DCLM val: Treatment 1.198 BPB (3.705 loss) vs Control 1.191 BPB (3.686 loss)

### Analysis
1. **SciQ is flat** (74.1% both arms) — surprising given OWM-only showed 73.2% vs 63.2% DCLM baseline. The control also reaches 74.1%, suggesting phase 0 pretraining (4 epochs of 203M DCLM) already saturates SciQ at this model size.
2. **MMLU is the only treatment win** (+1.4%) — OWM+Code may help with knowledge breadth
3. **Most benchmarks within noise** (0–0.6%) — no clear treatment advantage or disadvantage
4. **DCLM val loss slightly worse for treatment** (3.705 vs 3.686) — expected since treatment saw less DCLM in phase 1
5. **Continuous cosine LR worked correctly** — both arms resumed from step 1667 with matching LR schedules

### Conclusion
**H1 remains unsupported even with proper LR continuity and better treatment data.** Injecting OWM+Code mid-training does not meaningfully help reasoning benchmarks compared to pure DCLM training. The previous SciQ signal from OWM (73.2%) appears to be a domain knowledge effect that saturates with enough general pretraining, not a lasting advantage from procedural knowledge injection.

### Comparison with May 8 H1
| Change | May 8 | May 10 |
|--------|-------|--------|
| LR schedule | Fresh cosine per phase | Continuous cosine (initialize_from_step) |
| Treatment data | OpenThoughts (170M) | OWM+Code (437M) |
| Phase 0 | Paper's 16-epoch ckpt | 4-epoch fresh pretrain |
| DCLM data | Repeated across phases | Disjoint per phase |
| SciQ delta | +1.9% | 0.0% |
| Macro avg delta | -1.3% | +0.7% |

The improved design (continuous LR, better treatment data, disjoint data) eliminated the macro avg deficit but still shows no clear benefit from reasoning data injection.

---

## Research Direction: Revised Hypotheses (Post-May 10)

The original H1/H2/H3 hypotheses (May 5) have been refined based on accumulated experimental evidence across all runs (300M–1.4B, multiple data types and curriculum designs).

### H1: What Makes Reasoning Data Good for Pretraining?

**The problem:** Not all "reasoning data" is equal. OpenThoughts (long exploratory CoT traces) consistently hurts performance across all scales and curriculum orderings. OpenWebMath shows a SciQ gain (73.2% vs 63.2% baseline) but this saturates with enough general pretraining and does not transfer beyond science domains — consistent with domain knowledge transfer rather than general reasoning capability. Code alone hurts all benchmarks.

**The constraint:** Good reasoning data must teach something that (a) transfers beyond the domain it was trained on, and (b) is not confounded with domain familiarity — i.e., the gain should not disappear when the model sees enough general web text.

**What we know from the literature:**
- Content-free synthetic tasks (Percy's work, arxiv 2206.10139; Procedural Pretraining, arxiv 2601.21725) can close ~65% of the gap to natural pretraining, suggesting structural patterns matter even without semantic content
- Procedural knowledge — data demonstrating how to derive something step by step — is 10x overrepresented in influential pretraining documents for reasoning (Ruis et al., arxiv 2411.12580)
- OpenThoughts fails because its exploratory back-and-forth CoT is the wrong structure for a model starting from scratch with no world knowledge to anchor on

**What we don't know:** Whether real language data with explicit causal structure — as opposed to content-free synthetic tasks — can teach transferable reasoning capability. The causal bridge idea is the most natural candidate: by conditioning generation on two real document endpoints (causally related via Wikipedia wikilinks), the model is forced to construct relational understanding grounded in real-world events. This is neither content-free nor domain-specific — it is structured real language. Whether this teaches transferable reasoning is the core empirical question.

### H2: How Do We Retain Reasoning Capability Through General Pretraining?

**The problem:** Even if we solve H1 and identify good reasoning data, there are two distinct mechanisms by which the capability could be lost during subsequent general web text training:

**Sub-problem 2a — Catastrophic forgetting:** The model overwrites representations learned from reasoning data when exposed to web-scale text. The May 8 and May 10 H1 experiments are consistent with this — the SciQ gains from OWM disappear after phase 2 DCLM training. Replay (mixing a small fraction of reasoning data throughout web text training) is a standard mitigation but untested here.

**Sub-problem 2b — No training pressure to use reasoning circuits:** Steven Cao's point: even if reasoning circuits exist after phase 1, there is no mechanism during standard next-token prediction on web text that activates or reinforces those circuits. The model is not prompted to reason during web text training, so whatever was built in phase 1 sits dormant. This is a more fundamental problem than forgetting — replay does not solve it, because the problem is not forgetting but never using.

**What we don't know:** Whether there exists a training signal during web text exposure that both retains reasoning circuits and actively uses them. Possible directions include: perplexity-based filtering of web text (only train on documents the reasoning-capable model finds surprising, not documents it can predict via shortcuts), or a joint training objective that ties reasoning evaluation to web text prediction. Both are speculative.

### The Relationship Between H1 and H2

H1 is the more fundamental bottleneck. Until we have data that demonstrably teaches transferable reasoning (H1 solved), H2 is moot — there is nothing to retain. The causal bridge experiments address H1 first.

---

## Literature Review

See [papers/reasoning_data_summary.md](/papers/reasoning_data_summary.md) for full paper summaries, dataset inventory, and applicability analysis.

Key takeaways:
1. Pure reasoning data hurts; ~30% mixed with web data is optimal (Kang et al.)
2. Diversity of reasoning patterns matters more than domain specificity (NVIDIA Front-Loading)
3. Relational/combinatorial structure drives quality (EntiGraph)
4. Abstract reasoning from toy domains DOES transfer (Warm Up Before You Train)
5. Pretraining is the ceiling — post-training amplifies but cannot create (Echo Chamber, Front-Loading)

### Next Experiment Directions

**Direction A — Causal bridges (relational structure with real content):** Generate training data by conditioning on causally related document pairs (e.g., Wikipedia wikilinks). Combines EntiGraph's relational structure insight with real-world grounding. Most novel, most infrastructure-heavy.

**Direction B — Abstract procedural tasks (content-free structure):** Train on toy logic/reasoning tasks (e.g., Knights & Knaves). Cleanly isolates reasoning structure from domain knowledge. Risk: the paper uses post-training, not pretraining from scratch.

**Direction C — Diversity-optimized reasoning mix:** Instead of OWM+Code (2 domains), mix reasoning data from many domains at ~20-30% of total. Tests whether breadth of procedural patterns matters more than depth.
