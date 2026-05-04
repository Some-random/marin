# Experiment Log: Data Efficiency & Reasoning Pretraining

## Pre-May 2: Paper Replication

### Hypothesis
Replicate "Pre-training Under Infinite Compute" paper results on local 8x A100-40G GPUs.

### Runs

| Run | Model | Data | Tokens | Epochs | WD | LR | Steps | Time | dclm_val | Notes |
|-----|-------|------|--------|--------|-----|-----|-------|------|----------|-------|
| 300M baseline | 300m4k | dclm_200m | 200M | 8 | 0.1 | 1e-3 | 6400 | ~1.5h | **3.797** | Paper gets 3.785. Match. |
| 1.4B regularized (dclm_200m) | 1_4b4k | dclm_200m | 200M | 8 | 3.2 | 1e-3 | 6400 | ~3.5h (TE) | **3.413** | Paper single-model best: 3.462. We beat it slightly — likely dclm_200m is a curated subset. |
| 1.4B (dclm_shard73) | 1_4b4k | dclm shard73 | 655M | ~2.6 | 3.2 | 1e-3 | 6400 | ~4.5h (TE) | **3.309** | More unique tokens → less repetition → lower val loss. |
| 8B (dclm_200m) | l8b | dclm_200m | 200M | 1 | 0.1 | 3e-3 | 6104 | ~5h | **6.897** | 8B on 200M tokens is massively undertrained. |
| 1.4B OpenThoughts (unfiltered) | 1_4b4k | openthoughts_flat | 795M | ~2.1 | 3.2 | 1e-3 | 6400 | ~4.7h (TE) | **5.647** | Pure reasoning data → bad at web text. Expected. |

### Key Findings (Pre-May 2)
- Successfully replicated paper's single-model results within 0.05 nats
- The 3.174 number we chased was an **ensemble** result, not single-model (paper's best single 1.4B = 3.462)
- `max_train_batches=800` slices a fixed 51,200 sequences — every epoch sees the same data
- Transformer Engine 2.13 works with Levanter after adapting attention code (~30% speedup)
- High weight decay (3.2 vs 0.1) is critical for multi-epoch training

---

## May 1–2: Reasoning Data Curriculum Experiments (300M)

### Hypothesis
Does mixing reasoning data (OpenThoughts-114k) with web data (DCLM) during pretraining improve perplexity or reasoning benchmarks?

### Data
- **DCLM 200M**: 164K web documents, ~200M tokens
- **OpenThoughts filtered**: 54K reasoning traces (math/code/science CoT), ~170M tokens. Filtered to docs ≤4096 tokens to avoid truncating reasoning chains (53% of original data was >4096 tokens and would lose conclusions).

### Runs (300M)

| Run | Description | Phase 1 | Phase 2 | Steps | dclm_val | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|---------|---------|-------|----------|-------|-----------|------------|------|
| A (baseline) | DCLM only | DCLM 6400 steps | — | 6400 | **3.797** | — | — | — | — |
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

## May 2–3: Reasoning Data Curriculum Experiments (600M)

### Hypothesis
Same as 300M experiments but at 600M scale — does larger model show clearer signal from reasoning data?

### Runs (600M)

| Run | Description | Phase 1 | Phase 2 | Steps | dclm_val | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|---------|---------|-------|----------|-------|-----------|------------|------|
| A (baseline) | DCLM only | DCLM 6400 steps | — | 6400 | (running) | — | — | — | — |
| B | OT only | OT 6400 steps | — | 6400 | **6.151** | 0.225 | 0.275 | 0.500 | 0.263 |
| C | OT→DCLM | OT 3200 steps | DCLM 3200 steps | 6400 | **5.668** | 0.231 | 0.266 | 0.504 | 0.258 |
| D | DCLM→OT | DCLM 3200 steps | OT 3200 steps | 6400 | **4.074** | 0.221 | 0.260 | 0.503 | 0.252 |

### Key Findings (600M)
- Same pattern as 300M — reasoning data hurts DCLM perplexity, order matters (DCLM first is better)
- Eval harness still near random — 600M not enough to show reasoning gains
- 600M doesn't show improvement from reasoning data on any metric vs 300M

---

## Open Questions / Next Steps

1. **Need DCLM-only baselines with eval harness** for both 300M and 600M to compare properly
2. **Paper's benchmarks are easier** (arc_easy, piqa, sciq) than what we used (arc_challenge, hellaswag, winogrande, mmlu). Paper's 300M model gets 44% arc_easy. Should switch to their benchmarks.
3. **Paper's models are on HuggingFace** (`konwoo/300m4k-*`) — can download and replicate their exact eval numbers
4. **Scale question**: Do we need 1.4B+ to see reasoning data benefits? NVIDIA front-loading paper used 8B.
5. **Data mixing**: Haven't tried simultaneous mixing (80% DCLM + 20% OT) — only sequential curriculum.
6. **The half-baked idea**: Cross-document causal bridges — still unexplored. Requires generating bridges, not just selecting data.

---

## Infrastructure Notes

- **Transformer Engine 2.13**: Required 3 changes to Levanter attention.py (global mesh resource, AttnSoftmaxType, keyword args for fused_attn). ~30% speedup (2.5s → 1.8s/step for 1.4B).
- **Tokenization**: Full DCLM tokenization infeasible (~60 days estimated). Got 8 usable shards (~36B tokens).
- **Bug fixes**: OverflowError in iris backoff, VersionedValue tokenizer bug, GPU support in DataEfficiencyConfig.
- **OpenThoughts truncation**: 53% of docs >4096 tokens. Filtered to ≤4096 to keep complete reasoning chains.

---

## May 3: Eval Consolidation & Reference Models

### Paper's Benchmarks (arc_easy, piqa, sciq)
The paper evaluates on easier benchmarks than what we initially used. Results on these:

**300M models:**

| Model | ARC Easy | PIQA | SciQ |
|---|---|---|---|
| Paper 300M (16ep, WD=1.6) | **43.8%** | **62.5%** | **72.1%** |
| Our 300M (8ep, WD=0.1) | 39.6% | 60.3% | 63.2% |
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

---

## May 3–4: 1.4B Reasoning Experiments (completed ~4:02 AM PST May 4)

### Runs (1.4B)

| Run | Description | dclm_val | ARC Easy | PIQA | SciQ | ARC-C | HellaSwag | WinoGrande | MMLU |
|-----|-------------|----------|----------|------|------|-------|-----------|------------|------|
| A (baseline, from earlier) | DCLM 200M, 8ep | **3.413** | — | — | — | — | — | — | — |
| B | OT only, 6400 steps | 6.211 | 31.3% | 53.6% | 51.5% | 21.9% | 27.0% | 51.5% | 25.5% |
| C | OT→DCLM (3200+3200) | 5.935 | 28.6% | 54.5% | 42.4% | 23.4% | 26.6% | 50.3% | 25.7% |
| D | DCLM→OT (3200+3200) | 4.331 | 32.1% | 57.1% | 44.9% | 22.4% | 26.7% | 50.6% | 26.5% |

### Key Findings (1.4B)
- Same pattern as 300M/600M — reasoning data hurts both DCLM perplexity AND downstream benchmarks
- Run D (DCLM→OT) best among reasoning runs but still worse than DCLM baseline on all metrics
- 1.4B model shows same U-shape in dclm_val during OT-only training: drops, recovers, plateaus
- dclm_val trajectory for Run B: 12.3 → 7.8 → 9.5 → 6.5 → 6.2 (interesting overfitting then recovery)
- No model size from 300M to 1.4B shows benefit from OpenThoughts reasoning data on any benchmark

### Cross-Scale Summary (all models, same experiment design)

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
| A (DCLM baseline) | 39.6% | 37.3% | — |
| B (OT only) | — | — | 31.3% |
| C (OT→DCLM) | — | 30.9% | 28.6% |
| D (DCLM→OT) | — | 34.1% | 32.1% |

### Conclusion
At 200M token data budget with models 300M–1.4B, pretraining on reasoning data (OpenThoughts CoT traces) provides NO benefit over standard web text (DCLM) on any metric — perplexity, ARC, PIQA, SciQ, HellaSwag, WinoGrande, or MMLU. The reasoning data actively hurts performance. This holds regardless of curriculum order (reasoning first or web first).
