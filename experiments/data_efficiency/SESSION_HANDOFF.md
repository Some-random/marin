# Session Handoff — May 5, 2026

## Research Goal

**Core hypotheses:**

1. **We can teach models to reason first, before exposing them to general web data, during pre-training.**
2. **If we can preserve that reasoning capability, the model will learn more efficiently when it finally sees web data.**

The intuition: like teaching a child — if you teach them to reason and understand causality first, they should get more out of their experiences later. Most existing work does it the other way around (web text first, reasoning via SFT later).

**Two instantiations of hypothesis (1):**

- **Causal Bridge** (from `half-baked-idea.txt`): Generate synthetic causal connections between documents so the model has to construct relational understanding rather than just predict tokens. The model generates bridges between causally related documents, scored by an external model.

- **Reasoning-first curriculum**: Pre-train the model on reasoning data (procedural knowledge, code, math) before exposing it to general web text. Simpler than bridges — just data ordering.

Both are ways to achieve (1). If (2) holds, training on web data follows naturally.

**The bigger question**: Can we train a qualitatively different LLM — one that is more grounded and reasons more reliably — by changing what type of learning happens during pretraining, rather than just generating more text?

**Current exploration phase**: Testing whether reasoning-first curriculum works at all. Specifically:
- Does training on reasoning-heavy data (CoT traces, code, math) help?
- What TYPE of reasoning data helps? (Finding: procedural knowledge in code/math > explicit CoT traces)
- Can we preserve reasoning when switching to web data? (Finding so far: the model forgets)

**What we've established so far**:
1. Successfully replicated the "Pre-training Under Infinite Compute" paper (Kim et al. 2026)
2. OpenThoughts CoT traces DON'T help — they actively hurt all metrics
3. OpenWebMath (math web pages with formulas/procedures) DOES help SciQ (73.2% vs 63.2% baseline)
4. Code alone doesn't help either
5. The "Procedural Knowledge" paper (Ruis et al.) explains WHY: models learn reasoning from code/math that demonstrates HOW to solve things, not from explicit step-by-step solutions
6. Sequential curriculum (reasoning then web) doesn't preserve reasoning gains — the model forgets
7. **Untested**: Simultaneous mixing (80% web + 20% reasoning) — might preserve both

**Key open question**: Can we mix reasoning data WITH web data during pretraining (not sequentially) so the model learns both simultaneously? This is the NVIDIA front-loading approach scaled down to our setting.

## Current State

### Completed Experiments
All results in `experiments/data_efficiency/EXPERIMENT_LOG.md` and WandB (`dongwei_jiang/dongwei-data-efficiency`).

**300M Procedural Experiments (just completed):**

| Run | Data | ARC Easy | PIQA | SciQ | dclm_val |
|---|---|---|---|---|---|
| Baseline | DCLM 200M | 39.6% | 60.3% | 63.2% | 3.797 |
| Code only | Code 218M (Python/JS/C/C++) | 26.1% | 49.4% | 49.4% | 5.947 |
| **OpenWebMath only** | OWM 219M (math web pages) | 34.9% | 48.9% | **73.2%** | 4.304 |
| OpenThoughts only | OT 170M (CoT traces) | — | — | — | 6.187 |

**Key finding: OpenWebMath beats DCLM on SciQ (73.2% vs 63.2%).** First reasoning data to beat baseline on any benchmark.

### What Failed / Needs Redo
- 600M v2 (with correct LR=1e-3): Run B started but crashed. Needs restart.
- Missing: 80% DCLM + 20% OpenWebMath mixed experiment (the most promising next step)

### Running Processes
- Nothing currently running. GPUs idle.

## Key Paths
- Training scripts: `experiments/data_efficiency/run_reasoning_experiment.py`
- Pipeline: `experiments/data_efficiency/run_procedural_experiments.sh`
- Eval script: `experiments/data_efficiency/generate_benchmark_examples.py`
- Secrets: `.secrets` (WANDB_API_KEY, HF_TOKEN, GITHUB_TOKEN)
- Experiment log: `experiments/data_efficiency/EXPERIMENT_LOG.md`
- Papers: `papers/` (phi1, phi3, olmo2, front-loading reasoning, open thoughts, procedural knowledge)
- Tokenized data: `outputs/tokenized/` (dclm_200m, openthoughts_filtered, openwebmath, code_procedural)
- Checkpoints: `checkpoints/` (300m_hf, 600m_*, 1_4b_*, run_C/D_*)
- GitHub: `origin` = `https://github.com/Some-random/marin`, branch `data-efficiency`

## Next Steps (suggested)
1. Run 80% DCLM + 20% OpenWebMath mixed (300M) — fair comparison where OWM doesn't replace DCLM but supplements it
2. Fix 600M v2 experiments (LR=1e-3)
3. Maybe try 80% DCLM + 20% Code mixed
4. The half-baked idea (cross-document causal bridges) — still unexplored

## Paper Insight (just read)
"Procedural Knowledge in Pretraining Drives Reasoning" (arxiv:2411.12580):
- Models learn reasoning from **code and math that demonstrates procedures**, not from explicit CoT traces
- Code on StackExchange is 10x overrepresented in influential documents for reasoning
- The same procedural documents help across different reasoning questions of the same type
- This explains why OpenWebMath (math formulas/procedures) works but OpenThoughts (CoT traces) doesn't

## Config Notes
- 300M: LR=3e-3, WD=3.2
- 600M/1.4B: LR=1e-3, WD=3.2
- All: cosine schedule, min_lr_ratio=0.0, batch=64, seq_len=4096, 6400 steps
- Eval: arc_easy, piqa, sciq (paper's benchmarks) + dclm_val loss
- Start with: `CLAUDE_CODE_USE_BEDROCK=0 claude`
