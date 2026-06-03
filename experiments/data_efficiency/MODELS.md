# Models Reference: Architectures, Training Data, and Hyperparameters

Canonical record of what each model we evaluate in [EVALUATION.md](EVALUATION.md) was *actually trained on* and *how*. Eval numbers stay in EVALUATION.md; this file is the training-side companion.

Every claim about data mix, hyperparameters, or token counts in this file is sourced from the run script (path linked) or the model's published paper (citation linked). If a number isn't sourced, it doesn't go here.

---

## 1. Models we trained

### 1.4B base (x16) — `peach-thunder-100` / `6xx0hu3l`

- **Run script:** [`run_1_4b_wd1_6_x16_nocrossblock.py`](run_1_4b_wd1_6_x16_nocrossblock.py)
- **HF checkpoint:** `/fsx/users/dongweij/marin/checkpoints/1_4b_wd1_6_x16_nocrossblock_hf`
- **Levanter checkpoint:** `checkpoints/1_4b_wd1_6_x16_nocrossblock/`
- **Purpose:** Konwoo-style data-repetition baseline. Same hyperparams as `konwoo/1_4b4k-209Mx16-wd1.60` including `block_cross_document_attention=False` (the previously-missed config diff).

**Architecture (`llama_1_4b` in [experiments/llama.py](../../experiments/llama.py)):** Llama-style decoder. hidden 2048, FFN 7168, 16 layers, 16 heads, 8 KV heads (GQA), max_seq_len 4096, RoPE, SwiGLU, RMSNorm. ~1.4B parameters.

**Tokenizer:** `meta-llama/Meta-Llama-3.1-8B` (128,256 vocab).

**Data:**
- Source: tokenized `dclm_200m_train-d321eb` slice (~209M unique DCLM tokens from the same shard mix as the original 209M setup).
- Mix: 100% DCLM (`dclm_200m` weight=1.0). All paloma subsets and `dclm_200m_val` are eval-only (weight=0).
- Repetition: 16 epochs over the 209M slice → 3.36B total trained tokens.

**Hyperparameters:**
- Optimizer: AdamW (β₁=0.9, β₂=0.95)
- Learning rate: 1e-3, cosine schedule to `min_lr_ratio=0.0`
- Weight decay: **1.6** (repetition-overfit-aware; required for 16-epoch DCLM)
- Warmup: 0.01 of training (128 steps)
- Max grad norm: 1.0
- Batch size: 64 sequences × 4096 tokens = 262,144 tokens/step
- Steps: 12,800 → 3.36B trained tokens
- Seq len: 4096
- Precision: bfloat16 compute, fp32 params (`jmp.get_policy("p=f32,c=bfloat16")`)
- `block_cross_document_attention=False`, `shuffle=True`, `enforce_eos=True`, `seed=0`, `data_seed=0`
- Hardware: single 8-GPU node

---

### 1.4B code25 v2 (matched) — `sage-wildflower-106` / `joqfahkl`

- **Run script:** [`run_1_4b_25code_alg_v2.py`](run_1_4b_25code_alg_v2.py)
- **HF checkpoint:** `/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_v2_hf`
- **Levanter checkpoint:** `checkpoints/1_4b_25code_alg_v2/`
- **Purpose:** **Matched-token** 25% code-mix probe of the Aryabumi hypothesis. Replaces v1 (`eager-grass-104` / `p2n84bo3`, retracted June 1) which had 5× more unique tokens than the baseline.

**Architecture / Tokenizer / Hyperparams:** Identical to 1.4B base above. The only thing that differs is data.

**Data:**
- DCLM slice: tokenized `dclm_150m-*` (146.97M tokens, subsampled from the same source as `dclm_200m`).
- Code slice: tokenized `opc_algorithmic_50m-*` (54.59M tokens, subsample of `opc_algorithmic`).
- Total unique: 201.56M (matched to the baseline's 209M within ~4%).
- Sampling weights (proportional to unique sizes so epoch counts match exactly):
  - `dclm_150m`: 0.729 → 2.446B drawn → **16.64 epochs**
  - `opc_algorithmic_50m`: 0.271 → 0.910B drawn → **16.66 epochs**
- Total trained tokens: 3.355B over 12,800 steps (same budget as baseline).

The matching-epochs construction means this run isolates "swap 50M DCLM for 50M code" from the "extra unique tokens" confound that v1 had.

---

### A5 — 1.4B 1-epoch DCLM (`1ep-dclm-A5`)

- **Run script:** [`run_1_4b_1ep_dclm.py`](run_1_4b_1ep_dclm.py)
- **HF checkpoint (mid-train):** `/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf` (step-14672, ~50% trained)
- **HF checkpoint (final):** pending training completion
- **Levanter checkpoint:** `checkpoints/1_4b_1ep_dclm/tmgu1im8/`
- **Purpose:** 1-epoch (no-repetition) baseline for the matched-token code-mix experiment. Reference point that DOES NOT exhibit looping behavior we saw in the 16-epoch baseline.

**Architecture / Tokenizer:** Same `llama_1_4b` 1.4B Llama and Llama-3.1 tokenizer as above.

**Data:**
- Source: 7 full shards of `dclm_baseline-0206f1` (canonical Marin DCLM tokenization):
  - part-00006 (4.93B), part-00020 (4.95B), part-00026 (5.04B), part-00035 (5.00B), part-00042 (4.94B), part-00047 (5.00B), part-00071 (4.99B)
- Total available: **34.85B unique tokens**
- Mix: 100% DCLM, uniform 1/7 weight across the 7 shards.
- Trained tokens: 30.77B target (≈0.88 epoch over the 34.85B available)
- Eval streams (weight=0): `dclm_200m_val`, all 16 paloma subsets.

**Hyperparameters:**
- AdamW (β₁=0.9, β₂=0.95)
- LR: **3e-4** cosine to 0 (matches OLMo 2 7B; up from konwoo's 1e-3 to reflect 1-epoch regime)
- WD: **0.1** (down from 1.6; phi-1/phi-1.5/OLMo 2/Marin all use 0.1; WD=1.6 was a 16-epoch-specific overfit hack)
- Warmup: 0.01
- Max grad norm: 1.0
- Batch size: 256 sequences × 4096 tokens = 1,048,576 tokens/step (4 nodes × 8 GPUs × `per_device_parallelism=8`)
- Steps: 29,343 → 30.77B trained tokens
- Seq len: 4096
- `block_cross_document_attention=False`, `shuffle=True`, `enforce_eos=True`, `seed=0`, `data_seed=0`
- Hardware: 4-node DP across `gpu-st-p4d24xlarge-1..3 + gpu-dy-p4d24xlarge-1`

Hyperparameter rationale frozen 2026-06-01 (see file header).

---

### B4 — 1.4B 1-epoch code25 mix (`1ep-code25-B4`)

- **Run script:** [`run_1_4b_1ep_code25.py`](run_1_4b_1ep_code25.py)
- **HF checkpoint (mid-train):** `/fsx/users/dongweij/marin/checkpoints/1ep_code25_step14672_hf` (step-14672, ~50% trained)
- **HF checkpoint (final):** pending training completion
- **Levanter checkpoint:** `checkpoints/1_4b_1ep_code25/6zs6ybgt/`
- **Purpose:** Matched-compute counterpart to A5. Same target tokens, same hyperparams, only data mix differs (75% DCLM + 25% code split as Aryabumi paper used).

**Architecture / Tokenizer / Hyperparams:** Identical to A5 above. Only data differs.

**Data:**
- DCLM shards (75%): same 7 shards as A5 with uniform 1/7 weight × 0.75 total.
- Code components (25%, three sub-sources, each ≈1 epoch over its source at final step):
  - `aryabumi_synth`: 17.5% sampling weight → 5.385B trained / 5.4B available
  - `aryabumi_web`: 4.4% sampling weight → 1.354B trained / 1.35B available
  - `opc_algorithmic`: 3.1% sampling weight → 0.954B trained / 0.94B available
- Total trained tokens: 30.77B (matched to A5).
- DCLM portion at target: 23.08B drawn (≈0.66 epoch over the 34.85B available; smaller exposure than A5 by design).

Mix rationale (from script header): "75% DCLM / 17.5% synth / 4.4% web / 3.1% opc (each ~1 epoch)".

---

## 2. External reference models

### microsoft/phi-1 ([paper](https://arxiv.org/abs/2306.11644))

- **HF repo:** `microsoft/phi-1`
- **Parameters:** 1.3B
- **Architecture:** Decoder Transformer, 2048 hidden, 32 heads, 32 layers, 4096 max seq len. Code-only training.

**Data (per paper §2):**
- "CodeTextbook": filtered Stack + StackOverflow code (≈6B unique tokens, selected via GPT-4-annotated random forest quality classifier)
- "CodeExercises": ≈1B unique tokens of GPT-3.5-generated synthetic Python textbook content
- Total unique: ≈7B
- Training: ~8 epochs → ≈50B trained tokens
- Finetuning: additional ~180M tokens of GPT-3.5-generated coding exercises

**Hyperparameters (per paper):** AdamW, fp16, DeepSpeed ZeRO Stage 2. Trained for 4 days on 8 A100s.

**Why we eval it:** code-data-heavy baseline. Reports HumanEval pass@1 = 50.6%, MBPP = 55.5% in the paper. We've replicated HumanEval 54.3% via `bigcode-evaluation-harness` (within paper noise; see EVALUATION.md `‡‡` note).

---

### microsoft/phi-1_5 ([paper](https://arxiv.org/abs/2309.05463))

- **HF repo:** `microsoft/phi-1_5`
- **Parameters:** 1.3B
- **Architecture:** Same as phi-1.

**Data (per paper §2):**
- 7B tokens from phi-1's CodeTextbook
- ~20B newly created synthetic "textbook-like" NLP data via GPT-3.5, covering common-sense reasoning across ≈20K topics
- Total unique: ≈30B
- Training: ≈150B trained tokens (≈5 epochs)
- `phi-1.5-web` variant additionally uses 95B tokens of filtered web data (not used in the HF `phi-1_5` repo we eval).

**Hyperparameters (per paper):** Adam, constant LR 2e-4, weight decay 0.1, batch size 2048, fp16, DeepSpeed ZeRO Stage 2. ≈1.5K A100-80GB hours.

**Why we eval it:** synthetic-NL-textbook-heavy reference. The strongest small model on all of our NL benchmarks; the natural target if we want to try Phi-1.5-style cosmopedia training at our scale.

---

## 3. Training-data sources

For datasets that appear above, what's actually in the cache directory.

### DCLM (Marin canonical)
- **Cache:** `/fsx/users/dongweij/marin/outputs/tokenized/dclm_baseline-0206f1/` (Llama-3.1 vocab)
- **Per-shard structure:** each `train/part-XXXXX/` is a self-contained mini-cache with a `train -> .` symlink. The top-level cache wasn't finalized (missing merged ledger) so wiring is done per-shard.
- **Sizes verified:** 7 full shards (part-00006/20/26/35/42/47/71) totaling **34.85B tokens**; one partial shard (~0.7B); total ≈35.5B available.
- **Used in:** A5, B4 (75% portion).

### DCLM 200M slice
- **Cache:** `/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_train-d321eb` (~209M tokens)
- **Val cache:** `/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_val-415aea`
- **Used in:** 1.4B base (whole), 1.4B code25 v2 (a 150M subsample as `dclm_150m`), and as eval signal in 1ep runs.

### opc_algorithmic (OpenCoder algorithmic split)
- **Cache:** `/fsx/users/dongweij/marin/outputs/tokenized/opc_algorithmic-ffc825/` (≈0.94B tokens)
- **50M subsample cache:** `/fsx/users/dongweij/marin/outputs/tokenized/opc_algorithmic_50m-*/` (54.59M tokens)
- **Source:** OpenCoder "algorithmic" Python code subset.
- **Used in:** 1.4B code25 v2 (50M subsample), B4 (full 0.94B at ≈1 epoch).

### aryabumi_synth (synthetic code/textbook)
- **Cache:** `/fsx/users/dongweij/marin/outputs/tokenized/aryabumi_code_synth_full-0678c3/` (≈5.4B tokens)
- **Source:** synthetic code-mix from Aryabumi et al. ([To Code, or Not To Code?](https://arxiv.org/abs/2408.10914), Meta 2024). **Note:** I have not independently inspected the actual content of this cache vs paper claims; see [paper reading rules in CLAUDE.local.md](../../CLAUDE.local.md). Treat as "what the cache contains, which is supposed to be Aryabumi-style synth" until verified.
- **Sibling cache:** `aryabumi_code_synth_solution-660d21/` exists in the same dir (likely the solution-only slice) but is not used by B4.
- **Used in:** B4 (17.5% sampling weight).

### aryabumi_web (filtered web code)
- **Cache:** `/fsx/users/dongweij/marin/outputs/tokenized/aryabumi_code_web-591a44/` (≈1.35B tokens)
- **Source:** web-filtered code from Aryabumi et al.
- **Used in:** B4 (4.4% sampling weight).

### cosmopedia_v2 (tokenized locally, not yet used)
- **Cache:** `/fsx/users/dongweij/marin/outputs/tokenized/phi_1_5/cosmopedia_v2-21b787/` (111GB on disk; ~27.37B tokens reported in prior session)
- **Source:** HuggingFace `HuggingFaceTB/cosmopedia-v2` — phi-1.5-style GPT-generated NL textbook content.
- **Status:** tokenized, never used in a training run. Available if we choose Strategy A (phi-1.5-style synthetic NL textbook training) — see `next_steps_strategy.md`.

### Paloma (16 held-out subsets)
- **Cache:** `/fsx/users/dongweij/marin/outputs/tokenized/paloma/{subset}-*/`
- **Used in:** continuous eval (validation streams in all our training scripts; weight=0 in train mix).
- **Subsets:** 4chan, c4_100_domains, c4_en, dolma-v1_5, dolma_100_programing_languages, dolma_100_subreddits, falcon-refinedweb, gab, m2d2_s2orc_unsplit, m2d2_wikipedia_unsplit, manosphere_meta_sep, mc4, ptb, redpajama, twitterAAE_HELM_fixed, wikitext_103.

---

## 4. Cross-references

- **Eval results, per-task:** [EVALUATION.md](EVALUATION.md)
- **Chronological run log:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- **Monitoring runbook:** [monitor.md](monitor.md)
- **Strategy / next steps:** [next_steps_strategy.md](next_steps_strategy.md)
- **Paper notes:** [../../papers/reasoning_curriculum.md](../../papers/reasoning_curriculum.md)

## 5. Updating this doc

When a new model is trained: add a §1/§2 entry with run script, checkpoint paths, architecture, data, hyperparams, sources for each. When a new dataset enters the tokenized cache: add a §3 entry with cache path, source, used-in list. Token counts must be either grep-verified from the cache or quoted from the run script that consumed it.
