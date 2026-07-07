# Aryabumi Replication Plan — 1.4B scale

**Last updated**: 2026-05-25

## What we're testing

Aryabumi et al. (2408.10914) reported that at 470M / 2.8B parameter scale on 200B-token pretraining budgets:
- Adding 25% code to text gives +8.2% NL reasoning, +4.2% world knowledge, +6.6% generative win-rates, 12× code
- Synthetic code (only 10% of code budget) gives +9% NL reasoning and +44.9% code vs web-code-only baseline
- 100% code DROPS NL reasoning by 18.3% (vs 0% code) and world knowledge by 86% (vs 0% code)
- Pareto-best recipe is `balanced→text`: 50/50 code/text initialization then text-only continuation

**Our question**: do any of these effects hold at our scale (1.4B params, ~3.34B total training tokens, ~209M-token base data), achieving all three project objectives simultaneously — better reasoning, no NL regression, improved data efficiency?

## Caveats on scale

Aryabumi's setup is **~60× larger in total training tokens** than ours (200B vs 3.34B). It is genuinely unknown whether his recipe effects show up at our token budget. Negative results at our scale would NOT refute Aryabumi — they'd just show his effects require more compute than we're spending.

## Experimental design

All experiments share the same backbone of our wd=1.6/x16/block=False non-looping recipe, modifying only data composition:

| Run | Data mix | Tests |
|---|---|---|
| **Baseline (already done)** | 100% DCLM (`konwoo/dclm-164k-docs-train`) | Reference point. wandb: `divine-dream-99` (`iue9to5a`) for block=True, `peach-thunder-100` (`6xx0hu3l`) for block=False |
| **A. 25code_web** | 75% DCLM + 25% multi-language web code | Does Aryabumi's "25% sweet spot" replicate at our scale? Pure code (no NL pairing). |
| **B. 25code_synth_full** | 75% DCLM + 25% OpenCodeReasoning (problem + reasoning + solution) | Does code+NL paired data work even better? Tests Phi-style hypothesis. NOT a direct Aryabumi replication — Aryabumi's synthetic code was pure code, not paired. |

**Discriminating comparisons** (what each pairwise difference tells us):
- A vs Baseline: does *any* code help at our scale? (Aryabumi predicts yes: +8.2% NL reasoning)
- B vs A: does NL-paired code help more than pure code? (Phi predicts yes; Waheed's SFT result suggests NL-only paraphrase loses the benefit — TBD at pretraining scale)
- A vs Baseline on world-knowledge: does code add or hurt world knowledge? (Aryabumi: +4.2% at 25%)
- A vs Baseline on math/CoT: does code help GSM8K / BBH? (Aryabumi tests this implicitly via composite; we test it directly)

## Hyperparameters (identical across A and B)

| Param | Value |
|---|---|
| Model | `1_4b4k` (1.4B Llama, 16 layers, 2048 hidden, 16 heads, 8 KV heads) |
| Training data | 75% DCLM + 25% code source |
| Tokenizer | meta-llama/Meta-Llama-3.1-8B |
| LR | 1e-3 cosine to 0 |
| WD | 1.6 |
| min_lr_ratio | 0.0 |
| β₁, β₂ | 0.9, 0.95 |
| warmup | 0.01 |
| max_grad_norm | 1.0 |
| Batch size | 64 |
| Seq len | 4096 |
| `num_train_steps` | 12800 (≈3.34B tokens total) |
| seed | 0 |
| data_seed | 0 |
| block_cross_document_attention | False |
| `stop_strategy` | restart |

Note on data ratios with `stop_strategy=restart`:
- 25% × 3.34B = 836M code tokens consumed
- code_web has ~1.2B tokens (5 parquets); code_synth_full has ~2-3B tokens (30 parquets)
- Code source will NOT need to cycle to maintain 25% over the run
- DCLM (209M) WILL cycle ~12× to provide 75% × 3.34B = 2.5B text tokens
- This roughly matches our wd=1.6/x16 baseline (16 cycles of 209M = 3.34B), with code added

## Evaluation

Same set as our existing baseline runs (`run_1_4b_wd1_6_x16.py`), giving direct A/B comparison:
- During training: dclm_200m, dclm_200m_val, 16 Paloma subsets (Paloma macro for NL PPL)
- After training (HF-converted checkpoint, via `run_comprehensive_evals.py`):
  - **Reasoning quality**: gsm8k_cot (math), BBH (multi-task reasoning), HumanEval + MBPP (code generation), Minerva MATH, MathQA
  - **General NL**: ARC-E, ARC-C, MMLU, PIQA, HellaSwag, WinoGrande, SciQ, OpenBookQA, BoolQ, BLiMP, SocialIQA
  - **Generation diagnostics**: gsm8k_cot with `log_samples=True` (does the model loop? Coherent reasoning?)

## What success looks like (for each objective)

Per the three project objectives:

1. **Data efficiency**: Mixed run (A or B) achieves lower Paloma macro loss than baseline text-only at the same compute (same step count). Or matches Paloma at fewer effective text tokens.
2. **Reasoning quality**: GSM8K-CoT or HumanEval pass@1 strictly higher than text-only baseline AND no loop regression.
3. **General NL**: ARC/MMLU/PIQA/HellaSwag aggregate not lower (within noise, say within 1%) than text-only baseline.

If a run hits all three → it's a working H1 candidate. If any of the three regresses, it's not.

Aryabumi's prediction (extrapolated from his 470M/2.8B results):
- Run A: should achieve all three (+8.2% NL reasoning, +4.2% world knowledge, 12× code)
- Run B: untested by Aryabumi; speculative whether code+NL pairing improves or hurts

## Risks and what would refute Aryabumi at our scale

If both A and B fail to improve any objective vs baseline:
- Possible: scale-dependent effect (Aryabumi's gains require >> 3.34B tokens to manifest)
- Possible: our data subset (DCLM 200m) is qualitatively different from his SlimPajama 503B → different baseline dynamics
- Either way, document and consider scaling up training budget for a real test

If A loses NL but B doesn't → suggests code-NL pairing is what matters; Phi-style hypothesis confirmed at our scale

## Status of preparation

**Done**:
- [x] OpenCodeReasoning downloaded (30 parquets, 2.6 GB raw)
- [x] code_python already on disk (5 of 880 parquets, 1.7 GB raw)
- [x] Training scripts written: `run_1_4b_25code_web.py`, `run_1_4b_25code_synth_full.py`
- [x] Tokenization step written: `code_data.py`

**In progress**:
- [ ] JSONL conversion (`/tmp/convert_to_jsonl.py` running in background — produces `code_web.jsonl.gz`, `code_synth_solution.jsonl.gz`, `code_synth_full.jsonl.gz`)
- [ ] Tokenization (will run after JSONL via `code_data.py`)

**Blocked on completion of above**:
- [ ] Verification: tokenized output has expected token counts (~1.2B for code_web, ~1.5-3B for code_synth_full)
- [ ] Smoke test: load tokenized cache, verify can be opened by Levanter

**Then ready to launch** (waiting for user approval):
- Run A: `run_1_4b_25code_web.py` (~8h)
- Run B: `run_1_4b_25code_synth_full.py` (~8h)

Total compute needed: ~16h of 8× A100-40GB. Can run sequentially on our node, or one on Slurm if auto-scale gets fixed.

## Open questions for user

1. **Run order**: A then B sequential, or kick off both via separate GPU jobs? Latter only works if Slurm auto-scale gets fixed.
2. **Recipe to use**: wd=1.6 / x16 (our non-looping baseline). Should we also try wd=3.2 / x8 (Aryabumi's effective epoch count is ~1 since he has 200B tokens; our x8 vs x16 is a separate axis)?
3. **Should we also add an `aryabumi_code_synth_solution`** variant (pure code from OpenCodeReasoning, no NL)? This would isolate "code quality" from "NL pairing" — `solution`-only is a direct Aryabumi analogue (verified Python problems), while `_full` adds NL.

