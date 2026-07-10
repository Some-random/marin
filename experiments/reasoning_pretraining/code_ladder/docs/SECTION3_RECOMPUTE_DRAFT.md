# §3 recompute — CORRECTED-NUMBERS DRAFT (read-only; not the §3 fill)

**Generated 2026-07-10 from** `outputs/eval_results/retest_all_20260710_004423` (72 checkpoints × 3 tasks, 0 failures).
This is a REVIEW artifact. The actual §3 EVALUATION.md table fill is **HELD** pending your explicit *"fill the columns"*.

## What changed vs the old suite

| task | OLD (dropped) | NEW (this run) | why |
|---|---|---|---|
| commonsense_qa | letter-scored, 0-shot → ~20% (chance), fired ` A` ~98% | **commonsense_qa_text, 5-shot** (answer-text) | letter-prior collapse; text-scoring reads the answer |
| mmlu | letter-scored, 5-shot → ~24–25% (chance) | **mmlu_text, 0-shot** (answer-text, `all` config) | same letter-prior collapse |
| wsc | binary `wsc.fixed` (couldn't beat majority) | **wsc273, 0-shot** (referent-choice, Marin-aligned) | binary version was a collapse artifact |

**Also dropped** (Collapse, EVALUATION.md §1 G): cb, arc_challenge, logiqa, all Math (gsm8k/gsm8k_cot/minerva_math/gsm_symbolic/gsm_noop), all Aggregate (bbh/mmlu_pro/agieval_lsat_ar/gpqa). **Kept unchanged:** boolq (open-book 0-shot), winogrande (tripwire), sciq/arc_easy/piqa/social_iqa/hellaswag/copa/storycloze/quac/lambada/openbookqa_fact, code, PPL.

## Chance baselines (the OLD letter-scored numbers sat at these)
- commonsense_qa_text: **chance = 20%** (5-way). ✓ = clears chance.
- mmlu_text: **chance = 25%** (4-way). ✓ = clears chance.
- wsc273: **chance = 50%** (2-way). ✓ = clears chance.

## Per-model NEW numbers (all 72 checkpoints, alphabetical)

| checkpoint | csqa_text[5] acc | acc_norm | mmlu_text[0] acc | acc_norm | wsc273[0] acc |
|---|---|---|---|---|---|
| 1_4b_25code_alg_hf | 0.287 ✓ | 0.351 | 0.257 ✓ | 0.270 | 0.513 ✓ |
| 1_4b_25code_alg_v2_hf | 0.247 ✓ | 0.296 | 0.248 · | 0.264 | 0.527 ✓ |
| 1_4b_baseline_hf | 0.269 ✓ | 0.320 | 0.249 · | 0.267 | 0.524 ✓ |
| 1_4b_konwoo_match_hf | 0.330 ✓ | 0.369 | 0.250 · | 0.268 | 0.560 ✓ |
| 1_4b_run_B_hf | 0.182 · | 0.225 | 0.242 · | 0.265 | 0.524 ✓ |
| 1_4b_run_C_hf | 0.183 · | 0.234 | 0.233 · | 0.262 | 0.502 ✓ |
| 1_4b_run_D_hf | 0.183 · | 0.236 | 0.239 · | 0.259 | 0.502 ✓ |
| 1_4b_wd1_6_x16_hf | 0.292 ✓ | 0.340 | 0.244 · | 0.260 | 0.535 ✓ |
| 1_4b_wd1_6_x16_nocrossblock_hf | 0.282 ✓ | 0.339 | 0.250 ✓ | 0.263 | 0.516 ✓ |
| 1_4b_wd3_2_x16_nocrossblock_hf | 0.322 ✓ | 0.364 | 0.247 · | 0.267 | 0.509 ✓ |
| 1ep_code25_final_hf | 0.466 ✓ | 0.508 | 0.286 ✓ | 0.297 | 0.575 ✓ |
| 1ep_code25_step14672_hf | 0.456 ✓ | 0.498 | 0.279 ✓ | 0.287 | 0.593 ✓ |
| 1ep_dclm_final_hf | 0.485 ✓ | 0.523 | 0.290 ✓ | 0.302 | 0.586 ✓ |
| 1ep_dclm_step14672_hf | 0.446 ✓ | 0.482 | 0.284 ✓ | 0.294 | 0.568 ✓ |
| 300m_a5_step22887_hf | 0.292 ✓ | 0.350 | 0.255 ✓ | 0.270 | 0.542 ✓ |
| 300m_a5sp_step22887_hf | 0.251 ✓ | 0.308 | 0.252 ✓ | 0.264 | 0.535 ✓ |
| 300m_c5v2cont_step22887_hf | 0.180 · | 0.251 | 0.240 · | 0.264 | 0.516 ✓ |
| 300m_c5v3_step11443_hf | 0.286 ✓ | 0.334 | 0.252 ✓ | 0.265 | 0.527 ✓ |
| 300m_c5v4_step11443_hf | 0.254 ✓ | 0.315 | 0.251 ✓ | 0.262 | 0.513 ✓ |
| 300m_c5v6_step11443_hf | 0.253 ✓ | 0.318 | 0.253 ✓ | 0.265 | 0.498 · |
| 300m_c5v6_strict_step11443_hf | 0.267 ✓ | 0.320 | 0.251 ✓ | 0.266 | 0.505 ✓ |
| 300m_c5v7_step11443_hf | 0.241 ✓ | 0.301 | 0.250 · | 0.265 | 0.495 · |
| 300m_code_p1_half_step11443_hf | 0.173 · | 0.229 | 0.234 · | 0.259 | 0.502 ✓ |
| 300m_hf | 0.222 ✓ | 0.281 | 0.247 · | 0.262 | 0.513 ✓ |
| 4b_dclm_short_final_hf | 0.490 ✓ | 0.524 | 0.284 ✓ | 0.297 | 0.634 ✓ |
| 600m_a5_step45775_hf | 0.416 ✓ | 0.443 | 0.277 ✓ | 0.285 | 0.542 ✓ |
| 600m_a5sp_step45775_hf | 0.334 ✓ | 0.389 | 0.263 ✓ | 0.276 | 0.546 ✓ |
| 600m_baseline_hf | 0.225 ✓ | 0.281 | 0.243 · | 0.261 | 0.498 · |
| 600m_c5v3_step22887_hf | 0.373 ✓ | 0.423 | 0.263 ✓ | 0.276 | 0.568 ✓ |
| 600m_c5v6_step22887_hf | 0.359 ✓ | 0.409 | 0.262 ✓ | 0.274 | 0.542 ✓ |
| 600m_c5v6_strict_step22887_hf | 0.368 ✓ | 0.429 | 0.261 ✓ | 0.273 | 0.542 ✓ |
| 600m_c5v7_step22887_hf | 0.339 ✓ | 0.392 | 0.262 ✓ | 0.276 | 0.531 ✓ |
| 600m_code_p1_half_step22887_hf | 0.195 · | 0.267 | 0.244 · | 0.264 | 0.513 ✓ |
| 600m_run_C_phase2_hf | 0.177 · | 0.225 | 0.239 · | 0.264 | 0.491 · |
| 600m_run_D_phase2_hf | 0.192 · | 0.251 | 0.239 · | 0.255 | 0.495 · |
| a5_sp_audit_step29343_hf | 0.437 ✓ | 0.473 | 0.280 ✓ | 0.290 | 0.601 ✓ |
| a5_sp_step29343_hf | 0.402 ✓ | 0.450 | 0.271 ✓ | 0.287 | 0.553 ✓ |
| a5_step14672_hf | 0.446 ✓ | 0.482 | 0.284 ✓ | 0.294 | 0.568 ✓ |
| baseline_step10000_hf | 0.300 ✓ | 0.353 | 0.252 ✓ | 0.265 | 0.513 ✓ |
| c5_final_step29343_hf | 0.233 ✓ | 0.305 | 0.247 · | 0.262 | 0.516 ✓ |
| c5_stage1_step14672_hf | 0.229 ✓ | 0.290 | 0.244 · | 0.259 | 0.535 ✓ |
| c5v2_final_step29343_hf | 0.278 ✓ | 0.340 | 0.250 ✓ | 0.268 | 0.502 ✓ |
| c5v2_small_stage1_step6400_hf | 0.183 · | 0.254 | 0.238 · | 0.261 | 0.484 · |
| c5v2_small_step12799_hf | 0.205 ✓ | 0.277 | 0.244 · | 0.263 | 0.498 · |
| c5v2_stage1_step14672_hf | 0.272 ✓ | 0.326 | 0.251 ✓ | 0.266 | 0.524 ✓ |
| c5v3_half_p2_step14671_hf | 0.414 ✓ | 0.454 | 0.277 ✓ | 0.287 | 0.575 ✓ |
| c5v3_p2_a6_step14671_hf | 0.400 ✓ | 0.432 | 0.263 ✓ | 0.273 | 0.505 ✓ |
| c5v3_phase1_step14671_hf | 0.268 ✓ | 0.309 | 0.247 · | 0.265 | 0.520 ✓ |
| c5v3_small_phase1_step6399_hf | 0.188 · | 0.247 | 0.241 · | 0.266 | 0.487 · |
| c5v3_small_phase2_step6399_hf | 0.332 ✓ | 0.387 | 0.261 ✓ | 0.273 | 0.524 ✓ |
| c5v4_p2_audit_step14671_hf | 0.430 ✓ | 0.465 | 0.273 ✓ | 0.281 | 0.590 ✓ |
| c5v4_p2_step14671_hf | 0.425 ✓ | 0.459 | 0.273 ✓ | 0.283 | 0.586 ✓ |
| c5v5_step29343_hf | 0.279 ✓ | 0.342 | 0.253 ✓ | 0.271 | 0.516 ✓ |
| c5v6_phase2_step14671_hf | 0.437 ✓ | 0.484 | 0.277 ✓ | 0.287 | 0.601 ✓ |
| c5v6_strict_step14671_hf | 0.417 ✓ | 0.446 | 0.271 ✓ | 0.281 | 0.557 ✓ |
| c5v6new_final_hf | 0.445 ✓ | 0.467 | 0.280 ✓ | 0.290 | 0.593 ✓ |
| c5v6new_v7_step14671_hf | 0.447 ✓ | 0.479 | 0.278 ✓ | 0.288 | 0.615 ✓ |
| c5v7_final_hf | 0.427 ✓ | 0.456 | 0.276 ✓ | 0.285 | 0.564 ✓ |
| c5v8r_p1_step14671_hf | 0.224 ✓ | 0.279 | 0.241 · | 0.262 | 0.495 · |
| c5v8r_p2_step14671_hf | 0.422 ✓ | 0.444 | 0.276 ✓ | 0.282 | 0.564 ✓ |
| c5v8r_step14671_hf | 0.370 ✓ | 0.431 | 0.269 ✓ | 0.278 | 0.538 ✓ |
| code25b_clean_p2_15bt_step14671_hf | 0.434 ✓ | 0.486 | 0.280 ✓ | 0.292 | 0.557 ✓ |
| code25b_clean_p2_step4767_hf | 0.386 ✓ | 0.437 | 0.267 ✓ | 0.277 | 0.546 ✓ |
| code25b_clean_step23511_hf | 0.263 ✓ | 0.310 | 0.248 · | 0.269 | 0.531 ✓ |
| code25b_step23746_hf | 0.233 ✓ | 0.284 | 0.243 · | 0.262 | 0.549 ✓ |
| kb6hhnxn_mixed_hf | 0.234 ✓ | 0.283 | 0.248 · | 0.259 | 0.535 ✓ |
| run_C_phase2_hf | 0.169 · | 0.226 | 0.240 · | 0.263 | 0.509 ✓ |
| run_D_phase2_hf | 0.220 ✓ | 0.292 | 0.242 · | 0.262 | 0.476 · |
| v1_step10000_hf | 0.262 ✓ | 0.326 | 0.256 ✓ | 0.272 | 0.568 ✓ |
| v2_step10000_hf | 0.261 ✓ | 0.308 | 0.248 · | 0.263 | 0.538 ✓ |
| wd1_6_x8_final_hf | 0.287 ✓ | 0.339 | 0.254 ✓ | 0.266 | 0.553 ✓ |
| wd3_2_x16_step10000_hf | 0.305 ✓ | 0.340 | 0.254 ✓ | 0.264 | 0.549 ✓ |

## Summary
- **commonsense_qa_text acc**: 61/72 models clear chance (20%); range 0.169–0.490, median 0.287.
- **mmlu_text acc**: 43/72 models clear chance (25%); range 0.233–0.290, median 0.252.
- **wsc273 acc**: 62/72 models clear chance (50%); range 0.476–0.634, median 0.535.

## Failures
**None.** All 72 checkpoints produced all 3 tasks.

---
**Next:** on your *"fill the columns"*, I wire these into §3a/§3b/§3c (checkpoint→column), retire the Collapse rows, recompute the Means.