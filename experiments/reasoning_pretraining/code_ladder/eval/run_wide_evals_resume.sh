#!/usr/bin/env bash
# Resume the wide eval run that crashed on the gen-task import (missing
# HF_ALLOW_CODE_EVAL=1). Skips baseline_nocross logprob since it completed.
#
# Reuses the same output root as wide_eval_20260527_1343 so all results
# land in one place.

set -euo pipefail
cd /fsx/users/dongweij/marin
export HF_ALLOW_CODE_EVAL=1   # required by `evaluate` lib for humaneval/mbpp

TASKS_LOGPROB="arc_easy,arc_challenge,sciq,piqa,boolq,hellaswag,winogrande,openbookqa,commonsense_qa,social_iqa,logiqa,mmlu,gsm8k"
TASKS_GEN="humaneval,mbpp,gsm8k_cot,minerva_math"

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/wide_eval_20260527_1343
echo "Output root: $OUT_ROOT"

run_one() {
  local LABEL="$1" CKPT="$2" TASKS="$3" BATCH="$4" EXTRA="$5"
  local OUT="$OUT_ROOT/${LABEL}__$(echo "$TASKS" | tr ',' '_' | cut -c1-40)"
  local LOG="$OUT_ROOT/${LABEL}__$(echo "$TASKS" | tr ',' '_' | cut -c1-40).log"
  mkdir -p "$OUT"
  echo "=== [$LABEL] tasks=$TASKS batch=$BATCH start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval \
    --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH" \
    --log_samples \
    --output_path "$OUT" \
    $EXTRA \
    2>&1 | tee "$LOG"
  echo "=== [$LABEL] tasks=$TASKS DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
}

BASELINE=/fsx/users/dongweij/marin/checkpoints/1_4b_wd1_6_x16_nocrossblock_hf
CODE25=/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_hf

# Baseline logprob: SKIPPED (completed in the prior run).
# Baseline gen tasks:
run_one baseline_nocross "$BASELINE" "$TASKS_GEN" 8 "--confirm_run_unsafe_code"

# Code-mix: both passes.
run_one code25_alg "$CODE25" "$TASKS_LOGPROB" 32 ""
run_one code25_alg "$CODE25" "$TASKS_GEN" 8 "--confirm_run_unsafe_code"

echo "Resume complete. Results under $OUT_ROOT/"
