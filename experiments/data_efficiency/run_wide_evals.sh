#!/usr/bin/env bash
# Wide lm-eval suite on both 1.4B block=False checkpoints, 8-GPU data-parallel.
#
# Compares:
#   - 1_4b_wd1_6_x16_nocrossblock_hf  (peach-thunder-100 / 6xx0hu3l, 0% code)
#   - 1_4b_25code_alg_hf              (eager-grass-104 / p2n84bo3, 25% code-mix)
#
# Both use block_cross_document_attention=False, WD=1.6, x16, LR=1e-3 cosine.
# Only intentional difference: train_weights {dclm:1.0} vs {dclm:0.75, opc_alg:0.25}.
#
# Data-parallel via `accelerate launch --num_processes 8` — each GPU loads a full
# copy of the 1.4B model (fits in 40GB) and processes 1/8 of the requests.

set -euo pipefail

cd /fsx/users/dongweij/marin

# HumanEval/MBPP use the `evaluate` lib's code_eval metric which requires this
# env var (separate from lm-eval's --confirm_run_unsafe_code flag). Without it
# the humaneval task crashes at IMPORT time, before any work runs.
export HF_ALLOW_CODE_EVAL=1

# Logprob tasks (multiple-choice / yes-no): fast, big batches OK.
TASKS_LOGPROB="arc_easy,arc_challenge,sciq,piqa,boolq,hellaswag,winogrande,openbookqa,commonsense_qa,social_iqa,logiqa,mmlu,gsm8k"

# Generation tasks (free generation): slower per-example. Run as a separate
# invocation with smaller batch to avoid KV-cache pressure.
#   - HumanEval / MBPP: code generation. Require --confirm_run_unsafe_code
#     because lm-eval executes generated code to score pass@1.
#   - gsm8k_cot: 8-shot CoT math (also tells us if the model still avoids looping).
#   - minerva_math: competition math, ~12k problems — full run, no limit.
TASKS_GEN="humaneval,mbpp,gsm8k_cot,minerva_math"

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/wide_eval_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

declare -A CKPTS=(
  [baseline_nocross]=/fsx/users/dongweij/marin/checkpoints/1_4b_wd1_6_x16_nocrossblock_hf
  [code25_alg]=/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_hf
)

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

for LABEL in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$LABEL]}"
  # Logprob: bs=32 per device → effective 256.
  run_one "$LABEL" "$CKPT" "$TASKS_LOGPROB" 32 ""
  # Generation: bs=8 per device (KV-cache pressure) + flag to allow code execution.
  run_one "$LABEL" "$CKPT" "$TASKS_GEN" 8 "--confirm_run_unsafe_code"
done

echo "All evals complete. Results under $OUT_ROOT/"
echo "Per-task aggregates in <label>__<tasks>/results_*.json"
echo "Per-example samples in <label>__<tasks>/samples_*.jsonl"
