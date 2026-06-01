#!/usr/bin/env bash
# Run the full downstream benchmark suite on both step-10000 and final-step
# (step-12,799) of wd=3.2/x16 — same tasks/shots as the step-10000 baseline+v1+v2
# eval suite so all results land in EVALUATION.md with consistent settings.

cd /fsx/users/dongweij/marin
export HF_ALLOW_CODE_EVAL=1

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/wd3_2_x16_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

declare -A CKPTS=(
  [wd3_2_x16_step10000]=/fsx/users/dongweij/marin/checkpoints/wd3_2_x16_step10000_hf
  [wd3_2_x16_final]=/fsx/users/dongweij/marin/checkpoints/1_4b_wd3_2_x16_nocrossblock_hf
)

run_one() {
  local LABEL="$1" CKPT="$2" TASKS="$3" NSHOT="$4" BATCH="$5" EXTRA="$6"
  local OUT="$OUT_ROOT/${LABEL}__${NSHOT}shot__$(echo "$TASKS" | tr ',' '_' | cut -c1-30)"
  local LOG="$OUT.log"
  mkdir -p "$OUT"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" --num_fewshot "$NSHOT" --batch_size "$BATCH" \
    --log_samples --output_path "$OUT" \
    --include_path /fsx/users/dongweij/marin/experiments/data_efficiency \
    --trust_remote_code \
    $EXTRA > "$LOG" 2>&1 && \
    echo "=== [$LABEL] tasks=$TASKS DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ===" || \
    echo "=== [$LABEL] tasks=$TASKS FAILED-CONTINUE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
}

for LABEL in wd3_2_x16_step10000 wd3_2_x16_final; do
  M="${CKPTS[$LABEL]}"
  run_one "$LABEL" "$M" "arc_easy,arc_challenge" 25 16 ""
  run_one "$LABEL" "$M" "hellaswag" 10 16 ""
  run_one "$LABEL" "$M" "winogrande,mmlu,gsm8k" 5 16 ""
  run_one "$LABEL" "$M" "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa" 0 16 ""
  run_one "$LABEL" "$M" "openbookqa_fact" 0 16 ""
  run_one "$LABEL" "$M" "gsm8k_cot" 8 8 "--confirm_run_unsafe_code"
  run_one "$LABEL" "$M" "mbpp" 3 8 "--confirm_run_unsafe_code"
  run_one "$LABEL" "$M" "humaneval" 0 8 "--confirm_run_unsafe_code"
  run_one "$LABEL" "$M" "minerva_math" 4 8 ""
done

echo "All wd=3.2/x16 eval runs complete."
