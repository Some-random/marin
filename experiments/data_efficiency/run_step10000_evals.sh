#!/usr/bin/env bash
# Run the full downstream benchmark suite on step-10000 HF checkpoints of
# baseline_step10000, v1_step10000, v2_step10000 — same settings as the
# step-12,800 eval_fixes runs, so we can compare overfit-final vs near-peak.

cd /fsx/users/dongweij/marin
export HF_ALLOW_CODE_EVAL=1

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/step10000_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

declare -A CKPTS=(
  [baseline_step10000]=/fsx/users/dongweij/marin/checkpoints/baseline_step10000_hf
  [v1_step10000]=/fsx/users/dongweij/marin/checkpoints/v1_step10000_hf
  [v2_step10000]=/fsx/users/dongweij/marin/checkpoints/v2_step10000_hf
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
    $EXTRA > "$LOG" 2>&1 && \
    echo "=== [$LABEL] tasks=$TASKS DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ===" || \
    echo "=== [$LABEL] tasks=$TASKS FAILED-CONTINUE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
}

for LABEL in baseline_step10000 v1_step10000 v2_step10000; do
  M="${CKPTS[$LABEL]}"
  # 25-shot: arc
  run_one "$LABEL" "$M" "arc_easy,arc_challenge" 25 16 ""
  # 10-shot: hellaswag
  run_one "$LABEL" "$M" "hellaswag" 10 16 ""
  # 5-shot: winogrande, mmlu, gsm8k
  run_one "$LABEL" "$M" "winogrande,mmlu,gsm8k" 5 16 ""
  # 0-shot logprob: piqa, boolq, sciq, openbookqa, csqa, siqa, logiqa
  run_one "$LABEL" "$M" "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa" 0 16 ""
  # openbookqa_fact (custom)
  run_one "$LABEL" "$M" "openbookqa_fact" 0 16 ""
  # gsm8k_cot 8-shot
  run_one "$LABEL" "$M" "gsm8k_cot" 8 8 "--confirm_run_unsafe_code"
  # mbpp 3-shot, humaneval 0-shot, minerva 4-shot
  run_one "$LABEL" "$M" "mbpp" 3 8 "--confirm_run_unsafe_code"
  run_one "$LABEL" "$M" "humaneval" 0 8 "--confirm_run_unsafe_code"
  run_one "$LABEL" "$M" "minerva_math" 4 8 ""
done

echo "All step-10000 eval runs complete."
