#!/usr/bin/env bash
# Re-run the multi-shot benchmarks at Open-LLM-Leaderboard n-shot counts.
# Our prior wide-eval ran arc/hellaswag/winogrande at 0-shot (lm-eval defaults
# for those tasks), which is way below the standard. This pulls those numbers
# at the right n-shot to extract every drop of available signal.
#
# Tasks NOT in this rerun (kept from the 0-shot wide eval):
#  - piqa, boolq, sciq, openbookqa, commonsense_qa, social_iqa, logiqa: 0-shot is the standard
#  - mmlu (5-shot already), gsm8k (5-shot already)
#  - humaneval, mbpp, gsm8k_cot, minerva_math: gen tasks, shot count doesn't unblock floor

set -euo pipefail
cd /fsx/users/dongweij/marin

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/leaderboard_shots_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

declare -A CKPTS=(
  [baseline_nocross]=/fsx/users/dongweij/marin/checkpoints/1_4b_wd1_6_x16_nocrossblock_hf
  [code25_alg]=/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_hf
)

run_one() {
  local LABEL="$1" CKPT="$2" TASKS="$3" NSHOT="$4"
  local OUT="$OUT_ROOT/${LABEL}__${NSHOT}shot__$(echo "$TASKS" | tr ',' '_' | cut -c1-30)"
  local LOG="$OUT.log"
  mkdir -p "$OUT"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval \
    --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --num_fewshot "$NSHOT" \
    --batch_size 16 \
    --log_samples \
    --output_path "$OUT" \
    2>&1 | tee "$LOG"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
}

for LABEL in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$LABEL]}"
  # 25-shot: arc_easy, arc_challenge
  run_one "$LABEL" "$CKPT" "arc_easy,arc_challenge" 25
  # 10-shot: hellaswag
  run_one "$LABEL" "$CKPT" "hellaswag" 10
  # 5-shot: winogrande
  run_one "$LABEL" "$CKPT" "winogrande" 5
done

echo "All leaderboard-shot evals complete. Results under $OUT_ROOT/"
