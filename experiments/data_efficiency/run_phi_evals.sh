#!/usr/bin/env bash
# Run phi-1 and phi-1.5 on the SAME benchmark suite + n-shot settings as our
# 1.4B model runs, for apples-to-apples comparison.
#
# Models:
#   - microsoft/phi-1     (1.3B, ~7B training tokens, code-only)
#   - microsoft/phi-1_5   (1.3B, ~30B training tokens, code + NL)
#
# Settings match our `run_leaderboard_shots.sh` + `run_wide_evals.sh`:
#   25-shot: arc_easy, arc_challenge
#   10-shot: hellaswag
#   5-shot:  winogrande, mmlu, gsm8k
#   0-shot:  piqa, boolq, sciq, openbookqa, commonsense_qa, social_iqa, logiqa
#   gen:     humaneval, mbpp, gsm8k_cot, minerva_math

set -euo pipefail
cd /fsx/users/dongweij/marin
export HF_ALLOW_CODE_EVAL=1

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/phi_evals_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

# Map: HF model id -> short label for output dirs
declare -A MODELS=(
  [phi-1]=microsoft/phi-1
  [phi-1.5]=microsoft/phi-1_5
)

run_one() {
  local LABEL="$1" MODEL_ID="$2" TASKS="$3" NSHOT="$4" BATCH="$5" EXTRA="$6"
  local OUT="$OUT_ROOT/${LABEL}__${NSHOT}shot__$(echo "$TASKS" | tr ',' '_' | cut -c1-30)"
  local LOG="$OUT.log"
  mkdir -p "$OUT"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_ID,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --num_fewshot "$NSHOT" \
    --batch_size "$BATCH" \
    --log_samples \
    --output_path "$OUT" \
    $EXTRA \
    2>&1 | tee "$LOG"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
}

for LABEL in "${!MODELS[@]}"; do
  MODEL_ID="${MODELS[$LABEL]}"
  # 25-shot
  run_one "$LABEL" "$MODEL_ID" "arc_easy,arc_challenge" 25 16 ""
  # 10-shot
  run_one "$LABEL" "$MODEL_ID" "hellaswag" 10 16 ""
  # 5-shot
  run_one "$LABEL" "$MODEL_ID" "winogrande,mmlu,gsm8k" 5 16 ""
  # 0-shot logprob (lm-eval default for these)
  run_one "$LABEL" "$MODEL_ID" "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa" 0 16 ""
  # Generation tasks (smaller batch, code eval enabled)
  run_one "$LABEL" "$MODEL_ID" "humaneval,mbpp,gsm8k_cot,minerva_math" 0 8 "--confirm_run_unsafe_code"
done

echo "All phi evals complete. Results under $OUT_ROOT/"
