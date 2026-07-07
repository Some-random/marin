#!/usr/bin/env bash
# Cleanup re-runs after the main chain:
#   A. v2_final 4 batches that failed on the remote node (HF rate-limit + code_eval cache race)
#   B. wd1_6_x8_final 2 batches that failed on this node (Lustre flush race + stale GPU mem)
#   C. paloma_falcon-refinedweb for all 7 models with smaller batch (OOM at batch=16)
#   D. gsm8k_cot loop count on wd1_6_x8_final (limit=20, log_samples) — for looping investigation
# Runs on whichever node it's launched on; assumes HF caches and checkpoints already on /fsx.

cd /fsx/users/dongweij/marin
# HF_HOME unset — use default ~/.cache so token lookup works
export HF_TOKEN=$(cat /fsx/users/dongweij/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/cleanup_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

run_lm_eval() {
  local LABEL="$1" CKPT="$2" TASKS="$3" NSHOT="$4" BATCH="$5" EXTRA="$6" LIMIT="$7"
  local OUT="$OUT_ROOT/${LABEL}__${NSHOT}shot__$(echo "$TASKS" | tr ',' '_' | cut -c1-30)"
  mkdir -p "$OUT"
  local LIMIT_FLAG=""
  [ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT batch=$BATCH start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" --num_fewshot "$NSHOT" --batch_size "$BATCH" \
    --log_samples --output_path "$OUT" \
    --include_path /fsx/users/dongweij/marin/experiments/reasoning_pretraining/code_ladder/eval \
    --trust_remote_code \
    $LIMIT_FLAG $EXTRA > "$OUT.log" 2>&1 && \
    echo "=== [$LABEL] tasks=$TASKS DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ===" || \
    echo "=== [$LABEL] tasks=$TASKS FAILED-CONTINUE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
}

V2_FINAL_HF=/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_v2_hf
WD1_6_X8_HF=/fsx/users/dongweij/marin/checkpoints/wd1_6_x8_final_hf

# === A. v2_final failed batches ===
echo "=== A. v2_final failed-batch reruns start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
run_lm_eval "v2_final" "$V2_FINAL_HF" "winogrande,mmlu,gsm8k" 5 16 "" ""
run_lm_eval "v2_final" "$V2_FINAL_HF" "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa" 0 16 "" ""
run_lm_eval "v2_final" "$V2_FINAL_HF" "mbpp" 3 8 "--confirm_run_unsafe_code" ""
run_lm_eval "v2_final" "$V2_FINAL_HF" "humaneval" 0 8 "--confirm_run_unsafe_code" ""

# === B. wd1_6_x8_final failed batches ===
echo "=== B. wd1_6_x8_final failed-batch reruns start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
run_lm_eval "wd1_6_x8_final" "$WD1_6_X8_HF" "arc_easy,arc_challenge" 25 16 "" ""
run_lm_eval "wd1_6_x8_final" "$WD1_6_X8_HF" "winogrande,mmlu,gsm8k" 5 16 "" ""

# === C. paloma_falcon-refinedweb at batch=4 for all 7 missing-falcon models ===
echo "=== C. paloma_falcon-refinedweb reruns at batch=4 start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
declare -A FALCON_CKPTS=(
  [baseline_s10000]=/fsx/users/dongweij/marin/checkpoints/baseline_step10000_hf
  [v1_s10000]=/fsx/users/dongweij/marin/checkpoints/v1_step10000_hf
  [v2_s10000]=/fsx/users/dongweij/marin/checkpoints/v2_step10000_hf
  [v2_final]=/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_v2_hf
  [wd3_2_x16_s10000]=/fsx/users/dongweij/marin/checkpoints/wd3_2_x16_step10000_hf
  [wd3_2_x16_final]=/fsx/users/dongweij/marin/checkpoints/1_4b_wd3_2_x16_nocrossblock_hf
  [wd1_6_x8_final]=/fsx/users/dongweij/marin/checkpoints/wd1_6_x8_final_hf
)
for LABEL in baseline_s10000 v1_s10000 v2_s10000 v2_final wd3_2_x16_s10000 wd3_2_x16_final wd1_6_x8_final; do
  CKPT="${FALCON_CKPTS[$LABEL]}"
  OUT="$OUT_ROOT/${LABEL}__paloma_falcon-refinedweb"
  mkdir -p "$OUT"
  echo "  [$LABEL/falcon] start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')"
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks paloma_falcon-refinedweb --batch_size 4 --output_path "$OUT" \
    --trust_remote_code > "$OUT.log" 2>&1 && \
    echo "  [$LABEL/falcon] DONE" || echo "  [$LABEL/falcon] FAILED-CONTINUE"
done

# === D. wd1_6_x8_final gsm8k_cot looping count (limit=20, log_samples) ===
echo "=== D. wd1_6_x8_final looping smoke test start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
OUT="$OUT_ROOT/wd1_6_x8_final__loopcheck_gsm8k_cot"
mkdir -p "$OUT"
.venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
  -m lm_eval --model hf \
  --model_args "pretrained=$WD1_6_X8_HF,dtype=bfloat16,trust_remote_code=True" \
  --tasks gsm8k_cot --num_fewshot 8 --batch_size 4 --limit 20 \
  --log_samples --output_path "$OUT" \
  --trust_remote_code > "$OUT.log" 2>&1 && \
  echo "  [wd1_6_x8_final/loop] DONE" || echo "  [wd1_6_x8_final/loop] FAILED-CONTINUE"

echo "=== CLEANUP COMPLETE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
